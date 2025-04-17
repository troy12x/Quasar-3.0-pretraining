import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from native_sparse_attention import NativeSparseAttention

# Check if Flash Attention is available
TRY_FLASH_ATTENTION = True
HAS_FLASH_ATTENTION = False
if TRY_FLASH_ATTENTION:
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
        HAS_FLASH_ATTENTION = True
        print("Flash Attention is available and will be used for faster training")
    except ImportError:
        print("Flash Attention not available, falling back to standard attention")

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to queries and keys.
    
    This implementation is designed to be robust to dimension mismatches, which can occur
    when using different model configurations or when integrating with DeepSpeed.
    
    Args:
        q: Query tensor of shape [batch, seq, heads, head_dim]
        k: Key tensor of shape [batch, seq, heads, head_dim]
        cos: Cosine tensor from precomputed_freqs_cis
        sin: Sine tensor from precomputed_freqs_cis
        
    Returns:
        Tuple of rotary position embedded query and key tensors
    """
    # Get dimensions
    batch_size, seq_len, num_heads, head_dim = q.shape
    
    # Handle different cos/sin tensor shapes
    if cos.dim() == 2:
        # Shape [seq_len, dim]
        cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
        sin = sin.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1, dim]
    elif cos.dim() == 3:
        # Shape [batch, seq_len, dim]
        cos = cos.unsqueeze(2)  # [batch, seq_len, 1, dim]
        sin = sin.unsqueeze(2)  # [batch, seq_len, 1, dim]
    
    # Ensure the sequence length matches
    if cos.size(1) < seq_len:
        # Pad if needed (should be rare)
        pad_len = seq_len - cos.size(1)
        cos_padding = cos[:, -1:].expand(-1, pad_len, -1, -1)
        sin_padding = sin[:, -1:].expand(-1, pad_len, -1, -1)
        cos = torch.cat([cos, cos_padding], dim=1)
        sin = torch.cat([sin, sin_padding], dim=1)
    elif cos.size(1) > seq_len:
        # Truncate if needed
        cos = cos[:, :seq_len]
        sin = sin[:, :seq_len]
    
    # Ensure the feature dimension matches
    feature_dim = cos.size(-1)
    if feature_dim < head_dim:
        # Pad with zeros if needed
        pad_dim = head_dim - feature_dim
        cos = F.pad(cos, (0, pad_dim))
        sin = F.pad(sin, (0, pad_dim))
    elif feature_dim > head_dim:
        # Truncate if needed
        cos = cos[..., :head_dim]
        sin = sin[..., :head_dim]
    
    # Expand to match batch size and number of heads
    if cos.size(0) == 1 and batch_size > 1:
        cos = cos.expand(batch_size, -1, -1, -1)
        sin = sin.expand(batch_size, -1, -1, -1)
    
    if cos.size(2) == 1 and num_heads > 1:
        cos = cos.expand(-1, -1, num_heads, -1)
        sin = sin.expand(-1, -1, num_heads, -1)
    
    # Apply rotary embeddings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the frequency tensor for complex exponentials (cos, sin)."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs)
    cos, sin = torch.cos(freqs), torch.sin(freqs)
    return cos.float(), sin.float()

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # Store config for later reference
        self.hidden_size = config.hidden_size  # d
        self.num_heads = config.num_attention_heads  # nh
        self.head_dim = config.head_dim  # dh
        self.kv_compressed_dim = config.kv_compressed_dim  # dc
        self.query_compressed_dim = config.query_compressed_dim  # d'c
        self.rope_dim_per_head = config.rope_dim_per_head  # dRh
        
        # Projection matrices
        self.W_DKV = nn.Linear(self.hidden_size, self.kv_compressed_dim)
        self.W_UK = nn.Linear(self.kv_compressed_dim, self.num_heads * self.head_dim)
        self.W_UV = nn.Linear(self.kv_compressed_dim, self.num_heads * self.head_dim)
        
        # Query projection
        if self.query_compressed_dim is not None:
            self.W_DQ = nn.Linear(self.hidden_size, self.query_compressed_dim)
            self.W_UQ = nn.Linear(self.query_compressed_dim, self.num_heads * self.head_dim)
        else:
            self.W_Q = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        
        # Rotary projections if using partial rotary
        if self.rope_dim_per_head > 0 and self.rope_dim_per_head < self.head_dim:
            self.W_QR = nn.Linear(self.head_dim, self.rope_dim_per_head, bias=False)
            self.W_KR = nn.Linear(self.head_dim, self.rope_dim_per_head, bias=False)
        else:
            self.W_QR = None
            self.W_KR = None
        
        # Output projection
        self.W_O = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        # Dropout
        self.attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        
        # Precompute freqs for RoPE
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.max_seq_length = 0
        
        # TTM flag - only used for loss modulation, not to replace MLA
        self.use_ttm = getattr(config, 'use_ttm', False)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None, token_temperatures=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Initialize position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).unsqueeze(0)
            position_ids = position_ids.expand(batch_size, -1)
        
        # Initialize or extend position_ids based on past_key_value
        past_length = 0
        if past_key_value is not None:
            # past_key_value[0] shape: [batch_size, num_heads, seq_length, head_dim] or [batch_size, seq_length, num_heads, head_dim]
            if past_key_value[0].dim() >= 3:
                if past_key_value[0].dim() == 4 and past_key_value[0].shape[1] == self.num_heads:
                    # [batch_size, num_heads, seq_length, head_dim] format
                    past_length = past_key_value[0].shape[2]
                else:
                    # [batch_size, seq_length, ...] format
                    past_length = past_key_value[0].shape[1]
                
                # Ensure position_ids are correctly sliced
                if position_ids.shape[1] > seq_length:
                    position_ids = position_ids[:, -seq_length:]
        
        # Compute key and value
        c_kv = self.W_DKV(hidden_states)  # [batch, seq, dc]
        k = self.W_UK(c_kv)  # [batch, seq, nh*dh]
        v = self.W_UV(c_kv)  # [batch, seq, nh*dh]
        
        # Compute query
        if hasattr(self, 'W_DQ') and hasattr(self, 'W_UQ'):
            c_q = self.W_DQ(hidden_states)  # [batch, seq, d'c]
            q = self.W_UQ(c_q)  # [batch, seq, nh*dh]
        else:
            q = self.W_Q(hidden_states)  # [batch, seq, nh*dh]
        
        # Reshape q, k, v for multi-head attention
        q = q.view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply rotary position embeddings
        if self.rope_dim_per_head > 0:
            # Precompute freqs_cis if not already done
            if self.cos is None or self.sin is None or self.max_seq_length < seq_length + past_length:
                self.max_seq_length = max(self.max_seq_length, seq_length + past_length)
                # Only apply RoPE to a subset of dimensions if specified
                dim = min(self.rope_dim_per_head, self.head_dim)
                self.cos, self.sin = precompute_freqs_cis(dim, self.max_seq_length * 2)
                self.cos = self.cos.to(hidden_states.device)
                self.sin = self.sin.to(hidden_states.device)
            
            # Get the appropriate part of position embeddings
            # Ensure position_ids are within bounds
            max_pos = self.cos.size(0) - 1
            safe_pos_ids = torch.clamp(position_ids, 0, max_pos)
            
            # Use gather instead of index_select for better handling of batched position_ids
            # This handles both 1D and 2D position_ids correctly
            if safe_pos_ids.dim() == 2:
                # For batched position_ids [batch_size, seq_length]
                cos_seq = self.cos[None, :, :].expand(batch_size, -1, -1)
                sin_seq = self.sin[None, :, :].expand(batch_size, -1, -1)
                
                # Create indices for gather
                batch_indices = torch.arange(batch_size, device=safe_pos_ids.device)[:, None]
                batch_indices = batch_indices.expand(-1, seq_length)
                
                # Gather using the position_ids
                cos = cos_seq.gather(1, safe_pos_ids.unsqueeze(-1).expand(-1, -1, cos_seq.size(-1)))
                sin = sin_seq.gather(1, safe_pos_ids.unsqueeze(-1).expand(-1, -1, sin_seq.size(-1)))
            else:
                # For single position_ids [seq_length]
                cos = self.cos[safe_pos_ids]
                sin = self.sin[safe_pos_ids]
                # Reshape to [1, seq_length, dim]
                cos = cos.unsqueeze(0)
                sin = sin.unsqueeze(0)
            
            # Convert to query/key dtype
            cos = cos.to(q.dtype)
            sin = sin.to(q.dtype)
            
            # Apply partial rotary if needed
            if self.W_QR is not None and self.W_KR is not None:
                # Project q and k to lower dimension for RoPE
                q_flat = q.reshape(-1, self.head_dim)
                k_flat = k.reshape(-1, self.head_dim)
                
                q_r = self.W_QR(q_flat).reshape(batch_size, seq_length, self.num_heads, self.rope_dim_per_head)
                k_r = self.W_KR(k_flat).reshape(batch_size, seq_length, self.num_heads, self.rope_dim_per_head)
                
                # Apply rotary embeddings to the projected dimensions
                q_r, k_r = apply_rotary_pos_emb(q_r, k_r, cos, sin)
                
                # Combine the rotary and non-rotary parts
                q_rotary = q_r.reshape(-1, self.rope_dim_per_head)
                k_rotary = k_r.reshape(-1, self.rope_dim_per_head)
                
                # Use the original tensors but replace the rotary part
                q_flat[:, :self.rope_dim_per_head] = q_rotary
                k_flat[:, :self.rope_dim_per_head] = k_rotary
                
                # Reshape back
                q = q_flat.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
                k = k_flat.reshape(batch_size, seq_length, self.num_heads, self.head_dim)
                
                # Use the modified q and k as q_r and k_r
                q_r, k_r = q, k
            else:
                # Apply full rotary embeddings
                q_r, k_r = apply_rotary_pos_emb(q, k, cos, sin)
        else:
            q_r, k_r = q, k
        
        # Extend k, v with past_key_value if provided
        if past_key_value is not None:
            k_r = torch.cat([past_key_value[0], k_r], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        
        # Save current k, v for future use
        current_key_value = (k_r, v)
        
        # Transpose for attention computation
        q_r = q_r.transpose(1, 2)  # [batch, num_heads, seq_length, head_dim]
        k_r = k_r.transpose(1, 2)  # [batch, num_heads, kv_seq_length, head_dim]
        v = v.transpose(1, 2)   # [batch, num_heads, kv_seq_length, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(q_r, k_r.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand mask to match attention_scores dimensions
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.to(attention_scores.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores + attention_mask
        
        # Apply token temperature modulation if provided
        if token_temperatures is not None:
            # Reshape token_temperatures to match attention_scores
            temps = token_temperatures.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, seq]
            # Apply temperature scaling to attention scores
            attention_scores = attention_scores * temps
        
        # Convert scores to probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        
        # Apply attention dropout
        attention_probs = F.dropout(attention_probs, p=self.attention_dropout, training=self.training)
        
        # Compute context vectors - ensure matching dtype
        # Convert attention_probs to v's dtype to avoid mismatch
        attention_probs = attention_probs.to(dtype=v.dtype)
        context = torch.matmul(attention_probs, v)
        context = context.transpose(1, 2).contiguous()  # [batch, seq_length, num_heads, head_dim]
        
        # Reshape context back to [batch, seq, hidden_size]
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Apply output projection
        output = self.W_O(context)
        
        return output, current_key_value

class NSAAttention(nn.Module):
    """
    Native Sparse Attention implementation for Quasar model.
    Integrates with the existing architecture while using the NSA mechanism.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim)
        
        # Native Sparse Attention mechanism
        self.nsa = NativeSparseAttention(config)
        
        # Precompute freqs for RoPE
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self.rope_dim_per_head = config.rope_dim_per_head
        
        # Attention dropout for regularization
        self.attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Initialize or extend position_ids
        if position_ids is None:
            if past_key_value is None:
                position_ids = torch.arange(seq_length, device=hidden_states.device)
            else:
                position_ids = torch.arange(past_key_value[0].size(1), past_key_value[0].size(1) + seq_length, device=hidden_states.device)
        
        # Initialize RoPE freq cache if needed
        if self.cos is None or self.cos.size(0) < position_ids.max() + 1:
            max_seq_len = max(position_ids.max() + 1, self.config.max_position_embeddings)
            self.cos, self.sin = precompute_freqs_cis(self.rope_dim_per_head, max_seq_len)
            self.cos = self.cos.to(hidden_states.device)
            self.sin = self.sin.to(hidden_states.device)
        
        # Project to query, key, value
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply RoPE to query and key
        cos = self.cos[position_ids]
        sin = self.sin[position_ids]
        q, k = apply_rotary_pos_emb(
            q.transpose(1, 2),  # [batch, heads, seq, head_dim]
            k.transpose(1, 2),  # [batch, heads, seq, head_dim]
            cos.unsqueeze(0).unsqueeze(0),  # [1, 1, seq, head_dim]
            sin.unsqueeze(0).unsqueeze(0),  # [1, 1, seq, head_dim]
        )
        q = q.transpose(1, 2)  # [batch, seq, heads, head_dim]
        k = k.transpose(1, 2)  # [batch, seq, heads, head_dim]
        
        # Apply Native Sparse Attention
        output, past_key_value = self.nsa(q, k, v, attention_mask, position_ids, past_key_value)
        
        # Apply attention dropout
        output = F.dropout(output, p=self.attention_dropout, training=self.training)
        
        return output, past_key_value

class TokenTemperatureModulation(nn.Module):
    """
    Token Temperature Modulation (TTM) mechanism for enhancing attention to important tokens.
    This is implemented as a helper for Multi-Head Latent Attention, not as a replacement.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.use_ttm = getattr(config, 'use_ttm', False)
        
        # Temperature calculation networks
        self.frequency_scorer = nn.Linear(self.hidden_size, 1)
        self.position_scorer = nn.Linear(self.hidden_size, 1)
        self.context_scorer = nn.Linear(self.hidden_size, 1)
        
        # Weights for different components
        self.alpha = getattr(config, 'ttm_alpha', 0.5)  # Frequency weight
        self.beta = getattr(config, 'ttm_beta', 0.3)    # Position weight
        self.gamma = getattr(config, 'ttm_gamma', 0.2)  # Context weight
        
        # Initialize with proper scale
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with small weights to avoid disrupting attention at start
        nn.init.normal_(self.frequency_scorer.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_scorer.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.context_scorer.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.frequency_scorer.bias)
        nn.init.zeros_(self.position_scorer.bias)
        nn.init.zeros_(self.context_scorer.bias)
    
    def calculate_token_temperatures(self, hidden_states, input_ids=None):
        """
        Calculate temperature values for each token in the sequence.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            input_ids: Optional tensor of shape [batch_size, seq_length]
            
        Returns:
            temperatures: Tensor of shape [batch_size, seq_length]
        """
        if not self.use_ttm or not self.training:
            # Return ones if TTM is disabled or during inference
            return torch.ones(hidden_states.shape[0], hidden_states.shape[1], device=hidden_states.device)
        
        # Calculate frequency score (how common/rare a token is)
        frequency_score = self.frequency_scorer(hidden_states).squeeze(-1)
        
        # Calculate position score (importance based on position)
        position_score = self.position_scorer(hidden_states).squeeze(-1)
        
        # Calculate context score (importance based on surrounding context)
        context_score = self.context_scorer(hidden_states).squeeze(-1)
        
        # Combine scores with learned weights
        temperatures = (
            self.alpha * torch.sigmoid(frequency_score) + 
            self.beta * torch.sigmoid(position_score) + 
            self.gamma * torch.sigmoid(context_score)
        )
        
        # Normalize to have mean 1.0 to avoid changing overall attention scale
        temperatures = temperatures * (temperatures.shape[1] / temperatures.sum(dim=1, keepdim=True))
        
        return temperatures
    
    def calculate_ttm_loss(self, hidden_states, labels):
        """
        Calculate TTM loss to encourage focus on important tokens.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            labels: Tensor of shape [batch_size, seq_length]
            
        Returns:
            ttm_loss: Scalar tensor
        """
        if not self.use_ttm or not self.training:
            return torch.tensor(0.0, device=hidden_states.device)
        
        # Get token temperatures
        temperatures = self.calculate_token_temperatures(hidden_states)
        
        # Calculate entropy of temperature distribution
        # We want to minimize entropy to encourage clear focus on specific tokens
        temperature_probs = F.softmax(temperatures, dim=-1)
        entropy = -torch.sum(temperature_probs * torch.log(temperature_probs + 1e-10), dim=-1).mean()
        
        # Calculate variance of temperatures
        # We want to maximize variance to differentiate important from unimportant tokens
        variance = torch.var(temperatures, dim=-1).mean()
        
        # Combined loss: minimize entropy, maximize variance
        ttm_loss = entropy - 0.1 * variance
        
        return ttm_loss

class MultiTokenPrediction(nn.Module):
    """
    Multi-Token Prediction (MTP) head for predicting multiple tokens at once.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        
        # MTP projection head
        self.mtp_head = nn.Linear(self.hidden_size, self.vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize with same scale as main LM head
        nn.init.normal_(self.mtp_head.weight, mean=0.0, std=0.02)
        if hasattr(self.mtp_head, 'bias') and self.mtp_head.bias is not None:
            nn.init.zeros_(self.mtp_head.bias)
    
    def forward(self, hidden_states, attention_mask=None, labels=None):
        """
        Forward pass for MTP.
        
        Args:
            hidden_states: Tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask
            labels: Optional labels for loss calculation
            
        Returns:
            logits: Tensor of shape [batch_size, seq_length, vocab_size]
            loss: Optional loss if labels are provided
        """
        # Project hidden states to vocabulary
        logits = self.mtp_head(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None and self.training:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        return logits, loss

class QuasarExpertFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size)
        
    def forward(self, x):
        # SwiGLU activation
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class QuasarMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # MoE parameters
        self.num_experts = getattr(config, 'num_experts', 64)  # Ne
        self.num_shared_experts = getattr(config, 'num_shared_experts', 1)  # Ns
        self.num_routed_experts = getattr(config, 'num_routed_experts', 64)  # Nr
        self.top_k = getattr(config, 'top_k', 4)  # Kr
        self.alpha = getattr(config, 'load_balancing_alpha', 0.01)  # Alpha for sequence-wise auxiliary loss
        self.gamma = getattr(config, 'load_balancing_gamma', 0.01)  # Gamma for token-wise auxiliary loss
        
        # Shared experts
        self.shared_experts = nn.ModuleList([
            QuasarExpertFFN(config) for _ in range(self.num_shared_experts)
        ])
        
        # Routed experts
        self.routed_experts = nn.ModuleList([
            QuasarExpertFFN(config) for _ in range(self.num_routed_experts)
        ])
        
        # Router
        self.router = nn.Linear(self.hidden_size, self.num_routed_experts)
        self.expert_biases = nn.Parameter(torch.zeros(self.num_routed_experts))
        # gamma is already defined above
        
    def forward(self, x, update_biases=True):
        batch_size, seq_len = x.shape[:2]
        
        # Get router scores and add biases
        router_logits = self.router(x)  # [batch, seq, Nr]
        router_logits_with_bias = router_logits + self.expert_biases
        
        # Get top-K experts and their scores
        scores_with_bias, indices = torch.topk(router_logits_with_bias, self.top_k, dim=-1)  # [batch, seq, Kr]
        
        # Get original scores for selected experts (without bias)
        # This is important: route based on s_i,t + bi but compute gates using original s_i,t
        original_scores = torch.gather(router_logits, -1, indices)
        
        # Apply sigmoid to get affinity scores
        scores = torch.sigmoid(original_scores)  # Original scores for gates
        gates = F.normalize(scores, p=1, dim=-1)  # Normalize for weighted sum
        
        # Process with shared experts
        shared_output = torch.zeros_like(x)
        for i in range(self.num_shared_experts):
            shared_output += self.shared_experts[i](x)
        shared_output /= self.num_shared_experts
        
        # Process with routed experts - use a deterministic approach for gradient checkpointing
        routed_output = torch.zeros_like(x)
        
        # Deterministic processing of experts
        # This approach ensures the same number of tensors are created during forward and recomputation
        for k in range(self.top_k):
            expert_idx = indices[..., k]
            gate = gates[..., k:k+1]
            
            # Create a flattened version for processing
            flat_x = x.reshape(-1, x.size(-1))
            flat_expert_idx = expert_idx.reshape(-1)
            flat_gate = gate.reshape(-1, 1)
            
            # Process each expert in a deterministic way
            for i in range(self.num_routed_experts):
                # Create binary mask for this expert
                expert_mask = (flat_expert_idx == i)
                # Apply the mask to get inputs for this expert
                # Convert mask to same dtype as x to avoid dtype mismatch
                mask_tensor = expert_mask.unsqueeze(-1).to(dtype=x.dtype)
                masked_input = flat_x * mask_tensor
                # Process all inputs (most will be zeros)
                expert_output = self.routed_experts[i](masked_input)
                # Apply gate values (also masked)
                gated_output = expert_output * (flat_gate * mask_tensor)
                # Add to the output
                routed_output += gated_output.reshape(x.shape)
        
        # Combine outputs
        output = shared_output + routed_output
        
        # Update biases based on expert load - only during training
        if update_biases and self.training:
            # Calculate expert counts for batch-wise load balancing
            expert_counts = torch.zeros(self.num_routed_experts, device=x.device, dtype=x.dtype)
            
            # Deterministic approach to count experts
            flat_indices = indices.reshape(-1)
            for i in range(self.num_routed_experts):
                expert_counts[i] = (flat_indices == i).to(dtype=x.dtype).sum()
                    
            target_count = (batch_size * seq_len * self.top_k) / self.num_routed_experts
            load_diff = expert_counts - target_count
            
            # Update biases
            with torch.no_grad():
                self.expert_biases.data -= self.gamma * load_diff.to(self.expert_biases.dtype)
            
            # Compute sequence-wise auxiliary loss (L_Bal) - deterministic approach
            sequence_counts = torch.zeros(batch_size, self.num_routed_experts, device=x.device, dtype=x.dtype)
            
            # Reshape indices to [batch, seq*top_k] for deterministic processing
            reshaped_indices = indices.reshape(batch_size, -1)
            for i in range(self.num_routed_experts):
                sequence_counts[:, i] = (reshaped_indices == i).to(dtype=x.dtype).sum(dim=1)
            
            # Compute sequence-wise auxiliary loss (L_Bal)
            sequence_fractions = sequence_counts / (seq_len * self.top_k)
            target_fraction = 1.0 / self.num_routed_experts
            target_tensor = torch.full_like(sequence_fractions, target_fraction)
            sequence_loss = F.mse_loss(sequence_fractions, target_tensor)
            
            # Store the loss for later use in the training objective
            self.sequence_balance_loss = self.alpha * sequence_loss
        else:
            self.sequence_balance_loss = 0.0
        
        return output

class QuasarTransformerBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        
        # Memory-efficient initialization
        self._initialize_memory_efficient = True
        
        # Pre-normalization for attention
        self.pre_attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Attention mechanism (MLA or NSA)
        if config.use_nsa:
            self.attention = NSAAttention(config)
        else:
            self.attention = MultiHeadLatentAttention(config)
        
        # Post-attention normalization
        self.post_attention_norm = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
        # Feed-forward network (standard or MoE)
        use_moe_for_this_layer = config.use_moe and (self.layer_idx > 0 or not config.first_layer_no_moe)
        
        if use_moe_for_this_layer:
            self.ffn = QuasarMoE(config)
        else:
            self.ffn = QuasarExpertFFN(config)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None, token_temperatures=None):
        # Pre-norm for attention
        normed_states = self.pre_attention_norm(hidden_states)
        
        # Apply attention with MLA as the main mechanism, TTM just passes token temperatures
        attention_output, past_key_value = self.attention(
            normed_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids,
            token_temperatures=token_temperatures  # Just pass token temperatures, MLA decides how to use them
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Pre-norm for FFN/MoE
        normed_states = self.post_attention_norm(hidden_states)
        ffn_output = self.ffn(normed_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states, past_key_value

class QuasarConfig:
    def __init__(
        self,
        vocab_size=102400,  # DeepSeek
        hidden_size=2048,   # DeepSeek (dim)
        num_hidden_layers=27,  # DeepSeek (n_layers)
        num_attention_heads=16,  # DeepSeek (n_heads)
        head_dim=16,  # Keep as is unless you want v_head_dim=128
        intermediate_size=1532,  # DeepSeek (inter_dim)
        kv_compressed_dim=256,
        query_compressed_dim=512,
        rope_dim_per_head=32,
        max_position_embeddings=4096,  # Keep as is
        attention_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        use_moe=True,
        num_experts=66,  # 2 shared + 64 routed
        num_experts_per_token=4,  # Keep as is
        moe_balance_loss_weight=0.01,
        first_layer_no_moe=True,
        num_shared_experts=2,  # DeepSeek
        num_routed_experts=64,  # DeepSeek
        top_k=6,  # Reduced from 6 to prevent CUDA indexing errors
        load_balancing_alpha=0.01,
        load_balancing_gamma=0.01,
        use_ttm=False,  # Temporarily disabled to prevent CUDA memory errors
        ttm_loss_weight=0.01,
        mtp_loss_weight=0.1,
        use_mtp=True,
        use_rope=True,
        use_nsa=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.kv_compressed_dim = kv_compressed_dim
        self.query_compressed_dim = query_compressed_dim
        self.rope_dim_per_head = rope_dim_per_head
        self.max_position_embeddings = max_position_embeddings
        self.attention_dropout_prob = attention_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_token = num_experts_per_token
        self.moe_balance_loss_weight = moe_balance_loss_weight
        self.first_layer_no_moe = first_layer_no_moe
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k
        self.load_balancing_alpha = load_balancing_alpha
        self.load_balancing_gamma = load_balancing_gamma
        self.use_ttm = use_ttm
        self.ttm_loss_weight = ttm_loss_weight
        self.mtp_loss_weight = mtp_loss_weight
        self.use_mtp = use_mtp
        self.use_nsa = use_nsa
        
        # Process any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

class Quasar(nn.Module):
    def __init__(self, config, pbar=None, use_meta_device=False, ultra_lazy=False, distributed_init=False, rank=0, world_size=1):
        super().__init__()
        self.config = config
        
        # For very large models, we'll use DeepSpeed's initialization
        # This just creates the structure without allocating memory
        self.use_meta_device = use_meta_device
        self.ultra_lazy = ultra_lazy
        self.distributed_init = distributed_init
        self.rank = rank
        self.world_size = world_size
        
        if pbar:
            pbar.update(5)
            pbar.set_description("Creating token embeddings")
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Main transformer layers
        self.layers = nn.ModuleList([])
        total_layers = config.num_hidden_layers
        
        # This is the most time-consuming part, so we track progress here
        for i in range(total_layers):
            if pbar:
                # Update progress proportionally (70% of total progress)
                progress_per_layer = 70.0 / total_layers
                pbar.update(progress_per_layer)
                pbar.set_description(f"Creating transformer layer {i+1}/{total_layers}")
            
            # Memory-efficient initialization - clear CUDA cache after each layer
            layer = QuasarTransformerBlock(config, layer_idx=i)
            self.layers.append(layer)
            
            # Clear CUDA cache to prevent OOM errors during initialization
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Final layer norm
        self.final_norm = RMSNorm(config.hidden_size)
        
        if pbar:
            pbar.update(5)
            pbar.set_description("Creating output projection")
            
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # Weight tying
        
        # Multi-token prediction module
        self.mtp = MultiTokenPrediction(config)
        
        # Token Temperature Modulation (TTM) module
        if config.use_ttm:
            self.ttm = TokenTemperatureModulation(config)
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = False
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training."""
        self.gradient_checkpointing = True
    
    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing = False
        
    def forward(self, input_ids, attention_mask=None, past_key_values=None, labels=None, position_ids=None):
        batch_size, seq_length = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            # Create attention mask based on pad_token_id
            attention_mask = torch.ones_like(input_ids)
            if self.config.pad_token_id is not None:
                attention_mask = (input_ids != self.config.pad_token_id).long()
        
        # Initialize position_ids if not provided
        if position_ids is None:
            # Create position ids based on attention mask
            position_ids = torch.cumsum(attention_mask, dim=1) - 1
            # Set position ids for padding tokens to 0
            position_ids = position_ids * attention_mask
            position_ids = position_ids.to(input_ids.device)
        
        # Get input embeddings
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.embedding_dropout(hidden_states)
        
        # Initialize past_key_values if None
        if past_key_values is None:
            past_key_values = tuple([None] * len(self.layers))
            
        # Calculate token temperatures for TTM if enabled
        token_temperatures = None
        if hasattr(self, 'ttm') and labels is not None:
            token_temperatures = self.ttm.calculate_token_temperatures(hidden_states, input_ids)
        
        # Process through transformer layers
        all_hidden_states = []
        new_past_key_values = []
        moe_balance_loss = 0.0
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            all_hidden_states.append(hidden_states)
            
            if self.gradient_checkpointing and self.training:
                # Use gradient checkpointing for memory efficiency
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states, 
                    attention_mask, 
                    past_key_value, 
                    position_ids,
                    token_temperatures,
                    use_reentrant=False  # Set to False as recommended
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    position_ids=position_ids,
                    token_temperatures=token_temperatures
                )
            
            hidden_states, past_key_value = layer_outputs
            new_past_key_values.append(past_key_value)
            
            # Accumulate MoE balance loss if available
            if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'sequence_balance_loss'):
                moe_balance_loss += layer.ffn.sequence_balance_loss
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        all_hidden_states.append(hidden_states)
        
        # Calculate main loss (next token prediction)
        main_logits = self.lm_head(hidden_states)
        main_loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        # Calculate multi-token prediction loss if enabled
        mtp_loss = None
        if hasattr(self, 'mtp') and self.training and getattr(self.config, 'use_mtp', False):
            mtp_logits, mtp_loss = self.mtp(hidden_states, attention_mask, labels)
        
        # Calculate TTM loss if enabled
        ttm_loss = None
        if hasattr(self, 'ttm') and self.ttm.use_ttm and labels is not None:
            ttm_loss = self.ttm.calculate_ttm_loss(hidden_states, labels)
        
        # Combine losses
        loss = None
        if main_loss is not None:
            loss = main_loss
            
            if mtp_loss is not None:
                loss += self.config.mtp_loss_weight * mtp_loss
                
            if moe_balance_loss > 0:
                loss += self.config.moe_balance_loss_weight * moe_balance_loss
                
            if ttm_loss is not None:
                loss += self.config.ttm_loss_weight * ttm_loss
        
        return {
            'main_logits': main_logits,
            'loss': loss,
            'main_loss': main_loss,
            'mtp_loss': mtp_loss,
            'moe_balance_loss': moe_balance_loss,
            'ttm_loss': ttm_loss,
            'hidden_states': hidden_states,
            'past_key_values': tuple(new_past_key_values) if new_past_key_values else None
        }

def create_quasar_model(use_nsa=False, pbar=None, lazy_init=True, ultra_lazy=True, distributed_init=True):
    """Create a ~200B parameter Quasar model with ~17B active parameters through MoE.
    
    Args:
        use_nsa: Whether to use Native Sparse Attention instead of Multi-Head Latent Attention.
        pbar: Optional progress bar for tracking model creation.
        lazy_init: Whether to use lazy initialization to reduce memory usage during creation.
        ultra_lazy: Whether to use ultra-lazy initialization for extremely large models (200B+).
        distributed_init: Whether to use distributed initialization for multi-GPU setups.
    """
    # Start timing for more accurate progress estimation
    start_time = time.time()
    
    # Set memory-efficient initialization
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    
    # Try to free up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    if pbar:
        pbar.update(5)
        pbar.set_description("Initializing model configuration")
    
    config = QuasarConfig()
    config.use_nsa = use_nsa
    
    # Check if we're in a distributed environment
    is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
    
    # For multi-GPU distributed initialization of extremely large models
    if distributed_init and is_distributed:
        if pbar:
            pbar.update(15)
            pbar.set_description("Using distributed initialization for multi-GPU setup")
        
        # Get distributed info
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        
        if pbar and rank == 0:
            pbar.set_description(f"Distributed init across {world_size} GPUs (rank {rank})")
        
        # Calculate parameter counts analytically without creating any model
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        vocab_size = config.vocab_size
        ffn_dim = config.intermediate_size
        num_shared = config.num_shared_experts
        num_routed = config.num_routed_experts
        top_k = config.top_k
        
        # Embedding parameters
        embed_params = vocab_size * hidden_size
        
        # Attention parameters per layer
        attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        
        # FFN parameters per expert
        ffn_params = 3 * hidden_size * ffn_dim  # Up, Gate, Down projections
        
        # Layer norm parameters
        ln_params = 2 * hidden_size * num_layers
        
        # MoE parameters
        moe_params = ffn_params * (num_shared + num_routed)
        
        # Total parameters
        param_count = embed_params + (attn_params * num_layers) + (moe_params * num_layers) + ln_params
        
        # Active parameters
        shared_expert_params = param_count * (num_shared / (num_shared + num_routed))
        routed_expert_params = param_count * (num_routed / (num_shared + num_routed))
        active_param_count = shared_expert_params + (routed_expert_params * (top_k / num_routed))
        
        # Calculate per-GPU parameters (for ZeRO-3)
        per_gpu_params = param_count / world_size
        
        if pbar and rank == 0:
            pbar.update(25)
            pbar.set_description(f"Estimated {param_count/1e9:.2f}B params ({per_gpu_params/1e9:.2f}B per GPU)")
        
        # Synchronize all processes before creating model
        torch.distributed.barrier()
        
        # Create empty model shell - DeepSpeed will handle parameter initialization
        model = Quasar(config, distributed_init=True, rank=rank, world_size=world_size)
        
        if pbar and rank == 0:
            pbar.update(50)
            pbar.set_description("Model structure created, waiting for DeepSpeed init")
    
    # For extremely large models (200B+) on single GPU, use ultra-lazy initialization
    elif ultra_lazy:
        if pbar:
            pbar.update(15)
            pbar.set_description("Using ultra-lazy initialization for 200B+ model")
        
        # Calculate parameter counts analytically without creating any model
        hidden_size = config.hidden_size
        num_layers = config.num_hidden_layers
        num_heads = config.num_attention_heads
        vocab_size = config.vocab_size
        ffn_dim = config.intermediate_size
        num_shared = config.num_shared_experts
        num_routed = config.num_routed_experts
        top_k = config.top_k
        
        # Embedding parameters
        embed_params = vocab_size * hidden_size
        
        # Attention parameters per layer
        attn_params = 4 * hidden_size * hidden_size  # Q, K, V, O projections
        
        # FFN parameters per expert
        ffn_params = 3 * hidden_size * ffn_dim  # Up, Gate, Down projections
        
        # Layer norm parameters
        ln_params = 2 * hidden_size * num_layers
        
        # MoE parameters
        moe_params = ffn_params * (num_shared + num_routed)
        
        # Total parameters
        param_count = embed_params + (attn_params * num_layers) + (moe_params * num_layers) + ln_params
        
        # Active parameters
        shared_expert_params = param_count * (num_shared / (num_shared + num_routed))
        routed_expert_params = param_count * (num_routed / (num_shared + num_routed))
        active_param_count = shared_expert_params + (routed_expert_params * (top_k / num_routed))
        
        if pbar:
            pbar.update(75)
            pbar.set_description("Creating model structure (ultra-lazy)")
        
        # Create a minimal model structure for DeepSpeed
        # This is just a skeleton - DeepSpeed will handle the actual initialization
        model = Quasar(config, ultra_lazy=True)
        
    elif lazy_init:
        # For large models, use standard lazy initialization
        if pbar:
            pbar.update(10)
            pbar.set_description("Creating model skeleton (lazy initialization)")
        
        # Just create a minimal model for parameter counting
        # Actual parameters will be initialized by DeepSpeed
        config.num_hidden_layers = 2  # Temporarily reduce layers for counting
        temp_model = Quasar(config)
        
        # Estimate parameter count based on reduced model
        temp_param_count = sum(p.numel() for p in temp_model.parameters())
        # Scale up to full model size
        full_layers = QuasarConfig().num_hidden_layers
        param_count = temp_param_count * (full_layers / 2)
        
        # Calculate active parameters more accurately
        shared_expert_params = param_count * (config.num_shared_experts / (config.num_shared_experts + config.num_routed_experts))
        routed_expert_params = param_count * (config.num_routed_experts / (config.num_shared_experts + config.num_routed_experts))
        active_param_count = shared_expert_params + (routed_expert_params * (config.top_k / config.num_routed_experts))
        
        if pbar:
            pbar.update(80)
            pbar.set_description("Creating full model (lazy initialization)")
        
        # Delete temporary model to free memory
        del temp_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Restore full layer count for actual model creation
        config = QuasarConfig()
        config.use_nsa = use_nsa
        
        # Create actual model (DeepSpeed will handle memory-efficient initialization)
        model = Quasar(config)
    else:
        # Standard initialization (may cause OOM for very large models)
        model = Quasar(config, pbar=pbar)
    
    # Calculate and print parameter count
    if pbar:
        pbar.update(5)
        pbar.set_description("Calculating parameter counts")
        
    param_count = sum(p.numel() for p in model.parameters())
    
    # Calculate active parameters more accurately
    # For each token, we use all shared experts + top_k routed experts
    shared_expert_params = param_count * (config.num_shared_experts / (config.num_shared_experts + config.num_routed_experts))
    routed_expert_params = param_count * (config.num_routed_experts / (config.num_shared_experts + config.num_routed_experts))
    active_param_count = shared_expert_params + (routed_expert_params * (config.top_k / config.num_routed_experts))
    
    if pbar:
        # Ensure we reach 100%
        remaining = max(0, 100 - pbar.n)
        if remaining > 0:
            pbar.update(remaining)
        pbar.set_description("Model creation complete")
    
    print(f"Model created with {param_count/1e9:.2f}B total parameters")
    print(f"Approximately {active_param_count/1e9:.2f}B active parameters per token")
    print(f"Using {'Native Sparse Attention' if use_nsa else 'Multi-Head Latent Attention'}")
    print(f"MoE configuration: {config.num_shared_experts} shared + {config.num_routed_experts} routed experts, top-{config.top_k} routing")
    print(f"Model dimensions: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, heads={config.num_attention_heads}")
    
    return model
