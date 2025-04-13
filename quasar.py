import math
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
    # q, k: [batch, seq, heads, head_dim]
    # cos, sin: [seq_len, head_dim]
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
        self.W_KR = nn.Linear(self.head_dim, self.rope_dim_per_head)
        
        self.W_DQ = nn.Linear(self.hidden_size, self.query_compressed_dim)
        self.W_UQ = nn.Linear(self.query_compressed_dim, self.num_heads * self.head_dim)
        self.W_QR = nn.Linear(self.head_dim, self.rope_dim_per_head)
        
        self.W_O = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        # Attention dropout for regularization
        self.attention_dropout = getattr(config, 'attention_dropout_prob', 0.0)
        
        # Precompute freqs for RoPE
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        
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
        
        # Compress and project KV
        c_kv = self.W_DKV(hidden_states)  # [batch, seq, dc]
        k = self.W_UK(c_kv).view(batch_size, seq_length, self.num_heads, self.head_dim)
        v = self.W_UV(c_kv).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Compress and project Q
        c_q = self.W_DQ(hidden_states)  # [batch, seq, d'c]
        q = self.W_UQ(c_q).view(batch_size, seq_length, self.num_heads, self.head_dim)
        
        # Apply RoPE to decoupled components
        k_r = self.W_KR(k.reshape(-1, self.head_dim)).reshape(batch_size, seq_length, self.num_heads, self.rope_dim_per_head)
        q_r = self.W_QR(q.reshape(-1, self.head_dim)).reshape(batch_size, seq_length, self.num_heads, self.rope_dim_per_head)
        
        # Get cos/sin for current positions
        cos = self.cos[position_ids]  # [seq, rope_dim]
        sin = self.sin[position_ids]  # [seq, rope_dim]
        
        # Apply RoPE to decoupled components
        q_r, k_r = apply_rotary_pos_emb(
            q_r.transpose(1, 2),  # [batch, heads, seq, dRh]
            k_r.transpose(1, 2),  # [batch, heads, seq, dRh]
            cos.unsqueeze(0).unsqueeze(0),  # [1, 1, seq, dRh]
            sin.unsqueeze(0).unsqueeze(0),  # [1, 1, seq, dRh]
        )
        
        # Reshape back
        q_r = q_r.transpose(1, 2)  # [batch, seq, heads, dRh]
        k_r = k_r.transpose(1, 2)  # [batch, seq, heads, dRh]
        
        # Cache latent KV and decoupled RoPE key for generation
        if past_key_value is not None:
            c_kv = torch.cat([past_key_value[0], c_kv], dim=1)
            k_r = torch.cat([past_key_value[1], k_r], dim=1)
            # Recompute k and v with the full c_kv
            k_full = self.W_UK(c_kv).view(batch_size, -1, self.num_heads, self.head_dim)
            v_full = self.W_UV(c_kv).view(batch_size, -1, self.num_heads, self.head_dim)
            k, v = k_full, v_full
        
        # Compute attention scores and output
        q = q.transpose(1, 2)  # [batch, heads, seq, dh]
        k = k.transpose(1, 2)  # [batch, heads, seq, dh]
        v = v.transpose(1, 2)  # [batch, heads, seq, dh]
        
        # Process attention mask if provided
        attention_mask_for_sdp = None
        if attention_mask is not None:
            # Convert attention_mask to the format expected by scaled_dot_product_attention
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask_for_sdp = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert mask from 0/1 to -inf/0
            attention_mask_for_sdp = attention_mask_for_sdp.to(torch.float32)
            attention_mask_for_sdp = (1.0 - attention_mask_for_sdp) * torch.finfo(q.dtype).min
        
        # Use Flash Attention if available for faster training
        if HAS_FLASH_ATTENTION and attention_mask is None:
            # Flash Attention expects input in shape [batch_size, seq_len, num_heads, head_dim]
            q_flash = q.transpose(1, 2)  # [batch, seq, heads, dh]
            k_flash = k.transpose(1, 2)  # [batch, seq, heads, dh]
            v_flash = v.transpose(1, 2)  # [batch, seq, heads, dh]
            
            # Call Flash Attention
            attn_output = flash_attn_func(
                q_flash, k_flash, v_flash,
                dropout_p=self.attention_dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.head_dim)
            )
            
            # Reshape output
            attn_output = attn_output.reshape(batch_size, seq_length, -1)
        else:
            # Fall back to standard attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask_for_sdp,
                dropout_p=self.attention_dropout,
                scale=1.0 / math.sqrt(self.head_dim)
            )
            
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        
        output = self.W_O(attn_output)
        
        return output, (c_kv, k_r)

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
        self.num_shared_experts = config.num_shared_experts  # Ns
        self.num_routed_experts = config.num_routed_experts  # Nr
        self.top_k = config.top_k  # Kr
        self.alpha = config.load_balancing_alpha  # Alpha for sequence-wise auxiliary loss
        
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
        self.gamma = config.load_balancing_gamma
        
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
        shared_output = sum(expert(x) for expert in self.shared_experts)
        
        # Process with routed experts
        routed_output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = indices[..., k]
            gate = gates[..., k:k+1]
            for i in range(self.num_routed_experts):
                mask = (expert_idx == i)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.routed_experts[i](expert_input)
                    routed_output[mask] += gate[mask] * expert_output
        
        # Update biases based on expert load
        if update_biases and self.training:
            # Calculate expert counts for batch-wise load balancing
            expert_counts = torch.zeros(self.num_routed_experts, device=x.device)
            for i in range(self.num_routed_experts):
                expert_counts[i] = (indices == i).float().sum()
            target_count = (batch_size * seq_len * self.top_k) / self.num_routed_experts
            load_diff = expert_counts - target_count
            self.expert_biases.data -= self.gamma * load_diff
            
            # Compute sequence-wise auxiliary loss (L_Bal)
            # This is a complementary loss to prevent extreme sequence imbalance
            sequence_counts = torch.zeros(batch_size, self.num_routed_experts, device=x.device)
            for i in range(self.num_routed_experts):
                sequence_counts[:, i] = (indices == i).float().sum(dim=1)  # Sum over sequence dimension
            sequence_fractions = sequence_counts / (seq_len * self.top_k)
            target_fraction = 1.0 / self.num_routed_experts
            sequence_loss = F.mse_loss(sequence_fractions, torch.full_like(sequence_fractions, target_fraction))
            
            # Store the loss for later use in the training objective
            self.sequence_balance_loss = self.alpha * sequence_loss
        else:
            self.sequence_balance_loss = 0.0
        
        return x + shared_output + routed_output

class MultiTokenPrediction(nn.Module):
    def __init__(self, config):
        super().__init__()
        # D=1 MTP module to predict the next two tokens
        self.transformer = QuasarTransformerBlock(config)
        self.projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, hidden_states, attention_mask=None, labels=None):
        # Apply transformer block to get next token representations
        mtp_hidden = self.transformer(hidden_states, attention_mask)
        mtp_logits = self.projection(mtp_hidden)
        
        # Calculate MTP loss if labels are provided
        mtp_loss = None
        if labels is not None:
            # Get the labels for the next two tokens
            # For each position i, predict tokens at positions i+1 and i+2
            # Shift labels for next token prediction
            next_token_labels = labels[:, 1:] if labels.size(1) > 1 else None
            
            # Create padding mask - don't compute loss for padding tokens
            if next_token_labels is not None and hasattr(config, 'pad_token_id'):
                padding_mask = (next_token_labels != config.pad_token_id).float()
                if hasattr(config, 'eos_token_id'):
                    # Also compute loss for EOS token
                    eos_mask = (next_token_labels == config.eos_token_id).float()
                    padding_mask = padding_mask + eos_mask
                    padding_mask = torch.clamp(padding_mask, 0, 1)
                
                # Compute cross entropy loss with masking
                mtp_loss = F.cross_entropy(
                    mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)),
                    next_token_labels.reshape(-1),
                    reduction='none'
                )
                
                # Apply padding mask and compute mean
                mtp_loss = (mtp_loss * padding_mask.reshape(-1)).sum() / padding_mask.sum().clamp(min=1.0)
            else:
                # Fallback to standard loss if no padding mask
                mtp_loss = F.cross_entropy(
                    mtp_logits[:, :-1].reshape(-1, mtp_logits.size(-1)),
                    next_token_labels.reshape(-1),
                    ignore_index=-100
                ) if next_token_labels is not None else None
        
        return mtp_logits, mtp_loss

class QuasarTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_attention_norm = RMSNorm(config.hidden_size)
        
        # Choose attention mechanism based on config
        if config.use_nsa:
            self.attention = NSAAttention(config)
        else:
            self.attention = MultiHeadLatentAttention(config)
            
        self.post_attention_norm = RMSNorm(config.hidden_size)
        
        # Use MoE for all layers except the first if specified
        self.layer_idx = getattr(self, 'layer_idx', 0)
        use_moe_for_this_layer = config.use_moe and (self.layer_idx > 0 or not config.first_layer_no_moe)
        
        if use_moe_for_this_layer:
            self.ffn = QuasarMoE(config)
        else:
            self.ffn = QuasarExpertFFN(config)
        
        # Dropout for better regularization
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, attention_mask=None, past_key_value=None, position_ids=None):
        # Pre-norm for attention
        normed_states = self.pre_attention_norm(hidden_states)
        attention_output, past_key_value = self.attention(
            normed_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            position_ids=position_ids
        )
        hidden_states = hidden_states + self.dropout(attention_output)
        
        # Pre-norm for FFN/MoE
        normed_states = self.post_attention_norm(hidden_states)
        ffn_output = self.ffn(normed_states)
        hidden_states = hidden_states + self.dropout(ffn_output)
        
        return hidden_states

class QuasarConfig:
    def __init__(self):
        # Model dimensions for 140B model with 32B active parameters
        self.vocab_size = 128000
        self.hidden_size = 4096  # d (reduced from 6144)
        self.num_hidden_layers = 48  # L
        self.num_attention_heads = 32  # nh
        self.head_dim = 128  # dh
        self.kv_compressed_dim = 512  # dc
        self.query_compressed_dim = 1024  # d'c
        self.rope_dim_per_head = 32  # dRh
        self.intermediate_size = 14336  # 3.5x hidden_size
        self.max_position_embeddings = 4096
        
        # MoE parameters
        self.num_shared_experts = 1  # Ns
        self.num_routed_experts = 128  # Nr
        self.top_k = 4  # k
        self.load_balancing_gamma = 0.01  # γ
        self.first_layer_no_moe = True  # First layer uses standard FFN
        
        # Attention mechanism
        self.use_nsa = False  # Use Multi-Head Latent Attention by default
        
        # Other parameters
        self.initializer_range = 0.006
        self.use_moe = True  # Enable MoE for all layers except first
        
        # Training parameters
        self.mtp_loss_weight = 0.2  # lambda for MTP loss
        
        # Special token IDs - matching DeepSeek's implementation
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
        # Dropout rates - essential for regularization during pretraining
        self.hidden_dropout_prob = 0.1
        self.attention_dropout_prob = 0.1

class Quasar(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Embedding dropout
        self.embedding_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Main transformer layers
        self.layers = nn.ModuleList([])
        for i in range(config.num_hidden_layers):
            layer = QuasarTransformerBlock(config)
            layer.layer_idx = i  # Set layer index for MoE determination
            self.layers.append(layer)
        
        # Final layer norm
        self.final_norm = RMSNorm(config.hidden_size)
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.output_projection.weight = self.embed_tokens.weight  # Weight tying
        
        # Multi-token prediction module
        self.mtp = MultiTokenPrediction(config)
        
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
        
        # Process through transformer layers
        moe_balance_loss = 0.0
        for i, layer in enumerate(self.layers):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(layer, hidden_states, attention_mask, past_key_value, position_ids)
            else:
                hidden_states = layer(
                    hidden_states, 
                    attention_mask=attention_mask, 
                    past_key_value=past_key_value,
                    position_ids=position_ids
                )
            
            # Accumulate MoE balance loss if available
            if hasattr(layer.ffn, 'sequence_balance_loss'):
                moe_balance_loss += layer.ffn.sequence_balance_loss
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Main LM head output
        main_logits = self.output_projection(hidden_states)
        
        # Calculate main loss if labels are provided
        main_loss = None
        if labels is not None:
            # Shift labels for autoregressive prediction
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Create loss mask - don't compute loss for padding tokens
            loss_mask = (shift_labels != self.config.pad_token_id).float()
            if self.config.eos_token_id is not None:
                # Also compute loss for EOS token
                eos_mask = (shift_labels == self.config.eos_token_id).float()
                loss_mask = loss_mask + eos_mask
                loss_mask = torch.clamp(loss_mask, 0, 1)
            
            # Compute cross entropy loss with masking
            main_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            
            # Apply loss mask and compute mean
            main_loss = (main_loss * loss_mask.view(-1)).sum() / loss_mask.sum().clamp(min=1.0)
        
        # MTP output (predict next two tokens)
        mtp_logits, mtp_loss = self.mtp(hidden_states, attention_mask, labels)
        
        # Total loss
        loss = None
        if main_loss is not None and mtp_loss is not None:
            loss = main_loss + self.config.mtp_loss_weight * mtp_loss + moe_balance_loss
        
        return {
            'main_logits': main_logits,
            'mtp_logits': mtp_logits,
            'loss': loss,
            'hidden_states': hidden_states
        }

def create_quasar_model(use_nsa=False):
    """Create a ~140B parameter Quasar model with ~32B active parameters through MoE."""
    config = QuasarConfig()
    config.use_nsa = use_nsa
    model = Quasar(config)
    
    # Calculate and print parameter count
    param_count = sum(p.numel() for p in model.parameters())
    active_param_count = param_count / (config.num_routed_experts / config.top_k) if config.use_moe else param_count
    
    print(f"Model created with {param_count/1e9:.2f}B total parameters")
    print(f"Approximately {active_param_count/1e9:.2f}B active parameters per token")
    print(f"Using {'Native Sparse Attention' if use_nsa else 'Multi-Head Latent Attention'}")
    print(f"MoE configuration: {config.num_shared_experts} shared + {config.num_routed_experts} routed experts, top-{config.top_k} routing")
    print(f"Model dimensions: hidden_size={config.hidden_size}, layers={config.num_hidden_layers}, heads={config.num_attention_heads}")
    
    return model

# Example usage:
# model = create_quasar_model(use_nsa=True)  # Use Native Sparse Attention
# model = create_quasar_model()  # Use Multi-Head Latent Attention
# inputs = {"input_ids": torch.randint(0, 128000, (2, 512)), "attention_mask": torch.ones(2, 512)}
# outputs = model(**inputs)