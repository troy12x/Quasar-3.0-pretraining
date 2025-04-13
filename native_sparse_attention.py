import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenCompression(nn.Module):
    """
    Token Compression component of Native Sparse Attention.
    Compresses sequential blocks of keys/values using a learnable MLP.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.block_size = config.nsa_block_size  # l
        self.compressed_block_size = config.nsa_compressed_block_size  # l'
        self.stride = config.nsa_stride  # d
        
        # MLP for compression
        self.compress_mlp = nn.Sequential(
            nn.Linear(self.block_size * self.head_dim, config.nsa_compression_hidden_size),
            nn.GELU(),
            nn.Linear(config.nsa_compression_hidden_size, self.compressed_block_size * self.head_dim)
        )
        
        # Position encoding for compression
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, self.block_size, self.head_dim)
        )
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)
    
    def forward(self, k, v):
        """
        Compress keys and values into blocks.
        
        Args:
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            k_compressed: Compressed key tensor [batch_size, num_blocks, num_heads, compressed_block_size, head_dim]
            v_compressed: Compressed value tensor [batch_size, num_blocks, num_heads, compressed_block_size, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = k.shape
        
        # Calculate number of blocks
        num_blocks = max(1, (seq_len - self.block_size) // self.stride + 1)
        
        # Extract blocks with stride
        k_blocks = []
        v_blocks = []
        
        for i in range(num_blocks):
            start_idx = i * self.stride
            end_idx = min(start_idx + self.block_size, seq_len)
            
            # Handle padding if needed
            if end_idx - start_idx < self.block_size:
                pad_size = self.block_size - (end_idx - start_idx)
                k_block = F.pad(k[:, start_idx:end_idx], (0, 0, 0, 0, 0, pad_size))
                v_block = F.pad(v[:, start_idx:end_idx], (0, 0, 0, 0, 0, pad_size))
            else:
                k_block = k[:, start_idx:end_idx]
                v_block = v[:, start_idx:end_idx]
            
            # Add position encoding
            k_block = k_block + self.pos_encoding[:, :k_block.size(1)]
            
            k_blocks.append(k_block)
            v_blocks.append(v_block)
        
        # Stack blocks
        k_blocks = torch.stack(k_blocks, dim=1)  # [batch_size, num_blocks, block_size, num_heads, head_dim]
        v_blocks = torch.stack(v_blocks, dim=1)  # [batch_size, num_blocks, block_size, num_heads, head_dim]
        
        # Reshape for compression
        k_blocks = k_blocks.permute(0, 1, 3, 2, 4).contiguous()  # [batch_size, num_blocks, num_heads, block_size, head_dim]
        v_blocks = v_blocks.permute(0, 1, 3, 2, 4).contiguous()  # [batch_size, num_blocks, num_heads, block_size, head_dim]
        
        # Flatten block dimension for MLP
        k_flat = k_blocks.view(batch_size * num_blocks * num_heads, -1)  # [batch_size * num_blocks * num_heads, block_size * head_dim]
        v_flat = v_blocks.view(batch_size * num_blocks * num_heads, -1)  # [batch_size * num_blocks * num_heads, block_size * head_dim]
        
        # Compress with MLP
        k_compressed_flat = self.compress_mlp(k_flat)  # [batch_size * num_blocks * num_heads, compressed_block_size * head_dim]
        v_compressed_flat = self.compress_mlp(v_flat)  # [batch_size * num_blocks * num_heads, compressed_block_size * head_dim]
        
        # Reshape back
        k_compressed = k_compressed_flat.view(
            batch_size, num_blocks, num_heads, self.compressed_block_size, head_dim
        )
        v_compressed = v_compressed_flat.view(
            batch_size, num_blocks, num_heads, self.compressed_block_size, head_dim
        )
        
        return k_compressed, v_compressed


class TokenSelection(nn.Module):
    """
    Token Selection component of Native Sparse Attention.
    Selects important token blocks based on attention scores.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_selected_blocks = config.nsa_num_selected_blocks  # n
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.num_queries_per_kv = self.num_heads // self.num_key_value_heads
    
    def forward(self, q, k_compressed, v_compressed):
        """
        Select important token blocks based on attention scores.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k_compressed: Compressed key tensor [batch_size, num_blocks, num_heads, compressed_block_size, head_dim]
            v_compressed: Compressed value tensor [batch_size, num_blocks, num_heads, compressed_block_size, head_dim]
            
        Returns:
            k_selected: Selected key tensor [batch_size, seq_len, num_heads, num_selected_blocks * compressed_block_size, head_dim]
            v_selected: Selected value tensor [batch_size, seq_len, num_heads, num_selected_blocks * compressed_block_size, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        num_blocks = k_compressed.size(1)
        compressed_block_size = k_compressed.size(3)
        
        # Calculate attention scores between query and compressed keys
        # Reshape q to [batch_size, seq_len, num_heads, 1, head_dim]
        q_expanded = q.unsqueeze(3)
        
        # Compute attention scores
        # [batch_size, seq_len, num_heads, num_blocks, compressed_block_size]
        attention_scores = torch.matmul(
            q_expanded, k_compressed.transpose(-1, -2).unsqueeze(1)
        )
        
        # Scale attention scores
        attention_scores = attention_scores / math.sqrt(head_dim)
        
        # Compute block importance scores by summing over compressed_block_size
        # [batch_size, seq_len, num_heads, num_blocks]
        block_scores = attention_scores.sum(dim=-1)
        
        # For GQA/MQA, aggregate scores across heads in a group
        if self.num_queries_per_kv > 1:
            # Reshape to [batch_size, seq_len, num_key_value_heads, num_queries_per_kv, num_blocks]
            block_scores = block_scores.view(
                batch_size, seq_len, self.num_key_value_heads, self.num_queries_per_kv, num_blocks
            )
            # Sum across queries in each group
            block_scores = block_scores.sum(dim=3)
            # Expand back to match all query heads
            block_scores = block_scores.unsqueeze(3).expand(
                batch_size, seq_len, self.num_key_value_heads, self.num_queries_per_kv, num_blocks
            ).reshape(batch_size, seq_len, num_heads, num_blocks)
        
        # Select top-n blocks based on importance scores
        # [batch_size, seq_len, num_heads, num_selected_blocks]
        _, top_indices = torch.topk(
            block_scores, min(self.num_selected_blocks, num_blocks), dim=-1
        )
        
        # Gather selected blocks
        # We need to construct the selected key/value tensors
        # This is a bit tricky with gather, so we'll use a loop for clarity
        k_selected = []
        v_selected = []
        
        for b in range(batch_size):
            for t in range(seq_len):
                for h in range(num_heads):
                    # Get indices of selected blocks for this (batch, token, head)
                    indices = top_indices[b, t, h]  # [num_selected_blocks]
                    
                    # Gather selected blocks
                    k_blocks = k_compressed[b, indices, h]  # [num_selected_blocks, compressed_block_size, head_dim]
                    v_blocks = v_compressed[b, indices, h]  # [num_selected_blocks, compressed_block_size, head_dim]
                    
                    k_selected.append(k_blocks.reshape(-1, head_dim))  # [num_selected_blocks * compressed_block_size, head_dim]
                    v_selected.append(v_blocks.reshape(-1, head_dim))  # [num_selected_blocks * compressed_block_size, head_dim]
        
        # Reshape to match expected dimensions
        k_selected = torch.stack(k_selected).reshape(
            batch_size, seq_len, num_heads, self.num_selected_blocks * compressed_block_size, head_dim
        )
        v_selected = torch.stack(v_selected).reshape(
            batch_size, seq_len, num_heads, self.num_selected_blocks * compressed_block_size, head_dim
        )
        
        return k_selected, v_selected


class SlidingWindow(nn.Module):
    """
    Sliding Window component of Native Sparse Attention.
    Handles local context with recent tokens.
    """
    def __init__(self, config):
        super().__init__()
        self.window_size = config.nsa_window_size  # w
    
    def forward(self, k, v, position_ids=None):
        """
        Select the most recent w tokens.
        
        Args:
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, head_dim]
            position_ids: Optional position IDs for non-contiguous sequences
            
        Returns:
            k_window: Window key tensor [batch_size, seq_len, num_heads, window_size, head_dim]
            v_window: Window value tensor [batch_size, seq_len, num_heads, window_size, head_dim]
        """
        batch_size, seq_len, num_heads, head_dim = k.shape
        
        # For each position, select the window_size most recent tokens
        k_window = []
        v_window = []
        
        for t in range(seq_len):
            # Calculate window start (causal)
            window_start = max(0, t + 1 - self.window_size)
            
            # Select window
            k_t = k[:, window_start:t+1]  # [batch_size, window_size_t, num_heads, head_dim]
            v_t = v[:, window_start:t+1]  # [batch_size, window_size_t, num_heads, head_dim]
            
            # Pad if needed
            if t + 1 < self.window_size:
                pad_size = self.window_size - (t + 1)
                k_t = F.pad(k_t, (0, 0, 0, 0, 0, pad_size))
                v_t = F.pad(v_t, (0, 0, 0, 0, 0, pad_size))
            
            k_window.append(k_t)
            v_window.append(v_t)
        
        # Stack windows
        k_window = torch.stack(k_window, dim=1)  # [batch_size, seq_len, window_size, num_heads, head_dim]
        v_window = torch.stack(v_window, dim=1)  # [batch_size, seq_len, window_size, num_heads, head_dim]
        
        # Reorder dimensions
        k_window = k_window.permute(0, 1, 3, 2, 4)  # [batch_size, seq_len, num_heads, window_size, head_dim]
        v_window = v_window.permute(0, 1, 3, 2, 4)  # [batch_size, seq_len, num_heads, window_size, head_dim]
        
        return k_window, v_window


class GatingNetwork(nn.Module):
    """
    Gating Network for Native Sparse Attention.
    Computes gate scores for combining outputs from different attention branches.
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        
        # MLP for computing gate scores
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.head_dim, config.nsa_gate_hidden_size),
            nn.GELU(),
            nn.Linear(config.nsa_gate_hidden_size, 3)  # 3 gates for compression, selection, window
        )
    
    def forward(self, q):
        """
        Compute gate scores for each attention branch.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            
        Returns:
            gates: Gate scores [batch_size, seq_len, num_heads, 3]
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Compute gate scores
        gates = self.gate_mlp(q.reshape(-1, head_dim)).view(batch_size, seq_len, num_heads, 3)
        
        # Apply sigmoid to get values in [0, 1]
        gates = torch.sigmoid(gates)
        
        # Normalize gates to sum to 1
        gates = gates / (gates.sum(dim=-1, keepdim=True) + 1e-6)
        
        return gates


class NativeSparseAttention(nn.Module):
    """
    Native Sparse Attention (NSA) as described in the paper
    "Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention"
    """
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        
        # Components
        self.token_compression = TokenCompression(config)
        self.token_selection = TokenSelection(config)
        self.sliding_window = SlidingWindow(config)
        self.gating_network = GatingNetwork(config)
        
        # Output projection
        self.output_projection = nn.Linear(self.num_heads * self.head_dim, self.hidden_size)
        
        # Attention dropout
        self.attention_dropout = nn.Dropout(config.attention_dropout_prob)
        
        # Scaling factor for attention
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def _compute_attention(self, q, k, v, attention_mask=None):
        """
        Compute attention scores and output.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_heads, kv_len, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, kv_len, head_dim]
            attention_mask: Optional attention mask
            
        Returns:
            output: Attention output [batch_size, seq_len, num_heads, head_dim]
        """
        # Compute attention scores
        # q: [batch_size, seq_len, num_heads, 1, head_dim]
        # k: [batch_size, seq_len, num_heads, kv_len, head_dim]
        # scores: [batch_size, seq_len, num_heads, 1, kv_len]
        scores = torch.matmul(q.unsqueeze(3), k.transpose(-1, -2))
        
        # Scale scores
        scores = scores * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attn_weights = self.attention_dropout(attn_weights)
        
        # Compute attention output
        # attn_weights: [batch_size, seq_len, num_heads, 1, kv_len]
        # v: [batch_size, seq_len, num_heads, kv_len, head_dim]
        # output: [batch_size, seq_len, num_heads, 1, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # Remove singleton dimension
        output = output.squeeze(3)
        
        return output
    
    def forward(self, q, k, v, attention_mask=None, position_ids=None, past_key_value=None):
        """
        Forward pass for Native Sparse Attention.
        
        Args:
            q: Query tensor [batch_size, seq_len, num_heads, head_dim]
            k: Key tensor [batch_size, seq_len, num_heads, head_dim]
            v: Value tensor [batch_size, seq_len, num_heads, head_dim]
            attention_mask: Optional attention mask
            position_ids: Optional position IDs
            past_key_value: Optional past key/value for incremental decoding
            
        Returns:
            output: Attention output [batch_size, seq_len, hidden_size]
            past_key_value: Updated past key/value for incremental decoding
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Handle past key/value for incremental decoding
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        
        # Token Compression
        k_compressed, v_compressed = self.token_compression(k, v)
        
        # Token Selection
        k_selected, v_selected = self.token_selection(q, k_compressed, v_compressed)
        
        # Sliding Window
        k_window, v_window = self.sliding_window(k, v, position_ids)
        
        # Compute gate scores
        gates = self.gating_network(q)
        
        # Compute attention for each branch
        # Compression branch
        output_compressed = self._compute_attention(
            q, k_compressed.reshape(batch_size, 1, num_heads, -1, head_dim).expand(-1, seq_len, -1, -1, -1),
            v_compressed.reshape(batch_size, 1, num_heads, -1, head_dim).expand(-1, seq_len, -1, -1, -1),
            attention_mask
        )
        
        # Selection branch
        output_selected = self._compute_attention(q, k_selected, v_selected, attention_mask)
        
        # Window branch
        output_window = self._compute_attention(q, k_window, v_window, attention_mask)
        
        # Combine outputs using gates
        # gates: [batch_size, seq_len, num_heads, 3]
        output = (
            gates[:, :, :, 0:1] * output_compressed +
            gates[:, :, :, 1:2] * output_selected +
            gates[:, :, :, 2:3] * output_window
        )
        
        # Reshape for output projection
        output = output.reshape(batch_size, seq_len, num_heads * head_dim)
        
        # Apply output projection
        output = self.output_projection(output)
        
        # Update past key/value for incremental decoding
        past_key_value = (k, v)
        
        return output, past_key_value
