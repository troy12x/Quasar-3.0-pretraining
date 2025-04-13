"""
Parameter count calculator for Quasar model.
This script calculates the theoretical parameter counts without instantiating the model.
"""

def calculate_quasar_parameters():
    """Calculate the parameter counts for the 140B Quasar model with 32B active parameters."""
    # Model configuration (copied from QuasarConfig)
    config = {
        "vocab_size": 128000,
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "head_dim": 128,
        "kv_compressed_dim": 512,
        "query_compressed_dim": 1024,
        "intermediate_size": 4096,
        "num_shared_experts": 1,
        "num_routed_experts": 128,
        "top_k": 4,
        "use_moe": True,
        "first_layer_no_moe": True
    }
    
    # Parameter count for each component
    params = {}
    
    # 1. Embedding layer
    params["embedding"] = config["vocab_size"] * config["hidden_size"]
    
    # 2. Multi-Head Latent Attention (per layer)
    mla_params = {}
    mla_params["w_qkv"] = config["hidden_size"] * (config["hidden_size"] * 3)  # QKV projection
    mla_params["w_dkv"] = config["hidden_size"] * config["kv_compressed_dim"]  # KV compression
    mla_params["w_dq"] = config["hidden_size"] * config["query_compressed_dim"]  # Q compression
    mla_params["w_kr"] = config["kv_compressed_dim"] * config["num_attention_heads"] * config["head_dim"]  # K rotary
    mla_params["w_qr"] = config["query_compressed_dim"] * config["num_attention_heads"] * config["head_dim"]  # Q rotary
    mla_params["w_o"] = config["hidden_size"] * config["hidden_size"]  # Output projection
    
    mla_params["total"] = (
        mla_params["w_qkv"] + 
        mla_params["w_dkv"] + 
        mla_params["w_dq"] + 
        mla_params["w_kr"] + 
        mla_params["w_qr"] + 
        mla_params["w_o"]
    )
    
    # 3. Native Sparse Attention (per layer)
    nsa_params = {}
    nsa_params["w_qkv"] = config["hidden_size"] * (config["hidden_size"] * 3)  # QKV projection
    nsa_params["compression_mlp"] = (
        config["hidden_size"] * config["hidden_size"] * 2  # Approximate compression MLP size
    )
    nsa_params["gate_mlp"] = config["hidden_size"] * config["hidden_size"] // 4  # Approximate gate MLP size
    nsa_params["w_o"] = config["hidden_size"] * config["hidden_size"]  # Output projection
    
    nsa_params["total"] = (
        nsa_params["w_qkv"] + 
        nsa_params["compression_mlp"] + 
        nsa_params["gate_mlp"] + 
        nsa_params["w_o"]
    )
    
    # 4. Expert FFN (for first layer if first_layer_no_moe is True)
    expert_ffn_params = (
        config["hidden_size"] * config["intermediate_size"] +  # W1
        config["intermediate_size"] * config["hidden_size"] +  # W2
        config["hidden_size"] * config["intermediate_size"]    # W3
    )
    
    # 5. MoE (per layer)
    moe_params = {}
    # Shared experts
    moe_params["shared_experts"] = config["num_shared_experts"] * (
        config["hidden_size"] * config["intermediate_size"] +  # W1
        config["intermediate_size"] * config["hidden_size"] +  # W2
        config["hidden_size"] * config["intermediate_size"]    # W3
    )
    
    # Routed experts
    moe_params["routed_experts"] = config["num_routed_experts"] * (
        config["hidden_size"] * config["intermediate_size"] +  # W1
        config["intermediate_size"] * config["hidden_size"] +  # W2
        config["hidden_size"] * config["intermediate_size"]    # W3
    )
    
    # Router
    moe_params["router"] = config["hidden_size"] * config["num_routed_experts"]
    
    moe_params["total"] = moe_params["shared_experts"] + moe_params["routed_experts"] + moe_params["router"]
    
    # 6. Layer normalization (2 per layer)
    layer_norm_params = 2 * config["hidden_size"]
    
    # 7. Multi-token prediction
    mtp_params = mla_params["total"] + moe_params["total"] + layer_norm_params
    
    # Calculate total parameters
    # First layer (no MoE if first_layer_no_moe is True)
    first_layer_params = mla_params["total"] + (expert_ffn_params if config["first_layer_no_moe"] else moe_params["total"]) + layer_norm_params
    
    # Other layers
    other_layers_params = (config["num_hidden_layers"] - 1) * (mla_params["total"] + moe_params["total"] + layer_norm_params)
    
    # Total parameters
    total_params = params["embedding"] + first_layer_params + other_layers_params + mtp_params
    
    # Calculate active parameters
    if config["use_moe"]:
        # For MoE, active params = non-expert params + (routed_experts_params * top_k / num_routed_experts)
        routed_experts_params = moe_params["routed_experts"] * (config["num_hidden_layers"] - 1)
        active_ratio = config["top_k"] / config["num_routed_experts"]
        active_params = total_params - routed_experts_params + (routed_experts_params * active_ratio)
    else:
        active_params = total_params
    
    # Print results
    print("=" * 80)
    print("QUASAR 140B MODEL PARAMETER ANALYSIS")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Attention Heads: {config['num_attention_heads']}")
    print(f"  Head Dimension: {config['head_dim']}")
    print(f"  MoE: {config['num_shared_experts']} shared + {config['num_routed_experts']} routed experts, top-{config['top_k']}")
    
    print(f"\nParameter Counts (in billions):")
    print(f"  Embedding Layer: {params['embedding'] / 1e9:.2f}B")
    print(f"  MLA (per layer): {mla_params['total'] / 1e9:.2f}B")
    print(f"  NSA (per layer): {nsa_params['total'] / 1e9:.2f}B")
    print(f"  MoE (per layer): {moe_params['total'] / 1e9:.2f}B")
    print(f"  First Layer: {first_layer_params / 1e9:.2f}B")
    print(f"  Other Layers (total): {other_layers_params / 1e9:.2f}B")
    print(f"  MTP Module: {mtp_params / 1e9:.2f}B")
    
    print(f"\nTotal Parameters: {total_params / 1e9:.2f}B")
    print(f"Active Parameters: {active_params / 1e9:.2f}B")
    print(f"Active/Total Ratio: {active_params / total_params * 100:.2f}%")
    
    # Detailed MoE breakdown
    print(f"\nMoE Detailed Breakdown:")
    print(f"  Shared Experts: {moe_params['shared_experts'] / 1e9:.2f}B per layer")
    print(f"  Routed Experts: {moe_params['routed_experts'] / 1e9:.2f}B per layer")
    print(f"  Router: {moe_params['router'] / 1e9:.2f}B per layer")
    
    # Detailed MLA breakdown
    print(f"\nMLA Detailed Breakdown:")
    print(f"  QKV Projection: {mla_params['w_qkv'] / 1e9:.2f}B")
    print(f"  KV Compression: {mla_params['w_dkv'] / 1e9:.2f}B")
    print(f"  Q Compression: {mla_params['w_dq'] / 1e9:.2f}B")
    print(f"  K Rotary: {mla_params['w_kr'] / 1e9:.2f}B")
    print(f"  Q Rotary: {mla_params['w_qr'] / 1e9:.2f}B")
    print(f"  Output Projection: {mla_params['w_o'] / 1e9:.2f}B")
    
    return {
        "total_params": total_params,
        "active_params": active_params,
        "config": config
    }

if __name__ == "__main__":
    calculate_quasar_parameters()
