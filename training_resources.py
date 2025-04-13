"""
Training resource calculator for Quasar model.
This script calculates the computational resources required to train the Quasar model on 10 trillion tokens.
"""

import math
import os
from calculate_parameters import calculate_quasar_parameters

def calculate_training_resources():
    """Calculate the computational resources required to train the Quasar model on 10 trillion tokens."""
    # Get model parameters
    model_params = calculate_quasar_parameters()
    config = model_params["config"]
    total_params = model_params["total_params"]
    active_params = model_params["active_params"]
    
    # Check if DeepGEMM is available
    deepgemm_available = os.path.exists("DeepGEMM")
    
    # Training configuration with H100 GPUs and DeepGEMM kernel optimizations
    training_config = {
        "total_tokens": 10e12,  # 10 trillion tokens
        "batch_size": 2048,  # Global batch size
        "sequence_length": 4096,  # Sequence length
        "gradient_accumulation_steps": 1,  # Gradient accumulation steps
        "mixed_precision": "fp8",  # FP8 mixed precision training (with DeepGEMM)
        "optimizer": "AdamW",  # Optimizer
        "gpu_type": "H100",  # GPU type (H100 as specified by user)
        "gpu_memory": 80e9,  # H100 memory in bytes (80GB)
        "gpu_flops": 989e12,  # H100 FP8 FLOPS
        "gpu_count": 2048,  # Number of GPUs (same scale as DeepSeek-V3 paper)
        "deepgemm_speedup": 1.3 if deepgemm_available else 1.0,  # Speedup from DeepGEMM (based on benchmarks)
        "gpu_efficiency": 0.7 if deepgemm_available else 0.5,  # Higher efficiency with DeepGEMM
        "gpu_cost_per_hour": 3.5,  # Approximate cost per H100 GPU hour in USD
    }
    
    # Calculate FLOPs per token
    flops_per_token = calculate_flops_per_token(config, training_config)
    
    # Calculate total FLOPs for training
    total_flops = flops_per_token * training_config["total_tokens"]
    
    # Apply DeepGEMM speedup if available
    if deepgemm_available:
        total_flops = total_flops / training_config["deepgemm_speedup"]
        print(f"Applying DeepGEMM speedup: {training_config['deepgemm_speedup']}x faster computation")
    
    # Calculate memory requirements
    memory_requirements = calculate_memory_requirements(config, total_params, active_params, training_config)
    
    # Calculate training time and cost with H100 GPUs
    training_time_cost = calculate_training_time_cost(total_flops, memory_requirements, training_config, 
                                                     fixed_gpu_count=training_config["gpu_count"])
    
    # Calculate additional scenarios with different GPU counts
    training_time_cost_3000 = calculate_training_time_cost(total_flops, memory_requirements, training_config, fixed_gpu_count=3000)
    training_time_cost_4000 = calculate_training_time_cost(total_flops, memory_requirements, training_config, fixed_gpu_count=4000)
    
    # Print results
    print("=" * 80)
    print("QUASAR TRAINING RESOURCE ANALYSIS (WITH DEEPGEMM OPTIMIZATIONS)")
    print("=" * 80)
    
    print(f"\nModel Configuration:")
    print(f"  Total Parameters: {total_params / 1e9:.2f}B")
    print(f"  Active Parameters: {active_params / 1e9:.2f}B")
    print(f"  Hidden Size: {config['hidden_size']}")
    print(f"  Layers: {config['num_hidden_layers']}")
    print(f"  Attention Heads: {config['num_attention_heads']}")
    print(f"  MoE: {config['num_shared_experts']} shared + {config['num_routed_experts']} routed experts, top-{config['top_k']}")
    
    print(f"\nTraining Configuration:")
    print(f"  Total Tokens: {training_config['total_tokens'] / 1e12:.1f} trillion")
    print(f"  Batch Size: {training_config['batch_size']}")
    print(f"  Sequence Length: {training_config['sequence_length']}")
    print(f"  Mixed Precision: {training_config['mixed_precision']}")
    print(f"  GPU Type: {training_config['gpu_type']}")
    print(f"  GPU Count: {training_config['gpu_count']}")
    print(f"  DeepGEMM Optimizations: {'Enabled' if deepgemm_available else 'Not Available'}")
    
    print(f"\nComputational Requirements:")
    print(f"  FLOPs per Token: {flops_per_token / 1e9:.2f} GFLOPS")
    print(f"  Total Training FLOPs: {total_flops / 1e18:.2f} ExaFLOPS")
    if deepgemm_available:
        print(f"  DeepGEMM Speedup: {training_config['deepgemm_speedup']:.1f}x")
    
    print(f"\nMemory Requirements:")
    print(f"  Model States: {memory_requirements['model_states'] / 1e9:.2f} GB")
    print(f"  Optimizer States: {memory_requirements['optimizer_states'] / 1e9:.2f} GB")
    print(f"  Activations: {memory_requirements['activations'] / 1e9:.2f} GB")
    print(f"  Total Memory per GPU: {memory_requirements['per_gpu'] / 1e9:.2f} GB")
    print(f"  Minimum GPUs Required (Memory): {memory_requirements['min_gpus']}")
    
    print(f"\nTraining Time and Cost (With DeepGEMM Optimizations):")
    print(f"  GPU Count: {training_time_cost['gpu_count']}")
    print(f"  Training Time: {training_time_cost['training_time_days']:.2f} days")
    print(f"  Total GPU Hours: {training_time_cost['gpu_hours']:,.0f}")
    print(f"  Cost per GPU Hour: ${training_config['gpu_cost_per_hour']:.2f}")
    print(f"  Total Cost: ${training_time_cost['cost'] / 1e6:.2f} million")
    
    print(f"\nAlternative Scenarios:")
    print(f"  With 3,000 H100 GPUs:")
    print(f"    Training Time: {training_time_cost_3000['training_time_days']:.2f} days")
    print(f"    Total Cost: ${training_time_cost_3000['cost'] / 1e6:.2f} million")
    print(f"  With 4,000 H100 GPUs:")
    print(f"    Training Time: {training_time_cost_4000['training_time_days']:.2f} days")
    print(f"    Total Cost: ${training_time_cost_4000['cost'] / 1e6:.2f} million")
    
    return {
        "model_params": model_params,
        "flops_per_token": flops_per_token,
        "total_flops": total_flops,
        "memory_requirements": memory_requirements,
        "training_time_cost": training_time_cost,
        "training_time_cost_3000": training_time_cost_3000,
        "training_time_cost_4000": training_time_cost_4000,
        "deepgemm_available": deepgemm_available,
        "deepgemm_speedup": training_config["deepgemm_speedup"] if deepgemm_available else 1.0
    }

def calculate_flops_per_token(config, training_config):
    """Calculate the number of FLOPs required to process a single token.
    
    Note: This calculation incorporates optimizations from DeepGEMM for FP8 precision
    and efficient MoE operations if DeepGEMM is available.
    """
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    head_dim = config["head_dim"]
    seq_len = training_config["sequence_length"]
    
    # MoE parameters
    num_shared_experts = config["num_shared_experts"]
    num_routed_experts = config["num_routed_experts"]
    top_k = config["top_k"]
    intermediate_size = config["intermediate_size"]
    
    # MLA parameters
    kv_compressed_dim = config["kv_compressed_dim"]
    query_compressed_dim = config["query_compressed_dim"]
    
    # Calculate total and active parameters
    total_params, active_params = estimate_parameter_counts(config)
    
    # FP8 precision factor (reduces computation and memory requirements)
    fp8_factor = 0.8 if training_config["mixed_precision"] == "fp8" else 1.0
    
    # 1. Forward pass FLOPs
    
    # Attention FLOPs per layer
    # QKV projections
    qkv_flops = 3 * hidden_size * hidden_size * 2  # 2 FLOPs per multiply-add
    
    # Compressed latent vectors
    kv_compress_flops = hidden_size * kv_compressed_dim * 2
    q_compress_flops = hidden_size * query_compressed_dim * 2
    
    # Rotary projections
    kr_flops = kv_compressed_dim * num_heads * head_dim * 2
    qr_flops = query_compressed_dim * num_heads * head_dim * 2
    
    # Attention computation (Q*K^T, softmax, attention*V)
    # For MLA, this is done in the compressed space
    attn_flops = (
        # Q*K^T
        num_heads * seq_len * head_dim * seq_len * 2 +
        # Softmax (approximation)
        num_heads * seq_len * seq_len * 10 +
        # Attention*V
        num_heads * seq_len * seq_len * head_dim * 2
    )
    
    # Output projection
    output_flops = hidden_size * hidden_size * 2
    
    # Total attention FLOPs
    attention_flops = qkv_flops + kv_compress_flops + q_compress_flops + kr_flops + qr_flops + attn_flops + output_flops
    
    # MoE FLOPs per layer (considering only active experts)
    # Router
    router_flops = hidden_size * num_routed_experts * 2
    
    # Shared experts
    shared_expert_flops = num_shared_experts * (
        # W1
        hidden_size * intermediate_size * 2 +
        # Activation (SwiGLU)
        intermediate_size * 5 +
        # W2
        intermediate_size * hidden_size * 2 +
        # W3
        hidden_size * intermediate_size * 2
    )
    
    # Routed experts (only top-k are active per token)
    active_expert_ratio = top_k / num_routed_experts
    routed_expert_flops = active_expert_ratio * num_routed_experts * (
        # W1
        hidden_size * intermediate_size * 2 +
        # Activation (SwiGLU)
        intermediate_size * 5 +
        # W2
        intermediate_size * hidden_size * 2 +
        # W3
        hidden_size * intermediate_size * 2
    )
    
    # Total MoE FLOPs
    moe_flops = router_flops + shared_expert_flops + routed_expert_flops
    
    # Layer norm FLOPs (2 per layer)
    layer_norm_flops = 2 * hidden_size * 10  # Approximation for normalization operations
    
    # Total FLOPs per layer
    flops_per_layer = attention_flops + moe_flops + layer_norm_flops
    
    # Embedding and output layer FLOPs (amortized over sequence length)
    embedding_flops = 0  # Negligible compared to other operations
    output_layer_flops = hidden_size * config["vocab_size"] * 2 / seq_len  # Amortized per token
    
    # Total forward pass FLOPs
    forward_flops = num_layers * flops_per_layer + embedding_flops + output_layer_flops
    
    # 2. Backward pass FLOPs (typically 2-3x the forward pass)
    backward_flops = 2.5 * forward_flops
    
    # 3. Optimizer update FLOPs (typically small compared to forward/backward)
    optimizer_flops = 10 * active_params  # Approximation for AdamW updates
    
    # Apply FP8 precision factor if using FP8
    if training_config["mixed_precision"] == "fp8":
        forward_flops *= fp8_factor
        backward_flops *= fp8_factor
    
    # Total FLOPs per token
    total_flops_per_token = forward_flops + backward_flops + optimizer_flops / seq_len
    
    return total_flops_per_token

def calculate_memory_requirements(config, total_params, active_params, training_config):
    """Calculate the memory requirements for training."""
    # Memory for model parameters (based on precision)
    if training_config["mixed_precision"] == "fp8":
        model_size = total_params * 1  # 1 byte per parameter for FP8
    else:  # BF16
        model_size = total_params * 2  # 2 bytes per parameter for BF16
    
    # Memory for optimizer states (assuming AdamW with 8 bytes per parameter)
    # For MoE, we only need optimizer states for active parameters
    # With DeepGEMM's optimizations, we can reduce memory overhead
    if training_config["mixed_precision"] == "fp8" and os.path.exists("DeepGEMM"):
        optimizer_size = active_params * 6  # Reduced memory with DeepGEMM optimizations
    else:
        optimizer_size = active_params * 8  # 8 bytes per parameter (momentum + variance)
    
    # Memory for activations
    # Rough approximation: 5x the size of a single forward pass
    batch_size = training_config["batch_size"]
    seq_len = training_config["sequence_length"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    
    # Size of activations for a single sequence
    if training_config["mixed_precision"] == "fp8":
        # FP8 activations with DeepGEMM's fine-grained quantization
        activation_size_per_seq = 5 * num_layers * seq_len * hidden_size * 1  # 1 byte per activation for FP8
    else:
        activation_size_per_seq = 5 * num_layers * seq_len * hidden_size * 2  # 2 bytes per activation for BF16
    
    # Total activation size for the batch
    activation_size = activation_size_per_seq * batch_size
    
    # Total memory requirement
    total_memory = model_size + optimizer_size + activation_size
    
    # Memory per GPU (assuming ZeRO-3 style sharding)
    gpu_memory = training_config["gpu_memory"]
    min_gpus_for_memory = math.ceil(total_memory / gpu_memory)
    
    # Memory per GPU with optimal sharding
    memory_per_gpu = total_memory / min_gpus_for_memory
    
    return {
        "model_states": model_size,
        "optimizer_states": optimizer_size,
        "activations": activation_size,
        "total": total_memory,
        "per_gpu": memory_per_gpu,
        "min_gpus": min_gpus_for_memory
    }

def calculate_training_time_cost(total_flops, memory_requirements, training_config, fixed_gpu_count=None):
    """Calculate the training time and cost.
    
    Args:
        total_flops: Total FLOPs required for training
        memory_requirements: Memory requirements dictionary
        training_config: Training configuration dictionary
        fixed_gpu_count: If provided, use this fixed number of GPUs instead of calculating optimal count
    """
    # GPU performance
    gpu_flops = training_config["gpu_flops"] * training_config["gpu_efficiency"]
    gpu_cost_per_hour = training_config["gpu_cost_per_hour"]
    
    # Minimum GPUs required for memory
    min_gpus = memory_requirements["min_gpus"]
    
    # Calculate GPU count
    if fixed_gpu_count is not None:
        # Use fixed GPU count if provided
        if fixed_gpu_count < min_gpus:
            print(f"Warning: Fixed GPU count {fixed_gpu_count} is less than minimum required {min_gpus}")
            print(f"Using minimum required GPUs: {min_gpus}")
            gpu_count = min_gpus
        else:
            gpu_count = fixed_gpu_count
    else:
        # Calculate optimal GPU count (balance between memory and compute)
        # In practice, this would be determined by scaling tests
        gpu_count = max(min_gpus, 256)  # Assuming a large-scale training setup
    
    # Total compute power
    total_compute_flops = gpu_count * gpu_flops
    
    # Training time in seconds
    training_time_seconds = total_flops / total_compute_flops
    training_time_hours = training_time_seconds / 3600
    training_time_days = training_time_hours / 24
    
    # Total GPU hours
    gpu_hours = gpu_count * training_time_hours
    
    # Total cost
    cost = gpu_hours * gpu_cost_per_hour
    
    return {
        "gpu_count": gpu_count,
        "training_time_seconds": training_time_seconds,
        "training_time_hours": training_time_hours,
        "training_time_days": training_time_days,
        "gpu_hours": gpu_hours,
        "cost": cost
    }

def estimate_parameter_counts(config):
    """Estimate the total and active parameter counts based on the configuration."""
    # Model dimensions
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    vocab_size = config["vocab_size"]
    
    # MoE parameters
    num_shared_experts = config["num_shared_experts"]
    num_routed_experts = config["num_routed_experts"]
    top_k = config["top_k"]
    intermediate_size = config["intermediate_size"]
    use_moe = config["use_moe"]
    first_layer_no_moe = config["first_layer_no_moe"]
    
    # Embedding layer
    embedding_params = vocab_size * hidden_size
    
    # Attention parameters per layer
    attention_params = 4 * hidden_size * hidden_size  # Approximation for MLA
    
    # FFN parameters for non-MoE layer
    ffn_params = 3 * hidden_size * intermediate_size
    
    # MoE parameters per layer
    shared_expert_params = num_shared_experts * 3 * hidden_size * intermediate_size
    routed_expert_params = num_routed_experts * 3 * hidden_size * intermediate_size
    router_params = hidden_size * num_routed_experts
    moe_params = shared_expert_params + routed_expert_params + router_params
    
    # Layer norm parameters
    layer_norm_params = 2 * hidden_size
    
    # Calculate total parameters
    if use_moe:
        if first_layer_no_moe:
            first_layer_params = attention_params + ffn_params + layer_norm_params
            other_layers_params = (num_layers - 1) * (attention_params + moe_params + layer_norm_params)
        else:
            first_layer_params = attention_params + moe_params + layer_norm_params
            other_layers_params = (num_layers - 1) * (attention_params + moe_params + layer_norm_params)
    else:
        layer_params = attention_params + ffn_params + layer_norm_params
        first_layer_params = layer_params
        other_layers_params = (num_layers - 1) * layer_params
    
    # MTP module parameters (similar to a transformer layer)
    mtp_params = attention_params + (moe_params if use_moe else ffn_params) + layer_norm_params
    
    # Total parameters
    total_params = embedding_params + first_layer_params + other_layers_params + mtp_params
    
    # Calculate active parameters
    if use_moe:
        # For MoE, active params = non-expert params + (routed_experts_params * top_k / num_routed_experts)
        active_ratio = top_k / num_routed_experts
        routed_experts_total = routed_expert_params * (num_layers - (1 if first_layer_no_moe else 0))
        active_params = total_params - routed_experts_total + (routed_experts_total * active_ratio)
    else:
        active_params = total_params
    
    return total_params, active_params

if __name__ == "__main__":
    calculate_training_resources()
