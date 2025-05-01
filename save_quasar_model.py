#!/usr/bin/env python3
"""
Initialize and save a Quasar model without training.
This script creates a Quasar model with the same configuration as in profile_memory.py
and saves it to disk in a format compatible with Hugging Face's transformers library.
"""

import os
import torch
import argparse
import json
from quasar import QuasarConfig, Quasar
from transformers import PreTrainedTokenizerFast

def save_model_to_hf_format(model, config, output_dir, tokenizer_path=None, use_safetensors=True, max_shard_size="10GB"):
    """
    Save a Quasar model to disk in the Hugging Face format.
    
    Args:
        model: The Quasar model to save
        config: The QuasarConfig object
        output_dir: Directory to save the model to
        tokenizer_path: Path to the tokenizer.json file (optional)
        use_safetensors: Whether to use safetensors format (more efficient)
        max_shard_size: Maximum size of each shard in bytes or as a string (e.g., "10GB")
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the model weights
    print(f"Saving model weights to {output_dir}")
    
    # Convert max_shard_size to bytes if it's a string
    if isinstance(max_shard_size, str):
        if max_shard_size.endswith("GB"):
            max_shard_size = int(float(max_shard_size[:-2]) * 1024 * 1024 * 1024)
        elif max_shard_size.endswith("MB"):
            max_shard_size = int(float(max_shard_size[:-2]) * 1024 * 1024)
        elif max_shard_size.endswith("KB"):
            max_shard_size = int(float(max_shard_size[:-2]) * 1024)
    
    # Get model state dict
    state_dict = model.state_dict()
    
    # Calculate total size of model
    total_size = sum(param.numel() * param.element_size() for param in model.parameters())
    print(f"Total model size: {total_size / 1024 / 1024 / 1024:.2f} GB")
    
    # Determine if we need to shard
    if total_size > max_shard_size:
        print(f"Model size exceeds {max_shard_size / 1024 / 1024 / 1024:.2f} GB, saving in sharded format")
        
        # Sort state dict items by size for more balanced shards
        state_dict_items = sorted(
            state_dict.items(),
            key=lambda x: x[1].numel() * x[1].element_size(),
            reverse=True
        )
        
        # Create shards
        shards = []
        current_shard = {}
        current_size = 0
        
        for name, param in state_dict_items:
            param_size = param.numel() * param.element_size()
            
            # If this single parameter is larger than max_shard_size, it gets its own shard
            if param_size > max_shard_size:
                if current_shard:
                    shards.append(current_shard)
                shards.append({name: param})
                current_shard = {}
                current_size = 0
                continue
                
            # If adding this param would exceed max_shard_size, start a new shard
            if current_size + param_size > max_shard_size and current_shard:
                shards.append(current_shard)
                current_shard = {}
                current_size = 0
                
            # Add param to current shard
            current_shard[name] = param
            current_size += param_size
        
        # Add the last shard if it's not empty
        if current_shard:
            shards.append(current_shard)
            
        # Save shards
        for i, shard in enumerate(shards):
            if use_safetensors:
                try:
                    from safetensors.torch import save_file
                    shard_file = os.path.join(output_dir, f"model-{i:05d}-of-{len(shards):05d}.safetensors")
                    save_file(shard, shard_file)
                except ImportError:
                    print("safetensors not available, falling back to PyTorch format")
                    shard_file = os.path.join(output_dir, f"pytorch_model-{i:05d}-of-{len(shards):05d}.bin")
                    torch.save(shard, shard_file)
            else:
                shard_file = os.path.join(output_dir, f"pytorch_model-{i:05d}-of-{len(shards):05d}.bin")
                torch.save(shard, shard_file)
                
        # Create index file
        index = {"metadata": {}, "weight_map": {}}
        for i, shard in enumerate(shards):
            for name in shard.keys():
                if use_safetensors:
                    index["weight_map"][name] = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
                else:
                    index["weight_map"][name] = f"pytorch_model-{i:05d}-of-{len(shards):05d}.bin"
                    
        with open(os.path.join(output_dir, "pytorch_model.bin.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    else:
        # Save as a single file
        if use_safetensors:
            try:
                from safetensors.torch import save_file
                save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
            except ImportError:
                print("safetensors not available, falling back to PyTorch format")
                torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
        else:
            torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save the config
    config_dict = config.to_dict()
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    
    # Create a model card
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"# Quasar-3.0 Model\n\n")
        f.write(f"This is a Quasar-3.0 model with the following configuration:\n\n")
        f.write(f"- Hidden size: {config.hidden_size}\n")
        f.write(f"- Intermediate size: {config.intermediate_size}\n")
        f.write(f"- Num hidden layers: {config.num_hidden_layers}\n")
        f.write(f"- Num attention heads: {config.num_attention_heads}\n")
        # Only add key-value heads if the attribute exists
        if hasattr(config, 'num_key_value_heads'):
            f.write(f"- Num key value heads: {config.num_key_value_heads}\n")
        else:
            f.write(f"- Num key value heads: {config.num_attention_heads} (same as attention heads)\n")
        f.write(f"- MoE: {config.use_moe}\n")
        if config.use_moe:
            f.write(f"- Num experts: {config.num_experts}\n")
            f.write(f"- Num experts per token: {config.num_experts_per_token}\n")
            f.write(f"- Memory efficient: {config.use_memory_efficient_impl}\n")
    
    # Copy tokenizer if provided
    if tokenizer_path and os.path.exists(tokenizer_path):
        print(f"Copying tokenizer from {tokenizer_path}")
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        # If tokenizer_path is a directory, copy all files
        if os.path.isdir(tokenizer_path):
            import shutil
            for file in os.listdir(tokenizer_path):
                src_file = os.path.join(tokenizer_path, file)
                if os.path.isfile(src_file):
                    shutil.copy2(src_file, tokenizer_dir)
        else:
            # If it's a single file, copy it
            import shutil
            shutil.copy2(tokenizer_path, os.path.join(tokenizer_dir, "tokenizer.json"))
            
        # Create a tokenizer config
        if not os.path.exists(os.path.join(tokenizer_dir, "tokenizer_config.json")):
            tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            tokenizer.save_pretrained(tokenizer_dir)
    
    print(f"Model saved successfully to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Initialize and save a Quasar model without training")
    parser.add_argument("--output_dir", type=str, default="./quasar_model", help="Directory to save the model to")
    parser.add_argument("--hidden_size", type=int, default=2048, help="Hidden size of the model")
    parser.add_argument("--num_hidden_layers", type=int, default=24, help="Number of hidden layers")
    parser.add_argument("--num_attention_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--intermediate_size", type=int, default=5504, help="Intermediate size")
    parser.add_argument("--use_moe", action="store_true", help="Use Mixture of Experts")
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts (if use_moe is True)")
    parser.add_argument("--num_experts_per_token", type=int, default=2, help="Number of experts per token (if use_moe is True)")
    parser.add_argument("--memory_efficient", action="store_true", help="Use memory-efficient MoE implementation")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.json file or directory")
    parser.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="bf16", help="Data type for model weights")
    parser.add_argument("--use_safetensors", action="store_true", help="Use safetensors format for saving weights")
    parser.add_argument("--max_shard_size", type=str, default="10GB", help="Maximum size of each shard (e.g., '10GB', '500MB')")
    parser.add_argument("--cpu_offload", action="store_true", help="Initialize model on CPU to save GPU memory")
    args = parser.parse_args()
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set the dtype
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    elif args.dtype == "bf16":
        dtype = torch.bfloat16
    
    # Create the config
    config = QuasarConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        use_memory_efficient_impl=args.memory_efficient,
    )
    
    # Print the config
    print("\nModel Configuration:")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Num hidden layers: {config.num_hidden_layers}")
    print(f"Num attention heads: {config.num_attention_heads}")
    # Check if num_key_value_heads attribute exists
    if hasattr(config, 'num_key_value_heads'):
        print(f"Num key value heads: {config.num_key_value_heads}")
    else:
        print(f"Num key value heads: {config.num_attention_heads} (same as attention heads)")
    print(f"MoE: {config.use_moe}")
    if config.use_moe:
        print(f"Num experts: {config.num_experts}")
        print(f"Num experts per token: {config.num_experts_per_token}")
        print(f"Memory efficient: {config.use_memory_efficient_impl}")
    
    # Calculate model size
    param_count = (
        config.hidden_size * config.intermediate_size * 2 * config.num_hidden_layers +  # MLP
        config.hidden_size * config.hidden_size * 4 * config.num_hidden_layers +  # Attention
        config.hidden_size * 2 * config.num_hidden_layers  # Layer norms
    )
    if config.use_moe:
        # For MoE, the intermediate size is multiplied by the number of experts
        param_count += config.hidden_size * config.intermediate_size * 2 * config.num_hidden_layers * (config.num_experts - 1)
    
    print(f"\nApproximate parameter count: {param_count / 1e9:.2f} billion")
    
    # Create the model
    print("\nInitializing model...")
    
    if args.cpu_offload:
        print("Initializing model on CPU to save GPU memory...")
        with torch.device('cpu'):
            model = Quasar(config)
            # Convert to desired dtype on CPU
            model = model.to(dtype)
            print("Model initialized on CPU")
    else:
        # Initialize directly on GPU
        model = Quasar(config)
        # Move to device and convert to desired dtype
        model = model.to(device).to(dtype)
        print(f"Model initialized on {device}")
    
    # Save the model
    save_model_to_hf_format(
        model, 
        config, 
        args.output_dir, 
        args.tokenizer_path,
        use_safetensors=args.use_safetensors,
        max_shard_size=args.max_shard_size
    )

if __name__ == "__main__":
    main()
