import torch
from quasar import create_quasar_model
from tqdm import tqdm
import time
import argparse

def display_model_info(model_size="11B", use_nsa=False):
    """Display detailed information about the Quasar model."""
    print(f"\n{'='*50}")
    print(f"Creating Quasar {model_size} model...")
    print(f"{'='*50}\n")
    
    # Show loading bar for model creation
    with tqdm(total=100, desc="Creating model", ncols=100) as pbar:
        pbar.update(10)
        time.sleep(0.5)  # Simulate initialization
        
        # Create model
        model = create_quasar_model(use_nsa=use_nsa)
        
        pbar.update(70)
        time.sleep(0.5)  # Simulate processing
        
        # Calculate parameter counts
        total_params = sum(p.numel() for p in model.parameters())
        shared_params = sum(p.numel() for name, p in model.named_parameters() 
                          if not any(f'routed_experts.{i}' in name for i in range(model.config.num_routed_experts)))
        expert_params = (total_params - shared_params)
        active_params = shared_params + expert_params * (model.config.top_k / model.config.num_routed_experts)
        
        pbar.update(20)
    
    # Format parameter counts
    def format_params(params):
        if params >= 1e9:
            return f"{params/1e9:.2f}B"
        elif params >= 1e6:
            return f"{params/1e6:.2f}M"
        else:
            return f"{params:.2f}"
    
    # Print model information
    print(f"\n{'='*50}")
    print(f"QUASAR MODEL INFORMATION")
    print(f"{'='*50}")
    print(f"+ Model created with {format_params(total_params)} total parameters")
    print(f"+ Approximately {format_params(active_params)} active parameters per token")
    print(f"+ Using {'Native Sparse Attention' if use_nsa else 'Multi-Head Latent Attention'}")
    print(f"+ MoE configuration: {model.config.num_shared_experts} shared + {model.config.num_routed_experts} routed experts, top-{model.config.top_k} routing")
    print(f"+ Model dimensions: hidden_size={model.config.hidden_size}, layers={model.config.num_hidden_layers}, heads={model.config.num_attention_heads}")
    print(f"+ Token Temperature Modulation (TTM): {'Enabled' if model.config.use_ttm else 'Disabled'}")
    print(f"+ Multi-Token Prediction (MTP): {'Enabled' if hasattr(model.config, 'use_mtp') and model.config.use_mtp else 'Disabled'}")
    print(f"{'='*50}\n")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display Quasar model information")
    parser.add_argument("--use_nsa", action="store_true", help="Use Native Sparse Attention instead of MLA")
    args = parser.parse_args()
    
    model = display_model_info(use_nsa=args.use_nsa)
