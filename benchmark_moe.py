import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from quasar import QuasarMoE, QuasarExpertFFN, create_quasar_model
from dataclasses import dataclass

@dataclass
class DenseModelConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    layer_norm_epsilon: float = 1e-5
    use_moe: bool = False
    
@dataclass
class MoEModelConfig:
    hidden_size: int = 2048
    intermediate_size: int = 8192
    layer_norm_epsilon: float = 1e-5
    use_moe: bool = True
    num_experts: int = 64
    num_shared_experts: int = 1
    num_routed_experts: int = 64
    top_k: int = 4
    load_balancing_alpha: float = 0.01
    load_balancing_gamma: float = 0.01

class DenseFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size)
        
    def forward(self, x):
        # SwiGLU activation (same as in QuasarExpertFFN)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def benchmark_inference(model, input_tensor, num_runs=10, warmup=3):
    """Benchmark inference time for a model."""
    # Warmup runs
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_tensor)
    
    # Timed runs
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time

def run_benchmark(batch_sizes, seq_lengths, device="cuda"):
    """Run benchmarks for different batch sizes and sequence lengths."""
    dense_config = DenseModelConfig()
    moe_config = MoEModelConfig()
    
    results = {
        "batch_size": [],
        "seq_length": [],
        "dense_time": [],
        "moe_time": [],
        "speedup": []
    }
    
    for batch_size in batch_sizes:
        for seq_length in seq_lengths:
            print(f"Benchmarking with batch_size={batch_size}, seq_length={seq_length}")
            
            # Create input tensor
            input_tensor = torch.randn(batch_size, seq_length, dense_config.hidden_size, device=device)
            
            # Create models
            dense_model = DenseFFN(dense_config).to(device)
            moe_model = QuasarMoE(moe_config).to(device)
            
            # Ensure same parameter initialization for fair comparison
            with torch.no_grad():
                # Initialize the first expert in MoE with the same weights as dense model
                for i in range(moe_config.num_shared_experts):
                    moe_model.shared_experts[i].w1.weight.copy_(dense_model.w1.weight)
                    moe_model.shared_experts[i].w2.weight.copy_(dense_model.w2.weight)
                    moe_model.shared_experts[i].w3.weight.copy_(dense_model.w3.weight)
                    
                    moe_model.shared_experts[i].w1.bias.copy_(dense_model.w1.bias)
                    moe_model.shared_experts[i].w2.bias.copy_(dense_model.w2.bias)
                    moe_model.shared_experts[i].w3.bias.copy_(dense_model.w3.bias)
            
            # Benchmark
            dense_time = benchmark_inference(dense_model, input_tensor)
            moe_time = benchmark_inference(moe_model, input_tensor)
            speedup = dense_time / moe_time
            
            # Store results
            results["batch_size"].append(batch_size)
            results["seq_length"].append(seq_length)
            results["dense_time"].append(dense_time * 1000)  # Convert to ms
            results["moe_time"].append(moe_time * 1000)  # Convert to ms
            results["speedup"].append(speedup)
            
            print(f"  Dense model: {dense_time*1000:.2f} ms")
            print(f"  MoE model: {moe_time*1000:.2f} ms")
            print(f"  Speedup: {speedup:.2f}x")
    
    return results

def plot_results(results):
    """Plot benchmark results."""
    batch_sizes = sorted(list(set(results["batch_size"])))
    seq_lengths = sorted(list(set(results["seq_length"])))
    
    # Create a figure with subplots
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot inference time comparison
    for seq_length in seq_lengths:
        dense_times = []
        moe_times = []
        batch_list = []
        
        for i in range(len(results["batch_size"])):
            if results["seq_length"][i] == seq_length:
                batch_list.append(results["batch_size"][i])
                dense_times.append(results["dense_time"][i])
                moe_times.append(results["moe_time"][i])
        
        axs[0].plot(batch_list, dense_times, 'o-', label=f'Dense (seq_len={seq_length})')
        axs[0].plot(batch_list, moe_times, 's--', label=f'MoE (seq_len={seq_length})')
    
    axs[0].set_xlabel('Batch Size')
    axs[0].set_ylabel('Inference Time (ms)')
    axs[0].set_title('Inference Time Comparison')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot speedup
    for seq_length in seq_lengths:
        speedups = []
        batch_list = []
        
        for i in range(len(results["batch_size"])):
            if results["seq_length"][i] == seq_length:
                batch_list.append(results["batch_size"][i])
                speedups.append(results["speedup"][i])
        
        axs[1].plot(batch_list, speedups, 'o-', label=f'seq_len={seq_length}')
    
    axs[1].set_xlabel('Batch Size')
    axs[1].set_ylabel('Speedup (Dense / MoE)')
    axs[1].set_title('MoE Speedup Factor')
    axs[1].legend()
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('moe_benchmark_results.png')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Benchmark QuasarMoE vs Dense model')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8, 16, 32])
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[128, 512, 1024, 2048])
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda or cpu)')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = 'cpu'
    
    # Run benchmarks
    results = run_benchmark(args.batch_sizes, args.seq_lengths, args.device)
    
    # Plot results
    plot_results(results)
    
    # Print summary
    avg_speedup = np.mean(results["speedup"])
    max_speedup = np.max(results["speedup"])
    print(f"\nSummary:")
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Maximum speedup: {max_speedup:.2f}x")
    
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv('moe_benchmark_results.csv', index=False)
    print("Results saved to moe_benchmark_results.csv")

if __name__ == "__main__":
    main()
