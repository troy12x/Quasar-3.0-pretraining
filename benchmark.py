import os
import time
import torch
import argparse
import numpy as np
import psutil
import gc
from torch.cuda import max_memory_allocated, reset_peak_memory_stats
from quasar import create_1b_quasar_model, QuasarConfig

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_gb = process.memory_info().rss / (1024 ** 3)
    return memory_gb

def benchmark_forward_pass(model, batch_size, seq_length, num_iterations=10, warmup=3):
    """Benchmark forward pass of the model"""
    device = next(model.parameters()).device
    
    # Create dummy inputs
    input_ids = torch.randint(0, 128000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    labels = torch.randint(0, 128000, (batch_size, seq_length), device=device)
    
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    
    # Benchmark forward pass
    torch.cuda.synchronize()
    reset_peak_memory_stats()
    
    start_time = time.time()
    for _ in range(num_iterations):
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    # Calculate metrics
    avg_time = (end_time - start_time) / num_iterations
    tokens_per_second = (batch_size * seq_length) / avg_time
    peak_memory = max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory
    }

def benchmark_backward_pass(model, batch_size, seq_length, num_iterations=10, warmup=3):
    """Benchmark forward+backward pass of the model"""
    device = next(model.parameters()).device
    
    # Create dummy inputs
    input_ids = torch.randint(0, 128000, (batch_size, seq_length), device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    labels = torch.randint(0, 128000, (batch_size, seq_length), device=device)
    
    # Warmup
    for _ in range(warmup):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        # Clear gradients
        model.zero_grad()
    
    # Benchmark forward+backward pass
    torch.cuda.synchronize()
    reset_peak_memory_stats()
    
    start_time = time.time()
    for _ in range(num_iterations):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs["loss"]
        loss.backward()
        torch.cuda.synchronize()
        # Clear gradients
        model.zero_grad()
    
    end_time = time.time()
    
    # Calculate metrics
    avg_time = (end_time - start_time) / num_iterations
    tokens_per_second = (batch_size * seq_length) / avg_time
    peak_memory = max_memory_allocated() / (1024 ** 3)  # GB
    
    return {
        "avg_time_ms": avg_time * 1000,
        "tokens_per_second": tokens_per_second,
        "peak_memory_gb": peak_memory
    }

def benchmark_layer_by_layer(model, batch_size, seq_length):
    """Benchmark each layer of the model separately"""
    device = next(model.parameters()).device
    
    # Create dummy inputs
    hidden_states = torch.randn(batch_size, seq_length, model.config.hidden_size, device=device)
    attention_mask = torch.ones(batch_size, seq_length, device=device)
    
    layer_metrics = []
    
    # Benchmark each transformer layer
    for i, layer in enumerate(model.layers):
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = layer(hidden_states, attention_mask)
        
        # Benchmark
        torch.cuda.synchronize()
        reset_peak_memory_stats()
        
        start_time = time.time()
        for _ in range(5):
            with torch.no_grad():
                hidden_states = layer(hidden_states, attention_mask)
            torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Calculate metrics
        avg_time = (end_time - start_time) / 5
        peak_memory = max_memory_allocated() / (1024 ** 3)  # GB
        
        layer_metrics.append({
            "layer_idx": i,
            "avg_time_ms": avg_time * 1000,
            "peak_memory_gb": peak_memory
        })
    
    return layer_metrics

def benchmark_attention_mechanisms(batch_size, seq_length, num_iterations=10):
    """Benchmark different attention mechanisms"""
    results = {}
    
    # Benchmark MLA
    print("\n=== Benchmarking Multi-Head Latent Attention ===")
    model_mla = create_1b_quasar_model(use_nsa=False).cuda()
    model_mla.eval()
    
    print("Forward pass benchmark:")
    results["mla_forward"] = benchmark_forward_pass(model_mla, batch_size, seq_length, num_iterations)
    print(f"Average time: {results['mla_forward']['avg_time_ms']:.2f} ms")
    print(f"Tokens per second: {results['mla_forward']['tokens_per_second']:.2f}")
    print(f"Peak memory: {results['mla_forward']['peak_memory_gb']:.2f} GB")
    
    print("\nBackward pass benchmark:")
    results["mla_backward"] = benchmark_backward_pass(model_mla, batch_size, seq_length, num_iterations)
    print(f"Average time: {results['mla_backward']['avg_time_ms']:.2f} ms")
    print(f"Tokens per second: {results['mla_backward']['tokens_per_second']:.2f}")
    print(f"Peak memory: {results['mla_backward']['peak_memory_gb']:.2f} GB")
    
    # Clear memory
    del model_mla
    gc.collect()
    torch.cuda.empty_cache()
    
    # Benchmark NSA
    print("\n=== Benchmarking Native Sparse Attention ===")
    model_nsa = create_1b_quasar_model(use_nsa=True).cuda()
    model_nsa.eval()
    
    print("Forward pass benchmark:")
    results["nsa_forward"] = benchmark_forward_pass(model_nsa, batch_size, seq_length, num_iterations)
    print(f"Average time: {results['nsa_forward']['avg_time_ms']:.2f} ms")
    print(f"Tokens per second: {results['nsa_forward']['tokens_per_second']:.2f}")
    print(f"Peak memory: {results['nsa_forward']['peak_memory_gb']:.2f} GB")
    
    print("\nBackward pass benchmark:")
    results["nsa_backward"] = benchmark_backward_pass(model_nsa, batch_size, seq_length, num_iterations)
    print(f"Average time: {results['nsa_backward']['avg_time_ms']:.2f} ms")
    print(f"Tokens per second: {results['nsa_backward']['tokens_per_second']:.2f}")
    print(f"Peak memory: {results['nsa_backward']['peak_memory_gb']:.2f} GB")
    
    # Clear memory
    del model_nsa
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def benchmark_sequence_lengths(batch_size, seq_lengths, use_nsa=False):
    """Benchmark model with different sequence lengths"""
    results = {}
    
    print(f"\n=== Benchmarking {'NSA' if use_nsa else 'MLA'} with Different Sequence Lengths ===")
    model = create_1b_quasar_model(use_nsa=use_nsa).cuda()
    model.eval()
    
    for seq_length in seq_lengths:
        print(f"\nSequence length: {seq_length}")
        
        # Forward pass
        results[f"seq_{seq_length}_forward"] = benchmark_forward_pass(model, batch_size, seq_length, num_iterations=5)
        print(f"Forward - Average time: {results[f'seq_{seq_length}_forward']['avg_time_ms']:.2f} ms")
        print(f"Forward - Tokens per second: {results[f'seq_{seq_length}_forward']['tokens_per_second']:.2f}")
        print(f"Forward - Peak memory: {results[f'seq_{seq_length}_forward']['peak_memory_gb']:.2f} GB")
        
        # Backward pass
        results[f"seq_{seq_length}_backward"] = benchmark_backward_pass(model, batch_size, seq_length, num_iterations=5)
        print(f"Backward - Average time: {results[f'seq_{seq_length}_backward']['avg_time_ms']:.2f} ms")
        print(f"Backward - Tokens per second: {results[f'seq_{seq_length}_backward']['tokens_per_second']:.2f}")
        print(f"Backward - Peak memory: {results[f'seq_{seq_length}_backward']['peak_memory_gb']:.2f} GB")
    
    # Clear memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def benchmark_layer_breakdown(batch_size, seq_length, use_nsa=False):
    """Benchmark each layer of the model"""
    print(f"\n=== Layer-by-Layer Breakdown ({'NSA' if use_nsa else 'MLA'}) ===")
    model = create_1b_quasar_model(use_nsa=use_nsa).cuda()
    model.eval()
    
    layer_metrics = benchmark_layer_by_layer(model, batch_size, seq_length)
    
    print("\nLayer-by-layer performance:")
    for metric in layer_metrics:
        print(f"Layer {metric['layer_idx']}: {metric['avg_time_ms']:.2f} ms, {metric['peak_memory_gb']:.2f} GB")
    
    # Clear memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return layer_metrics

def main():
    parser = argparse.ArgumentParser(description="Benchmark Quasar model")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--seq_length", type=int, default=512, help="Sequence length for benchmarking")
    parser.add_argument("--num_iterations", type=int, default=10, help="Number of iterations for benchmarking")
    parser.add_argument("--benchmark_attention", action="store_true", help="Benchmark different attention mechanisms")
    parser.add_argument("--benchmark_seq_lengths", action="store_true", help="Benchmark different sequence lengths")
    parser.add_argument("--benchmark_layers", action="store_true", help="Benchmark layer-by-layer breakdown")
    parser.add_argument("--use_nsa", action="store_true", help="Use Native Sparse Attention for benchmarks")
    args = parser.parse_args()
    
    # Print system info
    print("=== System Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"CPU count: {os.cpu_count()}")
    print(f"RAM: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    print(f"Current memory usage: {get_memory_usage():.2f} GB")
    
    # Run benchmarks
    if args.benchmark_attention:
        benchmark_attention_mechanisms(args.batch_size, args.seq_length, args.num_iterations)
    
    if args.benchmark_seq_lengths:
        seq_lengths = [128, 512, 1024, 2048, 4096]
        benchmark_sequence_lengths(args.batch_size, seq_lengths, args.use_nsa)
    
    if args.benchmark_layers:
        benchmark_layer_breakdown(args.batch_size, args.seq_length, args.use_nsa)
    
    # If no specific benchmark is selected, run a basic benchmark
    if not (args.benchmark_attention or args.benchmark_seq_lengths or args.benchmark_layers):
        print("\n=== Basic Benchmark ===")
        model = create_1b_quasar_model(use_nsa=args.use_nsa).cuda()
        model.eval()
        
        print("Forward pass benchmark:")
        forward_metrics = benchmark_forward_pass(model, args.batch_size, args.seq_length, args.num_iterations)
        print(f"Average time: {forward_metrics['avg_time_ms']:.2f} ms")
        print(f"Tokens per second: {forward_metrics['tokens_per_second']:.2f}")
        print(f"Peak memory: {forward_metrics['peak_memory_gb']:.2f} GB")
        
        print("\nBackward pass benchmark:")
        backward_metrics = benchmark_backward_pass(model, args.batch_size, args.seq_length, args.num_iterations)
        print(f"Average time: {backward_metrics['avg_time_ms']:.2f} ms")
        print(f"Tokens per second: {backward_metrics['tokens_per_second']:.2f}")
        print(f"Peak memory: {backward_metrics['peak_memory_gb']:.2f} GB")
        
        # Clear memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
