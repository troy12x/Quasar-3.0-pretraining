import os
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quasar pretraining with simplified arguments")
    
    # Basic training settings
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb for logging")
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json", help="Path to tokenizer.json file")
    parser.add_argument("--use_nsa", action="store_true", help="Whether to use Native Sparse Attention")
    
    args = parser.parse_args()
    
    # Construct command to run pretrain.py with all arguments
    cmd = f"python pretrain.py " \
          f"--output_dir {args.output_dir} " \
          f"--batch_size {args.batch_size} " \
          f"--num_epochs {args.num_epochs} " \
          f"--learning_rate {args.learning_rate} " \
          f"--max_seq_length {args.max_seq_length} " \
          f"--tokenizer_path {args.tokenizer_path} "
    
    if args.use_wandb:
        cmd += "--use_wandb "
    
    if args.distributed:
        cmd += "--distributed "
        
    if args.use_nsa:
        cmd += "--use_nsa "
    
    # Print command and execute
    print(f"Running command: {cmd}")
    os.system(cmd)
