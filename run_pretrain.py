import os
import argparse
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Quasar pretraining with advanced training options")
    
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
    
    # Advanced training options
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, 
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--precision", type=str, choices=["fp32", "fp16", "bf16", "tf32"], default="fp32",
                        help="Precision for training (fp32, fp16, bf16, tf32)")
    parser.add_argument("--gradient_checkpointing", action="store_true", 
                        help="Enable gradient checkpointing to save memory at the expense of speed")
    parser.add_argument("--deepspeed", action="store_true", 
                        help="Enable DeepSpeed for distributed training optimizations")
    parser.add_argument("--deepspeed_config", type=str, default="./deepspeed_config.json",
                        help="Path to DeepSpeed configuration file")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adafactor"],
                        help="Optimizer to use for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay coefficient for L2 regularization")
    parser.add_argument("--warmup_steps", type=int, default=1000, 
                        help="Number of steps for linear warmup")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Save checkpoint every X updates steps")
    parser.add_argument("--eval_steps", type=int, default=1000, 
                        help="Run evaluation every X updates steps")
    parser.add_argument("--logging_steps", type=int, default=100, 
                        help="Log metrics every X updates steps")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for initialization")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for caching datasets")
    parser.add_argument("--run_name", type=str, default="quasar-pretrain",
                        help="Name of the run for wandb logging")
    
    args = parser.parse_args()
    
    # Construct command to run pretrain.py with all arguments
    cmd = f"python pretrain.py " \
          f"--output_dir {args.output_dir} " \
          f"--batch_size {args.batch_size} " \
          f"--num_epochs {args.num_epochs} " \
          f"--learning_rate {args.learning_rate} " \
          f"--max_seq_length {args.max_seq_length} " \
          f"--tokenizer_path {args.tokenizer_path} " \
          f"--gradient_accumulation_steps {args.gradient_accumulation_steps} " \
          f"--precision {args.precision} " \
          f"--optimizer {args.optimizer} " \
          f"--weight_decay {args.weight_decay} " \
          f"--warmup_steps {args.warmup_steps} " \
          f"--lr_scheduler {args.lr_scheduler} " \
          f"--save_steps {args.save_steps} " \
          f"--eval_steps {args.eval_steps} " \
          f"--logging_steps {args.logging_steps} " \
          f"--seed {args.seed} " \
          f"--num_workers {args.num_workers} " \
          f"--cache_dir {args.cache_dir} " \
          f"--run_name {args.run_name} "
    
    if args.use_wandb:
        cmd += "--use_wandb "
    
    if args.distributed:
        cmd += "--distributed "
        
    if args.use_nsa:
        cmd += "--use_nsa "
        
    if args.gradient_checkpointing:
        cmd += "--gradient_checkpointing "
        
    if args.deepspeed:
        cmd += f"--deepspeed --deepspeed_config {args.deepspeed_config} "
        
    if args.resume_from_checkpoint:
        cmd += f"--resume_from_checkpoint {args.resume_from_checkpoint} "
    
    # Print command and execute
    print(f"Running command: {cmd}")
    os.system(cmd)
