import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
import logging
import argparse
import wandb
import numpy as np
import json
from quasar import QuasarConfig, create_quasar_model
import torch.cuda.amp as amp

# Set multiprocessing start method to 'spawn' for CUDA compatibility
mp.set_start_method('spawn', force=True)

# Import DeepSpeed if available
try:
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class C4Dataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=2048, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load C4 dataset from Hugging Face
        logger.info(f"Loading C4 dataset ({split} split)...")
        try:
            self.dataset = load_dataset("eyad-silx/wiki-pretrain", "ab", split=split, cache_dir=cache_dir)
            logger.info(f"Loaded {len(self.dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load C4 dataset: {e}")
            logger.info("Falling back to a smaller dataset (wikitext)...")
            try:
                self.dataset = load_dataset("wikitext", "wikitext-103-v1", split=split, cache_dir=cache_dir)
                logger.info(f"Loaded {len(self.dataset)} examples from wikitext")
            except Exception as e2:
                logger.error(f"Failed to load fallback dataset: {e2}")
                # Create a small dummy dataset for testing
                logger.info("Using a dummy dataset for testing")
                from datasets import Dataset as HFDataset
                dummy_data = [{"text": "This is a dummy text for testing the Quasar model."} for _ in range(100)]
                self.dataset = HFDataset.from_dict({"text": [item["text"] for item in dummy_data]})
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Handle different dataset formats (C4 vs Wikitext)
        if "text" in self.dataset[idx]:
            text = self.dataset[idx]["text"]
        elif "page" in self.dataset[idx]:  # For Wikitext
            text = self.dataset[idx]["page"]
        else:
            # Get the first field that contains text
            for key, value in self.dataset[idx].items():
                if isinstance(value, str) and len(value) > 0:
                    text = value
                    break
            else:
                text = "This is a fallback text for empty examples."
        
        # Tokenize text
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # Create labels (shifted input_ids for causal language modeling)
        labels = input_ids.clone()
        # Mask out padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def setup_distributed(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def get_parameter_count(model):
    """Calculate the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())

def train(args, rank, world_size):
    # Set up distributed training if needed
    if args.distributed and not args.deepspeed:
        setup_distributed(rank, world_size)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb for tracking experiments
    if rank == 0 and args.use_wandb:
        wandb.init(
            project="quasar-pretrain",
            config=vars(args),
            name=args.run_name
        )
    
    # Load custom tokenizer from tokenizer.json
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # Set special tokens
    tokenizer.bos_token = "<｜begin▁of▁sentence｜>"
    tokenizer.eos_token = "<｜end▁of▁sentence｜>"
    tokenizer.pad_token = "<｜▁pad▁｜>"
    tokenizer.unk_token = "<｜▁unk▁｜>"
    
    # Create datasets and dataloaders
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    train_dataset = C4Dataset(
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    
    val_dataset = C4Dataset(
        tokenizer=tokenizer,
        split="validation",
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    
    if args.distributed and not args.deepspeed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    # Create model
    logger.info("Creating Quasar model...")
    model = create_quasar_model(use_nsa=args.use_nsa)
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing to save memory")
        model.gradient_checkpointing_enable()
    
    # Log parameter count
    param_count = get_parameter_count(model)
    logger.info(f"Model created with {param_count/1e9:.2f}B parameters")
    
    # Setup optimizer parameters
    optimizer_params = {
        "lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon
    }
    
    # Calculate total training steps
    total_steps = len(train_loader) * args.num_epochs
    if hasattr(args, 'warmup_steps') and args.warmup_steps > 0:
        warmup_steps = args.warmup_steps
    else:
        warmup_steps = int(total_steps * args.warmup_ratio)
    
    # DeepSpeed integration
    if args.deepspeed and DEEPSPEED_AVAILABLE:
        logger.info("Initializing DeepSpeed...")
        # Load DeepSpeed config
        ds_config = None
        if os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
        else:
            logger.warning(f"DeepSpeed config file {args.deepspeed_config} not found. Using default config.")
            ds_config = {
                "train_batch_size": int(args.batch_size * world_size * args.gradient_accumulation_steps),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "optimizer": {
                    "type": args.optimizer.capitalize(),
                    "params": optimizer_params
                },
                "fp16": {
                    "enabled": args.precision == "fp16" and args.precision != "bf16"
                },
                "bf16": {
                    "enabled": args.precision == "bf16" and args.precision != "fp16"
                },
                "scheduler": {
                    "type": "WarmupDecayLR",
                    "params": {
                        "warmup_min_lr": 0,
                        "warmup_max_lr": args.learning_rate,
                        "warmup_num_steps": warmup_steps,
                        "total_num_steps": total_steps
                    }
                },
                "zero_optimization": {
                    "stage": 3,
                    "offload_optimizer": {
                        "device": "cpu"
                    },
                    "offload_param": {
                        "device": "cpu"
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8,
                    "stage3_prefetch_bucket_size": 5e8,
                    "stage3_param_persistence_threshold": 1e6
                },
                "steps_per_print": args.logging_steps,
                "wall_clock_breakdown": False
            }
        
        # Move model to GPU before DeepSpeed initialization
        device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
        model = model.to(device)
        
        # Initialize DeepSpeed
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
            dist_init_required=True
        )
    else:
        # Standard training setup (non-DeepSpeed)
        # Move model to GPU and setup DDP if needed
        model = model.to(device)
        if args.distributed:
            model = DDP(model, device_ids=[rank])
        
        # Setup optimizer
        if args.optimizer == "adamw":
            optimizer = optim.AdamW(model.parameters(), **optimizer_params)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), **optimizer_params)
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "adafactor":
            optimizer = optim.Adafactor(model.parameters(), **optimizer_params)
        
        # Learning rate scheduler with warmup
        if args.lr_scheduler == "linear":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(
                    0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps))
                )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif args.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
        elif args.lr_scheduler == "constant":
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)
        elif args.lr_scheduler == "constant_with_warmup":
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    global_step = 0
    best_val_loss = float('inf')
    
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Loading checkpoint from {args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=f"cuda:{rank}")
        
        # Load model weights
        if isinstance(model, DDP):
            model.module.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint["model"])
            
        # Load optimizer and scheduler states
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        
        # Resume training state
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        logger.info(f"Resuming from epoch {start_epoch} (global step {global_step})")
    
    # Training loop
    logger.info("Starting training...")
    if args.precision in ["fp16", "bf16"] and not args.deepspeed:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        if args.distributed and not args.deepspeed:
            train_sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        # Zero gradients at the beginning of each epoch (for non-DeepSpeed)
        if not args.deepspeed:
            optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device (not needed for DeepSpeed)
            if not args.deepspeed:
                batch = {k: v.to(device) for k, v in batch.items()}
            
            # DeepSpeed handles mixed precision internally
            if args.deepspeed:
                # Ensure all tensors are on the correct device
                cuda_device = torch.device("cuda", rank if torch.cuda.is_available() else "cpu")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(cuda_device)
                
                # Forward pass with DeepSpeed
                outputs = model(**batch)
                loss = outputs["loss"]
                
                # Backward pass with DeepSpeed
                model.backward(loss)
                
                # Update parameters with DeepSpeed
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    model.step()
            else:
                # Standard training path (non-DeepSpeed)
                # Forward pass with appropriate precision
                if args.precision in ["fp16", "bf16"]:
                    with amp.autocast(dtype=torch.bfloat16 if args.precision == "bf16" else torch.float16):
                        outputs = model(**batch)
                        # Scale loss by gradient accumulation steps
                        loss = outputs["loss"] / args.gradient_accumulation_steps
                else:
                    outputs = model(**batch)
                    # Scale loss by gradient accumulation steps
                    loss = outputs["loss"] / args.gradient_accumulation_steps
                
                # Backward pass with mixed precision if enabled
                if args.precision in ["fp16", "bf16"]:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update parameters every gradient_accumulation_steps
                if (step + 1) % args.gradient_accumulation_steps == 0 or step == len(train_loader) - 1:
                    # Gradient clipping
                    if args.precision in ["fp16", "bf16"]:
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    
                    # Update parameters with mixed precision if enabled
                    if args.precision in ["fp16", "bf16"]:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    
                    # Update learning rate
                    scheduler.step()
                    
                    # Zero gradients
                    optimizer.zero_grad()
            
            # Get loss value for logging
            if args.deepspeed:
                loss_value = loss.item()
            else:
                loss_value = loss.item() * args.gradient_accumulation_steps  # Scale back to get the actual loss
            
            # Update progress
            global_step += 1
            epoch_loss += loss_value
            progress_bar.set_postfix({"loss": loss_value, "lr": scheduler.get_last_lr()[0] if not args.deepspeed else model.get_lr()[0]})
            
            # Log to wandb
            if rank == 0 and args.use_wandb and global_step % args.logging_steps == 0:
                wandb.log({
                    "train/loss": loss_value,
                    "train/learning_rate": scheduler.get_last_lr()[0] if not args.deepspeed else model.get_lr()[0],
                    "train/epoch": epoch + (progress_bar.n / len(progress_bar)),
                    "train/global_step": global_step
                })
            
            # Save checkpoint
            if rank == 0 and global_step % args.save_steps == 0:
                if args.deepspeed:
                    # DeepSpeed handles saving checkpoints
                    model.save_checkpoint(args.output_dir, f"checkpoint-{global_step}")
                else:
                    save_checkpoint(model, optimizer, scheduler, epoch, global_step, args)
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {epoch_loss:.4f}")
        
        # Validation
        val_loss = evaluate(model, val_loader, device, is_deepspeed=args.deepspeed)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Validation Loss: {val_loss:.4f}")
        
        # Log validation metrics
        if rank == 0 and args.use_wandb:
            wandb.log({
                "validation/loss": val_loss,
                "validation/epoch": epoch + 1,
                "validation/global_step": global_step
            })
        
        # Save best model
        if rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, args,
                filename=f"best_model.pt"
            )
    
    # Save final model
    if rank == 0:
        save_checkpoint(
            model, optimizer, scheduler, args.num_epochs-1, global_step, args,
            filename=f"final_model.pt"
        )
    
    # Clean up
    if args.distributed:
        cleanup_distributed()

def evaluate(model, dataloader, device, is_deepspeed=False):
    """Evaluate the model on the validation set."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device (not needed for DeepSpeed)
            if not is_deepspeed:
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                # Ensure all tensors are on the correct device for DeepSpeed
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
            
            # Forward pass
            if is_deepspeed:
                outputs = model(**batch)
            else:
                outputs = model(**batch)
                
            loss = outputs["loss"]
            
            total_loss += loss.item()
    
    # Calculate average loss
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args, filename=None):
    """Save model checkpoint."""
    if filename is None:
        filename = f"checkpoint-{global_step}.pt"
    
    checkpoint_path = os.path.join(args.output_dir, filename)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get model state dict (handle DDP case)
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Save checkpoint
    checkpoint = {
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "args": args,
        "best_val_loss": args.best_val_loss if hasattr(args, 'best_val_loss') else best_val_loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")

def main():
    parser = argparse.ArgumentParser(description="Pretrain Quasar model on C4 dataset")
    
    # Data arguments
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json", help="Path to tokenizer.json file")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache datasets")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Optimizer and scheduler arguments
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adafactor"],
                        help="Optimizer to use for training")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of steps for linear warmup")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=1000, help="Run evaluation every X updates steps")
    parser.add_argument("--run_name", type=str, default="quasar-pretrain", help="Run name for wandb")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use wandb")
    
    # Distributed training arguments
    parser.add_argument("--distributed", action="store_true", help="Whether to use distributed training")
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count(), help="Number of GPUs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--use_nsa", action="store_true", help="Whether to use Native Sparse Attention")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing to save memory")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of steps to accumulate gradients")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16", "fp8"], help="Precision for training")
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed for model parallelism")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed configuration file")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set a smaller number of workers if using DeepSpeed to avoid CUDA issues
    if args.deepspeed and args.num_workers > 0:
        args.num_workers = min(2, args.num_workers)
        logger.info(f"Using {args.num_workers} dataloader workers with DeepSpeed")
    
    # Launch training
    if args.deepspeed and DEEPSPEED_AVAILABLE:
        # DeepSpeed handles distributed training internally
        logger.info("Using DeepSpeed for distributed training")
        train(args, 0, args.world_size)
    elif args.distributed:
        logger.info(f"Using distributed training with {args.world_size} GPUs")
        mp.spawn(train, args=(args, args.world_size), nprocs=args.world_size, join=True)
    else:
        train(args, 0, 1)

if __name__ == "__main__":
    main()
