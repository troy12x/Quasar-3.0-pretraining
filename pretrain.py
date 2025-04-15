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
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
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

# Function to check if this is the main process
def is_main_process(local_rank):
    return local_rank in [-1, 0]

class MultiDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=8129, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        self.dataset_sizes = []
        self.total_size = 0
        
        # 1. Load all subsets from eyad-silx/wiki-pretrain
        logger.info(f"Loading wiki-pretrain datasets ({split} split)...")
        wiki_subsets = ["ar", "en"]
        for subset in wiki_subsets:
            try:
                dataset = load_dataset("eyad-silx/wiki-pretrain", subset, split=split, cache_dir=cache_dir)
                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.total_size += len(dataset)
                logger.info(f"Loaded wiki-pretrain/{subset} with {len(dataset)} examples")
            except Exception as e:
                logger.warning(f"Failed to load wiki-pretrain/{subset}: {e}")
        
        # 2. Load code pretraining data
        logger.info(f"Loading code pretraining data ({split} split)...")
        try:
            code_dataset = load_dataset("ZhentingNLP/code_pretraining_data", split=split, cache_dir=cache_dir)
            self.datasets.append(code_dataset)
            self.dataset_sizes.append(len(code_dataset))
            self.total_size += len(code_dataset)
            logger.info(f"Loaded code_pretraining_data with {len(code_dataset)} examples")
        except Exception as e:
            logger.warning(f"Failed to load code_pretraining_data: {e}")
        
        # 3. Load Arabic pretraining data as backup
        if not self.datasets:
            logger.info(f"Loading Arabic pretraining data as backup ({split} split)...")
            try:
                arabic_dataset = load_dataset("Ashmal/Arabic_Pretraining_10K", split=split, cache_dir=cache_dir)
                self.datasets.append(arabic_dataset)
                self.dataset_sizes.append(len(arabic_dataset))
                self.total_size += len(arabic_dataset)
                logger.info(f"Loaded Arabic_Pretraining_10K with {len(arabic_dataset)} examples")
            except Exception as e:
                logger.warning(f"Failed to load Arabic_Pretraining_10K: {e}")
        
        # Check if we have any datasets loaded
        if not self.datasets:
            logger.error("Failed to load any datasets! Training will not work.")
        else:
            logger.info(f"Successfully loaded {len(self.datasets)} datasets with a total of {self.total_size} examples")
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        dataset_idx = 0
        local_idx = idx
        
        while dataset_idx < len(self.dataset_sizes) and local_idx >= self.dataset_sizes[dataset_idx]:
            local_idx -= self.dataset_sizes[dataset_idx]
            dataset_idx += 1
        
        if dataset_idx >= len(self.datasets):
            raise IndexError(f"Index {idx} out of range for combined dataset of size {self.total_size}")
        
        # Get the item from the appropriate dataset
        item = self.datasets[dataset_idx][local_idx]
        
        # Extract text based on dataset format
        if "text" in item:
            text = item["text"]
        elif "page" in item:
            text = item["page"]
        elif "content" in item:  # For code dataset
            text = item["content"]
        else:
            # Get the first field that contains text
            for key, value in item.items():
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
    """Train the model."""
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")
    
    # Only log from the main process
    should_log = is_main_process(rank)
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb for tracking experiments
    if should_log and args.use_wandb:
        # Only initialize wandb on the main process (rank 0)
        if int(os.environ.get("LOCAL_RANK", "-1")) != 0:
            os.environ["WANDB_MODE"] = "disabled"
        else:
            wandb.init(
                project="quasar-pretrain",
                config=vars(args),
                name=args.run_name,
                tags=["quasar3", f"bs{args.batch_size}", f"lr{args.learning_rate}", 
                      f"precision-{args.precision}", "deepspeed" if args.deepspeed else "standard"]
            )
    
    # Load custom tokenizer from tokenizer.json
    if should_log:
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # Set special tokens
    tokenizer.bos_token = "<｜begin▁of▁sentence｜>"
    tokenizer.eos_token = "<｜end▁of▁sentence｜>"
    tokenizer.pad_token = "<｜▁pad▁｜>"
    tokenizer.unk_token = "<｜▁unk▁｜>"
    
    # Create datasets and dataloaders
    train_dataset = MultiDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=args.max_seq_length,
        cache_dir=args.cache_dir,
    )
    
    # Load a specific validation dataset
    val_dataset = None
    try:
        if should_log:
            logger.info("Loading specific validation dataset: Jackmin108/c4-en-validation-mini")
        # Create a simple validation dataset class that only loads the validation data
        class ValidationDataset(Dataset):
            def __init__(self, tokenizer, max_length=8129, cache_dir=None):
                self.tokenizer = tokenizer
                self.max_length = max_length
                # Load the specific validation dataset
                self.dataset = load_dataset("Jackmin108/c4-en-validation-mini", split="validation", cache_dir=cache_dir)
                logger.info(f"Loaded validation dataset with {len(self.dataset)} examples")
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                # Get text from the validation dataset
                if "text" in self.dataset[idx]:
                    text = self.dataset[idx]["text"]
                else:
                    # Fallback for other field names
                    for key, value in self.dataset[idx].items():
                        if isinstance(value, str) and len(value) > 0:
                            text = value
                            break
                    else:
                        text = "Validation example."
                
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
        
        # Create the validation dataset
        val_dataset = ValidationDataset(
            tokenizer=tokenizer,
            max_length=args.max_seq_length,
            cache_dir=args.cache_dir,
        )
        if should_log:
            logger.info(f"Successfully loaded validation dataset with {len(val_dataset)} examples")
    except Exception as e:
        if should_log:
            logger.warning(f"Failed to load validation dataset: {e}")
            logger.info("Training will continue without validation")
        val_dataset = None  # Ensure it's None even if partially initialized
    
    # Setup samplers for distributed training
    if args.distributed and not args.deepspeed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False) if val_dataset else None
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
    
    # Only create validation dataloader if validation dataset exists
    val_loader = None
    if val_dataset is not None and hasattr(val_dataset, 'dataset') and val_dataset.dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False
        )
    
    # Create model
    if should_log:
        logger.info("Creating Quasar model...")
    from tqdm import tqdm
    import time
    
    # Show loading bar for model creation with time estimation
    start_time = time.time()
    
    # Estimate total time based on model size (rough estimate)
    hidden_size = QuasarConfig().hidden_size
    num_layers = QuasarConfig().num_hidden_layers
    num_experts = QuasarConfig().num_routed_experts + QuasarConfig().num_shared_experts
    
    # Rough time estimate in seconds based on model size
    estimated_time = (hidden_size * num_layers * num_experts) / 1000000
    
    with tqdm(total=100, desc="Creating Quasar model", ncols=100, 
              bar_format='{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Create model with progress tracking
        model = create_quasar_model(use_nsa=args.use_nsa, pbar=pbar)
        
        # Ensure we reach 100%
        if pbar.n < 100:
            pbar.update(100 - pbar.n)
    
    # Log the actual time taken
    creation_time = time.time() - start_time
    if should_log:
        logger.info(f"Model creation completed in {creation_time:.2f} seconds")
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        if should_log:
            logger.info("Enabling gradient checkpointing to save memory")
        model.gradient_checkpointing_enable()
    
    # Log parameter count
    param_count = get_parameter_count(model)
    if should_log:
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
        if should_log:
            logger.info("Initializing DeepSpeed for data parallel training...")
            
            # Log GPU information
            gpu_count = torch.cuda.device_count()
            logger.info(f"Found {gpu_count} GPUs for training")
            for i in range(gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Load DeepSpeed config
        ds_config = None
        if os.path.exists(args.deepspeed_config):
            with open(args.deepspeed_config, 'r') as f:
                ds_config = json.load(f)
                if should_log:
                    logger.info(f"Loaded DeepSpeed config from {args.deepspeed_config}")
        else:
            if should_log:
                logger.warning(f"DeepSpeed config file {args.deepspeed_config} not found. Using default config.")
            # Calculate batch sizes properly to avoid DeepSpeed assertion errors
            micro_batch = args.batch_size
            grad_accum = args.gradient_accumulation_steps
            train_batch = micro_batch * grad_accum * world_size
            
            ds_config = {
                "train_batch_size": train_batch,
                "train_micro_batch_size_per_gpu": micro_batch,
                "gradient_accumulation_steps": grad_accum,
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
                    "stage": 2,  # Using stage 2 for better balance of memory and performance
                    "offload_optimizer": {
                        "device": "cpu"
                    },
                    "overlap_comm": True,
                    "contiguous_gradients": True,
                    "reduce_bucket_size": 5e8
                },
                "steps_per_print": args.logging_steps,
                "wall_clock_breakdown": False
            }
        
        # Ensure DeepSpeed config has proper data parallelism settings
        if "data_parallel_size" not in ds_config:
            ds_config["data_parallel_size"] = world_size
        
        # Ensure proper precision settings
        if args.precision == "bf16":
            ds_config["bf16"] = {"enabled": True}
            if "fp16" in ds_config:
                ds_config["fp16"]["enabled"] = False
        elif args.precision == "fp16":
            ds_config["fp16"] = {"enabled": True}
            if "bf16" in ds_config:
                ds_config["bf16"]["enabled"] = False
        
        # Modify DeepSpeed config to ensure proper warmup
        if 'scheduler' not in ds_config or ds_config['scheduler'] is None:
            ds_config['scheduler'] = {
                'type': 'WarmupLR',
                'params': {
                    'warmup_min_lr': 0,
                    'warmup_max_lr': args.learning_rate,
                    'warmup_num_steps': args.warmup_steps
                }
            }
        
        # Make sure optimizer config is properly set
        if 'optimizer' not in ds_config or ds_config['optimizer'] is None:
            ds_config['optimizer'] = {
                'type': 'AdamW',
                'params': {
                    'lr': args.learning_rate,
                    'betas': [args.adam_beta1, args.adam_beta2],
                    'eps': args.adam_epsilon,
                    'weight_decay': args.weight_decay
                }
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
        
        if should_log:
            logger.info(f"DeepSpeed initialized with config: {json.dumps(ds_config, indent=2)}")
            logger.info(f"All {world_size} GPUs will work on the same task using data parallelism")
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
            scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.min_learning_rate)
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
        if should_log:
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
        
        if should_log:
            logger.info(f"Resuming from epoch {start_epoch} (global step {global_step})")
    
    # Training loop
    if should_log:
        logger.info("Starting training...")
    if args.precision in ["fp16", "bf16"] and not args.deepspeed:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.amp.GradScaler("cuda", enabled=False)
    
    start_time = time.time()
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
            if should_log and args.use_wandb and global_step % args.logging_steps == 0:
                # Get learning rate
                if args.deepspeed:
                    current_lr = model.get_lr()[0]
                else:
                    current_lr = scheduler.get_last_lr()[0]
                
                # Get MoE balance loss if available
                moe_loss = 0.0
                if hasattr(model, 'module'):
                    if hasattr(model.module, 'moe_balance_loss'):
                        moe_loss = model.module.moe_balance_loss
                    elif hasattr(model.module, 'layers') and len(model.module.layers) > 0:
                        # Try to get from the first layer that might have it
                        for layer in model.module.layers:
                            if hasattr(layer, 'ffn') and hasattr(layer.ffn, 'sequence_balance_loss'):
                                moe_loss = layer.ffn.sequence_balance_loss
                                break
                
                # Log detailed metrics
                metrics = {
                    "loss": loss_value,  # Use simpler keys for main metrics
                    "learning_rate": current_lr,
                    "epoch": epoch + (progress_bar.n / len(progress_bar)),
                    "global_step": global_step,
                    "train/samples_per_second": args.batch_size / (time.time() - start_time),
                    "train/moe_balance_loss": moe_loss
                }
                
                # Add GPU memory usage if available
                try:
                    if torch.cuda.is_available():
                        metrics["system/gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                        metrics["system/gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                except:
                    pass
                
                # Log metrics to wandb
                wandb.log(metrics)
                
                # Also log to console for visibility
                if should_log and global_step % (args.logging_steps * 1) == 0:
                    logger.info(f"Step {global_step}: loss={loss_value:.4f}, lr={current_lr:.6f}, moe_loss={moe_loss:.6f}")
                
                # Reset timer for samples per second calculation
                start_time = time.time()
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if args.deepspeed:
                    # DeepSpeed handles saving checkpoints
                    try:
                        # Only log from rank 0 but all ranks save
                        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        if should_log:
                            logger.info(f"Saving DeepSpeed checkpoint to {checkpoint_path}")
                        
                        # Use simpler checkpoint saving to avoid NCCL issues
                        client_state = {"checkpoint_step": global_step, "epoch": epoch}
                        success = model.save_checkpoint(args.output_dir, f"checkpoint-{global_step}", client_state=client_state)
                        
                        # Only log from rank 0
                        if should_log:
                            if success:
                                logger.info(f"Successfully saved checkpoint at step {global_step}")
                            else:
                                logger.warning(f"Failed to save checkpoint at step {global_step}")
                    except Exception as e:
                        if should_log:
                            logger.warning(f"Error saving DeepSpeed checkpoint: {e}")
                            logger.info("Continuing training despite checkpoint error")
                else:
                    if should_log:
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step, args)
        
        # Calculate average epoch loss
        epoch_loss /= len(train_loader)
        if should_log:
            logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Average Loss: {epoch_loss:.4f}")
        
        # Validation
        if val_dataset is not None:
            val_loss = evaluate(model, val_loader, device, is_deepspeed=args.deepspeed)
            if should_log:
                logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Validation Loss: {val_loss:.4f}")
        
            # Log validation metrics
            if should_log and args.use_wandb:
                wandb.log({
                    "validation/loss": val_loss,
                    "validation/epoch": epoch + 1,
                    "validation/global_step": global_step
                })
        
            # Save best model
            if should_log and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, args,
                    filename=f"best_model.pt"
                )
    
    # Save final model
    if should_log:
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
        for batch in dataloader:
            # Move batch to device (not needed for DeepSpeed)
            if not is_deepspeed:
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                # Ensure all tensors are on the correct device for DeepSpeed
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
            
            # Forward pass with proper dtype handling
            try:
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    outputs = model(**batch)
                    loss = outputs["loss"]
            except RuntimeError as e:
                # If we encounter a dtype error, try again with explicit casting
                if "expected scalar type" in str(e):
                    logger.warning("Dtype mismatch during evaluation, retrying with explicit casting")
                    # Convert all inputs to the model's dtype
                    model_dtype = next(model.parameters()).dtype
                    batch = {k: v.to(dtype=model_dtype) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs["loss"]
                else:
                    # Re-raise if it's not a dtype error
                    raise
            
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
    
    # Add local_rank argument for distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    # Data arguments
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json", help="Path to tokenizer.json file")
    parser.add_argument("--max_seq_length", type=int, default=8129, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache datasets")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=4e-4, help="Peak learning rate")
    parser.add_argument("--min_learning_rate", type=float, default=4e-5, help="Minimum learning rate for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.01, help="Warmup ratio")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    
    # Optimizer and scheduler arguments
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd", "adafactor"],
                        help="Optimizer to use for training")
    parser.add_argument("--warmup_steps", type=int, default=200, help="Number of steps for linear warmup")
    parser.add_argument("--lr_scheduler", type=str, default="cosine", 
                        choices=["linear", "cosine", "constant", "constant_with_warmup"],
                        help="Learning rate scheduler type")
    
    # Logging and saving arguments
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--logging_steps", type=int, default=5, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Run evaluation every X updates steps")
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
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed for data parallel training")
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
        # Set environment variables for DeepSpeed
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "12355"
            
        # Get local rank from environment or args
        local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
        if local_rank == -1:
            local_rank = 0
            
        # Set required environment variables for distributed training if not already set
        if "RANK" not in os.environ:
            os.environ["RANK"] = str(local_rank)
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = str(args.world_size)
            
        # Log DeepSpeed configuration
        logger.info(f"Using DeepSpeed for data parallel training across {args.world_size} GPUs")
        logger.info(f"Local rank: {local_rank}, World size: {args.world_size}")
        logger.info(f"Environment variables: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")
        
        # Initialize distributed environment if not already done
        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(backend="nccl")
                logger.info("Successfully initialized process group with NCCL backend")
            except Exception as e:
                logger.warning(f"Failed to initialize with NCCL, trying Gloo backend: {e}")
                try:
                    # Fall back to Gloo backend which has better compatibility
                    torch.distributed.init_process_group(backend="gloo")
                    logger.info("Successfully initialized process group with Gloo backend")
                except Exception as e:
                    logger.error(f"Failed to initialize distributed environment: {e}")
                    logger.info("Falling back to non-distributed training")
                    train(args, 0, 1)
                    return
            
        # Train with DeepSpeed
        train(args, local_rank, args.world_size)
    elif args.distributed:
        logger.info(f"Using distributed training with {args.world_size} GPUs")
        mp.spawn(train, args=(args, args.world_size), nprocs=args.world_size, join=True)
    else:
        train(args, 0, 1)

if __name__ == "__main__":
    main()
