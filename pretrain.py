import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import bisect
import warnings
import numpy as np

# Simply filter the warnings - this is the most reliable approach
warnings.filterwarnings("ignore", message=".*The input object of type 'Tensor'.*")

# Define a robust data collator at module level for pickling compatibility
class RobustDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
    def __call__(self, examples):
        # Ensure all examples have the required keys
        required_keys = ["input_ids", "attention_mask", "labels"]
        for key in required_keys:
            if not all(key in example for example in examples):
                raise ValueError(f"All examples must have the '{key}' key")
        
        # Extract all tensors
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for example in examples:
            # Convert to lists if they're tensors
            if isinstance(example["input_ids"], torch.Tensor):
                input_ids = example["input_ids"].tolist()
                attention_mask = example["attention_mask"].tolist()
                labels = example["labels"].tolist()
            else:
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                labels = example["labels"]
            
            # Ensure all have the same length within each example
            min_len = min(len(input_ids), len(attention_mask), len(labels))
            input_ids = input_ids[:min_len]
            attention_mask = attention_mask[:min_len]
            labels = labels[:min_len]
            
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        # Find max length in this batch
        max_length = max(len(ids) for ids in input_ids_list)
        # Make max_length a multiple of 8 for tensor core efficiency
        if max_length % 8 != 0:
            max_length = ((max_length // 8) + 1) * 8
        
        # Pad all sequences to max_length
        for i in range(len(input_ids_list)):
            padding_length = max_length - len(input_ids_list[i])
            if padding_length > 0:
                input_ids_list[i] = input_ids_list[i] + [self.pad_token_id] * padding_length
                attention_mask_list[i] = attention_mask_list[i] + [0] * padding_length
                labels_list[i] = labels_list[i] + [-100] * padding_length
        
        # Convert to tensors
        batch = {
            "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
            "labels": torch.tensor(labels_list, dtype=torch.long)
        }
        
        return batch

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast, DataCollatorForLanguageModeling
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
def is_main_process(local_rank=None):
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    if local_rank is not None:
        return local_rank in [-1, 0]
    return True

# Function to check if distributed is initialized and get rank
def get_rank():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    return 0

# Function to check if distributed is initialized and get world size
def get_world_size():
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1

class MultiDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=1024, cache_dir=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.datasets = []
        self.dataset_sizes = []
        self.total_size = 0
        self.tokenized_datasets = []
        self.cumulative_sizes = [0]  # For faster binary search
        
        # 1. Load all subsets from eyad-silx/wiki-pretrain
        logger.info(f"Loading wiki-pretrain datasets ({split} split)...")
        wiki_subsets = ["en"]

        # Only have rank 0 download the dataset to avoid rate limiting
        is_main_process = torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True
        
        if is_main_process:
            logger.info("Main process downloading datasets...")
            for subset in wiki_subsets:
                try:
                    dataset = load_dataset("eyad-silx/wiki-pretrain", subset, split=split, cache_dir=cache_dir)
                    logger.info(f"Main process downloaded wiki-pretrain/{subset} with {len(dataset)} examples")
                except Exception as e:
                    logger.warning(f"Main process failed to download wiki-pretrain/{subset}: {e}")
        
        # Wait for main process to finish downloading
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Now all processes can load from cache
        for subset in wiki_subsets:
            try:
                dataset = load_dataset("eyad-silx/wiki-pretrain", subset, split=split, cache_dir=cache_dir)
                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.total_size += len(dataset)
                logger.info(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0} loaded wiki-pretrain/{subset} with {len(dataset)} examples")
            except Exception as e:
                logger.warning(f"Failed to load wiki-pretrain/{subset} from cache: {e}")

        
        # Check if we have any datasets loaded
        if not self.datasets:
            logger.error("Failed to load any datasets! Training will not work.")
        else:
            logger.info(f"Successfully loaded {len(self.datasets)} datasets with a total of {self.total_size} examples")
            
            # Pre-tokenize datasets for faster training
            logger.info("Pre-tokenizing datasets...")
            for i, dataset in enumerate(self.datasets):
                # Calculate cumulative sizes for binary search
                if i > 0:
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + self.dataset_sizes[i-1])
                
                # Define tokenization function
                def tokenize_function(examples):
                    # Handle different dataset formats
                    # For batched processing, examples will be a dict with keys as column names
                    # and values as lists
                    
                    # Try to find the text column
                    text_column = None
                    for column in dataset.column_names:
                        if column in ["text", "page", "content"]:
                            text_column = column
                            break
                    
                    if text_column is None:
                        # If no standard text column, use the first string column
                        for column in dataset.column_names:
                            if isinstance(examples[column][0], str):
                                text_column = column
                                break
                    
                    if text_column is None:
                        # Fallback to a dummy text
                        return self.tokenizer(
                            ["Fallback text"] * len(examples[dataset.column_names[0]]),
                            truncation=True,
                            max_length=self.max_length,
                            return_attention_mask=True
                        )
                    
                    return self.tokenizer(
                        examples[text_column],
                        truncation=True,
                        max_length=self.max_length,
                        return_attention_mask=True
                    )
                
                # Tokenize the dataset
                try:
                    # Use batched processing for speed
                    tokenized_dataset = dataset.map(
                        tokenize_function,
                        batched=True,
                        batch_size=10000,
                        num_proc=min(os.cpu_count() // 2, 40),  # Limit to avoid excessive processes
                        remove_columns=dataset.column_names
                    )
                    self.tokenized_datasets.append(tokenized_dataset)
                    logger.info(f"Pre-tokenized dataset {i+1}/{len(self.datasets)}")
                except Exception as e:
                    logger.warning(f"Failed to pre-tokenize dataset {i+1}: {e}")
                    # Fall back to on-the-fly tokenization for this dataset
                    self.tokenized_datasets.append(None)
    
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # Binary search to find dataset (much faster than linear search)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx) - 1
        local_idx = idx - self.cumulative_sizes[dataset_idx]
        
        if dataset_idx >= len(self.datasets):
            raise IndexError(f"Index {idx} out of range for combined dataset of size {self.total_size}")
        
        # Check if we have pre-tokenized data
        if self.tokenized_datasets[dataset_idx] is not None:
            # Get pre-tokenized data
            tokenized_item = self.tokenized_datasets[dataset_idx][local_idx]
            
            # Convert to tensors if needed (handle both list and tensor inputs)
            if isinstance(tokenized_item["input_ids"], torch.Tensor):
                input_ids = tokenized_item["input_ids"]
                attention_mask = tokenized_item["attention_mask"]
            else:
                input_ids = torch.tensor(tokenized_item["input_ids"], dtype=torch.long)
                attention_mask = torch.tensor(tokenized_item["attention_mask"], dtype=torch.long)
            
            # Create labels (shifted input_ids for causal language modeling)
            labels = input_ids.clone()
            # Mask out padding tokens in labels
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            # Ensure all tensors have the same length
            if len(input_ids) != len(labels) or len(input_ids) != len(attention_mask):
                # Truncate to the shortest length
                min_length = min(len(input_ids), len(labels), len(attention_mask))
                input_ids = input_ids[:min_length]
                attention_mask = attention_mask[:min_length]
                labels = labels[:min_length]
        else:
            # Fallback to on-the-fly tokenization
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
            
            # Tokenize text with explicit padding to max_length
            encodings = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",  # Use explicit padding to max_length
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
            def __init__(self, tokenizer, max_length=1024, cache_dir=None):
                self.tokenizer = tokenizer
                self.tokenizer.padding_side = "right"  # Ensure right padding for causal LM
                self.max_length = max_length
                # Load the specific validation dataset
                self.dataset = load_dataset("Jackmin108/c4-en-validation-mini", split="validation", cache_dir=cache_dir)
                logger.info(f"Loaded validation dataset with {len(self.dataset)} examples")
                
                # Pre-tokenize validation dataset for faster evaluation
                try:
                    def tokenize_function(examples):
                        # Handle validation dataset format
                        text_column = "text" if "text" in examples else next(iter(examples.keys()))
                        return self.tokenizer(
                            examples[text_column],
                            truncation=True,
                            max_length=self.max_length,
                            return_attention_mask=True
                        )
                        
                    self.tokenized_dataset = self.dataset.map(
                        tokenize_function,
                        batched=True,
                        batch_size=1000,
                        num_proc=os.cpu_count() // 6,
                        remove_columns=self.dataset.column_names
                    )
                    logger.info("Pre-tokenized validation dataset")
                except Exception as e:
                    logger.warning(f"Failed to pre-tokenize validation dataset: {e}")
                    self.tokenized_dataset = None
                
            def __len__(self):
                return len(self.dataset)
                
            def __getitem__(self, idx):
                # Check if we have pre-tokenized data
                if self.tokenized_dataset is not None:
                    # Get pre-tokenized data
                    tokenized_item = self.tokenized_dataset[idx]
                    
                    # Convert to tensors if needed (handle both list and tensor inputs)
                    if isinstance(tokenized_item["input_ids"], torch.Tensor):
                        input_ids = tokenized_item["input_ids"]
                        attention_mask = tokenized_item["attention_mask"]
                    else:
                        input_ids = torch.tensor(tokenized_item["input_ids"], dtype=torch.long)
                        attention_mask = torch.tensor(tokenized_item["attention_mask"], dtype=torch.long)
                    
                    # Create labels (shifted input_ids for causal language modeling)
                    labels = input_ids.clone()
                    # Mask out padding tokens in labels
                    labels[labels == self.tokenizer.pad_token_id] = -100
                    
                    return {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    }
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
                
                # Tokenize text with explicit padding to max_length
                encodings = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",  # Use explicit padding to max_length
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
    
    # Use our robust collator defined at module level
    data_collator = RobustDataCollator(tokenizer)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,  # Use the collator for dynamic padding
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
            collate_fn=data_collator,  # Use the same collator for validation
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
    
    # Set memory-efficient initialization
    if should_log:
        logger.info("Using ULTRA-LAZY initialization to prevent OOM errors for 200B+ parameter models")
        logger.info(f"Model config: {hidden_size} hidden size, {num_layers} layers, {num_experts} experts")
        logger.info("Note: Parameters will be initialized by DeepSpeed during model distribution")
    
    # Free up memory before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Set to lowest memory usage
        torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve some memory for system
    
    with tqdm(total=100, desc="Creating Quasar model", ncols=100, 
              bar_format='{l_bar}{bar}| {n:.0f}/{total:.0f} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        
        # Create model with ultra-lazy initialization to prevent OOM errors
        model = create_quasar_model(
            use_nsa=args.use_nsa, 
            pbar=pbar, 
            lazy_init=True,  # Use lazy initialization for large models
            ultra_lazy=True  # Use ultra-lazy initialization for 200B+ models
        )
        
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
        
        # Let DeepSpeed handle model placement - don't move to device first
        # This prevents OOM errors with large models
        
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
    parser = argparse.ArgumentParser(description="Quasar 3.0 Pretraining")

    # Data arguments
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output", help="Path to the tokenizer")
    parser.add_argument("--dataset", type=str, default="wiki-pretrain", help="Dataset to use for training")
    parser.add_argument("--val_dataset", type=str, default="Jackmin108/c4-en-validation-mini", help="Dataset to use for validation")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory for datasets")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="Adam beta2")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Adam epsilon")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps")
    parser.add_argument("--warmup_steps", type=int, default=1000, help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Number of steps between logging")
    parser.add_argument("--save_steps", type=int, default=5000, help="Number of steps between saving checkpoints")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Number of steps between evaluations")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 precision")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"], help="Precision to use for training")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "linear", "constant"], help="Learning rate scheduler")

    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--world_size", type=int, default=1, help="World size for distributed training")

    # DeepSpeed arguments
    parser.add_argument("--deepspeed", action="store_true", help="Use DeepSpeed")
    parser.add_argument("--deepspeed_config", type=str, default="deepspeed_config.json", help="DeepSpeed configuration file")

    # Logging arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="quasar-pretrain", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Weights & Biases entity name")
    parser.add_argument("--run_name", type=str, default=None, help="Run name for logging")

    # Parse arguments
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Set up logging
    if args.run_name is None:
        args.run_name = f"quasar3-training-{time.strftime('%Y%m%d-%H%M%S')}"

    # Set up distributed training
    if args.local_rank != -1 or "LOCAL_RANK" in os.environ:
        args.distributed = True
        if args.local_rank == -1:
            args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        args.world_size = int(os.environ.get("WORLD_SIZE", args.world_size))

    # Set precision
    if args.fp16:
        args.precision = "fp16"
    elif args.bf16:
        args.precision = "bf16"

    # Create output directory
    if is_main_process(args.local_rank):
        os.makedirs(args.output_dir, exist_ok=True)

    # Initialize distributed environment
    local_rank = args.local_rank

    # Check if we're using DeepSpeed or torchrun
    if args.deepspeed or "RANK" in os.environ:
        # Log distributed training configuration
        logger.info(f"Using distributed training across {args.world_size} GPUs")
        logger.info(f"Local rank: {local_rank}, World size: {args.world_size}")
        logger.info(f"Environment variables: RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, MASTER_ADDR={os.environ.get('MASTER_ADDR')}, MASTER_PORT={os.environ.get('MASTER_PORT')}")

        # Initialize distributed environment if not already done
        if not torch.distributed.is_initialized():
            try:
                # Try to use NCCL backend first (faster for GPU training)
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

        # Set device for this process
        torch.cuda.set_device(local_rank)

        # Update world_size based on initialized process group
        if torch.distributed.is_initialized():
            args.world_size = torch.distributed.get_world_size()
            args.local_rank = torch.distributed.get_rank()
            local_rank = args.local_rank

        # Train with distributed setup
        train(args, local_rank, args.world_size)
    elif args.distributed:
        logger.info(f"Using distributed training with {args.world_size} GPUs via mp.spawn")
        mp.spawn(train, args=(args, args.world_size), nprocs=args.world_size, join=True)
    else:
        logger.info("Using single GPU training")
        train(args, 0, 1)

if __name__ == "__main__":
    main()