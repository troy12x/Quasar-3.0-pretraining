#!/bin/bash

# Quasar 3.0 Training Script
# Simple and clean script for multi-GPU training

# Error handling
set -e

echo "========================================================="
echo "             QUASAR 3.0 TRAINING SCRIPT                  "
echo "========================================================="

# Get the number of available GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
    echo "Found $NUM_GPUS GPUs for training"
    
    # Display GPU information
    nvidia-smi --format=csv --query-gpu=index,name,memory.total,memory.free
else
    echo "Warning: nvidia-smi not found. Setting NUM_GPUS to 1"
    NUM_GPUS=1
fi

# Training parameters
MICRO_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=2
PRECISION="bf16"
OUTPUT_DIR="./checkpoints"
SAVE_STEPS=2000
LOGGING_STEPS=10
RUN_NAME="quasar3-training-$(date +%Y%m%d-%H%M%S)"

# Create output directory
mkdir -p $OUTPUT_DIR

# Calculate global batch size
GLOBAL_BATCH_SIZE=$((MICRO_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * NUM_GPUS))

echo "Training Configuration:"
echo "- Number of GPUs: $NUM_GPUS"
echo "- Micro batch size per GPU: $MICRO_BATCH_SIZE"
echo "- Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "- Global batch size: $GLOBAL_BATCH_SIZE"
echo "- Precision: $PRECISION"
echo "- Output directory: $OUTPUT_DIR"
echo "- Save checkpoint every $SAVE_STEPS steps"
echo "- Run name: $RUN_NAME"

# Use existing DeepSpeed config file
DEEPSPEED_CONFIG="deepspeed_config.json"
echo "Using existing DeepSpeed config: $DEEPSPEED_CONFIG"

# Performance optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Suppress duplicate logs in multi-GPU setup
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_DEBUG=WARN
export TORCH_SHOW_CPP_STACKTRACES=0
export CUDA_LAUNCH_BLOCKING=0

# Disable problematic DeepSpeed CUDA extensions
export DS_BUILD_OPS=0
export DS_BUILD_AIO=0
export DS_BUILD_CUFILE=0
export DISABLE_ADASUM=1
export DEEPSPEED_AIO=0

echo "Starting training..."
echo "========================================================="

# Use torchrun instead of deepspeed directly
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=127.0.0.1 \
    --master_port=29500 \
    pretrain.py \
    --deepspeed \
    --deepspeed_config=$DEEPSPEED_CONFIG \
    --batch_size=$MICRO_BATCH_SIZE \
    --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
    --precision=$PRECISION \
    --output_dir=$OUTPUT_DIR \
    --logging_steps=$LOGGING_STEPS \
    --save_steps=$SAVE_STEPS \
    --gradient_checkpointing \
    --use_wandb \
    --run_name=$RUN_NAME \
    --tokenizer_path=./tokenizer_output

echo "Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"