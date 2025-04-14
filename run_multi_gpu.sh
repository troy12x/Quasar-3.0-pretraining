#!/bin/bash

# Set environment variables to control logging
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NCCL_DEBUG=WARN
export PYTHONWARNINGS="ignore"

# Run the training script with multi-GPU configuration and log control
deepspeed pretrain.py \
    --batch_size 4 \
    --precision bf16 \
    --gradient_accumulation_steps 2 \
    --deepspeed \
    --deepspeed_config deepspeed_config.json \
    --gradient_checkpointing \
    --num_workers 0 \
    --use_wandb \
    --run_name "quasar3-multi-gpu" \
    --learning_rate 4e-4 \
    --warmup_steps 200 \
    --lr_scheduler cosine \
    --logging_steps 5 \
    --save_steps 1000 \
