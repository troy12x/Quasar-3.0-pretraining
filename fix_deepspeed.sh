#!/bin/bash

# Script to fix DeepSpeed installation issues for Quasar 3.0 training
echo "Fixing DeepSpeed installation for Quasar 3.0..."

# Install required dependencies
sudo apt-get update
sudo apt-get install -y pybind11-dev

# If the above doesn't work, try this alternative
if [ $? -ne 0 ]; then
    echo "Trying alternative installation method..."
    sudo apt-get install -y python3-pybind11
fi

# If you still have issues, you can disable the CPU Adam optimizer
echo "If you still encounter issues, run the training with these environment variables:"
echo "DS_BUILD_CPU_ADAM=0 DS_BUILD_FUSED_ADAM=0 python run_pretrain.py --batch_size 2 --precision bf16 --gradient_accumulation_steps 4 --deepspeed --gradient_checkpointing --num_workers 0 --use_wandb --run_name \"quasar3-training\""

echo "Installation complete!"
