#!/usr/bin/env python
"""
Setup script for Quasar 3.0 pretraining environment.
This script installs all necessary dependencies with compatible versions.
"""

import os
import subprocess
import sys

def install_dependencies():
    """Install all required dependencies for Quasar 3.0 pretraining."""
    print("Setting up Quasar 3.0 pretraining environment...")
    
    # Core dependencies with specific versions to ensure compatibility
    dependencies = [
        "numpy==1.23.4",  # Use NumPy 1.23.4 for best compatibility with DeepSpeed and other libraries
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "wandb>=0.15.0",
        "tqdm>=4.65.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
        "deepspeed>=0.9.0",
    ]
    
    # Install each dependency
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", dep])
    
    print("\nInstalling additional dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "accelerate", "bitsandbytes"])
    
    print("\nEnvironment setup complete!")
    print("You can now run pretraining with: python run_pretrain.py --batch_size 8 --precision bf16 --gradient_accumulation_steps 4 --deepspeed --gradient_checkpointing")

if __name__ == "__main__":
    install_dependencies()
