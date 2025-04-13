#!/bin/bash

# Script to generate text with the Quasar model checkpoint
CHECKPOINT_PATH="./checkpoints/checkpoint-3000"
TOKENIZER_PATH="./tokenizer.json"

# Set generation parameters
python generate_text.py \
    --checkpoint_path $CHECKPOINT_PATH \
    --tokenizer_path $TOKENIZER_PATH \
    --max_length 200 \
    --temperature 0.9 \
    --top_p 0.95

# The script will automatically test both Abkhaz prompts:
# - "Аԥсуа бызшәа" (Abkhaz language)
# - "Аԥсны ҳәынҭқарра" (Republic of Abkhazia)

echo "If the script fails, try running with CUDA_VISIBLE_DEVICES=0 prefix:"
echo "CUDA_VISIBLE_DEVICES=0 python generate_text.py --checkpoint_path $CHECKPOINT_PATH --tokenizer_path $TOKENIZER_PATH --max_length 200 --temperature 0.9 --top_p 0.95"
