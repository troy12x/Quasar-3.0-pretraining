import os
import torch
import argparse
import deepspeed
from transformers import PreTrainedTokenizerFast
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9):
    """Generate text from the model given a prompt."""
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    # Store the original prompt tokens for later
    prompt_tokens = input_ids.size(1)
    
    # Generate tokens
    generated_tokens = []
    
    logger.info(f"Starting generation with prompt: {prompt}")
    logger.info(f"Prompt has {prompt_tokens} tokens")
    
    with torch.no_grad():
        for i in range(max_length):
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get logits from the output dictionary
            if isinstance(outputs, dict) and "main_logits" in outputs:
                logits = outputs["main_logits"]
            else:
                # Handle different output formats
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
            
            # Get the logits for the last token
            next_token_logits = logits[0, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Create a mask for indices to remove
            indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
            indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True
            
            # Set the removed indices to negative infinity
            next_token_logits[indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Add to generated tokens
            token_id = next_token.item()
            generated_tokens.append(token_id)
            
            # Update input_ids and attention_mask
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token.unsqueeze(0))], dim=1)
            
            # Print progress every 10 tokens
            if (i + 1) % 10 == 0:
                logger.info(f"Generated {i + 1} tokens")
                
                # Try to decode what we have so far
                try:
                    current_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    logger.info(f"Current text: {current_text}")
                except Exception as e:
                    logger.warning(f"Error decoding tokens: {e}")
            
            # Stop if end of sequence token is generated or a special token
            if token_id == tokenizer.eos_token_id or token_id < 3:
                break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Return the full text (prompt + generated)
    return prompt + generated_text

def main():
    parser = argparse.ArgumentParser(description="Generate text with Quasar model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer.json", help="Path to tokenizer.json file")
    parser.add_argument("--prompt", type=str, default="تسوق عامل", help="Prompt for text generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for nucleus sampling")
    args = parser.parse_args()
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=args.tokenizer_path)
    
    # Create model with compatible configuration
    logger.info("Creating Quasar model with compatible configuration")
    from quasar import QuasarConfig, Quasar
    
    config = QuasarConfig()
    config.hidden_size = 1536
    config.num_hidden_layers = 12
    config.num_attention_heads = 12
    config.intermediate_size = 4096
    config.kv_compressed_dim = 192
    config.query_compressed_dim = 384
    config.num_experts = 16
    config.num_shared_experts = 1
    config.num_routed_experts = 16
    config.top_k = 4
    config.use_nsa = False
    
    # Add any missing NSA parameters to avoid errors
    config.nsa_block_size = 64
    config.nsa_window_size = 256
    config.nsa_sparse_block_size = 32
    
    model = Quasar(config)
    
    # Initialize DeepSpeed with minimal config
    logger.info("Initializing DeepSpeed")
    ds_config = {
        "train_batch_size": 1,
        "fp16": {"enabled": False},
        "bf16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "cpu"},
            "offload_param": {"device": "cpu"},
            "overlap_comm": True,
            "contiguous_gradients": True
        }
    }
    
    # Initialize DeepSpeed
    model_engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint_path}")
    model_engine.load_checkpoint(args.checkpoint_path)
    logger.info("Checkpoint loaded successfully")
    
    # Generate text for the specified prompt
    generated_text = generate_text(
        model_engine,
        tokenizer,
        args.prompt,
        max_length=args.max_length,
        temperature=args.temperature,
        top_p=args.top_p
    )
    
    # Print the generated text
    print("\n" + "="*50)
    print("GENERATED TEXT:")  
    print(generated_text)
    print("="*50 + "\n")
    
    # Also try additional Arabic prompts if not already specified
    arabic_prompts = ["تسوق عامل", "الرقبة ترطيب"]
    for prompt in arabic_prompts:
        if prompt != args.prompt:
            logger.info(f"Generating text for additional Arabic prompt: {prompt}")
            generated_text = generate_text(
                model_engine,
                tokenizer,
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            # Print the generated text
            print("\n" + "="*50)
            print(f"PROMPT: {prompt}")
            print("GENERATED TEXT:")
            print(generated_text)
            print("="*50 + "\n")
    
    logger.info("Text generation completed")

if __name__ == "__main__":
    main()
