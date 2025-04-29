#!/usr/bin/env python3
"""
Test the custom tokenizer and compare it with the original DeepSeek tokenizer
"""

import os
import json
from transformers import PreTrainedTokenizerFast

def test_tokenizer(tokenizer, text, tokenizer_name=""):
    """Test a tokenizer on a given text and print the results."""
    print(f"\n=== Testing {tokenizer_name} ===")
    
    # Encode the text
    tokens = tokenizer.encode(text)
    
    # Get token strings
    token_strings = tokenizer.convert_ids_to_tokens(tokens)
    
    # Print results
    print(f"Input text: '{text}'")
    print(f"Token IDs: {tokens}")
    print(f"Token strings: {token_strings}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Decoded text: '{tokenizer.decode(tokens)}'")
    
    return tokens, token_strings

def main():
    # Load our custom tokenizer
    custom_tokenizer_path = "./tokenizer_output/tokenizer.json"
    custom_hf_tokenizer_path = "./tokenizer_output/hf_tokenizer"
    
    # Load original DeepSeek tokenizer
    deepseek_tokenizer_path = "./deepseek_tok/tokenizer.json"
    
    # Test texts
    test_texts = [
        "strawberry",
        "Hello, world! How are you doing today?",
        "مرحبا بالعالم! كيف حالك اليوم؟",  # Arabic: "Hello world! How are you today?"
        "The Quasar-3.0 model uses a Mixture of Experts (MoE) architecture.",
        "1234 5678 9012"  # Numbers
    ]
    
    # Load tokenizers
    try:
        print("Loading custom tokenizer...")
        custom_tokenizer = PreTrainedTokenizerFast(tokenizer_file=custom_tokenizer_path)
        custom_tokenizer.bos_token = "<｜begin▁of▁sentence｜>"
        custom_tokenizer.eos_token = "<｜end▁of▁sentence｜>"
        custom_tokenizer.pad_token = "<｜▁pad▁｜>"
        custom_tokenizer.unk_token = "<｜▁unk▁｜>"
        
        print("Loading DeepSeek tokenizer...")
        deepseek_tokenizer = PreTrainedTokenizerFast(tokenizer_file=deepseek_tokenizer_path)
        deepseek_tokenizer.bos_token = "<｜begin▁of▁sentence｜>"
        deepseek_tokenizer.eos_token = "<｜end▁of▁sentence｜>"
        deepseek_tokenizer.pad_token = "<｜▁pad▁｜>"
        deepseek_tokenizer.unk_token = "<｜▁unk▁｜>"
        
        # Test each text
        for text in test_texts:
            custom_tokens, custom_strings = test_tokenizer(custom_tokenizer, text, "Custom Tokenizer")
            deepseek_tokens, deepseek_strings = test_tokenizer(deepseek_tokenizer, text, "DeepSeek Tokenizer")
            
            # Compare
            print("\n=== Comparison ===")
            if len(custom_tokens) == len(deepseek_tokens):
                print(f"✅ Both tokenizers produced the same number of tokens: {len(custom_tokens)}")
            else:
                print(f"❌ Different number of tokens: Custom={len(custom_tokens)}, DeepSeek={len(deepseek_tokens)}")
            
            print("-" * 50)
    
    except Exception as e:
        print(f"Error testing tokenizers: {e}")

if __name__ == "__main__":
    main()
