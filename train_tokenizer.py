#!/usr/bin/env python3
"""
Train a custom tokenizer for Quasar-3.0 using the Hugging Face dataset
miscovory/arabic_egypt_english_world_facts
"""

import os
import argparse
import tempfile
import json
from pathlib import Path
from typing import List, Optional

from datasets import load_dataset
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers, decoders
from tokenizers.pre_tokenizers import ByteLevel, Split, Sequence
from tokenizers.implementations import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast


def download_and_prepare_dataset(dataset_name: str, output_dir: str) -> List[str]:
    """Download a dataset from Hugging Face and prepare text files for tokenizer training.
    
    Args:
        dataset_name: Name of the Hugging Face dataset
        output_dir: Directory to save the text files
        
    Returns:
        List of paths to the text files
    """
    print(f"Downloading dataset: {dataset_name}")
    
    try:
        # Try loading the dataset
        dataset = load_dataset(dataset_name)
        print(f"Successfully loaded dataset {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise RuntimeError(f"Could not load dataset {dataset_name}. Error: {e}")
    
    # Get all splits (train, validation, test)
    splits = list(dataset.keys())
    print(f"Dataset splits: {splits}")
    
    # Create output files for each split
    all_files = []
    
    for split in splits:
        # Get columns
        columns = dataset[split].column_names
        print(f"Columns in {split}: {columns}")
        
        # Create output file
        output_file = os.path.join(output_dir, f"{split}.txt")
        all_files.append(output_file)
        
        # Extract text from all columns
        with open(output_file, "w", encoding="utf-8") as f:
            for example in dataset[split]:
                for column in columns:
                    if isinstance(example[column], str):
                        # Write each text field to the file
                        f.write(example[column] + "\n")
        
        print(f"Created {output_file} with text from {split} split")
    
    return all_files


def train_tokenizer(
    files: List[str],
    vocab_size: int = 128128,
    min_frequency: int = 2,
    output_dir: str = "./",
):
    """Train a BPE tokenizer on the given files.
    
    Args:
        files: List of text files to train on
        vocab_size: Size of the vocabulary
        min_frequency: Minimum frequency for a token to be included
        output_dir: Directory to save the tokenizer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define special tokens - matching those used in pretrain.py
    special_tokens = [
        "<｜begin▁of▁sentence｜>",  # Beginning of sequence (BOS)
        "<｜end▁of▁sentence｜>",   # End of sequence (EOS)
        "<｜▁pad▁｜>",            # Padding token
        "<｜▁unk▁｜>",            # Unknown token
        "<mask>",                # Mask token for MLM
        "<|im_start|>",          # Chat format markers
        "<|im_end|>",
    ]
    
    print(f"Training tokenizer with vocab_size={vocab_size}, min_frequency={min_frequency}")
    print(f"Special tokens: {special_tokens}")
    
    # Initialize a more customized tokenizer instead of the default ByteLevelBPETokenizer
    # This allows us to set specific pre-tokenizers like in DeepSeek's tokenizer
    
    # First create a basic tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Add proper normalizers for handling different scripts including Arabic
    # Note: We're not using Lowercase() to preserve capitalization
    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),              # Unicode normalization
        normalizers.StripAccents(),     # Remove accents
    ])
    
    # Add the pre-tokenizers from DeepSeek's tokenizer
    tokenizer_pre_tokenizers = [
        # Split numbers
        Split(pattern="\\p{N}{1,3}", behavior="isolated"),
        # Split Chinese/Japanese/Korean characters
        Split(pattern="[一-龥぀-ゟ゠-ヿ]+", behavior="isolated"),
        # Split Arabic characters - add specific handling for Arabic script
        Split(pattern="[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+", behavior="isolated"),
        # Split on punctuation, words, etc.
        Split(
            pattern="[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+| ?[\\p{P}\\p{S}]+[\r\n]*|\\s*[\r\n]+|\\s+(?!\\S)|\\s+",
            behavior="isolated"
        ),
        # ByteLevel tokenization
        ByteLevel(add_prefix_space=False)
    ]
    
    # Combine pre-tokenizers into a sequence
    tokenizer.pre_tokenizer = Sequence(tokenizer_pre_tokenizers)
    
    # Add post-processor to handle decoding properly
    tokenizer.decoder = decoders.ByteLevel()
    
    # Configure the trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True
    )
    
    # Train on the files
    tokenizer.train(files=files, trainer=trainer)
    
    # Save the tokenizer as a single JSON file
    tokenizer_json_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    
    # Also save the vocabulary and merges files for compatibility
    vocab_file = os.path.join(output_dir, "tokenizer-vocab.json")
    merges_file = os.path.join(output_dir, "tokenizer-merges.txt")
    
    # Extract vocabulary and merges from the tokenizer
    with open(vocab_file, "w", encoding="utf-8") as f:
        json.dump(tokenizer.get_vocab(), f, ensure_ascii=False, indent=2)
    
    # We can't easily extract merges from the tokenizer object
    # But the tokenizer.json file contains all necessary information
    
    print(f"Tokenizer saved to {tokenizer_json_path}")
    
    return tokenizer


def convert_to_hf_tokenizer(tokenizer_path: str, output_dir: str):
    """Convert the trained tokenizer to a Hugging Face PreTrainedTokenizerFast.
    
    Args:
        tokenizer_path: Path to the tokenizer.json file
        output_dir: Directory to save the HF tokenizer
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the tokenizer
    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Convert to HF tokenizer with the same special tokens as in pretrain.py
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<｜begin▁of▁sentence｜>",
        eos_token="<｜end▁of▁sentence｜>",
        pad_token="<｜▁pad▁｜>",
        unk_token="<｜▁unk▁｜>",
        mask_token="<mask>",
    )
    
    # Save the HF tokenizer
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Hugging Face tokenizer saved to {output_dir}")
    
    return hf_tokenizer


def test_tokenizer(tokenizer, texts):
    """Test the tokenizer on sample texts.
    
    Args:
        tokenizer: The tokenizer to test
        texts: List of sample texts
    """
    print("\nTesting tokenizer on sample texts:")
    
    for i, text in enumerate(texts):
        # Encode and decode
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        # Print results
        print(f"\nSample {i+1}:")
        print(f"Original: {text}")
        print(f"Token count: {len(encoded)}")
        print(f"Decoded: {decoded}")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for Quasar-3.0")
    parser.add_argument(
        "--dataset",
        type=str,
        default="miscovery/arabic_egypt_english_world_facts",
        help="Hugging Face dataset to use (default: miscovery/arabic_egypt_english_world_facts)",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=128128,
        help="Size of the vocabulary (default: 128128)",
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token (default: 2)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tokenizer_output",
        help="Directory to save the tokenizer (default: ./tokenizer_output)",
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a temporary directory for dataset files
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Created temporary directory: {temp_dir}")
        
        try:
            # Download and prepare dataset
            files = download_and_prepare_dataset(args.dataset, temp_dir)
            
            if not files:
                print("No files were created. Exiting.")
                return
            
            # Train tokenizer
            tokenizer = train_tokenizer(
                files=files,
                vocab_size=args.vocab_size,
                min_frequency=args.min_frequency,
                output_dir=args.output_dir,
            )
            
            # Convert to HF tokenizer
            hf_tokenizer_dir = os.path.join(args.output_dir, "hf_tokenizer")
            hf_tokenizer = convert_to_hf_tokenizer(
                os.path.join(args.output_dir, "tokenizer.json"),
                hf_tokenizer_dir,
            )
            
            # Test the tokenizer
            sample_texts = [
                "This is a test of the Quasar-3.0 tokenizer in English.",
                "هذا اختبار لمحلل الرموز Quasar-3.0 باللغة العربية الفصحى.",  # MSA Arabic
                "ده اختبار لمحلل الرموز Quasar-3.0 باللهجة المصرية."  # Egyptian Arabic
            ]
            test_tokenizer(hf_tokenizer, sample_texts)
            
            # Print statistics
            print("\nTokenizer Statistics:")
            print(f"Vocabulary size: {args.vocab_size}")
            print(f"Special tokens: {hf_tokenizer.all_special_tokens}")
            print(f"Special token IDs: {hf_tokenizer.all_special_ids}")
            
            print("\nTokenizer training complete!")
            print(f"Tokenizer files saved to {args.output_dir}")
            print(f"Hugging Face tokenizer saved to {hf_tokenizer_dir}")
            
        except Exception as e:
            print(f"Error during tokenizer training: {e}")


if __name__ == "__main__":
    main()