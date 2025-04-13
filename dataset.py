import os
import multiprocessing
from datasets import load_dataset, DatasetDict
from huggingface_hub import login
from functools import partial
import time

def process_split(args):
    """Process a single dataset split in parallel."""
    config_name, split_name, original_dataset_name, num_proc = args
    
    print(f"Starting download and processing of config {config_name}, split: {split_name}")
    start_time = time.time()
    
    try:
        # Load just this split for this config
        split_dataset = load_dataset(
            original_dataset_name,
            config_name,
            split=split_name,
            num_proc=num_proc  # Use multiple processes for loading
        )
        
        # Check if 'text' column exists
        if 'text' not in split_dataset.column_names:
            print(f"Warning: 'text' column not found in {config_name}/{split_name}. Available columns: {split_dataset.column_names}")
            return config_name, split_name, None
        
        # Keep only the 'text' column
        cleaned_split = split_dataset.select_columns(['text'])
        
        elapsed = time.time() - start_time
        print(f"Completed {config_name}/{split_name}: {len(cleaned_split)} examples in {elapsed:.2f} seconds")
        
        return config_name, split_name, cleaned_split
    
    except Exception as e:
        print(f"Error processing {config_name}/{split_name}: {e}")
        return config_name, split_name, None

def main():
    # Set your HuggingFace token
    HF_TOKEN = os.environ.get("HF_TOKEN")
    if not HF_TOKEN:
        print("Warning: HF_TOKEN not found in environment variables.")
        print("Please set your HuggingFace token using:")
        print("export HF_TOKEN=your_token_here")
        return
    
    # Login to HuggingFace
    login(token=HF_TOKEN)
    
    original_dataset_name = "HuggingFaceFW/clean-wikipedia"
    target_dataset_name = "eyad-silx/quasar-clean-wikipedia"
    
    # Get all available configs (languages)
    print(f"Getting all available configurations for {original_dataset_name}...")
    
    # Manually get the list of configs from the error message we saw earlier
    # This is more efficient than trying to load the entire dataset info
    config_names = ['ab', 'ace', 'ady', 'af', 'als', 'alt', 'am', 'ami', 'an', 'ang', 'anp', 'ar', 'arc', 'ary', 'arz', 'as', 'ast', 'atj', 'av', 'avk', 'awa', 'ay', 'az', 'azb', 'ba', 'ban', 'bar', 'bat-smg', 'bcl', 'be', 'be-x-old', 'bg', 'bh', 'bi', 'bjn', 'blk', 'bm', 'bn', 'bo', 'bpy', 'br', 'bs', 'bug', 'bxr', 'ca', 'cbk-zam', 'cdo', 'ce', 'ceb', 'ch', 'chr', 'chy', 'ckb', 'co', 'cr', 'crh', 'cs', 'csb', 'cu', 'cv', 'cy', 'da', 'dag', 'de', 'din', 'diq', 'dsb', 'dty', 'dv', 'dz', 'ee', 'el', 'eml', 'en', 'eo', 'es', 'et', 'eu', 'ext', 'fa', 'fat', 'ff', 'fi', 'fiu-vro', 'fj', 'fo', 'fon', 'fr', 'frp', 'frr', 'fur', 'fy', 'ga', 'gag', 'gan', 'gcr', 'gd', 'gl', 'glk', 'gn', 'gom', 'gor', 'got', 'gpe', 'gu', 'guc', 'gur', 'guw', 'gv', 'ha', 'hak', 'haw', 'he', 'hi', 'hif', 'hr', 'hsb', 'ht', 'hu', 'hy', 'hyw', 'ia', 'id', 'ie', 'ig', 'ik', 'ilo', 'inh', 'io', 'is', 'it', 'iu', 'ja', 'jam', 'jbo', 'jv', 'ka', 'kaa', 'kab', 'kbd', 'kbp', 'kcg', 'kg', 'ki', 'kk', 'kl', 'km', 'kn', 'ko', 'koi', 'krc', 'ks', 'ksh', 'ku', 'kv', 'kw', 'ky', 'la', 'lad', 'lb', 'lbe', 'lez', 'lfn', 'lg', 'li', 'lij', 'lld', 'lmo', 'ln', 'lo', 'lt', 'ltg', 'lv', 'mad', 'mai', 'mdf', 'mg', 'mhr', 'mi', 'min', 'mk', 'ml', 'mn', 'mni', 'mnw', 'mr', 'mrj', 'ms', 'mt', 'mwl', 'my', 'myv', 'mzn', 'nah', 'nap', 'nds', 'ne', 'new', 'nia', 'nl', 'nn', 'no', 'nov', 'nqo', 'nrm', 'nso', 'nv', 'ny', 'oc', 'olo', 'om', 'or', 'os', 'pa', 'pag', 'pam', 'pap', 'pcd', 'pcm', 'pdc', 'pfl', 'pi', 'pih', 'pl', 'pms', 'pnb', 'pnt', 'ps', 'pt', 'pwn', 'qu', 'rm', 'rmy', 'rn', 'ro', 'roa-rup', 'ru', 'rue', 'rw', 'sa', 'sah', 'sat', 'sc', 'scn', 'sco', 'sd', 'se', 'sg', 'sh', 'shi', 'shn', 'si', 'sk', 'skr', 'sl', 'sm', 'smn', 'sn', 'so', 'sq', 'sr', 'srn', 'ss', 'st', 'stq', 'su', 'sv', 'sw', 'szl', 'szy', 'ta', 'tay', 'tcy', 'te', 'tet', 'tg', 'th', 'ti', 'tk', 'tl', 'tly', 'tn', 'to', 'tpi', 'tr', 'trv', 'ts', 'tt', 'tum', 'tw', 'ty', 'tyv', 'udm', 'ug', 'uk', 'ur', 'uz', 've', 'vec', 'vep', 'vi', 'vls', 'vo', 'wa', 'war', 'wo', 'wuu', 'xal', 'xh', 'xmf', 'yi', 'yo', 'za', 'zea', 'zh', 'zh-classical', 'zh-min-nan', 'zh-yue', 'zu']
    
    # For testing, you might want to limit to just a few configurations initially
    # config_names = config_names[:5]  # Uncomment this to process only the first 5 configs
    
    print(f"Found {len(config_names)} configurations")
    
    # Create tasks for parallel processing
    tasks = []
    for config_name in config_names:
        try:
            # Get available splits for this config
            temp_dataset = load_dataset(original_dataset_name, config_name, trust_remote_code=True)
            split_names = list(temp_dataset.keys())
            print(f"Config {config_name} has {len(split_names)} splits: {split_names}")
            
            # Add tasks for each config/split combination
            for split_name in split_names:
                tasks.append((config_name, split_name, original_dataset_name, 4))  # Using 4 processes per task
        except Exception as e:
            print(f"Error loading configuration {config_name}: {e}")
    
    print(f"Total tasks to process: {len(tasks)}")
    
    # Create a pool of workers (adjust based on your CPU)
    num_cores = multiprocessing.cpu_count()
    pool_size = max(1, min(num_cores - 1, len(tasks)))  # Leave one core free, at least 1
    print(f"Using {pool_size} processes for parallel processing")
    
    # Process all configs/splits in parallel
    cleaned_dataset = DatasetDict()
    with multiprocessing.Pool(processes=pool_size) as pool:
        results = pool.map(process_split, tasks)
    
    # Organize results into a nested DatasetDict structure
    for config_name, split_name, split_data in results:
        if split_data is not None:
            config_key = f"{config_name}_{split_name}"
            cleaned_dataset[config_key] = split_data
    
    # Print dataset information
    print("\nCleaned dataset structure:")
    for key, dataset in cleaned_dataset.items():
        print(f"Split: {key}, Examples: {len(dataset)}, Columns: {dataset.column_names}")
    
    if not cleaned_dataset:
        print("No data was successfully processed. Exiting.")
        return
    
    # Push to HuggingFace
    print(f"\nPushing cleaned dataset to {target_dataset_name}...")
    
    cleaned_dataset.push_to_hub(
        target_dataset_name,
        token=HF_TOKEN,
        private=False,
        max_shard_size="500MB"  # Create smaller chunks for faster uploads
    )
    
    print(f"Dataset successfully pushed to {target_dataset_name}!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds")