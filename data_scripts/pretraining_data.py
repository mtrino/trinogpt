import os
import tiktoken
import multiprocessing as mp
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# 1. SETUP CONFIGURATION
OUTPUT_DIR = "data_mixture_2.5B"
os.makedirs(OUTPUT_DIR, exist_ok=True)
shard_size = int(1e8) # 100M tokens per shard

# Define our exact 2.5 Billion token mixture recipe
TARGET_RECIPE = {
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_key": "text",
        "target_tokens": 1_250_000_000  # 50%
    },
    "cosmopedia_code": {
        "path": "HuggingFaceTB/cosmopedia",
        "name": "stanford",  
        "split": "train",
        "text_key": "text",
        "target_tokens": 500_000_000     
    },
    "slimpajama_wiki": {
        "path": "DKYoon/SlimPajama-6B",
        "name": "default",
        "split": "train",
        "text_key": "text",
        "target_tokens": 500_000_000    # 20%
    },
    "openwebmath": {
        "path": "open-web-math/open-web-math",
        "name": "default",
        "split": "train",
        "text_key": "text",
        "target_tokens": 250_000_000    # 10%
    }
}

# 2. INITIALIZE TOKENIZER (GPT-2 for uint16 compatibility)
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']

def tokenize(text):
    """Tokenizes a single text string and returns a numpy array of uint16 tokens"""
    tokens = [eot] # the special <|endoftext|> delimits all documents
    tokens.extend(enc.encode_ordinary(text))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# 3. STREAM AND TOKENIZE PIPELINE
if __name__ == '__main__':
    nprocs = max(1, os.cpu_count() // 2)
    print(f"Starting tokenization pipeline using {nprocs} CPU cores...")

    for dataset_name, config in TARGET_RECIPE.items():
        print(f"\n--- Processing {dataset_name} (Target: {config['target_tokens']:,} tokens) ---")
        
        # Load dataset in streaming mode (0 bytes downloaded up front)
        dataset = load_dataset(
            config["path"], 
            name=config["name"], 
            split=config["split"], 
            streaming=True
        )

        # Generator to yield pure text strings for the multiprocessing pool
        def text_generator():
            for row in dataset:
                text = row[config["text_key"]]
                if text and len(text.strip()) > 0:
                    yield text

        shard_index = 0
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        global_token_count = 0
        progress_bar = None

        with mp.Pool(nprocs) as pool:
            for tokens in pool.imap(tokenize, text_generator(), chunksize=16):
                
                # Check if this document overshoots our global dataset cap
                tokens_needed = config["target_tokens"] - global_token_count
                if len(tokens) > tokens_needed:
                    tokens = tokens[:tokens_needed]

                # Is there enough space in the current shard for the new tokens?
                if token_count + len(tokens) < shard_size:
                    all_tokens_np[token_count:token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    global_token_count += len(tokens)
                    
                    # Update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit="tok", desc=f"{dataset_name} Shard {shard_index:03d}")
                    progress_bar.update(len(tokens))
                else:
                    # Write the current shard and start a new one
                    split = "val" if shard_index == 0 else "train"
                    filename = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}_{shard_index:06d}.npy")
                    
                    # Fill remainder of current shard
                    remainder = shard_size - token_count
                    progress_bar.update(remainder)
                    all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                    write_datafile(filename, all_tokens_np)
                    
                    # Setup for next shard
                    shard_index += 1
                    progress_bar = None
                    
                    # Populate the next shard with leftovers from the previous one
                    leftover_len = len(tokens) - remainder
                    all_tokens_np[0:leftover_len] = tokens[remainder:]
                    token_count = leftover_len
                    global_token_count += len(tokens)

                # Stop processing this dataset if we hit our target mixture cap
                if global_token_count >= config["target_tokens"]:
                    if progress_bar is not None:
                        progress_bar.close()
                    break

        # Write any remaining tokens as the last shard (will likely be < 100M tokens)
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(OUTPUT_DIR, f"{dataset_name}_{split}_{shard_index:06d}.npy")
            write_datafile(filename, all_tokens_np[:token_count])
            print(f"Saved final partial shard for {dataset_name}: {token_count:,} tokens.")

    print(f"\nPipeline complete! All binary .npy shards saved to {OUTPUT_DIR}")