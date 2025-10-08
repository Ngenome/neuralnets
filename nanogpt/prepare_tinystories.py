"""
Prepare TinyStories dataset for training.
TinyStories is a dataset of synthetic short stories for children, created by Microsoft Research.
Much smaller than OpenWebText (~300-500M tokens vs 1.78B).
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Paths
DATA_DIR = "nanogpt/data/tinystories"
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "val.bin")

# Processing config
num_proc = 8  # number of workers

# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")


def main():
    """Download and prepare TinyStories"""
    print("=" * 80)
    print("TinyStories Preparation")
    print("=" * 80)
    print("This will:")
    print("  1. Download TinyStories from HuggingFace (~500MB)")
    print("  2. Tokenize ~2.1M stories with GPT-2 tokenizer")
    print("  3. Save as .bin files for fast loading")
    print("=" * 80)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download dataset
    print("\nDownloading TinyStories from HuggingFace...")
    print("Using roneneldan/TinyStories dataset")
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc)

    # Use existing train/validation splits
    print("Using existing train/validation splits...")
    split_dataset = {
        'train': dataset['train'],
        'val': dataset['validation']
    }

    # Tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing dataset...")
    tokenized = {}
    for split_name, dset in split_dataset.items():
        print(f"\nTokenizing {split_name} split...")
        tokenized[split_name] = dset.map(
            process,
            remove_columns=['text'],
            desc=f"tokenizing {split_name}",
            num_proc=num_proc,
        )

    # Concatenate all the ids in each dataset into one large file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = TRAIN_FILE if split == 'train' else VAL_FILE

        # Skip if file already exists
        if os.path.exists(filename):
            print(f"⏭️  Skipping {filename} (already exists)")
            continue

        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()

        print(f"✅ Saved {filename}: {arr_len:,} tokens")

    print("\n" + "=" * 80)
    print("✅ TinyStories preparation complete!")
    print("=" * 80)
    print(f"Train: {TRAIN_FILE}")
    print(f"Val: {VAL_FILE}")
    print(f"Vocab size: {enc.n_vocab} (GPT-2)")
    print("\nTo train:")
    print("  uv run python -m nanogpt.train --name tinystories-v1")


if __name__ == '__main__':
    main()
