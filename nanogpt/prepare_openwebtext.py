"""
Prepare OpenWebText dataset for training.
Based on Karpathy's nanoGPT approach.
Downloads ~54GB from HuggingFace and tokenizes with GPT-2 tokenizer.
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Paths
DATA_DIR = "nanogpt/data/openwebtext"
TRAIN_FILE = os.path.join(DATA_DIR, "train.bin")
VAL_FILE = os.path.join(DATA_DIR, "val.bin")

# Processing config
num_proc = 8  # number of workers

# GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")


def main():
    """Download and prepare OpenWebText"""
    print("=" * 80)
    print("OpenWebText Preparation")
    print("=" * 80)
    print("This will:")
    print("  1. Download ~54GB from HuggingFace (cached)")
    print("  2. Tokenize ~8M documents with GPT-2 tokenizer")
    print("  3. Save as .bin files for fast loading")
    print("=" * 80)

    os.makedirs(DATA_DIR, exist_ok=True)

    # Download dataset (Parquet format, ~10GB, about 1.6M documents - 20% of full)
    print("\nDownloading OpenWebText from HuggingFace...")
    print("Using Bingsu/openwebtext_20p (20% subset in Parquet format)")
    dataset = load_dataset("Bingsu/openwebtext_20p", num_proc=num_proc)

    # Create train and validation splits
    print("Creating train/val splits...")
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenize the dataset
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing dataset...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Concatenate all the ids in each dataset into one large file
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = TRAIN_FILE if split == 'train' else VAL_FILE
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
    print("✅ OpenWebText preparation complete!")
    print("=" * 80)
    print(f"Train: {TRAIN_FILE}")
    print(f"Val: {VAL_FILE}")
    print(f"Vocab size: {enc.n_vocab} (GPT-2)")
    print("\nTo train:")
    print("  uv run python -m nanogpt.train --name openwebtext-v1 --dataset openwebtext")


if __name__ == '__main__':
    main()
