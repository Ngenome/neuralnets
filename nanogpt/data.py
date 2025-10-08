import torch
import requests
import numpy as np
from nanogpt.tokenizer import train_bpe_tokenizer, load_bpe_tokenizer, encode_text
from nanogpt.config import (
    data_url,
    data_file,
    tokenizer_vocab_size,
    tokenizer_path,
    batch_size,
    block_size,
    device,
)
import os


def download_data():
    """Download the training data"""
    if os.path.exists(data_file):
        print(f"✅ Data file '{data_file}' already exists")
        return

    print(f"Downloading data from {data_url}...")
    response = requests.get(data_url)

    if response.status_code == 200:
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Download complete! File saved as {data_file}")
    else:
        raise Exception(f"Failed to download. Status code: {response.status_code}")


def prepare_tokenizer():
    """Train or load BPE tokenizer"""
    if os.path.exists(tokenizer_path):
        print(f"✅ Loading existing tokenizer from {tokenizer_path}")
        return load_bpe_tokenizer(tokenizer_path)

    print("Training new BPE tokenizer...")
    tokenizer = train_bpe_tokenizer(
        data_file, vocab_size=tokenizer_vocab_size, save_path=tokenizer_path
    )
    return tokenizer


def load_data(tokenizer):
    """Load and tokenize the dataset"""
    with open(data_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Encode entire dataset using BPE tokenizer
    token_ids = encode_text(tokenizer, text)
    data = torch.tensor(token_ids, dtype=torch.long)

    # Split into train/val
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"✅ Dataset loaded: {len(data)} tokens")
    print(f"   Train: {len(train_data)} tokens")
    print(f"   Val: {len(val_data)} tokens")

    return train_data, val_data


def get_batch(split, train_data, val_data):
    """Generate a batch of data"""
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # Handle both torch tensors and numpy memmaps
    if isinstance(data, np.memmap):
        x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix])
    else:
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


def load_openwebtext():
    """Load pre-tokenized OpenWebText dataset"""
    data_dir = "nanogpt/data/openwebtext"
    train_file = os.path.join(data_dir, "train.bin")
    val_file = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(
            f"OpenWebText data not found. Run: uv run python -m nanogpt.prepare_openwebtext"
        )

    print(f"Loading OpenWebText dataset...")
    # Keep as memmap to avoid loading entire dataset into RAM
    train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint16, mode='r')

    print(f"✅ OpenWebText loaded: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # GPT-2 tokenizer vocab size
    vocab_size = 50257

    return train_data, val_data, None, vocab_size


def load_tinystories():
    """Load pre-tokenized TinyStories dataset"""
    data_dir = "nanogpt/data/tinystories"
    train_file = os.path.join(data_dir, "train.bin")
    val_file = os.path.join(data_dir, "val.bin")

    if not os.path.exists(train_file) or not os.path.exists(val_file):
        raise FileNotFoundError(
            f"TinyStories data not found. Run: uv run python -m nanogpt.prepare_tinystories"
        )

    print(f"Loading TinyStories dataset...")
    # Keep as memmap to avoid loading entire dataset into RAM
    train_data = np.memmap(train_file, dtype=np.uint16, mode='r')
    val_data = np.memmap(val_file, dtype=np.uint16, mode='r')

    print(f"✅ TinyStories loaded: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # GPT-2 tokenizer vocab size
    vocab_size = 50257

    return train_data, val_data, None, vocab_size


def prepare_data(dataset="shakespeare"):
    """Main function to prepare all data"""
    if dataset == "openwebtext":
        return load_openwebtext()
    elif dataset == "tinystories":
        return load_tinystories()
    else:
        download_data()
        tokenizer = prepare_tokenizer()
        train_data, val_data = load_data(tokenizer)
        vocab_size = tokenizer.get_vocab_size()
        return train_data, val_data, tokenizer, vocab_size
