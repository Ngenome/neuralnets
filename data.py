import torch
import requests
from tokenizer import train_bpe_tokenizer, load_bpe_tokenizer, encode_text
from config import (
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
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def prepare_data():
    """Main function to prepare all data"""
    download_data()
    tokenizer = prepare_tokenizer()
    train_data, val_data = load_data(tokenizer)
    vocab_size = tokenizer.get_vocab_size()
    return train_data, val_data, tokenizer, vocab_size
