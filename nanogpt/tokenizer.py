from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFD, Lowercase, StripAccents
import os

# 1. Train a new BPE tokenizer
def train_bpe_tokenizer(text_file, vocab_size=1000, save_path="bpe_tokenizer.json"):
    # Initialize tokenizer with empty BPE model
    tokenizer = Tokenizer(models.BPE())

    # Normalize text (optional: lowercase, remove accents)
    tokenizer.normalizer = None   # Shakespeare: keep casing & accents

    # Pre-tokenize by splitting into characters
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    # Decoder: reconstruct text from byte-level BPE tokens
    tokenizer.decoder = decoders.ByteLevel()

    # Special tokens (optional)
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens
    )

    # Train on your dataset
    tokenizer.train([text_file], trainer)

    # Save for later use
    tokenizer.save(save_path)

    print(f"✅ Trained and saved BPE tokenizer at {save_path}")
    return tokenizer


# 2. Load tokenizer
def load_bpe_tokenizer(path="bpe_tokenizer.json"):
    return Tokenizer.from_file(path)


# 3. Encode text → token IDs
def encode_text(tokenizer, text):
    return tokenizer.encode(text).ids


# 4. Decode token IDs → text
def decode_tokens(tokenizer, token_ids):
    return tokenizer.decode(token_ids)