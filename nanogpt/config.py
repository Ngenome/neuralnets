import torch

# Model hyperparameters (scaled for A10G)
n_embd = 384  # 4x larger
n_head = 6
n_layer = 8  # More depth
dropout = 0.2

# Training hyperparameters
batch_size = 128  # Reduced for larger model
block_size = 256  # Longer context
max_iters = 20000  # 4x more training
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200

# Data
data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
data_file = "input.txt"

# BPE tokenizer
tokenizer_vocab_size = 1000
tokenizer_path = "bpe_tokenizer.json"

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Random seed
seed = 1337
