# NanoGPT

A minimal GPT implementation for training a character-level language model on Shakespeare text.

## Project Structure

```
nanogpt/
├── config.py          # Hyperparameters and configuration
├── model.py           # GPT model architecture
├── data.py            # Data loading and preprocessing
├── tokenizer.py       # BPE tokenizer utilities
├── checkpoint.py      # Model checkpointing utilities
├── train.py           # Training script
├── inference.py       # Text generation script
└── pyproject.toml     # Project dependencies
```

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Set up Weights & Biases (optional but recommended):
```bash
export WANDB_API_KEY=your_api_key_here
# Or login interactively:
wandb login
```

## Usage

### Training

Train the model (downloads Shakespeare data automatically):

```bash
uv run train.py
```

The script will:
- Download training data
- Train/load BPE tokenizer
- Train the model with wandb logging
- Save checkpoints periodically

### Text Generation

Generate text from a trained checkpoint:

```bash
uv run inference.py --checkpoint model_YYYYMMDD_HHMMSS.pth
```

With a custom prompt:

```bash
uv run inference.py --checkpoint model_YYYYMMDD_HHMMSS.pth --prompt "To be, or not to be" --max-tokens 500
```

## Configuration

Edit `config.py` to adjust hyperparameters:

- `n_embd`: Embedding dimension (default: 96)
- `n_head`: Number of attention heads (default: 6)
- `n_layer`: Number of transformer layers (default: 6)
- `batch_size`: Batch size (default: 64)
- `learning_rate`: Learning rate (default: 3e-4)
- `max_iters`: Training iterations (default: 5000)

## Monitoring

Training metrics are logged to Weights & Biases. View your runs at:
https://wandb.ai/your-username/nanogpt
