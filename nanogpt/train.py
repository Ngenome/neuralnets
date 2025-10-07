import torch
import argparse
import glob
import os
from tqdm.auto import tqdm
import wandb
from nanogpt.model import GPTLanguageModel
from nanogpt.data import prepare_data, get_batch
from nanogpt.checkpoint import save_checkpoint
from nanogpt.config_manager import ExperimentConfig, ModelConfig, TrainingConfig, DataConfig


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(config.training.eval_iters)
        for k in range(config.training.eval_iters):
            X, Y = get_batch(split, train_data, val_data)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_latest_checkpoint(experiment_name):
    """Find the most recent checkpoint file for an experiment"""
    exp_dir = os.path.join("experiments", experiment_name)
    if not os.path.exists(exp_dir):
        return None

    checkpoints = glob.glob(os.path.join(exp_dir, "checkpoint_*.pth"))
    if not checkpoints:
        return None

    # Sort by timestamp in filename
    return max(
        checkpoints,
        key=lambda x: "_".join(os.path.basename(x).split("_")[1:3]).replace(".pth", ""),
    )


def train(config: ExperimentConfig, resume=False):
    """
    Train the model with given configuration.

    Args:
        config: ExperimentConfig with all hyperparameters
        resume: Whether to resume from latest checkpoint
    """
    # Set random seed
    torch.manual_seed(config.seed)

    # Prepare data
    print("Preparing data...")
    train_data, val_data, tokenizer, vocab_size = prepare_data()

    # Update vocab size in config
    config.model.vocab_size = vocab_size

    start_iter = 0

    if resume:
        checkpoint_path = get_latest_checkpoint(config.experiment_name)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
            # Load model first
            from nanogpt.checkpoint import load_checkpoint_for_training

            model = GPTLanguageModel(vocab_size)
            model.to(config.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.training.learning_rate
            )
            loaded_config, start_iter = load_checkpoint_for_training(
                checkpoint_path, model, optimizer, config.device
            )
            # Use loaded config but allow overriding experiment name
            config = loaded_config
            config.experiment_name = (
                config.experiment_name
            )  # Keep the requested experiment name
            print(f"Resuming from iteration {start_iter}")
        else:
            print(
                f"No checkpoint found for experiment '{config.experiment_name}', starting from scratch"
            )
            model = GPTLanguageModel(vocab_size)
            model.to(config.device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config.training.learning_rate
            )
    else:
        # Initialize model
        print(f"Initializing model with vocab_size={vocab_size}...")
        model = GPTLanguageModel(vocab_size)
        model.to(config.device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.training.learning_rate
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Experiment: {config.experiment_name}")
    print(f"Git hash: {config.git_hash}")

    # Initialize wandb
    wandb.init(
        project="nanogpt",
        name=config.experiment_name,
        resume="allow" if resume else None,
        config=config.to_dict(),
    )

    # Training loop
    print(f"\nStarting training on {config.device} from iteration {start_iter}...")
    for iter in tqdm(range(start_iter, config.training.max_iters)):
        # Evaluate loss periodically
        if (
            iter % config.training.eval_interval == 0
            or iter == config.training.max_iters - 1
        ):
            losses = estimate_loss(model, train_data, val_data, config)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

            # Log to wandb
            wandb.log(
                {
                    "train_loss": losses["train"],
                    "val_loss": losses["val"],
                    "iter": iter,
                }
            )

            # Save checkpoint
            if iter > 0:
                save_checkpoint(
                    model,
                    config,
                    optimizer=optimizer,
                    epoch=iter,
                    loss=losses["val"],
                    experiment_name=config.experiment_name,
                )

        # Training step
        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final checkpoint
    print("\nTraining complete!")
    save_checkpoint(
        model,
        config,
        optimizer=optimizer,
        epoch=config.training.max_iters,
        loss=losses["val"],
        experiment_name=config.experiment_name,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    parser.add_argument(
        "--name",
        type=str,
        default="shakespeare-v1",
        help="Experiment name for organization",
    )

    # Model hyperparameters
    parser.add_argument("--n-embd", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--n-head", type=int, default=6, help="Number of heads")
    parser.add_argument("--n-layer", type=int, default=8, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--block-size", type=int, default=256, help="Context length")
    parser.add_argument("--max-iters", type=int, default=20000, help="Max iterations")
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )

    args = parser.parse_args()

    # Build config from arguments
    config = ExperimentConfig(
        model=ModelConfig(
            n_embd=args.n_embd,
            n_head=args.n_head,
            n_layer=args.n_layer,
            dropout=args.dropout,
        ),
        training=TrainingConfig(
            batch_size=args.batch_size,
            block_size=args.block_size,
            max_iters=args.max_iters,
            learning_rate=args.learning_rate,
        ),
        data=DataConfig(),
        experiment_name=args.name,
    )

    train(config, resume=args.resume)
