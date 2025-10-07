import torch
import os
from datetime import datetime

CHECKPOINT_DIR = "experiments"


def save_checkpoint(
    model, config, optimizer=None, epoch=None, loss=None, experiment_name="default"
):
    """
    Save model checkpoint with full config and metadata.

    Args:
        model: The model to save
        config: ExperimentConfig object
        optimizer: Optional optimizer state
        epoch: Current training iteration
        loss: Current validation loss
        experiment_name: Name of experiment for folder organization
    """
    # Create experiment directory
    exp_dir = os.path.join(CHECKPOINT_DIR, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Format timestamp: YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_filename = f"checkpoint_{timestamp}.pth"
    checkpoint_path = os.path.join(exp_dir, checkpoint_filename)

    # Build checkpoint with everything needed for reproducibility
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config.to_dict(),
        "epoch": epoch,
        "loss": loss,
        "timestamp": timestamp,
    }

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Also save config as JSON for easy inspection
    config_path = os.path.join(exp_dir, f"config_{timestamp}.json")
    config.save_json(config_path)

    print(f"✅ Saved checkpoint: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(filepath, device="cuda"):
    """
    Load checkpoint with automatic config restoration.

    Args:
        filepath: Path to checkpoint file
        device: Device to load model onto

    Returns:
        model, config, checkpoint_dict
    """
    from config_manager import ExperimentConfig
    from model import GPTLanguageModel

    checkpoint = torch.load(filepath, map_location=device)

    # Load config from checkpoint
    if "config" in checkpoint:
        config = ExperimentConfig.from_dict(checkpoint["config"])
        print(f"✅ Loaded config from checkpoint")
        print(f"   Model: {config.model.n_embd}d, {config.model.n_layer}L")
        print(f"   Git hash: {config.git_hash}")
    else:
        # Backward compatibility: use legacy config for old checkpoints
        print("⚠️  Old checkpoint format detected, using legacy config")
        from config_manager import get_legacy_config

        config = get_legacy_config()

    # Rebuild model with correct architecture
    model = GPTLanguageModel(config.model.vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"✅ Loaded checkpoint {filepath}")
    if "epoch" in checkpoint:
        print(f"   Iteration: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"   Val loss: {checkpoint['loss']:.4f}")

    return model, config, checkpoint


def load_checkpoint_for_training(filepath, model, optimizer, device="cuda"):
    """
    Load checkpoint and restore training state.

    Args:
        filepath: Path to checkpoint
        model: Model instance to load weights into
        optimizer: Optimizer to restore state
        device: Device

    Returns:
        config, start_iteration
    """
    from config_manager import ExperimentConfig

    checkpoint = torch.load(filepath, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load config
    if "config" in checkpoint:
        config = ExperimentConfig.from_dict(checkpoint["config"])
    else:
        from config_manager import get_legacy_config

        config = get_legacy_config()

    start_iter = checkpoint.get("epoch", 0)

    print(f"✅ Loaded checkpoint for training: {filepath}")
    print(f"   Resuming from iteration: {start_iter}")

    return config, start_iter


# Legacy function for backward compatibility
def save_model_with_timestamp(
    model, optimizer=None, epoch=None, loss=None, prefix="model"
):
    """Legacy save function - kept for backward compatibility"""
    os.makedirs("checkpoints", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join("checkpoints", f"{prefix}_{timestamp}.pth")

    checkpoint = {"model_state_dict": model.state_dict()}

    if optimizer:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if loss is not None:
        checkpoint["loss"] = loss

    torch.save(checkpoint, filename)
    print(f"✅ Saved checkpoint: {filename}")
    return filename


def load_model_checkpoint(
    filename, model_class, vocab_size, device="cuda", optimizer=None
):
    """Legacy load function - kept for backward compatibility"""
    checkpoint = torch.load(filename, map_location=device)

    model = model_class(vocab_size).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"✅ Loaded checkpoint {filename}")
    if "epoch" in checkpoint:
        print(f"  epoch: {checkpoint['epoch']}")
    if "loss" in checkpoint:
        print(f"  loss: {checkpoint['loss']:.4f}")

    return model, optimizer, checkpoint
