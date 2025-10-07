import torch
import json
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional
from datetime import datetime


@dataclass
class ModelConfig:
    """Model architecture hyperparameters"""

    n_embd: int = 384
    n_head: int = 6
    n_layer: int = 8
    dropout: float = 0.2
    vocab_size: int = 1000  # Set at runtime from tokenizer


@dataclass
class TrainingConfig:
    """Training hyperparameters"""

    batch_size: int = 128
    block_size: int = 256
    max_iters: int = 20000
    eval_interval: int = 500
    learning_rate: float = 3e-4
    eval_iters: int = 200


@dataclass
class DataConfig:
    """Data configuration"""

    data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_file: str = "input.txt"
    tokenizer_vocab_size: int = 1000
    tokenizer_path: str = "bpe_tokenizer.json"


@dataclass
class ExperimentConfig:
    """Full experiment configuration"""

    model: ModelConfig
    training: TrainingConfig
    data: DataConfig
    experiment_name: str = "default"
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    git_hash: Optional[str] = None
    created_at: Optional[str] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()
        if self.git_hash is None:
            self.git_hash = get_git_hash()

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict):
        """Load from dictionary"""
        return cls(
            model=ModelConfig(**config_dict["model"]),
            training=TrainingConfig(**config_dict["training"]),
            data=DataConfig(**config_dict["data"]),
            experiment_name=config_dict.get("experiment_name", "default"),
            seed=config_dict.get("seed", 1337),
            device=config_dict.get("device", "cuda"),
            git_hash=config_dict.get("git_hash"),
            created_at=config_dict.get("created_at"),
        )

    def save_json(self, filepath):
        """Save config to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Saved config to {filepath}")

    @classmethod
    def load_json(cls, filepath):
        """Load config from JSON file"""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_git_hash():
    """Get current git commit hash"""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        return None


def get_default_config(experiment_name="default"):
    """Get default configuration"""
    return ExperimentConfig(
        model=ModelConfig(),
        training=TrainingConfig(),
        data=DataConfig(),
        experiment_name=experiment_name,
    )


# Legacy config.py values for backward compatibility
def get_legacy_config():
    """Returns config matching old config.py file"""
    return ExperimentConfig(
        model=ModelConfig(n_embd=384, n_head=6, n_layer=8, dropout=0.2),
        training=TrainingConfig(
            batch_size=128,
            block_size=256,
            max_iters=20000,
            eval_interval=500,
            learning_rate=3e-4,
            eval_iters=200,
        ),
        data=DataConfig(),
        experiment_name="legacy",
    )
