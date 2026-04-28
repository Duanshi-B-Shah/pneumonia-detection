"""Configuration loader using Pydantic models and YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator

# ── Sub-configs ──────────────────────────────────────────────────────


class DataConfig(BaseModel):
    root: str = "data/processed"
    image_size: int = 224
    num_workers: int = 4
    pin_memory: bool = True


class AugmentationConfig(BaseModel):
    horizontal_flip: bool = True
    rotation_degrees: int = 15
    brightness: float = 0.2
    contrast: float = 0.2
    affine_translate: list[float] = [0.05, 0.05]
    affine_scale: list[float] = [0.9, 1.1]


class ModelConfig(BaseModel):
    backbone: str = "efficientnet_b0"
    pretrained: bool = True
    num_classes: int = 1
    dropout: float = 0.3


class TrainingConfig(BaseModel):
    batch_size: int = 32
    epochs_frozen: int = 5
    epochs_unfrozen: int = 15
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine_warm_restarts"
    scheduler_t0: int = 5
    scheduler_tmult: int = 2


class LossConfig(BaseModel):
    type: str = "bce_weighted"
    pos_weight: float = 0.35


class EarlyStoppingConfig(BaseModel):
    patience: int = 5
    min_delta: float = 0.001
    monitor: str = "val_recall"


class CheckpointConfig(BaseModel):
    dir: str = "checkpoints"
    save_best: bool = True
    monitor: str = "val_recall"
    mode: str = "max"


class MLflowConfig(BaseModel):
    experiment_name: str = "pneumonia-detection"
    tracking_uri: str = "mlruns"
    log_model: bool = True


# ── Root config ──────────────────────────────────────────────────────


class Config(BaseModel):
    data: DataConfig = DataConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    loss: LossConfig = LossConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    mlflow: MLflowConfig = MLflowConfig()
    device: str = "auto"
    seed: int = 42

    @field_validator("device")
    @classmethod
    def resolve_device(cls, v: str) -> str:
        if v == "auto":
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return v


def load_config(path: str | Path) -> Config:
    """Load a YAML config file and return a validated Config object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)
