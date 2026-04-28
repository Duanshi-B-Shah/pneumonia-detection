"""Shared test fixtures for the pneumonia detection test suite."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from pneumonia.model.classifier import PneumoniaClassifier
from pneumonia.utils.config import Config, ModelConfig


@pytest.fixture
def model_config() -> ModelConfig:
    """Minimal model config for testing."""
    return ModelConfig(
        backbone="efficientnet_b0",
        pretrained=False,  # Don't download weights in tests
        num_classes=1,
        dropout=0.3,
    )


@pytest.fixture
def config() -> Config:
    """Full config with pretrained=False for testing."""
    cfg = Config()
    cfg.model.pretrained = False
    cfg.device = "cpu"
    cfg.data.pin_memory = False
    return cfg


@pytest.fixture
def model(model_config: ModelConfig) -> PneumoniaClassifier:
    """Untrained model instance for testing."""
    return PneumoniaClassifier(model_config).eval()


@pytest.fixture
def sample_image_path() -> str:
    """Create a temporary sample image for testing."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img.save(f.name)
        return f.name


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """Random batch tensor for model forward pass tests."""
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a temporary dataset directory structure for testing."""
    for split in ["train", "val", "test"]:
        for cls in ["NORMAL", "PNEUMONIA"]:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            # Create a few sample images per class
            for i in range(5):
                img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                img.save(cls_dir / f"sample_{i}.png")
    return tmp_path
