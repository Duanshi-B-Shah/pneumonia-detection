"""Tests for data loading and augmentation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pneumonia.data.augmentation import get_train_transforms, get_val_transforms
from pneumonia.data.dataset import ChestXrayDataset
from pneumonia.utils.config import AugmentationConfig, Config


class TestAugmentation:
    """Test augmentation pipelines."""

    def test_train_transforms_output_shape(self):
        """Train transforms should produce (3, 224, 224) tensors."""
        import numpy as np
        from PIL import Image

        aug_cfg = AugmentationConfig()
        transform = get_train_transforms(224, aug_cfg)

        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = transform(img)

        assert tensor.shape == (3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_val_transforms_output_shape(self):
        """Val transforms should produce (3, 224, 224) tensors."""
        import numpy as np
        from PIL import Image

        transform = get_val_transforms(224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        tensor = transform(img)

        assert tensor.shape == (3, 224, 224)

    def test_val_transforms_deterministic(self):
        """Val transforms should be deterministic (no randomness)."""
        import numpy as np
        from PIL import Image

        transform = get_val_transforms(224)
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))

        t1 = transform(img)
        t2 = transform(img)
        assert torch.allclose(t1, t2)


class TestChestXrayDataset:
    """Test dataset loading."""

    def test_load_dataset(self, sample_data_dir: Path, config: Config):
        """Dataset should load without errors."""
        config.data.root = str(sample_data_dir)
        dataset = ChestXrayDataset(root=sample_data_dir, split="train", config=config)
        assert len(dataset) == 10  # 5 NORMAL + 5 PNEUMONIA

    def test_class_weights(self, sample_data_dir: Path, config: Config):
        """Class weights should sum to 1."""
        config.data.root = str(sample_data_dir)
        dataset = ChestXrayDataset(root=sample_data_dir, split="train", config=config)
        weights = dataset.get_class_weights()
        assert len(weights) == 2
        assert abs(weights.sum().item() - 1.0) < 1e-5

    def test_sampler(self, sample_data_dir: Path, config: Config):
        """Weighted sampler should have same length as dataset."""
        config.data.root = str(sample_data_dir)
        dataset = ChestXrayDataset(root=sample_data_dir, split="train", config=config)
        sampler = dataset.get_sampler()
        assert len(list(sampler)) == len(dataset)

    def test_missing_split_raises(self, tmp_path: Path, config: Config):
        """Loading a non-existent split should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            ChestXrayDataset(root=tmp_path, split="nonexistent", config=config)
