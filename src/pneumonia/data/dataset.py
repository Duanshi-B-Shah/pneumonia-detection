"""Custom dataset and DataLoader factory for chest X-ray images."""
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

from pneumonia.data.augmentation import get_train_transforms, get_val_transforms
from pneumonia.utils.config import Config
from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


class ChestXrayDataset:
    """Wrapper around ImageFolder with proper transforms and class weighting."""

    # Class indices: ImageFolder sorts alphabetically → NORMAL=0, PNEUMONIA=1
    CLASS_NAMES: ClassVar[list[str]] = ["NORMAL", "PNEUMONIA"]

    def __init__(self, root: str | Path, split: str, config: Config) -> None:
        """Initialize dataset for a given split.

        Args:
            root: Root data directory containing train/val/test subdirs.
            split: One of 'train', 'val', 'test'.
            config: Project configuration.
        """
        self.root = Path(root) / split
        self.split = split
        self.config = config

        if not self.root.exists():
            raise FileNotFoundError(f"Data split directory not found: {self.root}")

        # Select transforms based on split
        if split == "train":
            transform = get_train_transforms(
                config.data.image_size, config.augmentation
            )
        else:
            transform = get_val_transforms(config.data.image_size)

        self.dataset = ImageFolder(root=str(self.root), transform=transform)
        self._log_stats()

    def _log_stats(self) -> None:
        """Log class distribution for this split."""
        targets = self.dataset.targets
        class_counts = {}
        for cls_idx, cls_name in enumerate(self.CLASS_NAMES):
            count = targets.count(cls_idx)
            class_counts[cls_name] = count
        logger.info(
            f"[{self.split}] Loaded {len(targets)} images: {class_counts}"
        )

    @property
    def targets(self) -> list[int]:
        return self.dataset.targets

    @property
    def num_samples(self) -> int:
        return len(self.dataset)

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for loss balancing."""
        targets = torch.tensor(self.targets)
        class_counts = torch.bincount(targets).float()
        weights = 1.0 / class_counts
        weights = weights / weights.sum()
        return weights

    def get_sampler(self) -> WeightedRandomSampler:
        """Build a weighted random sampler for balanced training batches."""
        targets = torch.tensor(self.targets)
        class_counts = torch.bincount(targets).float()
        sample_weights = 1.0 / class_counts[targets]
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def __len__(self) -> int:
        return len(self.dataset)


def create_dataloaders(config: Config) -> dict[str, DataLoader]:
    """Create train, val, and test DataLoaders.

    Args:
        config: Project configuration.

    Returns:
        Dictionary mapping split names to DataLoader instances.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        dataset = ChestXrayDataset(
            root=config.data.root,
            split=split,
            config=config,
        )

        shuffle = False
        sampler = None

        if split == "train":
            sampler = dataset.get_sampler()
        else:
            shuffle = False

        loaders[split] = DataLoader(
            dataset.dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            drop_last=(split == "train"),
        )

    return loaders
