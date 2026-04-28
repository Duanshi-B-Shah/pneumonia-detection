"""Stratified dataset re-splitting utility.

The original Kaggle dataset has a tiny validation set (16 images).
This module merges all images and creates a proper stratified split.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


def collect_all_images(source_dir: Path) -> tuple[list[Path], list[str]]:
    """Collect all images from train/val/test subdirs.

    Args:
        source_dir: Root directory with train/val/test subdirs.

    Returns:
        Tuple of (image_paths, labels).
    """
    paths = []
    labels = []

    for split_dir in source_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label = class_dir.name  # NORMAL or PNEUMONIA
            for img_path in class_dir.glob("*"):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    paths.append(img_path)
                    labels.append(label)

    logger.info(f"Collected {len(paths)} total images from {source_dir}")
    return paths, labels


def split_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, int]:
    """Re-split the dataset with stratification.

    Args:
        source_dir: Original dataset root (with train/val/test subdirs).
        output_dir: Output directory for the new split.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation. Test = 1 - train - val.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with counts per split per class.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    test_ratio = 1.0 - train_ratio - val_ratio

    assert test_ratio > 0, "train_ratio + val_ratio must be < 1.0"

    paths, labels = collect_all_images(source_dir)

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(val_ratio + test_ratio),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_val = val_ratio / (val_ratio + test_ratio)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1.0 - relative_val),
        stratify=temp_labels,
        random_state=seed,
    )

    # Copy files into new structure
    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    stats = {}
    for split_name, (split_paths, split_labels) in splits.items():
        stats[split_name] = {}
        for img_path, label in zip(split_paths, split_labels):
            dest_dir = output_dir / split_name / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / img_path.name

            # Handle filename collisions
            if dest_path.exists():
                stem = img_path.stem
                suffix = img_path.suffix
                counter = 1
                while dest_path.exists():
                    dest_path = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1

            shutil.copy2(img_path, dest_path)
            stats[split_name][label] = stats[split_name].get(label, 0) + 1

    for split_name, class_counts in stats.items():
        total = sum(class_counts.values())
        logger.info(f"[{split_name}] {total} images: {class_counts}")

    return stats
