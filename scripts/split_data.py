"""CLI script for stratified dataset re-splitting."""
from __future__ import annotations

import argparse

from pneumonia.data.split import split_dataset
from pneumonia.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-split chest X-ray dataset with stratification"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to original dataset (with train/val/test subdirs)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for new splits",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    setup_logging()

    stats = split_dataset(
        source_dir=args.input,
        output_dir=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\n═══ Split Complete ═══")
    for split_name, class_counts in stats.items():
        total = sum(class_counts.values())
        print(f"  {split_name:>5s}: {total:>5d} images — {class_counts}")


if __name__ == "__main__":
    main()
