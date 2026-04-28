"""Model evaluation — metrics computation, confusion matrix, ROC curve."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from pneumonia.model.classifier import PneumoniaClassifier
from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


class Evaluator:
    """Compute and visualize classification metrics."""

    CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

    def __init__(self, model: PneumoniaClassifier, device: str = "cpu") -> None:
        self.model = model
        self.device = device
        self.model.eval()

    @torch.no_grad()
    def predict(self, dataloader: DataLoader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference and collect predictions.

        Returns:
            Tuple of (all_labels, all_probs, all_preds).
        """
        all_labels = []
        all_probs = []

        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs >= 0.5).astype(int)

        return all_labels, all_probs, all_preds

    def compute_metrics(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        preds: np.ndarray,
    ) -> dict[str, float]:
        """Compute all classification metrics.

        Returns:
            Dictionary of metric name → value.
        """
        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "roc_auc": roc_auc_score(labels, probs),
        }

        logger.info("─── Evaluation Results ───")
        for name, value in metrics.items():
            logger.info(f"  {name:>12s}: {value:.4f}")

        logger.info("\nClassification Report:")
        logger.info(
            "\n"
            + classification_report(
                labels, preds, target_names=self.CLASS_NAMES, zero_division=0
            )
        )

        return metrics

    def plot_confusion_matrix(
        self,
        labels: np.ndarray,
        preds: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot and optionally save the confusion matrix."""
        cm = confusion_matrix(labels, preds)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, transparent=True)
            logger.info(f"Confusion matrix saved to {save_path}")

        plt.close(fig)

    def plot_roc_curve(
        self,
        labels: np.ndarray,
        probs: np.ndarray,
        save_path: Optional[str | Path] = None,
    ) -> None:
        """Plot and optionally save the ROC curve."""
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.3f})", color="#1E88E5")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, transparent=True)
            logger.info(f"ROC curve saved to {save_path}")

        plt.close(fig)

    def evaluate(
        self,
        dataloader: DataLoader,
        output_dir: Optional[str | Path] = None,
    ) -> dict[str, float]:
        """Full evaluation pipeline: predict → metrics → plots.

        Args:
            dataloader: Test DataLoader.
            output_dir: Directory to save plots (optional).

        Returns:
            Metrics dictionary.
        """
        labels, probs, preds = self.predict(dataloader)
        metrics = self.compute_metrics(labels, probs, preds)

        if output_dir:
            output_dir = Path(output_dir)
            self.plot_confusion_matrix(labels, preds, output_dir / "confusion_matrix.png")
            self.plot_roc_curve(labels, probs, output_dir / "roc_curve.png")

        return metrics


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate pneumonia detection model")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint")
    parser.add_argument("--output-dir", default="evaluation", help="Output directory")
    args = parser.parse_args()

    from pneumonia.data.dataset import create_dataloaders
    from pneumonia.model.classifier import load_model
    from pneumonia.utils.config import load_config

    config = load_config(args.config)
    checkpoint = args.checkpoint or f"{config.checkpoint.dir}/best_model.pth"

    model = load_model(config.model, checkpoint, device=config.device)
    loaders = create_dataloaders(config)

    evaluator = Evaluator(model, device=config.device)
    metrics = evaluator.evaluate(loaders["test"], output_dir=args.output_dir)

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
