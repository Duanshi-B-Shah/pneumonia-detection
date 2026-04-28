"""Training callbacks: early stopping and model checkpointing."""
from __future__ import annotations

from pathlib import Path

import torch

from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs with no improvement after which training stops.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' for loss-like metrics, 'max' for accuracy/recall.
    """

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        mode: str = "max",
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: float | None = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop.

        Args:
            score: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info(
                    f"Early stopping triggered after {self.patience} epochs "
                    f"without improvement. Best: {self.best_score:.4f}"
                )
                return True

        return False


class ModelCheckpoint:
    """Save model checkpoints when a monitored metric improves.

    Args:
        checkpoint_dir: Directory to save checkpoints.
        monitor: Metric name being tracked.
        mode: 'min' or 'max'.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_recall",
        mode: str = "max",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_score: float | None = None

    def __call__(
        self,
        score: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: dict | None = None,
    ) -> bool:
        """Save checkpoint if score improved.

        Returns:
            True if a new best checkpoint was saved.
        """
        if self.best_score is None:
            is_best = True
        elif self.mode == "max":
            is_best = score > self.best_score
        else:
            is_best = score < self.best_score

        if is_best:
            self.best_score = score
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                f"best_{self.monitor}": score,
            }
            if metrics:
                checkpoint["metrics"] = metrics

            path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, path)
            logger.info(
                f"Checkpoint saved: {self.monitor}={score:.4f} (epoch {epoch}) → {path}"
            )
            return True

        return False
