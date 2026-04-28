"""Two-phase training loop with MLflow experiment tracking."""

from __future__ import annotations

import argparse
import random

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from tqdm import tqdm

from pneumonia.data.dataset import create_dataloaders
from pneumonia.model.classifier import build_model
from pneumonia.training.callbacks import EarlyStopping, ModelCheckpoint
from pneumonia.training.evaluator import Evaluator
from pneumonia.utils.config import Config, load_config
from pneumonia.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    """Orchestrates the two-phase training pipeline.

    Phase 1: Backbone frozen — train classification head only.
    Phase 2: Full fine-tuning — all parameters trainable.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = config.device
        set_seed(config.seed)

        # Data
        self.loaders = create_dataloaders(config)

        # Model
        self.model = build_model(config.model, device=self.device)

        # Loss (weighted BCE)
        pos_weight = torch.tensor([1.0 / config.loss.pos_weight]).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # Callbacks
        self.early_stopping = EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            mode="max",
        )
        self.checkpoint = ModelCheckpoint(
            checkpoint_dir=config.checkpoint.dir,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
        )

        # Evaluator
        self.evaluator = Evaluator(self.model, device=self.device)

    def _build_optimizer(self) -> AdamW:
        """Build optimizer for trainable parameters only."""
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        return AdamW(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _build_scheduler(self, optimizer: AdamW) -> CosineAnnealingWarmRestarts:
        """Build learning rate scheduler."""
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.training.scheduler_t0,
            T_mult=self.config.training.scheduler_tmult,
        )

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: AdamW,
        scheduler: CosineAnnealingWarmRestarts,
    ) -> float:
        """Run one training epoch.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, labels in tqdm(dataloader, desc="Training", leave=False):
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)

            optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def _validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation and return metrics.

        Returns:
            Dictionary with val_loss, val_accuracy, val_recall, val_precision, val_f1.
        """
        self.model.eval()
        labels, probs, preds = self.evaluator.predict(dataloader)
        metrics = self.evaluator.compute_metrics(labels, probs, preds)

        # Compute val loss
        total_loss = 0.0
        num_batches = 0
        for images, batch_labels in dataloader:
            images = images.to(self.device)
            batch_labels = batch_labels.float().unsqueeze(1).to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, batch_labels)
            total_loss += loss.item()
            num_batches += 1

        val_loss = total_loss / max(num_batches, 1)

        return {
            "val_loss": val_loss,
            "val_accuracy": metrics["accuracy"],
            "val_recall": metrics["recall"],
            "val_precision": metrics["precision"],
            "val_f1": metrics["f1"],
            "val_roc_auc": metrics["roc_auc"],
        }

    def _train_phase(
        self,
        phase_name: str,
        num_epochs: int,
        start_epoch: int = 0,
    ) -> int:
        """Execute a training phase (frozen or unfrozen).

        Args:
            phase_name: Label for logging ('frozen' or 'finetune').
            num_epochs: Number of epochs for this phase.
            start_epoch: Global epoch counter offset.

        Returns:
            Next global epoch number.
        """
        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Phase: {phase_name} | Epochs: {num_epochs}")
        logger.info(f"{'=' * 60}")

        for epoch_i in range(num_epochs):
            epoch = start_epoch + epoch_i
            logger.info(f"\nEpoch {epoch + 1}")

            # Train
            train_loss = self._train_epoch(self.loaders["train"], optimizer, scheduler)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Validate
            val_metrics = self._validate(self.loaders["val"])
            for k, v in val_metrics.items():
                mlflow.log_metric(k, v, step=epoch)

            logger.info(
                f"  train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | "
                f"val_acc={val_metrics['val_accuracy']:.4f} | "
                f"val_recall={val_metrics['val_recall']:.4f}"
            )

            # Checkpoint
            monitor_value = val_metrics[f"val_{self.config.checkpoint.monitor.replace('val_', '')}"]
            self.checkpoint(
                score=monitor_value,
                model=self.model,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
            )

            # Early stopping
            if self.early_stopping(monitor_value):
                logger.info("Early stopping triggered.")
                break

        return start_epoch + num_epochs

    def train(self) -> dict[str, float]:
        """Execute full two-phase training.

        Returns:
            Final test set metrics.
        """
        mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)
        mlflow.set_experiment(self.config.mlflow.experiment_name)

        with mlflow.start_run():
            # Log config
            mlflow.log_params(
                {
                    "backbone": self.config.model.backbone,
                    "learning_rate": self.config.training.learning_rate,
                    "batch_size": self.config.training.batch_size,
                    "dropout": self.config.model.dropout,
                    "epochs_frozen": self.config.training.epochs_frozen,
                    "epochs_unfrozen": self.config.training.epochs_unfrozen,
                    "seed": self.config.seed,
                }
            )

            # Phase 1: Frozen backbone
            self.model.freeze_backbone()
            next_epoch = self._train_phase("frozen", self.config.training.epochs_frozen)

            # Reset early stopping for phase 2
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping.patience,
                min_delta=self.config.early_stopping.min_delta,
                mode="max",
            )

            # Phase 2: Full fine-tuning
            self.model.unfreeze_backbone()
            self._train_phase(
                "finetune", self.config.training.epochs_unfrozen, start_epoch=next_epoch
            )

            # Final evaluation on test set
            logger.info("\n" + "=" * 60)
            logger.info("Final evaluation on test set")
            logger.info("=" * 60)

            test_metrics = self.evaluator.evaluate(self.loaders["test"], output_dir="evaluation")
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)

            if self.config.mlflow.log_model:
                mlflow.pytorch.log_model(self.model, "model")

            return test_metrics


def main() -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description="Train pneumonia detection model")
    parser.add_argument(
        "--config",
        default="configs/train_config.yaml",
        help="Path to training config YAML",
    )
    args = parser.parse_args()

    setup_logging()
    config = load_config(args.config)

    trainer = Trainer(config)
    test_metrics = trainer.train()

    logger.info("\n─── Final Test Metrics ───")
    for k, v in test_metrics.items():
        logger.info(f"  {k:>12s}: {v:.4f}")


if __name__ == "__main__":
    main()
