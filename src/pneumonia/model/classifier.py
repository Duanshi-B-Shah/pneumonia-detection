"""EfficientNet-based binary classifier for chest X-ray images."""
from __future__ import annotations

import timm
import torch
import torch.nn as nn

from pneumonia.utils.config import ModelConfig
from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


class PneumoniaClassifier(nn.Module):
    """Binary classifier using an EfficientNet backbone with a custom head.

    Architecture:
        - Backbone: EfficientNet-B0 (pretrained on ImageNet)
        - Head: Global Average Pooling → Dropout → Linear(1280, 1) → Sigmoid
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Load backbone (no built-in classification head)
        self.backbone = timm.create_model(
            config.backbone,
            pretrained=config.pretrained,
            num_classes=0,  # removes the default head
        )

        # Get feature dimension from the backbone
        self.feature_dim = self.backbone.num_features
        logger.info(f"Backbone: {config.backbone}, features: {self.feature_dim}")

        # Custom classification head
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout),
            nn.Linear(self.feature_dim, config.num_classes),
        )

        self._initialize_head()

    def _initialize_head(self) -> None:
        """Xavier initialization for the custom head."""
        for module in self.head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, 1) — apply sigmoid for probabilities.
        """
        features = self.backbone(x)  # (B, feature_dim)
        logits = self.head(features)  # (B, 1)
        return logits

    def freeze_backbone(self) -> None:
        """Freeze all backbone parameters (for initial training phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen — training head only")

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters (for fine-tuning phase)."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — full fine-tuning")

    def get_gradcam_target_layer(self) -> nn.Module:
        """Return the target layer for Grad-CAM visualization.

        For EfficientNet, this is the last block in the feature extractor.
        """
        # timm EfficientNet uses .blocks as the feature extractor
        if hasattr(self.backbone, "blocks"):
            return self.backbone.blocks[-1]
        # Fallback for other architectures
        if hasattr(self.backbone, "features"):
            return self.backbone.features[-1]
        raise AttributeError(
            f"Cannot find target layer for Grad-CAM in {self.config.backbone}"
        )


def build_model(config: ModelConfig, device: str = "cpu") -> PneumoniaClassifier:
    """Build and return the classifier model.

    Args:
        config: Model configuration.
        device: Target device.

    Returns:
        Initialized PneumoniaClassifier on the target device.
    """
    model = PneumoniaClassifier(config)
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model built: {total_params:,} total params, {trainable:,} trainable"
    )
    return model


def load_model(
    config: ModelConfig,
    checkpoint_path: str,
    device: str = "cpu",
) -> PneumoniaClassifier:
    """Load a model from a checkpoint.

    Args:
        config: Model configuration.
        checkpoint_path: Path to the .pth checkpoint.
        device: Target device.

    Returns:
        Model with loaded weights.
    """
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle both raw state_dict and wrapped checkpoint
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info(f"Model loaded from {checkpoint_path}")
    return model
