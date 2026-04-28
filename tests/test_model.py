"""Tests for the PneumoniaClassifier model."""

from __future__ import annotations

import torch

from pneumonia.model.classifier import PneumoniaClassifier, build_model
from pneumonia.utils.config import ModelConfig


class TestPneumoniaClassifier:
    """Test suite for the classifier model."""

    def test_forward_pass_shape(self, model: PneumoniaClassifier, sample_batch: torch.Tensor):
        """Model output should have shape (B, 1)."""
        output = model(sample_batch)
        assert output.shape == (4, 1), f"Expected (4, 1), got {output.shape}"

    def test_forward_pass_single(self, model: PneumoniaClassifier):
        """Single image forward pass."""
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 1)

    def test_sigmoid_output_range(self, model: PneumoniaClassifier, sample_batch: torch.Tensor):
        """Sigmoid of logits should be in [0, 1]."""
        logits = model(sample_batch)
        probs = torch.sigmoid(logits)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_freeze_backbone(self, model: PneumoniaClassifier):
        """After freezing, backbone params should not require grad."""
        model.freeze_backbone()
        for param in model.backbone.parameters():
            assert not param.requires_grad
        # Head should still be trainable
        for param in model.head.parameters():
            assert param.requires_grad

    def test_unfreeze_backbone(self, model: PneumoniaClassifier):
        """After unfreezing, all params should require grad."""
        model.freeze_backbone()
        model.unfreeze_backbone()
        for param in model.backbone.parameters():
            assert param.requires_grad

    def test_gradcam_target_layer(self, model: PneumoniaClassifier):
        """Should return a valid nn.Module for Grad-CAM."""
        layer = model.get_gradcam_target_layer()
        assert isinstance(layer, torch.nn.Module)

    def test_build_model(self, model_config: ModelConfig):
        """build_model should return a model on the specified device."""
        model = build_model(model_config, device="cpu")
        assert isinstance(model, PneumoniaClassifier)
        # Check all params are on CPU
        for param in model.parameters():
            assert param.device.type == "cpu"

    def test_model_param_count(self, model: PneumoniaClassifier):
        """EfficientNet-B0 should have ~5.3M params."""
        total = sum(p.numel() for p in model.parameters())
        assert total > 4_000_000, "Model seems too small"
        assert total < 10_000_000, "Model seems too large for B0"
