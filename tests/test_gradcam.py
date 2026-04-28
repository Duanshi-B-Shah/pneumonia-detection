"""Tests for Grad-CAM visualization."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pneumonia.model.classifier import PneumoniaClassifier
from pneumonia.model.gradcam import GradCAMExplainer


class TestGradCAMExplainer:
    """Test Grad-CAM heatmap generation."""

    def test_explain_returns_heatmap_and_overlay(
        self, model: PneumoniaClassifier, sample_image_path: str
    ):
        """explain() should return a heatmap and overlay array."""
        explainer = GradCAMExplainer(model, image_size=224, device="cpu")
        heatmap, overlay = explainer.explain(sample_image_path)

        # Heatmap should be 2D float in [0, 1]
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

        # Overlay should be RGB uint8
        assert overlay.ndim == 3
        assert overlay.shape[2] == 3
        assert overlay.dtype == np.uint8

    def test_explain_and_save(
        self, model: PneumoniaClassifier, sample_image_path: str, tmp_path: Path
    ):
        """explain_and_save() should create an image file."""
        explainer = GradCAMExplainer(model, image_size=224, device="cpu")
        output = tmp_path / "gradcam_output.png"
        result_path = explainer.explain_and_save(sample_image_path, output)

        assert result_path.exists()
        assert result_path.stat().st_size > 0

    def test_heatmap_dimensions_match_image_size(
        self, model: PneumoniaClassifier, sample_image_path: str
    ):
        """Heatmap spatial dims should match the model's input size."""
        image_size = 224
        explainer = GradCAMExplainer(model, image_size=image_size, device="cpu")
        heatmap, _ = explainer.explain(sample_image_path)

        # pytorch-grad-cam returns heatmap at the feature map resolution
        # then upsamples to input size
        assert heatmap.shape[0] == image_size
        assert heatmap.shape[1] == image_size
