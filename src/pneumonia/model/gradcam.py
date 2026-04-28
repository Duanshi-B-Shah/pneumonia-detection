"""Grad-CAM visualization for model interpretability.

Generates heatmap overlays showing which image regions drove the prediction.
Uses the pytorch-grad-cam library for robust, hook-based extraction.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from pneumonia.data.augmentation import get_inference_transforms
from pneumonia.model.classifier import PneumoniaClassifier
from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)


class GradCAMExplainer:
    """Generates Grad-CAM heatmaps for pneumonia predictions.

    Usage:
        explainer = GradCAMExplainer(model, image_size=224)
        heatmap, overlay = explainer.explain(image_path)
    """

    def __init__(
        self,
        model: PneumoniaClassifier,
        image_size: int = 224,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.image_size = image_size
        self.device = device
        self.transform = get_inference_transforms(image_size)

        # Get the target layer for Grad-CAM
        target_layer = model.get_gradcam_target_layer()
        self.cam = GradCAM(model=model, target_layers=[target_layer])

        logger.info("GradCAMExplainer initialized")

    def _load_and_preprocess(self, image_path: str | Path) -> tuple[np.ndarray, torch.Tensor]:
        """Load image and prepare both raw (for overlay) and tensor (for model).

        Returns:
            Tuple of (rgb_image_normalized_0_1, input_tensor).
        """
        img = Image.open(image_path).convert("RGB")
        img_resized = img.resize((self.image_size, self.image_size))

        # Normalized RGB for overlay (0-1 range)
        rgb_img = np.array(img_resized).astype(np.float32) / 255.0

        # Tensor for model input
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        return rgb_img, input_tensor

    def explain(
        self,
        image_path: str | Path,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate Grad-CAM heatmap for a given image.

        Args:
            image_path: Path to the chest X-ray image.
            target_class: Class index to explain (None = predicted class).

        Returns:
            Tuple of (raw_heatmap, overlay_image). Both are numpy arrays.
            - raw_heatmap: (H, W) float array in [0, 1]
            - overlay_image: (H, W, 3) uint8 array — heatmap on top of original
        """
        rgb_img, input_tensor = self._load_and_preprocess(image_path)

        # For binary classification with sigmoid, target_class is not used
        # by pytorch-grad-cam in the same way. We use targets=None for the
        # highest-scoring class.
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=None)
        heatmap = grayscale_cam[0, :]  # (H, W)

        # Create overlay
        overlay = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

        return heatmap, overlay

    def explain_and_save(
        self,
        image_path: str | Path,
        output_path: str | Path,
        target_class: int | None = None,
    ) -> Path:
        """Generate and save the Grad-CAM overlay to disk.

        Args:
            image_path: Path to input image.
            output_path: Path to save the overlay.
            target_class: Class index to explain.

        Returns:
            Path to the saved overlay image.
        """
        _, overlay = self.explain(image_path, target_class)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        overlay_img = Image.fromarray(overlay)
        overlay_img.save(output_path)
        logger.info(f"Grad-CAM overlay saved to {output_path}")

        return output_path
