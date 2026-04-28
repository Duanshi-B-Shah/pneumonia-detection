"""Inference pipeline for single and batch predictions."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from pneumonia.data.augmentation import get_inference_transforms
from pneumonia.model.classifier import PneumoniaClassifier, load_model
from pneumonia.model.gradcam import GradCAMExplainer
from pneumonia.utils.logging import get_logger

logger = get_logger(__name__)

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


class Predictor:
    """End-to-end inference pipeline with optional Grad-CAM explanations.

    Usage:
        predictor = Predictor(model, image_size=224, device="cpu")
        result = predictor.predict("xray.jpg")
        # result = {"label": "PNEUMONIA", "confidence": 0.94, "latency_ms": 123}
    """

    def __init__(
        self,
        model: PneumoniaClassifier,
        image_size: int = 224,
        device: str = "cpu",
        threshold: float = 0.5,
        enable_gradcam: bool = True,
    ) -> None:
        self.model = model
        self.model.eval()
        self.device = device
        self.threshold = threshold
        self.image_size = image_size
        self.transform = get_inference_transforms(image_size)

        self.explainer: GradCAMExplainer | None = None
        if enable_gradcam:
            self.explainer = GradCAMExplainer(model, image_size, device)

    def _preprocess(self, image_path: str | Path) -> torch.Tensor:
        """Load and preprocess a single image."""
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        return tensor

    def predict(
        self,
        image_path: str | Path,
        gradcam_output_path: str | Path | None = None,
    ) -> dict:
        """Run inference on a single image.

        Args:
            image_path: Path to the chest X-ray image.
            gradcam_output_path: If set, save Grad-CAM overlay to this path.

        Returns:
            Dictionary with keys: label, confidence, class_index, latency_ms,
            and optionally gradcam_path.
        """
        start = time.perf_counter()

        input_tensor = self._preprocess(image_path)

        with torch.no_grad():
            logits = self.model(input_tensor)
            prob = torch.sigmoid(logits).item()

            class_idx = 1 if prob >= self.threshold else 0
        label = CLASS_NAMES[class_idx]
        confidence = prob if class_idx == 1 else 1.0 - prob

        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "label": label,
            "confidence": round(confidence, 4),
            "probability_pneumonia": round(prob, 4),
            "class_index": class_idx,
            "latency_ms": round(latency_ms, 1),
        }

        # Generate Grad-CAM if requested
        if gradcam_output_path and self.explainer:
            self.explainer.explain_and_save(image_path, gradcam_output_path)
            result["gradcam_path"] = str(gradcam_output_path)

        logger.info(
            f"Prediction: {label} ({confidence:.2%}) | {latency_ms:.1f}ms | {image_path}"
        )

        return result

    def predict_batch(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[dict]:
        """Run inference on all images in a directory.

        Args:
            image_dir: Directory containing images.
            output_dir: Directory to save Grad-CAM overlays (optional).

        Returns:
            List of prediction result dictionaries.
        """
        image_dir = Path(image_dir)
        results = []

        extensions = {".jpg", ".jpeg", ".png"}
        image_files = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        logger.info(f"Batch inference: {len(image_files)} images from {image_dir}")

        for img_path in image_files:
            gradcam_path = None
            if output_dir:
                gradcam_path = Path(output_dir) / f"{img_path.stem}_gradcam.png"

            result = self.predict(img_path, gradcam_output_path=gradcam_path)
            result["file"] = str(img_path.name)
            results.append(result)

        return results


def main() -> None:
    """CLI entry point for prediction."""
    parser = argparse.ArgumentParser(description="Run pneumonia inference")
    parser.add_argument("image", help="Path to image or directory")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", default="configs/inference_config.yaml")
    parser.add_argument("--gradcam-dir", default=None, help="Save Grad-CAM overlays")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from pneumonia.utils.config import load_config
    from pneumonia.utils.logging import setup_logging

    setup_logging()

    config = load_config(args.config)
    model = load_model(config.model, args.checkpoint, device=args.device)
    predictor = Predictor(model, config.data.image_size, args.device)

    image_path = Path(args.image)
    if image_path.is_dir():
        results = predictor.predict_batch(image_path, output_dir=args.gradcam_dir)
        for r in results:
            print(f"  {r['file']:>40s} → {r['label']} ({r['confidence']:.2%})")
    else:
        result = predictor.predict(image_path, gradcam_output_path=args.gradcam_dir)
        print(result)


if __name__ == "__main__":
    main()
