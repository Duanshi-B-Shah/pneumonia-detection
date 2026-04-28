"""Export trained model to ONNX format for optimized inference."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from pneumonia.model.classifier import load_model
from pneumonia.utils.config import load_config
from pneumonia.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    config_path: str = "configs/inference_config.yaml",
    image_size: int = 224,
) -> Path:
    """Export PyTorch model to ONNX.

    Args:
        checkpoint_path: Path to the .pth checkpoint.
        output_path: Path for the output .onnx file.
        config_path: Path to config YAML.
        image_size: Input image size.

    Returns:
        Path to the exported ONNX file.
    """
    config = load_config(config_path)
    model = load_model(config.model, checkpoint_path, device="cpu")
    model.eval()

    # Dummy input matching expected shape
    dummy_input = torch.randn(1, 3, image_size, image_size)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
        opset_version=17,
    )

    logger.info(f"ONNX model exported to {output_path}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Verify
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        logger.info("ONNX model verification: PASSED ✓")
    except ImportError:
        logger.warning("onnx package not installed — skipping verification")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--config", default="configs/inference_config.yaml")
    args = parser.parse_args()

    setup_logging()
    export_to_onnx(args.checkpoint, args.output, args.config)


if __name__ == "__main__":
    main()
