"""FastAPI application for pneumonia detection."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from api.routes import router, set_predictor
from pneumonia.utils.logging import setup_logging

logger = setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    config_path = Path("configs/inference_config.yaml")
    if config_path.exists():
        from pneumonia.inference.predictor import Predictor
        from pneumonia.model.classifier import load_model
        from pneumonia.utils.config import load_config

        config = load_config(config_path)
        model = load_model(
            config.model,
            checkpoint_path="checkpoints/best_model.pth",
            device="cpu",
        )
        predictor = Predictor(
            model=model,
            image_size=config.data.image_size,
            device="cpu",
        )
        set_predictor(predictor)
        logger.info("Model loaded and predictor ready")
    else:
        logger.warning(f"Config not found at {config_path} — running without model")

    yield

    logger.info("Shutting down")


app = FastAPI(
    title="Pneumonia Detection API",
    description="Classify chest X-rays as Normal or Pneumonia with Grad-CAM explanations",
    version="0.1.0",
    lifespan=lifespan,
)

# Serve Grad-CAM overlay images (only mount if dir exists)
static_dir = Path("static")
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles

    app.mount("/static", StaticFiles(directory="static"), name="static")

# Register routes
app.include_router(router, prefix="/api/v1")


def main():
    """Run the server directly."""
    import uvicorn

    # Ensure static dir exists for production
    static_dir.mkdir(exist_ok=True)
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
