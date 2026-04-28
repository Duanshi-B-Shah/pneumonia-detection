"""API route definitions."""
from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.schemas import BatchPredictionResponse, HealthResponse, PredictionResponse

router = APIRouter()

# Predictor is injected at app startup (see app.py)
predictor = None
GRADCAM_DIR = Path("static/gradcam")


def set_predictor(pred):
    """Set the global predictor instance (called from app startup)."""
    global predictor
    predictor = pred


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=predictor is not None,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Predict pneumonia from a chest X-ray image.

    Accepts JPEG or PNG images. Returns prediction label, confidence,
    and a Grad-CAM heatmap overlay URL.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Use JPEG or PNG.",
        )

    # Save uploaded file temporarily
    suffix = Path(file.filename or "image.jpg").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Generate unique filename for Grad-CAM output
        gradcam_filename = f"{uuid.uuid4().hex}.png"
        gradcam_path = GRADCAM_DIR / gradcam_filename
        GRADCAM_DIR.mkdir(parents=True, exist_ok=True)

        result = predictor.predict(
            tmp_path,
            gradcam_output_path=str(gradcam_path),
        )

        return PredictionResponse(
            label=result["label"],
            confidence=result["confidence"],
            probability_pneumonia=result["probability_pneumonia"],
            class_index=result["class_index"],
            latency_ms=result["latency_ms"],
            gradcam_url=f"/static/gradcam/{gradcam_filename}",
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)):
    """Batch prediction for multiple images."""
    import time

    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    results = []

    for file in files:
        suffix = Path(file.filename or "image.jpg").suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = predictor.predict(tmp_path)
            results.append({
                "filename": file.filename or "unknown",
                "label": result["label"],
                "confidence": result["confidence"],
                "probability_pneumonia": result["probability_pneumonia"],
            })
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    latency_ms = (time.perf_counter() - start) * 1000

    return BatchPredictionResponse(
        results=results,
        total=len(results),
        latency_ms=round(latency_ms, 1),
    )
