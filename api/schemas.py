"""Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Response schema for single image prediction."""

    label: str = Field(..., description="Predicted class: NORMAL or PNEUMONIA")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probability_pneumonia: float = Field(..., description="Raw probability of pneumonia")
    class_index: int = Field(..., description="Class index: 0=NORMAL, 1=PNEUMONIA")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    gradcam_url: str | None = Field(None, description="URL to Grad-CAM overlay image")


class BatchPredictionItem(BaseModel):
    """Single item in a batch prediction response."""

    filename: str
    label: str
    confidence: float
    probability_pneumonia: float


class BatchPredictionResponse(BaseModel):
    """Response schema for batch prediction."""

    results: list[BatchPredictionItem]
    total: int
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    model_loaded: bool = True
    version: str = "0.1.0"
