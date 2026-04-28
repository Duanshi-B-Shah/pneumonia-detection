"""Tests for the FastAPI endpoints."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from api.app import app

    return TestClient(app)


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Create a sample image as bytes for upload testing."""
    img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


class TestHealthEndpoint:
    """Test the health check endpoint."""

    def test_health_returns_200(self, client):
        """Health endpoint should return 200 with status."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data

    def test_health_reports_no_model_in_test(self, client):
        """In test mode (no checkpoint), model_loaded should be False."""
        response = client.get("/api/v1/health")
        data = response.json()
        # During tests, no model is loaded
        assert data["model_loaded"] is False


class TestPredictEndpoint:
    """Test the prediction endpoint."""

    def test_predict_returns_503_without_model(self, client, sample_image_bytes):
        """Should return 503 when no model is loaded."""
        response = client.post(
            "/api/v1/predict",
            files={"file": ("test.png", sample_image_bytes, "image/png")},
        )
        # Without a loaded model, the API returns 503 Service Unavailable
        assert response.status_code == 503

    def test_predict_requires_file(self, client):
        """Should return 422 when no file is provided."""
        response = client.post("/api/v1/predict")
        assert response.status_code == 422

    def test_batch_returns_503_without_model(self, client, sample_image_bytes):
        """Batch endpoint should return 503 without model."""
        response = client.post(
            "/api/v1/batch",
            files=[("files", ("test.png", sample_image_bytes, "image/png"))],
        )
        assert response.status_code == 503


class TestAPISchema:
    """Test API response schemas."""

    def test_health_response_has_version(self, client):
        """Health response should include version."""
        response = client.get("/api/v1/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"
