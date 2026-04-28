# ─── Stage 1: Builder ─────────────────────────────────────────────────
FROM python:3.9-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[api]"


# ─── Stage 2: Runtime ─────────────────────────────────────────────────
FROM python:3.9-slim AS runtime

WORKDIR /app

# System libs needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ src/
COPY api/ api/
COPY configs/ configs/
# Checkpoints are optional (not present in CI, mounted at runtime)
RUN mkdir -p checkpoints

# Create static directory for Grad-CAM outputs
RUN mkdir -p static/gradcam

# Non-root user
RUN useradd -m -r appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health')"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
