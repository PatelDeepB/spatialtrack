# Multi-stage production Dockerfile optimized for Hugging Face Spaces (Free CPU tier)
# ------------------------------------------------------------------------------
# Stage 1: Build & Dependencies
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system compilation prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies into isolated prefix
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --prefix=/install ".[app]"

# ------------------------------------------------------------------------------
# Stage 2: Runtime Image
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runner

WORKDIR /app

# Install lightweight runtime system dependencies for OpenCV and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code and configurations
COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY configs/ configs/
COPY scripts/ scripts/

# Install spatialtrack in editable mode and download ONNX INT8 model weights
RUN pip install --no-cache-dir -e . && \
    python scripts/download_model.py

# Configure non-root user for security compliance
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Hugging Face Spaces default port
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl --fail http://localhost:7860/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
            "--server.port=7860", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
