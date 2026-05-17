# Multi-stage Dockerfile for multilingual sentiment analysis API
# Stage 1 (builder): installs all Python dependencies
# Stage 2 (final): copies source and runs the FastAPI server
#
# Usage:
#   docker build -t multilingual-sentiment .
#   docker run -p 8080:8080 -e MODEL_PATH=/app/models/xlmr-finetuned multilingual-sentiment

# ─── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

# System dependencies for building native Python extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy only requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies into a prefix for easy copying
RUN pip install --upgrade pip setuptools wheel && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ─── Stage 2: Final runtime image ─────────────────────────────────────────────
FROM python:3.11-slim

# Runtime OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY src/ ./src/
COPY configs/ ./configs/

# Create directories for model storage and logs
RUN mkdir -p /app/models /app/logs /tmp/model && \
    chown -R appuser:appuser /app /tmp/model

# Switch to non-root user
USER appuser

# ─── Environment ──────────────────────────────────────────────────────────────
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models/xlmr-finetuned \
    DEVICE=cpu \
    NUM_LABELS=3 \
    MAX_LENGTH=128 \
    FP16=false \
    BATCH_SIZE=32 \
    MAX_BATCH_TEXTS=512 \
    # Reduce tokenizer parallelism warnings in containers
    TOKENIZERS_PARALLELISM=false \
    # HuggingFace cache inside container
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    # Disable progress bars for cleaner logs
    HF_DATASETS_OFFLINE=0

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# ─── Entrypoint ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.serving.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--access-log"]
