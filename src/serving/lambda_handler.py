"""
AWS Lambda handler for the multilingual sentiment API.

Uses Mangum to adapt the FastAPI ASGI app for Lambda + API Gateway.
Optimizes cold start by downloading the model from S3 to /tmp on first invocation
and reusing the cached /tmp copy on subsequent warm invocations.

Deploy:
  1. Build Docker image with Dockerfile
  2. Push to ECR
  3. Create Lambda function from container image
  4. Set env vars: MODEL_BUCKET, MODEL_KEY, NUM_LABELS, DEVICE
  5. Set LAMBDA_TIMEOUT=30s, MEMORY=3008MB for <150ms p95 latency

Environment variables:
  MODEL_BUCKET   S3 bucket containing the model artifacts
  MODEL_KEY      S3 object key prefix (e.g., models/xlmr-finetuned/)
  MODEL_PATH     Local path to use; defaults to /tmp/model
  NUM_LABELS     Number of sentiment classes (default: 3)
  DEVICE         torch device (default: cpu for Lambda)
  FP16           Enable FP16 (default: false; Lambda CPU doesn't benefit)
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ─── Environment configuration ────────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "/tmp/model")
MODEL_BUCKET = os.environ.get("MODEL_BUCKET", "")
MODEL_KEY = os.environ.get("MODEL_KEY", "models/xlmr-finetuned/")

# ─── Cold start flag ──────────────────────────────────────────────────────────
# Lambda reuses the execution environment across invocations in the same instance.
# We track whether the model has been downloaded in this instance to avoid
# redundant S3 downloads.
_model_downloaded: bool = False
_model_download_start: Optional[float] = None


def download_model_from_s3(
    bucket: str,
    key: str,
    local_path: str,
) -> bool:
    """
    Download model artifacts from S3 to local_path.

    Downloads all objects under the given S3 prefix (key) to local_path.
    Skips download if local_path already contains model files (warm instance).

    Args:
        bucket: S3 bucket name
        key: S3 object key prefix (directory-like, ending with /)
        local_path: Local filesystem path to download into

    Returns:
        True if download occurred, False if cache hit (warm start)
    """
    local = Path(local_path)

    # Check if model is already downloaded (warm Lambda instance)
    config_file = local / "config.json"
    if config_file.exists():
        logger.info(f"Model cache hit at {local_path} (warm Lambda instance)")
        return False

    logger.info(f"Cold start: downloading model from s3://{bucket}/{key} → {local_path}")
    local.mkdir(parents=True, exist_ok=True)

    try:
        import boto3
        from botocore.config import Config

        s3 = boto3.client(
            "s3",
            config=Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                max_pool_connections=20,
            ),
        )

        # List all objects under the prefix
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=key)

        object_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                object_keys.append(obj["Key"])

        if not object_keys:
            logger.error(f"No objects found at s3://{bucket}/{key}")
            raise FileNotFoundError(f"No model files at s3://{bucket}/{key}")

        total_size_bytes = 0
        download_start = time.time()

        for obj_key in object_keys:
            # Compute relative path within model directory
            relative = obj_key[len(key):]
            if not relative:
                continue  # skip the directory placeholder itself

            dest = local / relative
            dest.parent.mkdir(parents=True, exist_ok=True)

            logger.info(f"  Downloading {obj_key} → {dest}")
            s3.download_file(bucket, obj_key, str(dest))

            total_size_bytes += dest.stat().st_size

        elapsed = time.time() - download_start
        total_mb = total_size_bytes / (1024 * 1024)
        logger.info(
            f"Model downloaded: {len(object_keys)} files, "
            f"{total_mb:.1f} MB in {elapsed:.1f}s "
            f"({total_mb / elapsed:.1f} MB/s)"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to download model from S3: {e}")
        raise


def _ensure_model_downloaded() -> None:
    """
    Ensure model is available at MODEL_PATH.

    On the first invocation of a Lambda cold start, downloads from S3.
    On warm starts, the /tmp directory persists across invocations
    so this is a no-op.
    """
    global _model_downloaded, _model_download_start

    if _model_downloaded:
        return

    if not MODEL_BUCKET:
        logger.info(
            "MODEL_BUCKET not set. Assuming model already at MODEL_PATH "
            f"({MODEL_PATH}). Skipping S3 download."
        )
        _model_downloaded = True
        return

    _model_download_start = time.time()
    downloaded = download_model_from_s3(
        bucket=MODEL_BUCKET,
        key=MODEL_KEY,
        local_path=MODEL_PATH,
    )

    if downloaded:
        cold_start_ms = (time.time() - _model_download_start) * 1000
        logger.info(f"Cold start model download: {cold_start_ms:.0f}ms")

    _model_downloaded = True


# ─── Initialize model before first request ────────────────────────────────────
# This runs once per Lambda container instance (cold start).
# Subsequent invocations skip the download and model load is fast.
_ensure_model_downloaded()

# Set MODEL_PATH env var so the FastAPI app picks it up in its lifespan handler
os.environ["MODEL_PATH"] = MODEL_PATH

# Import the FastAPI app AFTER setting MODEL_PATH
from src.serving.api import app  # noqa: E402

# ─── Mangum adapter ───────────────────────────────────────────────────────────
# Mangum wraps the ASGI app for Lambda + API Gateway / Function URL.
# lifespan='off' because Lambda containers don't run the lifespan protocol;
# instead we initialize the model explicitly above via _ensure_model_downloaded().
try:
    from mangum import Mangum

    handler = Mangum(app, lifespan="off")
    logger.info("Mangum handler initialized successfully")

except ImportError:
    logger.error(
        "mangum not installed. Install with: pip install mangum. "
        "Falling back to raw ASGI handler."
    )

    # Minimal fallback that returns a 500 with instructions
    async def handler(event, context):  # type: ignore[misc]
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": "mangum_not_installed",
                "message": "Install mangum: pip install mangum>=0.17.0",
            }),
            "headers": {"Content-Type": "application/json"},
        }


# ─── Lambda entry point ───────────────────────────────────────────────────────
# AWS Lambda invokes `handler(event, context)` automatically.
# The Mangum instance is callable and acts as the Lambda handler.
