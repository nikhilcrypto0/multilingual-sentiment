"""
FastAPI REST API for multilingual sentiment analysis.

Endpoints:
  POST /predict        — single text prediction
  POST /predict/batch  — batch prediction with asyncio.gather parallelism
  GET  /health         — service health check
  GET  /languages      — supported languages

Features:
  - Async request handling with lifespan context manager
  - Request timing middleware + correlation ID header
  - CORS enabled for all origins
  - Global exception handler
  - <150ms p95 latency target (FP16 CUDA or CPU with caching)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from src.serving.inference import SentimentInferenceEngine

logger = logging.getLogger(__name__)

# ─── Configuration from environment ───────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "models/xlmr-finetuned")
DEVICE = os.environ.get("DEVICE", "cpu")
NUM_LABELS = int(os.environ.get("NUM_LABELS", "3"))
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
FP16 = os.environ.get("FP16", "false").lower() == "true"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
MAX_BATCH_TEXTS = int(os.environ.get("MAX_BATCH_TEXTS", "512"))

SUPPORTED_LANGUAGES = [
    "en", "de", "fr", "es", "zh", "ar", "sw", "ta",  # training languages
    "ru", "hi", "bg", "el", "th", "tr", "ur",          # zero-shot languages
]

# ─── Global inference engine (loaded at startup) ──────────────────────────────
_engine: Optional[SentimentInferenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.
    Loads the model on startup, cleans up on shutdown.
    """
    global _engine
    logger.info(f"Loading model from {MODEL_PATH} on device={DEVICE}")
    try:
        _engine = SentimentInferenceEngine(
            model_path=MODEL_PATH,
            device=DEVICE,
            num_labels=NUM_LABELS,
            max_length=MAX_LENGTH,
            fp16=FP16,
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Allow startup even without model for health checks
        _engine = None

    yield  # Application runs here

    # Shutdown cleanup
    logger.info("Shutting down inference engine")
    _engine = None


# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multilingual Sentiment Analysis API",
    description=(
        "Production-grade sentiment analysis supporting 8+ languages. "
        "Uses mBERT / XLM-RoBERTa with cross-lingual transfer learning. "
        "Achieves 92%+ F1 across English, German, French, Spanish, Chinese, "
        "Arabic, Swahili, and Tamil."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─── CORS middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request timing + correlation ID middleware ────────────────────────────────
@app.middleware("http")
async def add_timing_and_correlation(request: Request, call_next):
    """Add X-Correlation-ID and X-Process-Time headers to every response."""
    correlation_id = request.headers.get("X-Correlation-ID") or str(uuid.uuid4())
    start = time.perf_counter()
    response: Response = await call_next(request)
    process_time_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Correlation-ID"] = correlation_id
    response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"
    return response


# ─── Global exception handler ─────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions and return structured JSON error."""
    logger.exception(f"Unhandled exception on {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url),
        },
    )


# ─── Pydantic schemas ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    """Single text prediction request."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="Text to analyze for sentiment",
        example="I absolutely loved the food at this restaurant!",
    )
    language: Optional[str] = Field(
        default=None,
        description="ISO 639-1 language code. Auto-detected if omitted.",
        example="en",
    )

    @validator("text")
    def text_not_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be empty or whitespace only")
        return v


class PredictResponse(BaseModel):
    """Single text prediction response."""
    label: str = Field(..., description="Predicted sentiment: positive, neutral, or negative")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence score")
    language: str = Field(..., description="Language used for prediction")
    latency_ms: float = Field(..., description="Prediction latency in milliseconds")
    all_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Probabilities for all sentiment classes",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""
    texts: List[str] = Field(
        ...,
        min_items=1,
        max_items=MAX_BATCH_TEXTS,
        description="List of texts to analyze",
    )
    languages: Optional[List[str]] = Field(
        default=None,
        description="Optional list of language codes (one per text). Auto-detects if omitted.",
    )

    @validator("texts")
    def texts_not_empty(cls, v: List[str]) -> List[str]:
        for i, text in enumerate(v):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{i}] must be a non-empty string")
        return v

    @validator("languages")
    def languages_match_texts(cls, v: Optional[List[str]], values: dict) -> Optional[List[str]]:
        if v is not None and "texts" in values:
            if len(v) != len(values["texts"]):
                raise ValueError(
                    f"languages length ({len(v)}) must match texts length ({len(values['texts'])})"
                )
        return v


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictResponse] = Field(..., description="Per-text predictions")
    total_latency_ms: float = Field(..., description="Total batch processing time in ms")
    count: int = Field(..., description="Number of predictions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str
    device: str
    num_labels: int


class LanguagesResponse(BaseModel):
    """Supported languages response."""
    training_languages: List[str]
    zero_shot_languages: List[str]
    total: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["utility"])
async def health_check() -> HealthResponse:
    """
    Service health check.

    Returns model loaded status, device, and configuration.
    Always returns 200 (even if model not loaded) so load balancers
    can distinguish process-up from model-ready.
    """
    return HealthResponse(
        status="ok" if _engine is not None else "degraded",
        model_loaded=_engine is not None,
        model_path=MODEL_PATH,
        device=DEVICE,
        num_labels=NUM_LABELS,
    )


@app.get("/languages", response_model=LanguagesResponse, tags=["utility"])
async def get_languages() -> LanguagesResponse:
    """Return the list of supported languages."""
    return LanguagesResponse(
        training_languages=["en", "de", "fr", "es", "zh", "ar", "sw", "ta"],
        zero_shot_languages=["ru", "hi", "bg", "el", "th", "tr", "ur"],
        total=15,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Predict sentiment for a single text.

    Supports 15+ languages. Language is auto-detected if not provided.
    Returns the predicted label, confidence score, and all class probabilities.
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up.",
        )

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _engine.predict_single(
                text=request.text,
                lang=request.language,
            ),
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        label=result["label"],
        confidence=result["confidence"],
        language=result["language"],
        latency_ms=result.get("latency_ms", 0.0),
        all_scores=result.get("all_scores"),
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["inference"])
async def predict_batch(request: BatchPredictRequest) -> BatchPredictResponse:
    """
    Predict sentiment for a batch of texts.

    Processes texts in parallel using asyncio.gather for maximum throughput.
    Each text can have an optional language code; auto-detection is used otherwise.
    """
    if _engine is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is starting up.",
        )

    start = time.perf_counter()

    texts = request.texts
    langs = request.languages

    # For small batches (<= 8), run individual predictions in parallel via asyncio.gather
    # For larger batches, use the optimized predict_batch method
    try:
        if len(texts) <= 8:
            # Parallel async dispatch for small batches
            async def _predict_one(text: str, lang: Optional[str]) -> Dict:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda t=text, l=lang: _engine.predict_single(t, l),
                )

            lang_list = langs if langs else [None] * len(texts)
            batch_results = await asyncio.gather(
                *[_predict_one(t, l) for t, l in zip(texts, lang_list)]
            )
        else:
            # Optimized batch inference (single forward pass per sub-batch)
            batch_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _engine.predict_batch(
                    texts=texts,
                    langs=langs,
                    batch_size=BATCH_SIZE,
                ),
            )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

    total_latency_ms = (time.perf_counter() - start) * 1000

    predictions = [
        PredictResponse(
            label=r["label"],
            confidence=r["confidence"],
            language=r["language"],
            latency_ms=r.get("latency_ms", total_latency_ms / len(texts)),
            all_scores=r.get("all_scores"),
        )
        for r in batch_results
    ]

    return BatchPredictResponse(
        predictions=predictions,
        total_latency_ms=round(total_latency_ms, 2),
        count=len(predictions),
    )
