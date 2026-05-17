"""
Tests for the FastAPI sentiment analysis API.

Uses httpx AsyncClient with a mocked inference engine so tests run
without loading actual model weights. Tests cover:
  - Health endpoint
  - Single prediction (English + multilingual)
  - Batch prediction
  - Language auto-detection
  - Input validation (422 on invalid inputs)
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport


# ─── Fixtures ─────────────────────────────────────────────────────────────────

def _make_mock_engine(
    label: str = "positive",
    confidence: float = 0.92,
    language: str = "en",
) -> MagicMock:
    """Create a mock SentimentInferenceEngine."""
    engine = MagicMock()

    # predict_single returns a dict
    engine.predict_single.return_value = {
        "label": label,
        "confidence": confidence,
        "language": language,
        "latency_ms": 12.5,
        "all_scores": {"negative": 0.04, "neutral": 0.04, "positive": confidence},
    }

    # predict_batch returns a list of dicts
    engine.predict_batch.side_effect = lambda texts, langs=None, batch_size=32: [
        {
            "label": label,
            "confidence": confidence,
            "language": langs[i] if langs else language,
            "latency_ms": 10.0,
            "all_scores": {"negative": 0.04, "neutral": 0.04, "positive": confidence},
        }
        for i, _ in enumerate(texts)
    ]

    return engine


@pytest.fixture
def mock_engine():
    """Fixture providing a default positive-sentiment mock engine."""
    return _make_mock_engine()


@pytest_asyncio.fixture
async def client(mock_engine):
    """
    Create an async test client with the inference engine mocked.
    Patches src.serving.api._engine so the FastAPI app uses the mock.
    """
    from src.serving.api import app

    with patch("src.serving.api._engine", mock_engine):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


# ─── Health endpoint ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient):
    """Health check should return 200 with model_loaded=True when engine is set."""
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert "device" in body
    assert "num_labels" in body


@pytest.mark.asyncio
async def test_health_endpoint_without_model():
    """Health check returns degraded status when engine is None."""
    from src.serving.api import app

    with patch("src.serving.api._engine", None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False


# ─── Single prediction endpoint ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_single_prediction_english(client: AsyncClient, mock_engine: MagicMock):
    """Single English prediction returns expected structure."""
    mock_engine.predict_single.return_value = {
        "label": "positive",
        "confidence": 0.97,
        "language": "en",
        "latency_ms": 8.3,
        "all_scores": {"negative": 0.01, "neutral": 0.02, "positive": 0.97},
    }

    response = await client.post(
        "/predict",
        json={"text": "I absolutely love this product!", "language": "en"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "positive"
    assert body["confidence"] == pytest.approx(0.97, abs=1e-4)
    assert body["language"] == "en"
    assert "latency_ms" in body
    assert "all_scores" in body
    mock_engine.predict_single.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "de", "fr", "es"])
async def test_single_prediction_multilingual(
    language: str,
    mock_engine: MagicMock,
):
    """Parametrized test: predictions work for en, de, fr, es."""
    mock_engine.predict_single.return_value = {
        "label": "positive",
        "confidence": 0.88,
        "language": language,
        "latency_ms": 15.0,
        "all_scores": {"negative": 0.06, "neutral": 0.06, "positive": 0.88},
    }

    from src.serving.api import app

    with patch("src.serving.api._engine", mock_engine):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/predict",
                json={
                    "text": "This is a great product with excellent quality.",
                    "language": language,
                },
            )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"positive", "neutral", "negative"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert body["language"] == language


# ─── Batch prediction endpoint ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_batch_prediction(client: AsyncClient, mock_engine: MagicMock):
    """Batch prediction returns one result per input text."""
    texts = [
        "Great product!",
        "Terrible experience.",
        "It was okay, nothing special.",
    ]
    langs = ["en", "en", "en"]

    mock_engine.predict_batch.return_value = [
        {"label": "positive", "confidence": 0.95, "language": "en", "latency_ms": 5.0,
         "all_scores": {"negative": 0.02, "neutral": 0.03, "positive": 0.95}},
        {"label": "negative", "confidence": 0.91, "language": "en", "latency_ms": 5.0,
         "all_scores": {"negative": 0.91, "neutral": 0.05, "positive": 0.04}},
        {"label": "neutral", "confidence": 0.72, "language": "en", "latency_ms": 5.0,
         "all_scores": {"negative": 0.15, "neutral": 0.72, "positive": 0.13}},
    ]

    response = await client.post(
        "/predict/batch",
        json={"texts": texts, "languages": langs},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert len(body["predictions"]) == 3
    assert "total_latency_ms" in body

    assert body["predictions"][0]["label"] == "positive"
    assert body["predictions"][1]["label"] == "negative"
    assert body["predictions"][2]["label"] == "neutral"


@pytest.mark.asyncio
async def test_batch_prediction_no_languages(client: AsyncClient, mock_engine: MagicMock):
    """Batch prediction without explicit languages triggers auto-detection."""
    texts = ["Excellent!", "Schlecht.", "Bien."]

    response = await client.post(
        "/predict/batch",
        json={"texts": texts},
    )

    assert response.status_code == 200
    assert response.json()["count"] == 3


# ─── Language auto-detection ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_language_auto_detection(client: AsyncClient, mock_engine: MagicMock):
    """Omitting language triggers auto-detection (handled by inference engine)."""
    mock_engine.predict_single.return_value = {
        "label": "positive",
        "confidence": 0.85,
        "language": "de",  # auto-detected as German
        "latency_ms": 20.0,
        "all_scores": {"negative": 0.08, "neutral": 0.07, "positive": 0.85},
    }

    response = await client.post(
        "/predict",
        json={"text": "Das ist ein fantastisches Produkt!"},  # German, no lang provided
    )

    assert response.status_code == 200
    body = response.json()
    assert body["language"] == "de"  # auto-detected
    # predict_single called with lang=None (auto-detect)
    call_kwargs = mock_engine.predict_single.call_args
    assert call_kwargs.kwargs.get("lang") is None or call_kwargs[1].get("lang") is None


# ─── Input validation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_input_empty_text_returns_422(client: AsyncClient):
    """Empty text should return 422 Unprocessable Entity."""
    response = await client.post("/predict", json={"text": ""})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_input_missing_text_returns_422(client: AsyncClient):
    """Missing required 'text' field returns 422."""
    response = await client.post("/predict", json={"language": "en"})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_input_whitespace_only_returns_422(client: AsyncClient):
    """Whitespace-only text returns 422."""
    response = await client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_batch_mismatched_languages_returns_422(client: AsyncClient):
    """Batch request with mismatched texts/languages length returns 422."""
    response = await client.post(
        "/predict/batch",
        json={"texts": ["text1", "text2"], "languages": ["en"]},  # mismatch
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_invalid_batch_empty_texts_returns_422(client: AsyncClient):
    """Empty batch returns 422."""
    response = await client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


# ─── Languages endpoint ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_languages_endpoint(client: AsyncClient):
    """Languages endpoint returns training and zero-shot language lists."""
    response = await client.get("/languages")
    assert response.status_code == 200
    body = response.json()
    assert "training_languages" in body
    assert "zero_shot_languages" in body
    assert "en" in body["training_languages"]
    assert "sw" in body["training_languages"]  # Swahili (low-resource)
    assert "ta" in body["training_languages"]  # Tamil (low-resource)
    assert len(body["training_languages"]) == 8
    assert len(body["zero_shot_languages"]) == 7


# ─── Response headers ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_correlation_id_header(client: AsyncClient):
    """Responses include X-Correlation-ID header."""
    response = await client.get("/health")
    assert "x-correlation-id" in response.headers


@pytest.mark.asyncio
async def test_process_time_header(client: AsyncClient):
    """Responses include X-Process-Time-Ms header."""
    response = await client.get("/health")
    assert "x-process-time-ms" in response.headers
    latency = float(response.headers["x-process-time-ms"])
    assert latency >= 0


# ─── 503 when engine not loaded ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_predict_returns_503_when_no_model():
    """Prediction returns 503 when model is not loaded."""
    from src.serving.api import app

    with patch("src.serving.api._engine", None):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            response = await ac.post(
                "/predict",
                json={"text": "Test text", "language": "en"},
            )
    assert response.status_code == 503
