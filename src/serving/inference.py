"""
Inference engine for multilingual sentiment analysis.

Provides:
  - ModelCache: singleton for loaded model + tokenizer
  - TokenizerCache: LRU-cached tokenized inputs by text hash + language
  - SentimentInferenceEngine: batch/single prediction with FP16, torch.no_grad

Designed for low-latency serving with <150ms p95 latency target.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
BINARY_LABEL_MAP: Dict[int, str] = {0: "negative", 1: "positive"}


class ModelCache:
    """
    Singleton holding the loaded model and tokenizer in memory.

    Prevents reloading the model on every request in a long-lived process
    (e.g., a FastAPI server). Thread-safe via Python's GIL for the load step.
    """

    _instance: Optional["ModelCache"] = None
    _model: Optional[torch.nn.Module] = None
    _tokenizer: Optional[PreTrainedTokenizerBase] = None
    _model_path: Optional[str] = None
    _num_labels: int = 3

    def __new__(cls) -> "ModelCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(
        self,
        model_path: str,
        device: str = "cpu",
        num_labels: int = 3,
    ) -> None:
        """
        Load model and tokenizer from model_path into memory.

        Args:
            model_path: Local directory or HuggingFace model ID
            device: Torch device string
            num_labels: Number of sentiment classes
        """
        if self._model_path == model_path:
            logger.info("ModelCache: model already loaded")
            return

        logger.info(f"ModelCache: loading model from {model_path}")
        start = time.time()

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
        )
        self._model.eval()
        self._model.to(device)
        self._model_path = model_path
        self._num_labels = num_labels

        elapsed = time.time() - start
        logger.info(f"ModelCache: model loaded in {elapsed:.2f}s on {device}")

    @property
    def model(self) -> Optional[torch.nn.Module]:
        return self._model

    @property
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._tokenizer is not None

    @property
    def num_labels(self) -> int:
        return self._num_labels


# Global singleton
_model_cache = ModelCache()


def _text_hash(text: str) -> str:
    """Compute MD5 hash of text for cache keying."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class TokenizerCache:
    """
    LRU cache for tokenized inputs keyed by (text_hash, language).

    Avoids re-tokenizing identical texts across concurrent requests.
    Particularly effective for health-check or repeated query patterns.
    """

    def __init__(self, maxsize: int = 1024) -> None:
        self._maxsize = maxsize
        self._cache: Dict[Tuple[str, str], Dict] = {}
        self._access_order: List[Tuple[str, str]] = []

    def get(self, text: str, lang: str) -> Optional[Dict]:
        """Retrieve cached tokenization or None."""
        key = (_text_hash(text), lang)
        if key in self._cache:
            # Move to end (LRU update)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, text: str, lang: str, tokenized: Dict) -> None:
        """Store tokenized result in cache."""
        key = (_text_hash(text), lang)
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self._maxsize:
            # Evict least recently used
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[key] = tokenized
        self._access_order.append(key)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()

    def __len__(self) -> int:
        return len(self._cache)


# Global tokenizer cache
_tokenizer_cache = TokenizerCache(maxsize=1024)


def _detect_language(text: str) -> str:
    """
    Auto-detect language using langdetect with 'en' fallback.

    Args:
        text: Input text

    Returns:
        ISO 639-1 language code
    """
    try:
        from langdetect import detect
        if len(text.strip()) < 10:
            return "en"
        return detect(text)
    except Exception:
        return "en"


class SentimentInferenceEngine:
    """
    Production inference engine for multilingual sentiment analysis.

    Features:
      - Singleton model loading via ModelCache
      - LRU tokenization caching (1024 entries)
      - torch.no_grad() context for inference
      - FP16 autocast for CUDA devices
      - Async-compatible (no blocking I/O in hot path)

    Example:
        engine = SentimentInferenceEngine("models/xlmr-finetuned", device="cuda")
        result = engine.predict_single("I love this!", lang="en")
        # {'label': 'positive', 'confidence': 0.97, 'language': 'en'}
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        num_labels: int = 3,
        max_length: int = 128,
        fp16: bool = False,
    ) -> None:
        """
        Args:
            model_path: Path to saved model directory or HuggingFace model ID
            device: Torch device ('cpu', 'cuda', 'mps')
            num_labels: Number of sentiment classes (2 or 3)
            max_length: Max tokenization length
            fp16: Enable FP16 autocast (CUDA only)
        """
        self.model_path = model_path
        self.device = device
        self.num_labels = num_labels
        self.max_length = max_length
        self.fp16 = fp16 and device == "cuda"

        self.label_map = LABEL_MAP if num_labels == 3 else BINARY_LABEL_MAP

        # Load model into singleton cache
        _model_cache.load(model_path, device=device, num_labels=num_labels)

        self._tokenizer_cache = _tokenizer_cache
        logger.info(
            f"SentimentInferenceEngine ready: device={device}, fp16={self.fp16}, "
            f"num_labels={num_labels}"
        )

    @property
    def model(self) -> torch.nn.Module:
        if not _model_cache.is_loaded:
            raise RuntimeError("Model not loaded. Call SentimentInferenceEngine.__init__ first.")
        return _model_cache.model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        if not _model_cache.is_loaded:
            raise RuntimeError("Tokenizer not loaded.")
        return _model_cache.tokenizer

    def _tokenize(self, text: str, lang: str = "en") -> Dict[str, torch.Tensor]:
        """
        Tokenize a single text with LRU cache.

        Args:
            text: Raw input text
            lang: Language code (used as cache key dimension)

        Returns:
            Dict with input_ids, attention_mask tensors on self.device
        """
        cached = self._tokenizer_cache.get(text, lang)
        if cached is not None:
            return {k: v.to(self.device) for k, v in cached.items()}

        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Store CPU tensors in cache (device-agnostic)
        self._tokenizer_cache.put(text, lang, {k: v.cpu() for k, v in encoded.items()})

        return {k: v.to(self.device) for k, v in encoded.items()}

    def predict_single(
        self,
        text: str,
        lang: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Predict sentiment for a single text.

        Args:
            text: Raw input text
            lang: ISO 639-1 language code. Auto-detects if None.

        Returns:
            Dict with keys:
              - label: 'positive', 'neutral', or 'negative'
              - confidence: float [0, 1]
              - language: detected or provided language code
        """
        start_ns = time.perf_counter_ns()

        if lang is None:
            lang = _detect_language(text)

        encoded = self._tokenize(text, lang)

        with torch.no_grad():
            if self.fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**encoded)
            else:
                outputs = self.model(**encoded)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if isinstance(logits, tuple):
            logits = logits[-1]

        probabilities = F.softmax(logits, dim=-1)
        confidence, predicted_class = probabilities.max(dim=-1)

        label_idx = int(predicted_class.item())
        label = self.label_map.get(label_idx, "unknown")
        conf = float(confidence.item())

        latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        return {
            "label": label,
            "confidence": round(conf, 4),
            "language": lang,
            "latency_ms": round(latency_ms, 2),
            "all_scores": {
                self.label_map.get(i, str(i)): round(float(p), 4)
                for i, p in enumerate(probabilities[0].tolist())
            },
        }

    def predict_batch(
        self,
        texts: List[str],
        langs: Optional[List[str]] = None,
        batch_size: int = 32,
    ) -> List[Dict[str, object]]:
        """
        Predict sentiment for a batch of texts.

        Processes in sub-batches of batch_size for memory efficiency.

        Args:
            texts: List of raw input texts
            langs: Optional list of language codes (one per text)
            batch_size: Sub-batch size for inference

        Returns:
            List of prediction dicts (same format as predict_single)
        """
        if not texts:
            return []

        start_ns = time.perf_counter_ns()

        # Detect languages if not provided
        if langs is None:
            langs = [_detect_language(t) for t in texts]
        elif len(langs) != len(texts):
            raise ValueError(f"langs length {len(langs)} != texts length {len(texts)}")

        all_results: List[Dict[str, object]] = []

        # Process in sub-batches
        for batch_start in range(0, len(texts), batch_size):
            batch_texts = texts[batch_start: batch_start + batch_size]
            batch_langs = langs[batch_start: batch_start + batch_size]

            # Tokenize the sub-batch
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                if self.fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**encoded)
                else:
                    outputs = self.model(**encoded)

            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            if isinstance(logits, tuple):
                logits = logits[-1]

            probabilities = F.softmax(logits, dim=-1)
            confidences, predicted_classes = probabilities.max(dim=-1)

            for i, (text, lang) in enumerate(zip(batch_texts, batch_langs)):
                label_idx = int(predicted_classes[i].item())
                label = self.label_map.get(label_idx, "unknown")
                conf = float(confidences[i].item())

                all_results.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "language": lang,
                    "all_scores": {
                        self.label_map.get(j, str(j)): round(float(p), 4)
                        for j, p in enumerate(probabilities[i].tolist())
                    },
                })

        total_latency_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
        logger.debug(
            f"Batch inference: {len(texts)} texts in {total_latency_ms:.1f}ms "
            f"({total_latency_ms / len(texts):.1f}ms/text)"
        )

        return all_results
