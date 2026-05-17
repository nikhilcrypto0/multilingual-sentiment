"""
Benchmarking suite for multilingual sentiment models.

Runs systematic evaluations on XNLI test sets per language,
zero-shot cross-lingual transfer benchmarks, head-to-head model comparisons,
and generates structured JSON + tabular reports.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.data.dataset_loader import (
    TRAIN_LANGUAGES,
    ZERO_SHOT_LANGUAGES,
    load_xnli,
)
from src.data.preprocessing import create_tokenized_dataset
from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_baseline_comparison,
    compute_f1_per_language,
    print_metrics_table,
)

logger = logging.getLogger(__name__)


class Benchmarker:
    """
    Systematic benchmarking for multilingual sentiment classifiers.

    Supports:
      - XNLI per-language evaluation
      - Zero-shot evaluation on unseen languages
      - Side-by-side model comparison
      - JSON + tabular report generation
    """

    def __init__(
        self,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 128,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            device: Torch device ('cuda', 'cpu', 'mps'). Auto-detects if None.
            batch_size: Inference batch size
            max_length: Max tokenization length
            cache_dir: HuggingFace cache directory
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        logger.info(f"Benchmarker initialized on device: {self.device}")

    def _run_inference(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        dataset: Dataset,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Run batch inference on a dataset.

        Args:
            model: Pre-trained classification model
            tokenizer: Matching tokenizer
            dataset: Tokenized or raw dataset with 'text', 'label', 'language' columns

        Returns:
            (predictions, labels, languages) as numpy arrays / list
        """
        model = model.to(self.device)
        model.eval()

        # Tokenize if not already done
        if "input_ids" not in dataset.column_names:
            dataset = create_tokenized_dataset(
                dataset,
                tokenizer=tokenizer,
                max_length=self.max_length,
                num_proc=1,
            )

        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=(self.device == "cuda"),
        )

        all_preds: List[int] = []
        all_labels: List[int] = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].tolist()

                # Support both HF models and our custom classifiers
                if hasattr(model, "forward"):
                    try:
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                        )
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                    except TypeError:
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    raise RuntimeError("Model must have a forward() method")

                if isinstance(logits, tuple):
                    logits = logits[1]  # (loss, logits) tuple

                preds = logits.argmax(dim=-1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch_labels)

        # Recover language info from original dataset
        dataset.reset_format()
        if "language" in dataset.column_names:
            languages = dataset["language"]
        else:
            languages = ["unknown"] * len(all_labels)

        return (
            np.array(all_preds),
            np.array(all_labels),
            list(languages),
        )

    def run_xnli_benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        languages: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Evaluate model on XNLI test set per language.

        Args:
            model: Sentiment classification model
            tokenizer: Matching tokenizer
            languages: Languages to evaluate; defaults to TRAIN_LANGUAGES

        Returns:
            (per_lang_df, aggregate_metrics) tuple
        """
        if languages is None:
            languages = TRAIN_LANGUAGES

        logger.info(f"Running XNLI benchmark for {len(languages)} languages...")
        start_time = time.time()

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_langs: List[str] = []

        for lang in languages:
            logger.info(f"  Benchmarking XNLI: {lang}")
            try:
                test_ds = load_xnli(
                    [lang],
                    split="test",
                    cache_dir=self.cache_dir,
                )
                preds, labels, langs = self._run_inference(model, tokenizer, test_ds)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_langs.extend(langs)
            except Exception as e:
                logger.warning(f"  Failed to benchmark {lang}: {e}")

        if not all_preds:
            logger.error("No predictions generated during XNLI benchmark")
            return pd.DataFrame(), {}

        per_lang_df = compute_f1_per_language(all_preds, all_labels, all_langs)
        aggregate = compute_aggregate_metrics(per_lang_df)

        elapsed = time.time() - start_time
        logger.info(
            f"XNLI benchmark done in {elapsed:.1f}s. Macro F1: {aggregate.get('macro_f1', 'N/A'):.4f}"
        )
        return per_lang_df, aggregate

    def run_zero_shot_benchmark(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        zero_shot_langs: Optional[List[str]] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Evaluate on languages not seen during training (zero-shot transfer).

        Args:
            model: Model trained on TRAIN_LANGUAGES only
            tokenizer: Matching tokenizer
            zero_shot_langs: Languages to test; defaults to ZERO_SHOT_LANGUAGES

        Returns:
            (per_lang_df, aggregate_metrics) tuple
        """
        if zero_shot_langs is None:
            zero_shot_langs = ZERO_SHOT_LANGUAGES

        logger.info(f"Running zero-shot benchmark for {len(zero_shot_langs)} languages...")

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_langs: List[str] = []

        for lang in zero_shot_langs:
            logger.info(f"  Zero-shot eval: {lang}")
            try:
                test_ds = load_xnli([lang], split="test", cache_dir=self.cache_dir)
                preds, labels, langs = self._run_inference(model, tokenizer, test_ds)
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
                all_langs.extend(langs)
            except Exception as e:
                logger.warning(f"  Failed zero-shot for {lang}: {e}")

        if not all_preds:
            logger.error("No zero-shot predictions generated")
            return pd.DataFrame(), {}

        per_lang_df = compute_f1_per_language(all_preds, all_labels, all_langs)
        aggregate = compute_aggregate_metrics(per_lang_df)

        logger.info(
            f"Zero-shot macro F1: {aggregate.get('macro_f1', 'N/A'):.4f} "
            f"over {len(zero_shot_langs)} languages"
        )
        return per_lang_df, aggregate

    def compare_models(
        self,
        mbert_model: PreTrainedModel,
        xlmr_model: PreTrainedModel,
        tokenizer_mbert: PreTrainedTokenizerBase,
        tokenizer_xlmr: PreTrainedTokenizerBase,
        languages: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Run side-by-side XNLI benchmark comparison of mBERT vs XLM-RoBERTa.

        Args:
            mbert_model: Fine-tuned mBERT model
            xlmr_model: Fine-tuned XLM-RoBERTa model
            tokenizer_mbert: mBERT tokenizer
            tokenizer_xlmr: XLM-R tokenizer
            languages: Languages to compare; defaults to TRAIN_LANGUAGES

        Returns:
            Combined DataFrame with per-language metrics for both models
        """
        if languages is None:
            languages = TRAIN_LANGUAGES

        logger.info("Running mBERT vs XLM-R comparison benchmark...")

        mbert_df, mbert_agg = self.run_xnli_benchmark(mbert_model, tokenizer_mbert, languages)
        xlmr_df, xlmr_agg = self.run_xnli_benchmark(xlmr_model, tokenizer_xlmr, languages)

        comparison_dict = compute_baseline_comparison(mbert_df, xlmr_df)

        logger.info("\nmBERT vs XLM-RoBERTa Comparison:")
        logger.info(f"  mBERT macro F1:  {mbert_agg.get('macro_f1', 'N/A'):.4f}")
        logger.info(f"  XLM-R macro F1:  {xlmr_agg.get('macro_f1', 'N/A'):.4f}")
        logger.info(
            f"  Avg improvement: {comparison_dict.get('avg_improvement', 'N/A'):.4f} "
            f"({comparison_dict.get('avg_improvement_pct', 'N/A'):.1f}%)"
        )

        # Build side-by-side DataFrame
        comparison_df = pd.DataFrame(comparison_dict["per_language"])
        return comparison_df

    def generate_benchmark_report(
        self,
        results: Dict[str, Any],
        output_path: str,
    ) -> str:
        """
        Save benchmark results to JSON and print formatted table.

        Args:
            results: Dict containing benchmark results (from run_xnli_benchmark, etc.)
            output_path: Path to save JSON report (e.g., 'reports/benchmark.json')

        Returns:
            Path to the saved JSON file
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        # Convert DataFrames to records for JSON serialization
        serializable_results: Dict[str, Any] = {}
        for key, value in results.items():
            if isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict(orient="records")
            elif isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, np.floating):
                serializable_results[key] = float(value)
            elif isinstance(value, np.integer):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value

        # Add timestamp
        from datetime import datetime
        serializable_results["generated_at"] = datetime.now().isoformat()

        with open(output_path, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        logger.info(f"Benchmark report saved to {output_path}")

        # Print summary table
        print(f"\n{'='*70}")
        print("  BENCHMARK REPORT SUMMARY")
        print(f"{'='*70}")

        for key, value in serializable_results.items():
            if key == "generated_at":
                continue
            if isinstance(value, dict):
                print(f"\n  [{key}]")
                for k, v in value.items():
                    if isinstance(v, float):
                        print(f"    {k:<30} {v:.4f}")
                    elif not isinstance(v, (list, dict)):
                        print(f"    {k:<30} {v}")
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                print(f"\n  [{key}]")
                df = pd.DataFrame(value)
                print(df.to_string(index=False, float_format="%.4f"))

        print(f"\n  Report saved to: {output_path}")
        print(f"{'='*70}\n")

        return output_path
