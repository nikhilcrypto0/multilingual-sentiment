#!/usr/bin/env python3
"""
Benchmark both mBERT and XLM-RoBERTa models on XNLI test sets.
Prints a side-by-side comparison table and saves a JSON report.

Usage:
    python scripts/run_benchmark.py \
        --mbert-path models/mbert-finetuned \
        --xlmr-path models/xlmr-finetuned \
        --output reports/comparison.json

Reports include:
  - Per-language F1 for each model
  - XLM-R vs mBERT improvement per language
  - Zero-shot transfer results
  - Macro and weighted averages
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark mBERT and XLM-RoBERTa on multilingual sentiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mbert-path",
        type=str,
        default="models/mbert-finetuned",
        help="Path to fine-tuned mBERT model directory",
    )
    parser.add_argument(
        "--xlmr-path",
        type=str,
        default="models/xlmr-finetuned",
        help="Path to fine-tuned XLM-RoBERTa model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/comparison.json",
        help="Path to save benchmark report JSON",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Override languages to benchmark (default: all 8 training languages)",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        default=True,
        help="Also run zero-shot evaluation on 7 additional languages",
    )
    parser.add_argument(
        "--no-zero-shot",
        action="store_true",
        help="Skip zero-shot evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=3,
        choices=[2, 3],
        help="Number of sentiment classes",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, num_labels: int = 3):
    """Load a fine-tuned model and its tokenizer from disk."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
    )
    model.eval()
    logger.info(f"Loaded model: {model.__class__.__name__}")
    return model, tokenizer


def print_comparison_table(
    mbert_df,
    xlmr_df,
    comparison_df,
    mbert_agg: dict,
    xlmr_agg: dict,
) -> None:
    """Print a formatted side-by-side comparison table."""
    import pandas as pd

    print("\n" + "=" * 80)
    print("  MULTILINGUAL SENTIMENT BENCHMARK — mBERT vs XLM-RoBERTa")
    print("=" * 80)
    print(f"\n{'Language':<12} {'mBERT F1':>10} {'XLM-R F1':>10} {'Δ F1':>10} {'Δ%':>8}")
    print("-" * 55)

    for _, row in comparison_df.iterrows():
        delta = row.get("f1_improvement", 0)
        delta_pct = row.get("f1_improvement_pct", 0)
        sign = "+" if delta > 0 else ""
        print(
            f"  {row['language']:<10} "
            f"{row['f1_mbert']:>10.4f} "
            f"{row['f1_xlmr']:>10.4f} "
            f"{sign}{delta:>9.4f} "
            f"{sign}{delta_pct:>7.1f}%"
        )

    print("-" * 55)
    print(
        f"  {'MACRO AVG':<10} "
        f"{mbert_agg.get('macro_f1', 0):>10.4f} "
        f"{xlmr_agg.get('macro_f1', 0):>10.4f} "
        f"{xlmr_agg.get('macro_f1', 0) - mbert_agg.get('macro_f1', 0):>+10.4f}"
    )
    print(
        f"  {'WEIGHTED':<10} "
        f"{mbert_agg.get('weighted_f1', 0):>10.4f} "
        f"{xlmr_agg.get('weighted_f1', 0):>10.4f} "
        f"{xlmr_agg.get('weighted_f1', 0) - mbert_agg.get('weighted_f1', 0):>+10.4f}"
    )
    print("=" * 80)
    print(f"\n  mBERT best lang:  {mbert_agg.get('best_lang')} (F1={mbert_agg.get('best_f1'):.4f})")
    print(f"  mBERT worst lang: {mbert_agg.get('worst_lang')} (F1={mbert_agg.get('worst_f1'):.4f})")
    print(f"  XLM-R best lang:  {xlmr_agg.get('best_lang')} (F1={xlmr_agg.get('best_f1'):.4f})")
    print(f"  XLM-R worst lang: {xlmr_agg.get('worst_lang')} (F1={xlmr_agg.get('worst_f1'):.4f})")
    print()


def main() -> None:
    args = parse_args()
    run_zero_shot = args.zero_shot and not args.no_zero_shot

    from src.data.dataset_loader import TRAIN_LANGUAGES, ZERO_SHOT_LANGUAGES
    from src.evaluation.benchmark import Benchmarker
    from src.evaluation.metrics import compute_baseline_comparison, print_metrics_table

    languages = args.languages or TRAIN_LANGUAGES

    logger.info("=" * 60)
    logger.info("MULTILINGUAL SENTIMENT BENCHMARK")
    logger.info(f"  mBERT path:  {args.mbert_path}")
    logger.info(f"  XLM-R path:  {args.xlmr_path}")
    logger.info(f"  Languages:   {languages}")
    logger.info(f"  Zero-shot:   {run_zero_shot}")
    logger.info(f"  Output:      {args.output}")
    logger.info("=" * 60)

    # Load both models
    mbert_model, mbert_tokenizer = load_model_and_tokenizer(args.mbert_path, args.num_labels)
    xlmr_model, xlmr_tokenizer = load_model_and_tokenizer(args.xlmr_path, args.num_labels)

    # Initialize benchmarker
    benchmarker = Benchmarker(
        device=args.device,
        batch_size=args.batch_size,
    )

    # Run XNLI benchmark for both models
    logger.info("Benchmarking mBERT on XNLI test set...")
    mbert_per_lang, mbert_agg = benchmarker.run_xnli_benchmark(
        mbert_model, mbert_tokenizer, languages
    )

    logger.info("Benchmarking XLM-R on XNLI test set...")
    xlmr_per_lang, xlmr_agg = benchmarker.run_xnli_benchmark(
        xlmr_model, xlmr_tokenizer, languages
    )

    # Model comparison
    comparison_dict = compute_baseline_comparison(mbert_per_lang, xlmr_per_lang)
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_dict["per_language"])

    # Print table
    print_comparison_table(mbert_per_lang, xlmr_per_lang, comparison_df, mbert_agg, xlmr_agg)

    # Per-model tables
    print_metrics_table(mbert_per_lang, mbert_agg, "mBERT (bert-base-multilingual-cased)")
    print_metrics_table(xlmr_per_lang, xlmr_agg, "XLM-RoBERTa (xlm-roberta-base)")

    # Zero-shot evaluation
    zs_mbert_per_lang = None
    zs_xlmr_per_lang = None

    if run_zero_shot:
        logger.info("Running zero-shot benchmark (mBERT)...")
        zs_mbert_per_lang, zs_mbert_agg = benchmarker.run_zero_shot_benchmark(
            mbert_model, mbert_tokenizer, ZERO_SHOT_LANGUAGES
        )

        logger.info("Running zero-shot benchmark (XLM-R)...")
        zs_xlmr_per_lang, zs_xlmr_agg = benchmarker.run_zero_shot_benchmark(
            xlmr_model, xlmr_tokenizer, ZERO_SHOT_LANGUAGES
        )

        print("\n" + "=" * 60)
        print("  ZERO-SHOT TRANSFER RESULTS")
        print("=" * 60)
        print_metrics_table(zs_mbert_per_lang, zs_mbert_agg, "mBERT Zero-Shot")
        print_metrics_table(zs_xlmr_per_lang, zs_xlmr_agg, "XLM-R Zero-Shot")

    # Compile full report
    report = {
        "benchmark_metadata": {
            "generated_at": datetime.now().isoformat(),
            "mbert_path": args.mbert_path,
            "xlmr_path": args.xlmr_path,
            "languages": languages,
            "zero_shot_languages": ZERO_SHOT_LANGUAGES if run_zero_shot else [],
        },
        "mbert": {
            "per_language": mbert_per_lang.to_dict(orient="records"),
            "aggregate": mbert_agg,
        },
        "xlmr": {
            "per_language": xlmr_per_lang.to_dict(orient="records"),
            "aggregate": xlmr_agg,
        },
        "comparison": {
            "per_language": comparison_dict["per_language"],
            "avg_f1_mbert": comparison_dict["avg_f1_mbert"],
            "avg_f1_xlmr": comparison_dict["avg_f1_xlmr"],
            "avg_improvement": comparison_dict["avg_improvement"],
            "avg_improvement_pct": comparison_dict["avg_improvement_pct"],
            "xlmr_wins": comparison_dict["xlmr_wins"],
            "mbert_wins": comparison_dict["mbert_wins"],
        },
    }

    if run_zero_shot and zs_mbert_per_lang is not None:
        report["zero_shot"] = {
            "mbert": {
                "per_language": zs_mbert_per_lang.to_dict(orient="records"),
                "aggregate": zs_mbert_agg,
            },
            "xlmr": {
                "per_language": zs_xlmr_per_lang.to_dict(orient="records"),
                "aggregate": zs_xlmr_agg,
            },
        }

    # Save report
    benchmarker.generate_benchmark_report(report, args.output)

    logger.info("=" * 60)
    logger.info("Benchmark complete!")
    logger.info(f"  mBERT macro F1:  {mbert_agg.get('macro_f1', 'N/A'):.4f}")
    logger.info(f"  XLM-R macro F1:  {xlmr_agg.get('macro_f1', 'N/A'):.4f}")
    logger.info(
        f"  Avg improvement: {comparison_dict.get('avg_improvement_pct', 'N/A'):.1f}%"
    )
    logger.info(f"  Report: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
