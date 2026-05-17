"""
Evaluation metrics for multilingual sentiment analysis.

Provides per-language F1/precision/recall breakdowns, aggregate statistics,
model comparison utilities, and visualization helpers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)

LABEL_NAMES = ["negative", "neutral", "positive"]


def compute_f1_per_language(
    predictions: List[int],
    labels: List[int],
    languages: List[str],
    average: str = "macro",
    num_labels: int = 3,
) -> pd.DataFrame:
    """
    Compute per-language F1, precision, recall, and support.

    Args:
        predictions: List of predicted label indices
        labels: List of ground-truth label indices
        languages: List of language codes (one per example)
        average: Sklearn averaging strategy ('macro', 'weighted', 'micro')
        num_labels: Number of label classes

    Returns:
        DataFrame with columns: language, f1, precision, recall, support, accuracy
    """
    assert len(predictions) == len(labels) == len(languages), (
        "predictions, labels, and languages must have the same length"
    )

    preds_arr = np.array(predictions)
    labels_arr = np.array(labels)
    langs_arr = np.array(languages)

    unique_langs = sorted(set(languages))
    rows = []

    for lang in unique_langs:
        mask = langs_arr == lang
        lang_preds = preds_arr[mask]
        lang_labels = labels_arr[mask]
        support = int(mask.sum())

        if support == 0:
            continue

        f1 = f1_score(lang_labels, lang_preds, average=average, zero_division=0)
        precision = precision_score(lang_labels, lang_preds, average=average, zero_division=0)
        recall = recall_score(lang_labels, lang_preds, average=average, zero_division=0)
        accuracy = float((lang_preds == lang_labels).mean())

        rows.append({
            "language": lang,
            "f1": round(float(f1), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "accuracy": round(accuracy, 4),
            "support": support,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    logger.info(f"Per-language metrics computed for {len(df)} languages")
    return df


def compute_aggregate_metrics(per_lang_df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute macro and weighted aggregate metrics from per-language DataFrame.

    Args:
        per_lang_df: Output DataFrame from compute_f1_per_language

    Returns:
        Dict with keys: macro_f1, macro_precision, macro_recall, macro_accuracy,
                        weighted_f1, weighted_precision, weighted_recall, weighted_accuracy,
                        min_f1_lang, min_f1, max_f1_lang, max_f1
    """
    if per_lang_df.empty:
        return {}

    total_support = per_lang_df["support"].sum()

    # Macro average (equal weight per language)
    macro_f1 = per_lang_df["f1"].mean()
    macro_prec = per_lang_df["precision"].mean()
    macro_rec = per_lang_df["recall"].mean()
    macro_acc = per_lang_df["accuracy"].mean()

    # Weighted average (weight by support)
    weights = per_lang_df["support"] / total_support
    weighted_f1 = (per_lang_df["f1"] * weights).sum()
    weighted_prec = (per_lang_df["precision"] * weights).sum()
    weighted_rec = (per_lang_df["recall"] * weights).sum()
    weighted_acc = (per_lang_df["accuracy"] * weights).sum()

    # Best and worst languages
    best_row = per_lang_df.loc[per_lang_df["f1"].idxmax()]
    worst_row = per_lang_df.loc[per_lang_df["f1"].idxmin()]

    return {
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_accuracy": round(float(macro_acc), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "weighted_precision": round(float(weighted_prec), 4),
        "weighted_recall": round(float(weighted_rec), 4),
        "weighted_accuracy": round(float(weighted_acc), 4),
        "best_lang": str(best_row["language"]),
        "best_f1": round(float(best_row["f1"]), 4),
        "worst_lang": str(worst_row["language"]),
        "worst_f1": round(float(worst_row["f1"]), 4),
        "total_support": int(total_support),
        "num_languages": len(per_lang_df),
    }


def compute_baseline_comparison(
    mbert_results: pd.DataFrame,
    xlmr_results: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compare mBERT vs XLM-RoBERTa results per language.

    Args:
        mbert_results: Per-language DataFrame from mBERT evaluation
        xlmr_results: Per-language DataFrame from XLM-RoBERTa evaluation

    Returns:
        Dict with per-language improvement (xlmr - mbert) and summary stats
    """
    # Merge on language
    merged = mbert_results.merge(
        xlmr_results,
        on="language",
        suffixes=("_mbert", "_xlmr"),
    )

    merged["f1_improvement"] = merged["f1_xlmr"] - merged["f1_mbert"]
    merged["f1_improvement_pct"] = (merged["f1_improvement"] / merged["f1_mbert"]) * 100

    avg_improvement = merged["f1_improvement"].mean()
    avg_improvement_pct = merged["f1_improvement_pct"].mean()

    # Languages where XLM-R wins
    xlmr_wins = merged[merged["f1_improvement"] > 0]["language"].tolist()
    mbert_wins = merged[merged["f1_improvement"] < 0]["language"].tolist()
    ties = merged[merged["f1_improvement"] == 0]["language"].tolist()

    comparison = {
        "per_language": merged[[
            "language", "f1_mbert", "f1_xlmr", "f1_improvement", "f1_improvement_pct"
        ]].to_dict(orient="records"),
        "avg_f1_mbert": round(float(mbert_results["f1"].mean()), 4),
        "avg_f1_xlmr": round(float(xlmr_results["f1"].mean()), 4),
        "avg_improvement": round(float(avg_improvement), 4),
        "avg_improvement_pct": round(float(avg_improvement_pct), 2),
        "xlmr_wins": xlmr_wins,
        "mbert_wins": mbert_wins,
        "ties": ties,
        "xlmr_better_count": len(xlmr_wins),
        "mbert_better_count": len(mbert_wins),
    }

    logger.info(
        f"Comparison: XLM-R avg F1 improvement = {avg_improvement:.4f} "
        f"({avg_improvement_pct:.1f}%)"
    )
    return comparison


def plot_language_comparison(
    results_dict: Dict[str, pd.DataFrame],
    metric: str = "f1",
    save_path: Optional[str] = None,
    title: str = "Per-Language F1 Comparison",
    figsize: Tuple[int, int] = (12, 6),
) -> None:
    """
    Plot per-language metric comparison as a grouped bar chart.

    Args:
        results_dict: Mapping of model_name → per-language DataFrame
        metric: Column to plot (default 'f1')
        save_path: Path to save figure; if None, displays interactively
        title: Chart title
        figsize: Figure size (width, height) in inches
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed. Skipping plot.")
        return

    # Build combined DataFrame for seaborn
    dfs = []
    for model_name, df in results_dict.items():
        model_df = df[["language", metric]].copy()
        model_df["model"] = model_name
        dfs.append(model_df)

    combined = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=figsize)
    sns.set_theme(style="whitegrid", palette="muted")

    ax = sns.barplot(
        data=combined,
        x="language",
        y=metric,
        hue="model",
        dodge=True,
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Language", fontsize=12)
    ax.set_ylabel(f"{metric.upper()} Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")

    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def print_metrics_table(
    per_lang_df: pd.DataFrame,
    aggregate: Optional[Dict[str, float]] = None,
    model_name: str = "Model",
) -> None:
    """
    Print a formatted metrics table to stdout.

    Args:
        per_lang_df: Per-language metrics DataFrame
        aggregate: Optional aggregate metrics dict
        model_name: Name to display in the header
    """
    print(f"\n{'='*60}")
    print(f"  {model_name} — Per-Language Sentiment Metrics")
    print(f"{'='*60}")
    print(per_lang_df.to_string(index=False, float_format="%.4f"))

    if aggregate:
        print(f"\n{'─'*60}")
        print("  Aggregate Metrics")
        print(f"{'─'*60}")
        for k, v in aggregate.items():
            if isinstance(v, float):
                print(f"  {k:<30} {v:.4f}")
            else:
                print(f"  {k:<30} {v}")
    print(f"{'='*60}\n")
