"""
Unit tests for evaluation metrics utilities.

Tests compute_f1_per_language and compute_aggregate_metrics with
synthetic predictions to verify correctness without requiring real models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    compute_aggregate_metrics,
    compute_baseline_comparison,
    compute_f1_per_language,
)


# ─── Synthetic data fixtures ──────────────────────────────────────────────────

@pytest.fixture
def perfect_binary_data():
    """Perfect predictions for 2 languages, binary labels."""
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    preds = [0, 0, 1, 1, 0, 0, 1, 1]  # 100% accuracy
    langs = ["en", "en", "en", "en", "de", "de", "de", "de"]
    return preds, labels, langs


@pytest.fixture
def imperfect_multiclass_data():
    """Imperfect 3-class predictions across 3 languages."""
    # English: 4 examples, 3 correct
    # French: 4 examples, 2 correct
    # Spanish: 4 examples, 4 correct
    labels = [0, 1, 2, 0,  0, 1, 2, 0,  0, 1, 2, 0]
    preds  = [0, 1, 2, 1,  0, 2, 2, 1,  0, 1, 2, 0]  # en:3/4, fr:2/4, es:4/4
    langs  = ["en","en","en","en", "fr","fr","fr","fr", "es","es","es","es"]
    return preds, labels, langs


@pytest.fixture
def single_language_data():
    """Single language data for edge case testing."""
    np.random.seed(42)
    n = 100
    labels = np.random.randint(0, 3, n).tolist()
    preds = np.random.randint(0, 3, n).tolist()
    langs = ["en"] * n
    return preds, labels, langs


# ─── compute_f1_per_language tests ────────────────────────────────────────────

class TestComputeF1PerLanguage:

    def test_returns_dataframe(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        result = compute_f1_per_language(preds, labels, langs)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        result = compute_f1_per_language(preds, labels, langs)
        expected_cols = {"language", "f1", "precision", "recall", "accuracy", "support"}
        assert expected_cols.issubset(set(result.columns))

    def test_perfect_predictions_f1_is_one(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        result = compute_f1_per_language(preds, labels, langs)
        assert len(result) == 2  # en + de
        assert (result["f1"] == 1.0).all()
        assert (result["accuracy"] == 1.0).all()

    def test_number_of_languages(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs)
        assert len(result) == 3  # en, fr, es

    def test_support_counts_correct(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs)
        total_support = result["support"].sum()
        assert total_support == len(preds)

    def test_per_language_support(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs)
        for lang in ["en", "fr", "es"]:
            row = result[result["language"] == lang]
            assert len(row) == 1
            assert row.iloc[0]["support"] == 4

    def test_f1_within_bounds(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs)
        assert (result["f1"] >= 0.0).all()
        assert (result["f1"] <= 1.0).all()

    def test_sorted_by_f1_descending(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs)
        f1_values = result["f1"].tolist()
        assert f1_values == sorted(f1_values, reverse=True)

    def test_single_language(self, single_language_data):
        preds, labels, langs = single_language_data
        result = compute_f1_per_language(preds, labels, langs)
        assert len(result) == 1
        assert result.iloc[0]["language"] == "en"
        assert result.iloc[0]["support"] == 100

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            compute_f1_per_language([0, 1], [0, 1, 2], ["en", "en"])

    def test_empty_predictions_for_missing_language(self):
        """Language not in langs should not appear in output."""
        preds = [0, 1, 0, 1]
        labels = [0, 1, 0, 1]
        langs = ["en", "en", "de", "de"]
        result = compute_f1_per_language(preds, labels, langs)
        assert "fr" not in result["language"].values

    def test_binary_average(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        result = compute_f1_per_language(preds, labels, langs, average="binary")
        # Perfect binary predictions → F1 = 1.0
        assert (result["f1"] == 1.0).all()

    def test_weighted_average(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        result = compute_f1_per_language(preds, labels, langs, average="weighted")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3


# ─── compute_aggregate_metrics tests ─────────────────────────────────────────

class TestComputeAggregateMetrics:

    def test_returns_dict(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        assert isinstance(result, dict)

    def test_required_keys(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        required = {
            "macro_f1", "macro_precision", "macro_recall", "macro_accuracy",
            "weighted_f1", "weighted_precision", "weighted_recall", "weighted_accuracy",
            "best_lang", "best_f1", "worst_lang", "worst_f1",
            "total_support", "num_languages",
        }
        assert required.issubset(set(result.keys()))

    def test_perfect_predictions_macro_f1_is_one(self, perfect_binary_data):
        preds, labels, langs = perfect_binary_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        assert result["macro_f1"] == pytest.approx(1.0)
        assert result["macro_accuracy"] == pytest.approx(1.0)

    def test_macro_f1_equals_mean_per_lang_f1(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        expected_macro = per_lang["f1"].mean()
        assert result["macro_f1"] == pytest.approx(expected_macro, abs=1e-4)

    def test_total_support_correct(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        assert result["total_support"] == len(preds)

    def test_num_languages(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        assert result["num_languages"] == 3

    def test_best_lang_has_highest_f1(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        best_row = per_lang[per_lang["language"] == result["best_lang"]]
        assert best_row.iloc[0]["f1"] == per_lang["f1"].max()

    def test_worst_lang_has_lowest_f1(self, imperfect_multiclass_data):
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        worst_row = per_lang[per_lang["language"] == result["worst_lang"]]
        assert worst_row.iloc[0]["f1"] == per_lang["f1"].min()

    def test_empty_dataframe_returns_empty_dict(self):
        result = compute_aggregate_metrics(pd.DataFrame())
        assert result == {}

    def test_weighted_f1_between_macro_extremes(self, imperfect_multiclass_data):
        """Weighted F1 should be between min and max per-language F1."""
        preds, labels, langs = imperfect_multiclass_data
        per_lang = compute_f1_per_language(preds, labels, langs)
        result = compute_aggregate_metrics(per_lang)
        assert per_lang["f1"].min() <= result["weighted_f1"] <= per_lang["f1"].max()


# ─── compute_baseline_comparison tests ───────────────────────────────────────

class TestComputeBaselineComparison:

    def _make_df(self, data: dict) -> pd.DataFrame:
        return pd.DataFrame(data)

    def test_returns_dict(self):
        mbert_df = self._make_df({
            "language": ["en", "de", "fr"],
            "f1": [0.90, 0.85, 0.88],
            "precision": [0.91, 0.84, 0.87],
            "recall": [0.89, 0.86, 0.89],
            "accuracy": [0.90, 0.85, 0.88],
            "support": [100, 100, 100],
        })
        xlmr_df = self._make_df({
            "language": ["en", "de", "fr"],
            "f1": [0.93, 0.91, 0.92],
            "precision": [0.93, 0.90, 0.92],
            "recall": [0.93, 0.92, 0.92],
            "accuracy": [0.93, 0.91, 0.92],
            "support": [100, 100, 100],
        })
        result = compute_baseline_comparison(mbert_df, xlmr_df)
        assert isinstance(result, dict)

    def test_xlmr_wins_detected(self):
        mbert_df = self._make_df({
            "language": ["en"],
            "f1": [0.85],
            "precision": [0.85],
            "recall": [0.85],
            "accuracy": [0.85],
            "support": [100],
        })
        xlmr_df = self._make_df({
            "language": ["en"],
            "f1": [0.93],
            "precision": [0.93],
            "recall": [0.93],
            "accuracy": [0.93],
            "support": [100],
        })
        result = compute_baseline_comparison(mbert_df, xlmr_df)
        assert "en" in result["xlmr_wins"]
        assert result["xlmr_better_count"] == 1
        assert result["mbert_better_count"] == 0

    def test_improvement_values(self):
        mbert_df = self._make_df({
            "language": ["en"],
            "f1": [0.80],
            "precision": [0.80],
            "recall": [0.80],
            "accuracy": [0.80],
            "support": [100],
        })
        xlmr_df = self._make_df({
            "language": ["en"],
            "f1": [0.883],  # 8.3% above mBERT baseline
            "precision": [0.883],
            "recall": [0.883],
            "accuracy": [0.883],
            "support": [100],
        })
        result = compute_baseline_comparison(mbert_df, xlmr_df)
        assert result["avg_improvement"] == pytest.approx(0.083, abs=1e-3)


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_all_same_predictions(self):
        """All predictions identical — F1 may be 0 for unseen classes."""
        preds = [0] * 20
        labels = [0, 1, 2] * 6 + [0, 1]
        langs = ["en"] * len(preds)
        result = compute_f1_per_language(preds, labels, langs)
        assert len(result) == 1
        assert 0.0 <= result.iloc[0]["f1"] <= 1.0

    def test_single_example(self):
        """Single example doesn't crash."""
        result = compute_f1_per_language([1], [1], ["en"])
        assert len(result) == 1
        assert result.iloc[0]["support"] == 1

    def test_many_languages(self):
        """Performance test: 15 languages with 1000 examples each."""
        import random
        random.seed(0)
        n_per_lang = 1000
        lang_list = ["en", "de", "fr", "es", "zh", "ar", "sw", "ta",
                     "ru", "hi", "bg", "el", "th", "tr", "ur"]
        preds = [random.randint(0, 2) for _ in range(n_per_lang * len(lang_list))]
        labels = [random.randint(0, 2) for _ in range(n_per_lang * len(lang_list))]
        langs = lang_list * n_per_lang

        result = compute_f1_per_language(preds, labels, langs)
        assert len(result) == 15

        agg = compute_aggregate_metrics(result)
        assert agg["num_languages"] == 15
        assert agg["total_support"] == n_per_lang * len(lang_list)
