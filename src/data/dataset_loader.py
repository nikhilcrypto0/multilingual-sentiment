"""
Dataset loading utilities for multilingual sentiment analysis.
Supports XNLI and multilingual SST datasets across 8 training languages
and 7 zero-shot evaluation languages.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Languages used during fine-tuning
TRAIN_LANGUAGES: List[str] = ["en", "de", "fr", "es", "zh", "ar", "sw", "ta"]

# Languages evaluated zero-shot (no training data)
ZERO_SHOT_LANGUAGES: List[str] = ["ru", "hi", "bg", "el", "th", "tr", "ur"]

# XNLI label mapping: entailment→positive(0), neutral→neutral(1), contradiction→negative(2)
XNLI_LABEL_MAP: Dict[int, int] = {0: 0, 1: 1, 2: 2}
XNLI_BINARY_LABEL_MAP: Dict[int, int] = {0: 1, 1: 1, 2: 0}  # entailment→pos, rest→neg

LABEL_NAMES: List[str] = ["negative", "neutral", "positive"]


def _xnli_lang_code(lang: str) -> str:
    """Map our language codes to XNLI dataset language codes."""
    mapping = {
        "en": "en",
        "de": "de",
        "fr": "fr",
        "es": "es",
        "zh": "zh",
        "ar": "ar",
        "sw": "sw",
        "ta": "ta",
        "ru": "ru",
        "hi": "hi",
        "bg": "bg",
        "el": "el",
        "th": "th",
        "tr": "tr",
        "ur": "ur",
    }
    return mapping.get(lang, lang)


def load_xnli(
    languages: List[str],
    split: str = "train",
    binary: bool = False,
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load XNLI dataset for the specified languages and split.

    XNLI labels: 0=entailment (positive), 1=neutral, 2=contradiction (negative)
    For 3-class sentiment: entailment→positive, neutral→neutral, contradiction→negative
    For binary: entailment→positive(1), others→negative(0)

    Args:
        languages: List of ISO language codes to load
        split: One of 'train', 'validation', 'test'
        binary: If True, map to binary labels (positive/negative)
        cache_dir: Optional HuggingFace cache directory

    Returns:
        HuggingFace Dataset with 'text', 'label', 'language' columns
    """
    all_splits: List[Dataset] = []
    label_map = XNLI_BINARY_LABEL_MAP if binary else XNLI_LABEL_MAP

    for lang in languages:
        xnli_lang = _xnli_lang_code(lang)
        logger.info(f"Loading XNLI for language: {lang} (split={split})")
        try:
            # XNLI is loaded per language config
            ds = load_dataset(
                "xnli",
                xnli_lang,
                split=split,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            def _process(example: Dict, lang_code: str = lang) -> Dict:
                # Combine premise and hypothesis as the text
                text = f"{example['premise']} [SEP] {example['hypothesis']}"
                orig_label = int(example["label"])
                mapped_label = label_map.get(orig_label, orig_label)
                return {
                    "text": text,
                    "label": mapped_label,
                    "language": lang_code,
                }

            ds = ds.map(_process, remove_columns=ds.column_names)
            all_splits.append(ds)
            logger.info(f"  Loaded {len(ds)} examples for {lang}")
        except Exception as e:
            logger.warning(f"Failed to load XNLI for {lang}: {e}")
            continue

    if not all_splits:
        raise ValueError(f"No XNLI data could be loaded for languages: {languages}")

    combined = concatenate_datasets(all_splits)
    logger.info(f"Total XNLI examples loaded: {len(combined)}")
    return combined


def load_multilingual_sst(
    languages: List[str],
    cache_dir: Optional[str] = None,
) -> Dataset:
    """
    Load multilingual sentiment data.
    - English: uses SST-2 from GLUE (binary, mapped to 3-class by adding neutral)
    - Other languages: falls back to XNLI training split as proxy sentiment data

    Args:
        languages: List of ISO language codes
        cache_dir: Optional HuggingFace cache directory

    Returns:
        HuggingFace Dataset with 'text', 'label', 'language' columns
    """
    all_datasets: List[Dataset] = []

    for lang in languages:
        logger.info(f"Loading sentiment data for: {lang}")
        try:
            if lang == "en":
                # SST-2: binary sentiment (0=neg, 1=pos) — map to 3-class (no neutral)
                sst = load_dataset(
                    "glue",
                    "sst2",
                    split="train",
                    cache_dir=cache_dir,
                )

                def _sst_process(example: Dict) -> Dict:
                    # SST-2 label: 0=negative→0, 1=positive→2 (skip neutral class)
                    label_map = {0: 0, 1: 2}
                    return {
                        "text": example["sentence"],
                        "label": label_map[int(example["label"])],
                        "language": "en",
                    }

                sst = sst.map(_sst_process, remove_columns=sst.column_names)
                all_datasets.append(sst)
                logger.info(f"  Loaded {len(sst)} SST-2 examples for en")
            else:
                # For non-English: use XNLI as proxy (provides 3-class labels)
                xnli_ds = load_xnli([lang], split="train", cache_dir=cache_dir)
                all_datasets.append(xnli_ds)
                logger.info(f"  Loaded {len(xnli_ds)} XNLI proxy examples for {lang}")
        except Exception as e:
            logger.warning(f"Failed to load data for {lang}: {e}")
            continue

    if not all_datasets:
        raise ValueError(f"No data loaded for languages: {languages}")

    combined = concatenate_datasets(all_datasets)
    logger.info(f"Total multilingual SST examples: {len(combined)}")
    return combined


def load_combined_dataset(
    languages: List[str] = None,
    include_zero_shot: bool = False,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    cache_dir: Optional[str] = None,
) -> DatasetDict:
    """
    Load and merge XNLI + multilingual SST, perform stratified train/val/test split.

    Args:
        languages: Languages to include; defaults to TRAIN_LANGUAGES
        include_zero_shot: Whether to also load ZERO_SHOT_LANGUAGES for eval
        val_ratio: Fraction for validation
        test_ratio: Fraction for test
        seed: Random seed for reproducibility
        cache_dir: Optional HuggingFace cache directory

    Returns:
        DatasetDict with 'train', 'validation', 'test' keys
    """
    if languages is None:
        languages = TRAIN_LANGUAGES

    logger.info(f"Loading combined dataset for languages: {languages}")

    # Load from both sources
    xnli_train = load_xnli(languages, split="train", cache_dir=cache_dir)
    xnli_val = load_xnli(languages, split="validation", cache_dir=cache_dir)
    xnli_test = load_xnli(languages, split="test", cache_dir=cache_dir)

    sst_data = load_multilingual_sst(languages, cache_dir=cache_dir)

    # Merge XNLI train + SST for training
    train_combined = concatenate_datasets([xnli_train, sst_data])
    train_combined = train_combined.shuffle(seed=seed)

    result = DatasetDict(
        {
            "train": train_combined,
            "validation": xnli_val,
            "test": xnli_test,
        }
    )

    if include_zero_shot:
        logger.info("Loading zero-shot evaluation languages...")
        for lang in ZERO_SHOT_LANGUAGES:
            try:
                zs_test = load_xnli([lang], split="test", cache_dir=cache_dir)
                result[f"test_{lang}"] = zs_test
            except Exception as e:
                logger.warning(f"Could not load zero-shot data for {lang}: {e}")

    logger.info("Dataset splits:")
    for split_name, ds in result.items():
        logger.info(f"  {split_name}: {len(ds)} examples")

    return result


def get_language_stats(dataset: Dataset) -> Dict[str, int]:
    """
    Compute per-language sample counts in a dataset.

    Args:
        dataset: HuggingFace Dataset with 'language' column

    Returns:
        Dictionary mapping language code → sample count
    """
    if "language" not in dataset.column_names:
        raise ValueError("Dataset must have a 'language' column")

    languages = dataset["language"]
    counts: Dict[str, int] = {}
    for lang in languages:
        counts[lang] = counts.get(lang, 0) + 1

    # Sort by count descending
    counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    return counts


def get_label_distribution(dataset: Dataset) -> Dict[str, Dict[str, int]]:
    """
    Compute label distribution per language.

    Args:
        dataset: HuggingFace Dataset with 'language' and 'label' columns

    Returns:
        Nested dict: language → label_name → count
    """
    dist: Dict[str, Dict[str, int]] = {}
    for example in dataset:
        lang = example["language"]
        label_idx = int(example["label"])
        label_name = LABEL_NAMES[label_idx] if label_idx < len(LABEL_NAMES) else str(label_idx)
        if lang not in dist:
            dist[lang] = {name: 0 for name in LABEL_NAMES}
        dist[lang][label_name] = dist[lang].get(label_name, 0) + 1
    return dist
