"""
Preprocessing utilities for multilingual text data.
Handles Unicode normalization, script-specific cleaning, language detection,
tokenization, and dynamic padding for multilingual sentiment analysis.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorMixin

logger = logging.getLogger(__name__)

# CJK Unicode ranges for special handling
CJK_RANGES = [
    (0x4E00, 0x9FFF),    # CJK Unified Ideographs
    (0x3400, 0x4DBF),    # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
    (0x2A700, 0x2B73F),  # CJK Extension C
    (0x2B740, 0x2B81F),  # CJK Extension D
    (0xF900, 0xFAFF),    # CJK Compatibility Ideographs
    (0x3000, 0x303F),    # CJK Symbols and Punctuation
    (0xFF00, 0xFFEF),    # Halfwidth/Fullwidth Forms
]

# Arabic Unicode range
ARABIC_RANGE = (0x0600, 0x06FF)

# Tamil Unicode range
TAMIL_RANGE = (0x0B80, 0x0BFF)


def _is_cjk_char(char: str) -> bool:
    """Check if a character is in a CJK Unicode block."""
    cp = ord(char)
    return any(start <= cp <= end for start, end in CJK_RANGES)


def _normalize_cjk(text: str) -> str:
    """Add spaces around CJK characters to ensure correct tokenization."""
    output = []
    for char in text:
        if _is_cjk_char(char):
            output.append(f" {char} ")
        else:
            output.append(char)
    return "".join(output)


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text: strip diacritics (tashkeel) for cleaner input."""
    # Arabic diacritics range: 0x064B–0x065F
    diacritics_pattern = re.compile(r"[ً-ٰٟ]")
    text = diacritics_pattern.sub("", text)
    # Normalize Alef variants to plain Alef
    text = re.sub(r"[إأآ]", "ا", text)
    # Normalize Teh marbuta
    text = re.sub(r"ة", "ه", text)
    return text


def _normalize_tamil(text: str) -> str:
    """Normalize Tamil text: handle common Unicode normalization."""
    # Use NFC normalization for Tamil to handle composed forms
    return unicodedata.normalize("NFC", text)


class MultilingualPreprocessor:
    """
    Handles language-aware text cleaning and preprocessing for multilingual
    sentiment analysis. Applies script-specific normalization and unicode
    standardization before tokenization.
    """

    def __init__(self, lower_case: bool = False) -> None:
        """
        Args:
            lower_case: Whether to lowercase text (not recommended for cased models)
        """
        self.lower_case = lower_case

    def clean_text(self, text: str, lang: str = "en") -> str:
        """
        Clean and normalize text with language-specific handling.

        Pipeline:
        1. Unicode NFC normalization
        2. Remove control characters
        3. Script-specific normalization (CJK, Arabic, Tamil)
        4. Collapse excessive whitespace
        5. Optional lowercasing

        Args:
            text: Raw input text
            lang: ISO 639-1 language code

        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            text = str(text)

        # Step 1: Unicode normalization (NFC)
        text = unicodedata.normalize("NFC", text)

        # Step 2: Remove control characters (except newline/tab)
        text = "".join(
            char for char in text
            if unicodedata.category(char) not in ("Cc", "Cf") or char in ("\n", "\t")
        )

        # Step 3: Replace newlines/tabs with spaces
        text = re.sub(r"[\n\t\r]+", " ", text)

        # Step 4: Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text)

        # Step 5: Remove email addresses
        text = re.sub(r"\S+@\S+\.\S+", "[EMAIL]", text)

        # Step 6: Script-specific normalization
        if lang == "zh":
            text = _normalize_cjk(text)
        elif lang == "ar":
            text = _normalize_arabic(text)
        elif lang == "ta":
            text = _normalize_tamil(text)

        # Step 7: Collapse multiple spaces
        text = re.sub(r"\s+", " ", text).strip()

        # Step 8: Optional lowercasing
        if self.lower_case:
            text = text.lower()

        return text

    def detect_language(self, text: str, fallback: str = "en") -> str:
        """
        Detect the language of a text using langdetect with a fallback.

        Args:
            text: Input text
            fallback: Language code to return if detection fails

        Returns:
            ISO 639-1 language code
        """
        try:
            from langdetect import detect, lang_detect_exception

            if len(text.strip()) < 10:
                # Too short for reliable detection
                return fallback

            detected = detect(text)
            return detected
        except Exception as e:
            logger.debug(f"Language detection failed for text '{text[:50]}...': {e}")
            return fallback

    def preprocess_dataset(
        self,
        dataset: Dataset,
        text_column: str = "text",
        lang_column: str = "language",
    ) -> Dataset:
        """
        Apply cleaning to all examples in a dataset.

        Args:
            dataset: Input dataset
            text_column: Name of the text column
            lang_column: Name of the language column (used for script-specific cleaning)

        Returns:
            Cleaned dataset
        """
        def _clean_example(example: Dict[str, Any]) -> Dict[str, Any]:
            lang = example.get(lang_column, "en")
            example[text_column] = self.clean_text(example[text_column], lang)
            return example

        return dataset.map(_clean_example, desc="Cleaning text")


def create_tokenized_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
    text_column: str = "text",
    label_column: str = "label",
    batch_size: int = 1000,
    num_proc: int = 4,
) -> Dataset:
    """
    Tokenize a dataset using the provided HuggingFace tokenizer.

    Returns a dataset with 'input_ids', 'attention_mask', and 'labels' columns.
    Token type IDs are included if the tokenizer produces them.

    Args:
        dataset: Input HuggingFace Dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length (truncates/pads to this length)
        text_column: Column containing raw text
        label_column: Column containing integer labels
        batch_size: Batch size for mapping operation
        num_proc: Number of parallel processes for tokenization

    Returns:
        Tokenized HuggingFace Dataset
    """

    def _tokenize_batch(examples: Dict[str, List]) -> Dict[str, List]:
        tokenized = tokenizer(
            examples[text_column],
            padding=False,  # Dynamic padding applied in collator
            truncation=True,
            max_length=max_length,
            return_token_type_ids=tokenizer.model_type in ("bert", "albert", "deberta"),
        )
        tokenized["labels"] = examples[label_column]
        return tokenized

    # Determine columns to remove (keep only tokenized + labels)
    columns_to_remove = [
        col for col in dataset.column_names
        if col not in [text_column, label_column]
    ]
    # We'll also remove text/label after tokenizing
    all_remove = list(set(dataset.column_names) - {"labels"})

    tokenized = dataset.map(
        _tokenize_batch,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=all_remove,
        desc="Tokenizing dataset",
    )

    tokenized.set_format(type="torch")
    return tokenized


@dataclass
class DataCollatorForSentiment(DataCollatorMixin):
    """
    Custom data collator with dynamic padding for sentiment classification.

    Pads sequences to the longest sequence in each batch rather than a
    fixed maximum length, improving training efficiency on shorter sequences.
    Supports both single and multi-GPU training with optional label smoothing.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = 8  # Align to 8 for tensor cores
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of features with dynamic padding.

        Args:
            features: List of dicts with input_ids, attention_mask, labels, etc.

        Returns:
            Padded batch dict with torch tensors
        """
        import torch

        # Separate labels from features for padding
        label_name = "label" if "label" in features[0] else "labels"
        labels = [feature.pop(label_name) for feature in features] if label_name in features[0] else None

        # Pad input features
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # Re-attach labels
        if labels is not None:
            batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch
