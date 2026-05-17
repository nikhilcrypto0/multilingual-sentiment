"""
mBERT-based multilingual sentiment classifier.
Uses bert-base-multilingual-cased with a custom classification head,
dropout regularization, and support for layer-wise LR decay.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.models.base_classifier import BaseSentimentClassifier

logger = logging.getLogger(__name__)

MBERT_MODEL_NAME = "bert-base-multilingual-cased"


class MBertClassifier(BaseSentimentClassifier):
    """
    Multilingual BERT classifier for sentiment analysis.

    Architecture:
      - bert-base-multilingual-cased backbone (12 layers, 768 hidden, 110M params)
      - Dropout on pooled [CLS] representation
      - Linear classification head → num_labels logits
      - Cross-entropy loss for supervised fine-tuning

    Supports:
      - 3-class (negative/neutral/positive) or binary classification
      - Gradient checkpointing for memory efficiency
      - Layer freezing and layer-wise LR via base class
    """

    def __init__(
        self,
        num_labels: int = 3,
        model_name: str = MBERT_MODEL_NAME,
        dropout_prob: float = 0.1,
        use_gradient_checkpointing: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            num_labels: Number of output classes (2 or 3)
            model_name: HuggingFace model identifier
            dropout_prob: Dropout probability applied to pooled output
            use_gradient_checkpointing: Enable gradient checkpointing to reduce VRAM
            cache_dir: Optional HuggingFace cache directory
        """
        super().__init__(num_labels=num_labels)

        self.model_name = model_name
        self.dropout_prob = dropout_prob

        # Load pre-trained mBERT with sequence classification head
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            cache_dir=cache_dir,
        )

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        # Extract the built-in classifier head from the model
        # BertForSequenceClassification has: bert → pooler → dropout → classifier
        # We expose self.classifier for layer group extraction
        if hasattr(self.encoder, "classifier"):
            self.classifier = self.encoder.classifier
        else:
            # Fallback: build our own classification head
            hidden_size = config.hidden_size
            self.dropout = nn.Dropout(p=dropout_prob)
            self.classifier = nn.Linear(hidden_size, num_labels)
            # Register on encoder for forward pass
            self.encoder.classifier = self.classifier

        if use_gradient_checkpointing:
            base = self._get_encoder_base()
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for mBERT")

        logger.info(
            f"Initialized MBertClassifier: {self.count_parameters()['total']:,} total params"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through mBERT.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs [batch, seq_len]
            labels: Ground-truth labels [batch] (required for loss)

        Returns:
            (loss, logits) if labels provided, else logits [batch, num_labels]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,
            **kwargs,
        )

        if labels is not None:
            return outputs.loss, outputs.logits
        return outputs.logits

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract pooled [CLS] token embeddings from mBERT.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs [batch, seq_len]

        Returns:
            Pooled output tensor [batch, hidden_size]
        """
        base_model = self._get_encoder_base()
        with torch.no_grad() if not self.training else torch.enable_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        return outputs.pooler_output

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and config to directory."""
        self.encoder.save_pretrained(save_directory)
        logger.info(f"Model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_labels: int = 3,
        **kwargs: Any,
    ) -> "MBertClassifier":
        """
        Load a fine-tuned MBertClassifier from a saved directory.

        Args:
            model_path: Path to saved model directory
            num_labels: Number of output labels

        Returns:
            Loaded MBertClassifier instance
        """
        instance = cls(num_labels=num_labels, model_name=model_path, **kwargs)
        logger.info(f"Loaded MBertClassifier from {model_path}")
        return instance

    @staticmethod
    def get_tokenizer(
        model_name: str = MBERT_MODEL_NAME,
        cache_dir: Optional[str] = None,
    ) -> Any:
        """Load the mBERT tokenizer."""
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
