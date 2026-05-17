"""
XLM-RoBERTa-based multilingual sentiment classifier.
Uses xlm-roberta-base with a custom classification head,
attention weight extraction for interpretability, and layer-wise LR support.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.models.base_classifier import BaseSentimentClassifier

logger = logging.getLogger(__name__)

XLMR_MODEL_NAME = "xlm-roberta-base"


class XLMRClassifier(BaseSentimentClassifier):
    """
    XLM-RoBERTa classifier for multilingual sentiment analysis.

    Architecture:
      - xlm-roberta-base backbone (12 layers, 768 hidden, 278M params)
      - Dropout on pooled [CLS] token representation
      - Linear classification head → num_labels logits
      - Cross-entropy loss for supervised fine-tuning

    Extra capabilities vs mBERT:
      - get_attention_weights(): extract per-layer, per-head attention scores
        for interpretability and cross-lingual transfer analysis
      - Better performance on low-resource languages (sw, ta) due to
        larger multilingual pretraining corpus

    Supports:
      - 3-class (negative/neutral/positive) or binary classification
      - Gradient checkpointing
      - Layer freezing and layer-wise LR via base class
    """

    def __init__(
        self,
        num_labels: int = 3,
        model_name: str = XLMR_MODEL_NAME,
        dropout_prob: float = 0.1,
        use_gradient_checkpointing: bool = False,
        output_attentions: bool = False,
        cache_dir: Optional[str] = None,
    ) -> None:
        """
        Args:
            num_labels: Number of output classes (2 or 3)
            model_name: HuggingFace model identifier
            dropout_prob: Dropout probability on pooled representation
            use_gradient_checkpointing: Enable gradient checkpointing to reduce VRAM
            output_attentions: Whether to output attention weights (needed for
                               get_attention_weights())
            cache_dir: Optional HuggingFace cache directory
        """
        super().__init__(num_labels=num_labels)

        self.model_name = model_name
        self.dropout_prob = dropout_prob
        self._output_attentions = output_attentions

        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob,
            output_attentions=output_attentions,
            cache_dir=cache_dir,
        )

        self.encoder = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=True,
        )

        # XLM-RoBERTa uses RobertaForSequenceClassification
        # which has: roberta → pooler → dropout → out_proj (2 linear layers)
        if hasattr(self.encoder, "classifier"):
            self.classifier = self.encoder.classifier
        else:
            hidden_size = config.hidden_size
            self.dropout = nn.Dropout(p=dropout_prob)
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.encoder.classifier = self.classifier

        if use_gradient_checkpointing:
            base = self._get_encoder_base()
            if hasattr(base, "gradient_checkpointing_enable"):
                base.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for XLM-R")

        logger.info(
            f"Initialized XLMRClassifier: {self.count_parameters()['total']:,} total params"
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
        Forward pass through XLM-RoBERTa.

        Note: XLM-R does not use token_type_ids (always zeros); they are ignored.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Ignored for RoBERTa (kept for interface compatibility)
            labels: Ground-truth labels [batch]

        Returns:
            (loss, logits) if labels provided, else logits [batch, num_labels]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # RoBERTa does not use token_type_ids
            labels=labels,
            output_attentions=self._output_attentions,
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
        Extract [CLS] token embeddings from the last hidden state.

        XLM-RoBERTa uses the first token (position 0) as the sequence representation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            [CLS] token embedding [batch, hidden_size]
        """
        base_model = self._get_encoder_base()
        ctx = torch.no_grad() if not self.training else torch.enable_grad()
        with ctx:
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        # Use [CLS] token (position 0) from last hidden state
        return outputs.last_hidden_state[:, 0, :]

    def get_attention_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        Extract attention weights from all layers for interpretability.

        Must be called with output_attentions=True in __init__.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            List of attention tensors, one per layer.
            Each tensor shape: [batch, num_heads, seq_len, seq_len]

        Raises:
            RuntimeError: If model was not initialized with output_attentions=True
        """
        if not self._output_attentions:
            raise RuntimeError(
                "Attention weights not available. "
                "Initialize XLMRClassifier with output_attentions=True"
            )

        base_model = self._get_encoder_base()
        with torch.no_grad():
            outputs = base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                return_dict=True,
            )

        # outputs.attentions is a tuple of tensors, one per layer
        return list(outputs.attentions)

    def get_cross_lingual_attention_analysis(
        self,
        source_input_ids: torch.Tensor,
        source_attention_mask: torch.Tensor,
        target_input_ids: torch.Tensor,
        target_attention_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Analyze attention pattern differences between source and target language inputs.
        Useful for understanding cross-lingual transfer quality.

        Args:
            source_input_ids: Source language token IDs
            source_attention_mask: Source language attention mask
            target_input_ids: Target language token IDs
            target_attention_mask: Target language attention mask

        Returns:
            Dict with 'source_attention', 'target_attention', 'mean_difference'
        """
        source_attentions = self.get_attention_weights(source_input_ids, source_attention_mask)
        target_attentions = self.get_attention_weights(target_input_ids, target_attention_mask)

        # Compute mean absolute difference in attention patterns (last layer)
        last_layer_src = source_attentions[-1].mean(dim=1)  # avg over heads
        last_layer_tgt = target_attentions[-1].mean(dim=1)

        # Pool to scalar difference (compare min-dim tensors)
        min_seq = min(last_layer_src.shape[-1], last_layer_tgt.shape[-1])
        diff = (
            last_layer_src[:, :min_seq, :min_seq] - last_layer_tgt[:, :min_seq, :min_seq]
        ).abs().mean().item()

        return {
            "source_attention": source_attentions,
            "target_attention": target_attentions,
            "mean_last_layer_difference": diff,
            "num_layers": len(source_attentions),
        }

    def save_pretrained(self, save_directory: str) -> None:
        """Save model and config to directory."""
        self.encoder.save_pretrained(save_directory)
        logger.info(f"XLM-R model saved to {save_directory}")

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        num_labels: int = 3,
        **kwargs: Any,
    ) -> "XLMRClassifier":
        """
        Load a fine-tuned XLMRClassifier from a saved directory.

        Args:
            model_path: Path to saved model directory
            num_labels: Number of output labels

        Returns:
            Loaded XLMRClassifier instance
        """
        instance = cls(num_labels=num_labels, model_name=model_path, **kwargs)
        logger.info(f"Loaded XLMRClassifier from {model_path}")
        return instance

    @staticmethod
    def get_tokenizer(
        model_name: str = XLMR_MODEL_NAME,
        cache_dir: Optional[str] = None,
    ) -> Any:
        """Load the XLM-RoBERTa tokenizer."""
        return AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
