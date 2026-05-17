"""
Abstract base class for multilingual sentiment classifiers.
Provides shared interface for encoder freezing, layer group extraction,
and embedding retrieval used by both mBERT and XLM-RoBERTa classifiers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseSentimentClassifier(nn.Module, ABC):
    """
    Abstract base for multilingual sentiment classifiers.

    Subclasses must implement:
      - forward(): compute loss and/or logits
      - get_embeddings(): return CLS/pooled token embeddings

    Provides:
      - freeze_encoder(): freeze bottom N transformer layers
      - get_layer_groups(): list of (name, param_iterator) for layer-wise LR
    """

    def __init__(self, num_labels: int = 3) -> None:
        """
        Args:
            num_labels: Number of sentiment classes (2=binary, 3=neg/neu/pos)
        """
        super().__init__()
        self.num_labels = num_labels
        # Subclasses must set self.encoder and self.classifier
        self.encoder: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Run forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs (BERT only) [batch, seq_len]
            labels: Ground-truth label indices [batch] for loss computation

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        ...

    @abstractmethod
    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract CLS / pooled token embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Segment IDs (BERT only)

        Returns:
            Embedding tensor [batch, hidden_size]
        """
        ...

    def freeze_encoder(self, num_layers: int) -> None:
        """
        Freeze the bottom `num_layers` transformer encoder layers.

        This is useful for efficient fine-tuning where only top layers are updated.
        Embeddings are always frozen along with the specified layers.

        Args:
            num_layers: Number of encoder layers to freeze (counting from bottom).
                        Pass 0 to freeze only embeddings, -1 to freeze nothing.
        """
        if self.encoder is None:
            logger.warning("freeze_encoder called but self.encoder is None")
            return

        if num_layers < 0:
            return

        # Freeze embedding layer
        encoder_base = self._get_encoder_base()
        if hasattr(encoder_base, "embeddings"):
            for param in encoder_base.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze encoder embeddings")

        if num_layers == 0:
            return

        # Freeze transformer layers
        transformer_layers = self._get_transformer_layers()
        freeze_count = min(num_layers, len(transformer_layers))

        for layer in transformer_layers[:freeze_count]:
            for param in layer.parameters():
                param.requires_grad = False

        logger.info(
            f"Froze {freeze_count}/{len(transformer_layers)} encoder layers"
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for full fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True
        logger.info("Unfroze all model parameters")

    def get_layer_groups(self) -> List[Tuple[str, List[nn.Parameter]]]:
        """
        Return ordered list of (group_name, parameters) for layer-wise LR scheduling.

        Order: [embeddings, layer_0, layer_1, ..., layer_N, pooler, classifier]
        The optimizer assigns decreasing LR to earlier groups (embeddings get lowest).

        Returns:
            List of (name, params) tuples in bottom-to-top order
        """
        groups: List[Tuple[str, List[nn.Parameter]]] = []
        encoder_base = self._get_encoder_base()

        # Embedding layer group
        if hasattr(encoder_base, "embeddings"):
            groups.append((
                "embeddings",
                list(encoder_base.embeddings.parameters()),
            ))

        # Individual transformer layer groups
        transformer_layers = self._get_transformer_layers()
        for i, layer in enumerate(transformer_layers):
            groups.append((
                f"encoder_layer_{i}",
                list(layer.parameters()),
            ))

        # Pooler group (if exists)
        if hasattr(encoder_base, "pooler") and encoder_base.pooler is not None:
            groups.append((
                "pooler",
                list(encoder_base.pooler.parameters()),
            ))

        # Classification head
        if self.classifier is not None:
            groups.append((
                "classifier",
                list(self.classifier.parameters()),
            ))

        return groups

    def _get_encoder_base(self) -> nn.Module:
        """
        Navigate to the core transformer model (e.g., bert.encoder, roberta).

        Returns:
            The base transformer module (BertModel, RobertaModel, etc.)
        """
        if self.encoder is None:
            raise RuntimeError("self.encoder must be set by the subclass")

        # Try common attribute names for the base model
        for attr in ("bert", "roberta", "xlm_roberta", "deberta", "model"):
            if hasattr(self.encoder, attr):
                return getattr(self.encoder, attr)

        # Assume encoder IS the base model
        return self.encoder

    def _get_transformer_layers(self) -> nn.ModuleList:
        """
        Return the list of transformer encoder layers.

        Returns:
            ModuleList of encoder layers
        """
        base = self._get_encoder_base()
        if hasattr(base, "encoder") and hasattr(base.encoder, "layer"):
            return base.encoder.layer
        if hasattr(base, "encoder") and hasattr(base.encoder, "layers"):
            return base.encoder.layers
        raise RuntimeError(
            "Cannot locate transformer layers. Expected base.encoder.layer or base.encoder.layers"
        )

    def count_parameters(self) -> Dict[str, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def __repr__(self) -> str:
        param_counts = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"num_labels={self.num_labels}, "
            f"total_params={param_counts['total']:,}, "
            f"trainable_params={param_counts['trainable']:,})"
        )
