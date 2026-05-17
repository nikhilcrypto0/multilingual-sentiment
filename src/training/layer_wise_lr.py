"""
Layer-wise learning rate decay optimizer factory.

Implements the technique from "Fine-Tuning Pretrained Language Models:
Weight Initializations, Data Orders, and Early Stopping" where each
encoder layer receives a learning rate scaled by a decay factor relative
to its depth: LR_i = base_lr * decay_factor^(num_layers - i)

This means:
  - Bottom layers (embeddings, layer 0) → lowest LR (most stable features)
  - Top layers (layer N-1, pooler, classifier) → highest LR (task-specific)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW

logger = logging.getLogger(__name__)


def _get_bert_layer_groups(
    model: nn.Module,
    base_lr: float,
    lr_decay_factor: float,
    weight_decay: float,
) -> List[Dict[str, Any]]:
    """
    Build per-layer optimizer parameter groups for BERT-family models.

    Layer ordering (bottom-to-top):
      0: embeddings
      1..N: encoder layers 0..N-1
      N+1: pooler
      N+2: classifier head

    Args:
        model: The PyTorch model (MBertClassifier or XLMRClassifier)
        base_lr: Learning rate for the top layer (classifier)
        lr_decay_factor: Multiplicative decay per layer going down
        weight_decay: Weight decay applied to non-bias/non-norm params

    Returns:
        List of optimizer param group dicts with 'params' and 'lr' keys
    """
    # Navigate to the underlying transformer
    encoder = model.encoder if hasattr(model, "encoder") else model

    # Detect the base model (bert / roberta attribute)
    base_model = None
    for attr in ("bert", "roberta", "xlm_roberta", "deberta"):
        if hasattr(encoder, attr):
            base_model = getattr(encoder, attr)
            break
    if base_model is None:
        base_model = encoder

    # Collect transformer layers
    transformer_layers: Optional[nn.ModuleList] = None
    if hasattr(base_model, "encoder"):
        enc = base_model.encoder
        for attr in ("layer", "layers"):
            if hasattr(enc, attr):
                transformer_layers = getattr(enc, attr)
                break

    num_layers = len(transformer_layers) if transformer_layers is not None else 12
    logger.info(f"Building layer-wise optimizer with {num_layers} transformer layers")
    logger.info(f"Base LR: {base_lr:.2e}, decay factor: {lr_decay_factor}")

    # We'll track which parameters have been added to avoid duplicates
    added_params: set = set()

    def _no_decay(name: str) -> bool:
        """Parameters that should NOT have weight decay."""
        return any(nd in name for nd in ("bias", "LayerNorm.weight", "layer_norm.weight"))

    def _make_groups(
        named_params: Iterable[Tuple[str, nn.Parameter]],
        lr: float,
        group_name: str,
    ) -> List[Dict[str, Any]]:
        decay_params = []
        no_decay_params = []
        for name, param in named_params:
            if id(param) in added_params:
                continue
            if not param.requires_grad:
                continue
            added_params.add(id(param))
            if _no_decay(name):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        groups = []
        if decay_params:
            groups.append({
                "params": decay_params,
                "lr": lr,
                "weight_decay": weight_decay,
                "name": f"{group_name}_decay",
            })
        if no_decay_params:
            groups.append({
                "params": no_decay_params,
                "lr": lr,
                "weight_decay": 0.0,
                "name": f"{group_name}_no_decay",
            })
        return groups

    param_groups: List[Dict[str, Any]] = []

    # --- Embedding layer: lowest LR ---
    # LR = base_lr * decay^num_layers  (furthest from top)
    emb_lr = base_lr * (lr_decay_factor ** num_layers)
    if hasattr(base_model, "embeddings"):
        emb_groups = _make_groups(
            base_model.embeddings.named_parameters(),
            lr=emb_lr,
            group_name="embeddings",
        )
        param_groups.extend(emb_groups)
        logger.debug(f"  embeddings: lr={emb_lr:.2e}")

    # --- Transformer encoder layers ---
    if transformer_layers is not None:
        for i, layer in enumerate(transformer_layers):
            # Layer i from the bottom gets LR = base_lr * decay^(num_layers - i)
            # (layer 0 = bottom, layer N-1 = top)
            layer_lr = base_lr * (lr_decay_factor ** (num_layers - i))
            layer_groups = _make_groups(
                layer.named_parameters(),
                lr=layer_lr,
                group_name=f"encoder_layer_{i}",
            )
            param_groups.extend(layer_groups)
            logger.debug(f"  encoder_layer_{i}: lr={layer_lr:.2e}")

    # --- Pooler: slightly below top LR ---
    # LR = base_lr * decay^1
    pooler_lr = base_lr * lr_decay_factor
    if hasattr(base_model, "pooler") and base_model.pooler is not None:
        pooler_groups = _make_groups(
            base_model.pooler.named_parameters(),
            lr=pooler_lr,
            group_name="pooler",
        )
        param_groups.extend(pooler_groups)
        logger.debug(f"  pooler: lr={pooler_lr:.2e}")

    # --- Classification head: full base_lr ---
    if hasattr(encoder, "classifier") and encoder.classifier is not None:
        clf_groups = _make_groups(
            encoder.classifier.named_parameters(),
            lr=base_lr,
            group_name="classifier",
        )
        param_groups.extend(clf_groups)
        logger.debug(f"  classifier: lr={base_lr:.2e}")

    # --- Catch any remaining parameters ---
    remaining = []
    for name, param in model.named_parameters():
        if id(param) not in added_params and param.requires_grad:
            remaining.append(param)
            added_params.add(id(param))

    if remaining:
        param_groups.append({
            "params": remaining,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "remaining",
        })
        logger.debug(f"  remaining params: {len(remaining)}, lr={base_lr:.2e}")

    # Summary
    total_params = sum(len(g["params"]) for g in param_groups)
    logger.info(
        f"Layer-wise optimizer: {len(param_groups)} groups, {total_params} param tensors"
    )

    return param_groups


def get_layer_wise_optimizer(
    model: nn.Module,
    base_lr: float,
    lr_decay_factor: float = 0.9,
    weight_decay: float = 0.01,
    adam_epsilon: float = 1e-8,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
) -> AdamW:
    """
    Build an AdamW optimizer with layer-wise learning rate decay.

    Each transformer layer receives a different LR:
      LR(layer_i) = base_lr * lr_decay_factor^(num_layers - i)

    where layer 0 is the bottom (closest to embeddings) and
    layer num_layers-1 is the top (closest to the classifier).

    The classifier head always receives base_lr (no decay).
    Embeddings receive base_lr * decay^num_layers (maximum decay).

    No weight decay is applied to bias parameters and LayerNorm weights.

    Args:
        model: The sentiment classifier model
        base_lr: Learning rate for the top layer / classifier
        lr_decay_factor: Multiplicative decay per layer (0 < factor <= 1.0)
        weight_decay: L2 regularization coefficient
        adam_epsilon: Adam epsilon for numerical stability
        adam_beta1: Adam beta1 (momentum)
        adam_beta2: Adam beta2 (RMS)

    Returns:
        Configured AdamW optimizer instance
    """
    assert 0.0 < lr_decay_factor <= 1.0, "lr_decay_factor must be in (0, 1]"
    assert base_lr > 0, "base_lr must be positive"

    param_groups = _get_bert_layer_groups(
        model=model,
        base_lr=base_lr,
        lr_decay_factor=lr_decay_factor,
        weight_decay=weight_decay,
    )

    optimizer = AdamW(
        param_groups,
        lr=base_lr,  # default lr (overridden per group)
        betas=(adam_beta1, adam_beta2),
        eps=adam_epsilon,
        weight_decay=weight_decay,
    )

    logger.info(
        f"AdamW optimizer created with {len(optimizer.param_groups)} param groups"
    )
    return optimizer


def log_lr_schedule(optimizer: AdamW) -> None:
    """Log the learning rate for each parameter group."""
    for i, group in enumerate(optimizer.param_groups):
        name = group.get("name", f"group_{i}")
        logger.info(f"  [{i}] {name}: lr={group['lr']:.3e}, wd={group.get('weight_decay', 0):.4f}")
