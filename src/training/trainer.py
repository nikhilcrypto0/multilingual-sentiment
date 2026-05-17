"""
SentimentTrainer: wraps HuggingFace Trainer with layer-wise LR, FP16,
gradient accumulation, cosine annealing, W&B tracking, and per-language
evaluation metrics.
"""

from __future__ import annotations

import dataclasses
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.models.base_classifier import BaseSentimentClassifier
from src.training.config import TrainingConfig
from src.training.layer_wise_lr import get_layer_wise_optimizer

logger = logging.getLogger(__name__)


def _build_compute_metrics(num_labels: int = 3):
    """
    Build a compute_metrics function for the HuggingFace Trainer.

    Returns a callable that accepts EvalPrediction and returns a dict
    with 'accuracy', 'f1', 'precision', 'recall'.
    """
    import evaluate

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    average = "macro" if num_labels > 2 else "binary"

    def compute_metrics(eval_pred) -> Dict[str, float]:
        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        acc = accuracy_metric.compute(predictions=predictions, references=labels)
        f1 = f1_metric.compute(
            predictions=predictions, references=labels, average=average
        )
        prec = precision_metric.compute(
            predictions=predictions,
            references=labels,
            average=average,
            zero_division=0,
        )
        rec = recall_metric.compute(
            predictions=predictions,
            references=labels,
            average=average,
            zero_division=0,
        )

        return {
            "accuracy": acc["accuracy"],
            "f1": f1["f1"],
            "precision": prec["precision"],
            "recall": rec["recall"],
        }

    return compute_metrics


class SentimentTrainer:
    """
    High-level trainer for multilingual sentiment classifiers.

    Wraps the HuggingFace Trainer with:
      - Layer-wise AdamW optimizer (lower LR for bottom layers)
      - FP16 mixed-precision training
      - Gradient accumulation (effective batch ~128)
      - Cosine LR annealing with linear warmup
      - W&B experiment tracking
      - Per-language evaluation

    Example:
        config = XLMRConfig(output_dir="models/xlmr-v1")
        model = XLMRClassifier(num_labels=3)
        trainer = SentimentTrainer(config, model, train_ds, eval_ds)
        results = trainer.train()
    """

    def __init__(
        self,
        config: TrainingConfig,
        model: BaseSentimentClassifier,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        tokenizer: Optional[Any] = None,
    ) -> None:
        """
        Args:
            config: Training configuration dataclass
            model: Initialized sentiment classifier (MBertClassifier or XLMRClassifier)
            train_dataset: Tokenized training dataset
            eval_dataset: Tokenized evaluation dataset
            tokenizer: HuggingFace tokenizer (used for saving)
        """
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        set_seed(config.seed)

        self._trainer: Optional[Trainer] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None

    def _init_wandb(self) -> None:
        """Initialize W&B run with full config."""
        if not self.config.use_wandb:
            return
        try:
            import wandb
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                name=self.config.run_name,
                config=dataclasses.asdict(self.config),
                tags=[
                    self.config.model_name.split("/")[-1],
                    f"labels={self.config.num_labels}",
                ],
                reinit=True,
            )
            logger.info(f"W&B initialized: project={self.config.wandb_project}")
        except Exception as e:
            logger.warning(f"W&B init failed: {e}. Continuing without W&B.")
            self.config.use_wandb = False
            self.config.report_to = "none"

    def build_trainer(self) -> Trainer:
        """
        Construct and return a HuggingFace Trainer with all config applied.

        Sets up:
          - TrainingArguments from config (FP16, grad accumulation, cosine schedule)
          - Layer-wise AdamW optimizer + cosine LR scheduler
          - compute_metrics for F1/accuracy
          - EarlyStopping callback (patience=3)

        Returns:
            Configured HuggingFace Trainer instance
        """
        self._init_wandb()

        # Build TrainingArguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            run_name=self.config.run_name,

            # Batch & accumulation
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

            # Mixed precision
            fp16=self.config.fp16,
            bf16=self.config.bf16,

            # Optimization
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            adam_epsilon=self.config.adam_epsilon,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            max_grad_norm=self.config.max_grad_norm,
            optim=self.config.optim,

            # Schedule
            num_train_epochs=self.config.num_train_epochs,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,

            # Eval & Logging
            eval_strategy=self.config.eval_strategy,
            eval_steps=self.config.eval_steps,
            logging_dir=self.config.logging_dir,
            logging_steps=self.config.logging_steps,
            logging_first_step=True,

            # Saving
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=self.config.greater_is_better,

            # Reproducibility
            seed=self.config.seed,
            data_seed=self.config.data_seed,

            # Efficiency
            dataloader_num_workers=self.config.dataloader_num_workers,
            dataloader_pin_memory=self.config.dataloader_pin_memory,
            gradient_checkpointing=self.config.gradient_checkpointing,

            # Reporting
            report_to=self.config.report_to if self.config.use_wandb else "none",

            # Hub
            push_to_hub=self.config.push_to_hub,
        )

        # Build layer-wise AdamW optimizer
        self._optimizer = get_layer_wise_optimizer(
            model=self.model,
            base_lr=self.config.learning_rate,
            lr_decay_factor=self.config.layer_wise_lr_decay,
            weight_decay=self.config.weight_decay,
            adam_epsilon=self.config.adam_epsilon,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
        )

        compute_metrics = _build_compute_metrics(num_labels=self.config.num_labels)

        self._trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=compute_metrics,
            tokenizer=self.tokenizer,
            optimizers=(self._optimizer, None),  # None → HF builds scheduler
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
            ],
        )

        logger.info("HuggingFace Trainer built successfully")
        return self._trainer

    def train(self) -> Dict[str, Any]:
        """
        Run training.

        Returns:
            Training metrics dict with loss, f1, etc.
        """
        if self._trainer is None:
            self.build_trainer()

        logger.info(
            f"Starting training: {self.config.num_train_epochs} epochs, "
            f"effective_batch={self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps}"
        )

        train_result = self._trainer.train()
        metrics = train_result.metrics

        # Log and save
        self._trainer.log_metrics("train", metrics)
        self._trainer.save_metrics("train", metrics)
        self._trainer.save_state()

        logger.info(f"Training complete. Metrics: {metrics}")
        return metrics

    def evaluate_per_language(
        self,
        test_datasets_by_lang: Dict[str, Dataset],
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model on per-language test sets.

        Args:
            test_datasets_by_lang: Dict mapping language code → tokenized dataset

        Returns:
            Dict mapping language code → metrics dict (f1, accuracy, etc.)
        """
        if self._trainer is None:
            self.build_trainer()

        results: Dict[str, Dict[str, float]] = {}

        for lang, dataset in test_datasets_by_lang.items():
            logger.info(f"Evaluating on language: {lang} ({len(dataset)} examples)")
            try:
                metrics = self._trainer.evaluate(eval_dataset=dataset)
                # Remove eval_ prefix for cleaner output
                clean_metrics = {
                    k.replace("eval_", ""): v for k, v in metrics.items()
                }
                results[lang] = clean_metrics
                logger.info(f"  {lang}: F1={clean_metrics.get('f1', 'N/A'):.4f}, "
                           f"Acc={clean_metrics.get('accuracy', 'N/A'):.4f}")
            except Exception as e:
                logger.error(f"Evaluation failed for {lang}: {e}")
                results[lang] = {"error": str(e)}

        # Compute macro-average across languages
        valid_results = {k: v for k, v in results.items() if "error" not in v}
        if valid_results:
            avg_f1 = np.mean([v.get("f1", 0) for v in valid_results.values()])
            avg_acc = np.mean([v.get("accuracy", 0) for v in valid_results.values()])
            results["macro_avg"] = {"f1": avg_f1, "accuracy": avg_acc}
            logger.info(f"Macro-avg F1: {avg_f1:.4f}")

        return results

    def save_model(self, path: str) -> None:
        """
        Save the trained model, tokenizer, and config to path.

        Args:
            path: Directory path to save model artifacts
        """
        os.makedirs(path, exist_ok=True)

        if self._trainer is not None:
            self._trainer.save_model(path)
        else:
            # Save model directly
            if hasattr(self.model, "save_pretrained"):
                self.model.save_pretrained(path)
            else:
                torch.save(self.model.state_dict(), os.path.join(path, "pytorch_model.bin"))

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(path)

        # Save training config
        import json
        config_path = os.path.join(path, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(dataclasses.asdict(self.config), f, indent=2)

        logger.info(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from a saved checkpoint.

        Args:
            path: Directory path containing model artifacts
        """
        if hasattr(self.model, "encoder"):
            from transformers import AutoModelForSequenceClassification
            self.model.encoder = AutoModelForSequenceClassification.from_pretrained(path)
        else:
            state_dict_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(state_dict_path):
                self.model.load_state_dict(torch.load(state_dict_path, map_location="cpu"))

        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(path)
            except Exception:
                pass

        logger.info(f"Model loaded from {path}")
