"""
Task-Adaptive Pre-Training (TAPT) for multilingual sentiment analysis.

TAPT fine-tunes the language model backbone on unlabeled in-domain text
using masked language modeling (MLM) before the downstream sentiment
fine-tuning stage. This approach (Gururangan et al., 2020) provides an
average 8.3% F1 improvement over direct fine-tuning from the mBERT checkpoint.

Usage:
    config = TAPTConfig(output_dir="models/tapt-xlmr")
    trainer = TAPTTrainer(config)
    adapted_model_path = trainer.run_tapt(
        model_name="xlm-roberta-base",
        tapt_texts=my_corpus_texts,
    )
    # Use adapted_model_path as model_name in XLMRConfig
"""

from __future__ import annotations

import logging
import os
import random
from typing import Dict, List, Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.training.config import TAPTConfig

logger = logging.getLogger(__name__)


class TAPTTrainer:
    """
    Runs Task-Adaptive Pre-Training (TAPT) via masked language modeling.

    Steps:
    1. Load unlabeled in-domain text (e.g., review/sentiment corpus)
    2. Tokenize and create MLM training examples
    3. Fine-tune the backbone model with MLM objective
    4. Save the adapted model for downstream fine-tuning
    """

    def __init__(self, config: TAPTConfig) -> None:
        """
        Args:
            config: TAPT configuration dataclass
        """
        self.config = config
        self._tokenizer: Optional[AutoTokenizer] = None

    def prepare_tapt_dataset(
        self,
        texts: List[str],
        tokenizer: AutoTokenizer,
        mlm_probability: Optional[float] = None,
    ) -> Dataset:
        """
        Create a tokenized MLM dataset from raw unlabeled texts.

        Filters out texts that are too short, shuffles, and tokenizes
        without padding (dynamic padding applied in DataCollatorForLanguageModeling).

        Args:
            texts: List of raw text strings for TAPT corpus
            tokenizer: HuggingFace tokenizer
            mlm_probability: Override config.mlm_probability if provided

        Returns:
            HuggingFace Dataset with 'input_ids' and 'attention_mask' columns
        """
        mlm_prob = mlm_probability or self.config.mlm_probability

        # Filter short texts
        min_len = self.config.min_text_length
        filtered = [t.strip() for t in texts if isinstance(t, str) and len(t.strip()) >= min_len]
        logger.info(f"TAPT corpus: {len(texts)} raw → {len(filtered)} after filtering")

        # Optional cap
        if self.config.max_texts is not None and len(filtered) > self.config.max_texts:
            random.seed(self.config.seed)
            random.shuffle(filtered)
            filtered = filtered[: self.config.max_texts]
            logger.info(f"Capped TAPT corpus to {len(filtered)} examples")

        if not filtered:
            raise ValueError("TAPT corpus is empty after filtering. Check min_text_length.")

        # Create HF dataset
        raw_dataset = Dataset.from_dict({"text": filtered})

        def _tokenize_batch(examples: Dict[str, List]) -> Dict[str, List]:
            return tokenizer(
                examples["text"],
                padding=False,
                truncation=True,
                max_length=self.config.max_length,
                return_special_tokens_mask=True,
            )

        tokenized = raw_dataset.map(
            _tokenize_batch,
            batched=True,
            batch_size=1000,
            remove_columns=["text"],
            desc="Tokenizing TAPT corpus",
        )

        # Remove examples that are too short after tokenization
        tokenized = tokenized.filter(
            lambda x: len(x["input_ids"]) >= 8,
            desc="Filtering short tokenized sequences",
        )

        logger.info(f"TAPT tokenized dataset: {len(tokenized)} examples")
        return tokenized

    def run_tapt(
        self,
        model_name: str,
        tapt_texts: List[str],
        output_dir: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> str:
        """
        Run the full TAPT pre-training pipeline:
        1. Load model + tokenizer
        2. Prepare MLM dataset from tapt_texts
        3. Train with MLM objective
        4. Save adapted model to output_dir

        Args:
            model_name: HuggingFace model name or local path (base model to adapt)
            tapt_texts: List of in-domain text strings for MLM pre-training
            output_dir: Where to save the adapted model (overrides config)
            cache_dir: HuggingFace cache directory

        Returns:
            Path to the saved adapted model directory
        """
        save_dir = output_dir or self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)

        logger.info(f"Starting TAPT on model: {model_name}")
        logger.info(f"  MLM probability: {self.config.mlm_probability}")
        logger.info(f"  TAPT epochs: {self.config.tapt_epochs}")
        logger.info(f"  TAPT lr: {self.config.tapt_lr}")

        # Load tokenizer and base model for MLM
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

        self._tokenizer = tokenizer

        # Prepare dataset
        tapt_dataset = self.prepare_tapt_dataset(
            texts=tapt_texts,
            tokenizer=tokenizer,
            mlm_probability=self.config.mlm_probability,
        )

        # Split into train/eval (95/5)
        split = tapt_dataset.train_test_split(test_size=0.05, seed=self.config.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

        logger.info(f"TAPT train: {len(train_dataset)}, eval: {len(eval_dataset)}")

        # MLM data collator handles dynamic masking
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.config.mlm_probability,
            pad_to_multiple_of=8,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=save_dir,
            num_train_epochs=self.config.tapt_epochs,
            per_device_train_batch_size=self.config.per_device_batch_size,
            per_device_eval_batch_size=self.config.per_device_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.tapt_lr,
            weight_decay=self.config.weight_decay,
            adam_epsilon=self.config.adam_epsilon,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
            max_grad_norm=self.config.max_grad_norm,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type="cosine",
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.config.seed,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to="none",  # No W&B for TAPT
        )

        # HF Trainer for MLM
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        logger.info("Starting TAPT training...")
        train_result = trainer.train()

        # Log final perplexity
        eval_results = trainer.evaluate()
        import math
        perplexity = math.exp(eval_results.get("eval_loss", 0))
        logger.info(f"TAPT final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
        logger.info(f"TAPT perplexity: {perplexity:.2f}")

        # Save adapted model and tokenizer
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)

        # Save training metrics
        metrics = train_result.metrics
        metrics.update(eval_results)
        metrics["tapt_perplexity"] = perplexity
        trainer.log_metrics("tapt", metrics)
        trainer.save_metrics("tapt", metrics)

        logger.info(f"TAPT completed. Adapted model saved to: {save_dir}")
        return save_dir

    @staticmethod
    def load_tapt_texts_from_dataset(
        languages: Optional[List[str]] = None,
        max_per_language: int = 10000,
        cache_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Build a TAPT corpus from XNLI training data.

        Extracts premise sentences from the XNLI dataset to create
        an in-domain MLM corpus representing the target text distribution.

        Args:
            languages: Languages to extract text from; defaults to TRAIN_LANGUAGES
            max_per_language: Maximum number of texts per language
            cache_dir: HuggingFace cache directory

        Returns:
            List of raw text strings for TAPT training
        """
        from datasets import load_dataset as hf_load_dataset
        from src.data.dataset_loader import TRAIN_LANGUAGES

        if languages is None:
            languages = TRAIN_LANGUAGES

        all_texts: List[str] = []

        for lang in languages:
            logger.info(f"Extracting TAPT texts for {lang}...")
            try:
                ds = hf_load_dataset(
                    "xnli",
                    lang,
                    split="train",
                    cache_dir=cache_dir,
                    trust_remote_code=True,
                )
                # Extract premise sentences as in-domain text
                texts = [
                    example["premise"]
                    for example in ds
                    if isinstance(example.get("premise"), str) and len(example["premise"]) >= 20
                ]
                # Random sample if too many
                if len(texts) > max_per_language:
                    random.shuffle(texts)
                    texts = texts[:max_per_language]

                all_texts.extend(texts)
                logger.info(f"  {lang}: {len(texts)} TAPT texts")
            except Exception as e:
                logger.warning(f"  Failed to load TAPT texts for {lang}: {e}")

        logger.info(f"Total TAPT corpus size: {len(all_texts)} texts")
        return all_texts
