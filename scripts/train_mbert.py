#!/usr/bin/env python3
"""
Train mBERT for multilingual sentiment analysis.

Usage:
    python scripts/train_mbert.py \
        --config configs/mbert_config.yaml \
        --output-dir models/mbert-finetuned

The script:
1. Loads and validates the YAML config
2. Loads combined XNLI + SST datasets for all 8 training languages
3. Tokenizes with mBERT tokenizer
4. Optionally runs TAPT if tapt.enabled=true in config
5. Fine-tunes with gradient accumulation, FP16, cosine LR, layer-wise decay
6. Evaluates per-language F1 on XNLI test set
7. Saves model + metrics report
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

# Ensure src/ is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune mBERT for multilingual sentiment analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mbert_config.yaml",
        help="Path to YAML training configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output_dir from config",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B tracking (overrides config)",
    )
    parser.add_argument(
        "--tapt",
        action="store_true",
        help="Run TAPT before fine-tuning (overrides config tapt.enabled)",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Override training languages (space-separated ISO codes)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable FP16 training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed",
    )
    return parser.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load and validate YAML configuration."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return cfg


def build_training_config(yaml_cfg: dict, args: argparse.Namespace):
    """Build MBertConfig dataclass from YAML + CLI overrides."""
    from src.training.config import MBertConfig

    training_cfg = yaml_cfg.get("training", {})
    model_cfg = yaml_cfg.get("model", {})
    data_cfg = yaml_cfg.get("data", {})
    wandb_cfg = yaml_cfg.get("wandb", {})
    tok_cfg = yaml_cfg.get("tokenizer", {})

    config = MBertConfig(
        model_name=model_cfg.get("name", "bert-base-multilingual-cased"),
        num_labels=model_cfg.get("num_labels", 3),
        max_length=tok_cfg.get("max_length", 128),
        per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 16),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        weight_decay=training_cfg.get("weight_decay", 0.01),
        adam_epsilon=float(training_cfg.get("adam_epsilon", 1e-8)),
        adam_beta1=training_cfg.get("adam_beta1", 0.9),
        adam_beta2=training_cfg.get("adam_beta2", 0.999),
        max_grad_norm=training_cfg.get("max_grad_norm", 1.0),
        num_train_epochs=training_cfg.get("num_train_epochs", 5),
        warmup_ratio=training_cfg.get("warmup_ratio", 0.1),
        lr_scheduler_type=training_cfg.get("lr_scheduler_type", "cosine"),
        layer_wise_lr_decay=training_cfg.get("layer_wise_lr_decay", 0.9),
        fp16=training_cfg.get("fp16", True),
        bf16=training_cfg.get("bf16", False),
        logging_steps=training_cfg.get("logging_steps", 100),
        eval_steps=training_cfg.get("eval_steps", 500),
        save_steps=training_cfg.get("save_steps", 1000),
        save_total_limit=training_cfg.get("save_total_limit", 3),
        seed=training_cfg.get("seed", 42),
        output_dir=training_cfg.get("output_dir", "models/mbert-finetuned"),
        logging_dir=training_cfg.get("logging_dir", "logs/mbert"),
        run_name=training_cfg.get("run_name", "mbert-sentiment-v1"),
        use_wandb=wandb_cfg.get("use_wandb", True),
        wandb_project=wandb_cfg.get("project", "multilingual-sentiment"),
        report_to=wandb_cfg.get("report_to", "wandb"),
        languages=data_cfg.get("languages", ["en", "de", "fr", "es", "zh", "ar", "sw", "ta"]),
        dataloader_num_workers=training_cfg.get("dataloader_num_workers", 4),
        gradient_checkpointing=training_cfg.get("gradient_checkpointing", False),
    )

    # Apply CLI overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.no_wandb:
        config.use_wandb = False
        config.report_to = "none"
    if args.languages:
        config.languages = args.languages
    if args.num_epochs is not None:
        config.num_train_epochs = args.num_epochs
    if args.fp16:
        config.fp16 = True
    if args.seed is not None:
        config.seed = args.seed

    return config


def run_tapt_if_enabled(yaml_cfg: dict, args: argparse.Namespace, config) -> str:
    """Run TAPT if requested, return adapted model path."""
    from src.training.tapt import TAPTTrainer
    from src.training.config import TAPTConfig

    tapt_cfg_yaml = yaml_cfg.get("tapt", {})
    tapt_enabled = tapt_cfg_yaml.get("enabled", False) or args.tapt

    if not tapt_enabled:
        return config.model_name

    logger.info("=" * 60)
    logger.info("RUNNING TASK-ADAPTIVE PRE-TRAINING (TAPT)")
    logger.info("=" * 60)

    tapt_config = TAPTConfig(
        mlm_probability=tapt_cfg_yaml.get("mlm_probability", 0.15),
        tapt_epochs=tapt_cfg_yaml.get("tapt_epochs", 3),
        tapt_lr=float(tapt_cfg_yaml.get("tapt_lr", 5e-5)),
        output_dir=tapt_cfg_yaml.get("tapt_output_dir", "models/mbert-tapt-adapted"),
        fp16=config.fp16,
        seed=config.seed,
    )

    # Collect TAPT corpus from XNLI
    tapt_texts = TAPTTrainer.load_tapt_texts_from_dataset(
        languages=config.languages,
        max_per_language=10000,
    )

    trainer = TAPTTrainer(tapt_config)
    adapted_path = trainer.run_tapt(
        model_name=config.model_name,
        tapt_texts=tapt_texts,
        output_dir=tapt_config.output_dir,
    )

    logger.info(f"TAPT complete. Adapted model at: {adapted_path}")
    return adapted_path


def main() -> None:
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)
    config = build_training_config(yaml_cfg, args)

    logger.info("=" * 60)
    logger.info(f"mBERT SENTIMENT TRAINING")
    logger.info(f"  Model:      {config.model_name}")
    logger.info(f"  Languages:  {config.languages}")
    logger.info(f"  Epochs:     {config.num_train_epochs}")
    logger.info(f"  Eff batch:  {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    logger.info(f"  LR:         {config.learning_rate:.2e}")
    logger.info(f"  Output:     {config.output_dir}")
    logger.info("=" * 60)

    # Optionally run TAPT first
    model_name_to_use = run_tapt_if_enabled(yaml_cfg, args, config)
    if model_name_to_use != config.model_name:
        logger.info(f"Using TAPT-adapted model: {model_name_to_use}")
        config.model_name = model_name_to_use

    # Load dataset
    logger.info("Loading datasets...")
    from src.data.dataset_loader import load_combined_dataset
    from src.data.preprocessing import MultilingualPreprocessor, create_tokenized_dataset

    data_cfg = yaml_cfg.get("data", {})
    dataset_dict = load_combined_dataset(
        languages=config.languages,
        include_zero_shot=data_cfg.get("include_zero_shot_eval", False),
        seed=config.seed,
    )

    logger.info(f"Dataset loaded: {list(dataset_dict.keys())}")
    for split, ds in dataset_dict.items():
        logger.info(f"  {split}: {len(ds)} examples")

    # Initialize model and tokenizer
    logger.info("Initializing mBERT model...")
    from src.models.mbert_classifier import MBertClassifier

    model = MBertClassifier(
        num_labels=config.num_labels,
        model_name=config.model_name,
        dropout_prob=yaml_cfg.get("model", {}).get("dropout_prob", 0.1),
        use_gradient_checkpointing=yaml_cfg.get("model", {}).get("use_gradient_checkpointing", False),
    )
    tokenizer = MBertClassifier.get_tokenizer(config.model_name)
    logger.info(f"Model: {model}")

    # Tokenize datasets
    logger.info("Tokenizing datasets...")
    preprocessor = MultilingualPreprocessor(lower_case=False)
    train_ds = preprocessor.preprocess_dataset(dataset_dict["train"])
    eval_ds = preprocessor.preprocess_dataset(dataset_dict["validation"])

    train_tokenized = create_tokenized_dataset(
        train_ds, tokenizer, max_length=config.max_length
    )
    eval_tokenized = create_tokenized_dataset(
        eval_ds, tokenizer, max_length=config.max_length
    )
    logger.info(
        f"Tokenized — train: {len(train_tokenized)}, eval: {len(eval_tokenized)}"
    )

    # Build and run trainer
    from src.training.trainer import SentimentTrainer

    trainer = SentimentTrainer(
        config=config,
        model=model,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        tokenizer=tokenizer,
    )
    trainer.build_trainer()

    logger.info("Starting training...")
    train_metrics = trainer.train()
    logger.info(f"Training metrics: {train_metrics}")

    # Per-language evaluation
    logger.info("Running per-language evaluation...")
    test_datasets_by_lang: dict = {}
    for lang in config.languages:
        lang_key = f"test_{lang}" if f"test_{lang}" in dataset_dict else "test"
        if lang_key == "test":
            from datasets import Dataset
            test_full = dataset_dict["test"]
            lang_test = test_full.filter(lambda x: x["language"] == lang)
            if len(lang_test) > 0:
                lang_tokenized = create_tokenized_dataset(
                    lang_test, tokenizer, max_length=config.max_length
                )
                test_datasets_by_lang[lang] = lang_tokenized

    per_lang_results = trainer.evaluate_per_language(test_datasets_by_lang)
    logger.info("Per-language results:")
    for lang, metrics in per_lang_results.items():
        if isinstance(metrics, dict) and "f1" in metrics:
            logger.info(f"  {lang}: F1={metrics['f1']:.4f}")

    # Save model
    trainer.save_model(config.output_dir)
    logger.info(f"Model saved to {config.output_dir}")

    # Save benchmark report
    eval_cfg = yaml_cfg.get("evaluation", {})
    report_path = eval_cfg.get("report_path", "reports/mbert_benchmark.json")
    os.makedirs(os.path.dirname(report_path) if os.path.dirname(report_path) else ".", exist_ok=True)

    import json
    report = {
        "model": config.model_name,
        "output_dir": config.output_dir,
        "train_metrics": train_metrics,
        "per_language_results": per_lang_results,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Report saved to {report_path}")

    logger.info("=" * 60)
    logger.info("mBERT training complete!")
    if "macro_avg" in per_lang_results:
        logger.info(
            f"Macro-avg F1: {per_lang_results['macro_avg'].get('f1', 'N/A'):.4f}"
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
