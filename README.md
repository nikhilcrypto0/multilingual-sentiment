# Multilingual Sentiment Analysis

Fine-tuned mBERT and XLM-R classifiers for sentiment analysis across multiple languages, with a FastAPI inference server and AWS Lambda deployment support.

## Features

- **Two model architectures** — mBERT and XLM-RoBERTa, benchmarked head-to-head
- **Task-Adaptive Pre-Training (TAPT)** — domain-specific continued pre-training before fine-tuning
- **Layer-wise learning rate decay** — improves fine-tuning stability on transformer models
- **ONNX export + optimization** — reduced latency for production inference
- **FastAPI serving** — REST API with async inference endpoints
- **AWS Lambda handler** — serverless deployment via Mangum adapter
- **Evaluation suite** — per-language metrics, cross-lingual benchmarks

## Stack

| Layer | Technology |
|-------|-----------|
| Models | `transformers` (mBERT, XLM-R) |
| Training | PyTorch, Accelerate, W&B |
| Serving | FastAPI, Uvicorn |
| Optimization | ONNX Runtime, Optimum |
| Deployment | AWS Lambda (Mangum) |

## Project Structure

```
src/
├── models/       # mBERT and XLM-R classifier heads
├── training/     # Trainer, TAPT, layer-wise LR
├── data/         # Dataset loaders and preprocessing
├── evaluation/   # Metrics and cross-lingual benchmarks
└── serving/      # FastAPI app and Lambda handler
configs/          # Model and training hyperparameters
scripts/          # Training and benchmark entry points
```

## Quickstart

```bash
pip install -r requirements.txt

# Train XLM-R
python scripts/train_xlmr.py --config configs/xlmr_config.yaml

# Run benchmark
python scripts/run_benchmark.py

# Start inference server
uvicorn src.serving.api:app --reload
```

## API

```
POST /predict
{ "text": "This is amazing!", "lang": "en" }

→ { "label": "positive", "score": 0.97, "lang": "en" }
```
