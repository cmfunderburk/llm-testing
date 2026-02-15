# LLM Learning Lab

A hands-on learning environment to build deep intuition about how LLMs work through instrumented experiments — from pretraining GPT models from scratch to fine-tuning, attention visualization, and representation probing.

## Purpose

This is a **learning vehicle**, not a production pipeline. The goal is to build mental models deep enough to:
- Read research papers fluently
- Understand training dynamics and inference
- Make informed decisions about fine-tuning approaches
- Develop experimental intuition about LLM behavior

## Quick Start

```bash
# Install dependencies
uv sync

# Launch the dashboard (API server + React frontend)
./run-dashboard.sh

# Open http://localhost:5173 and navigate to Pretraining
```

Or run pretraining directly from the CLI:

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

See [GETTING-STARTED.md](GETTING-STARTED.md) for detailed setup and [TRAINING-GUIDE.md](TRAINING-GUIDE.md) for a full pretraining walkthrough.

## Structure

```
llm-testing/
├── experiments/
│   ├── pretraining/             # GPT pretraining from scratch
│   │   ├── train.py             # Main training loop
│   │   ├── model.py             # GPT architecture (educational)
│   │   ├── config.py            # Model configs (nano/small/medium)
│   │   ├── data.py              # Dataset & DataLoader
│   │   ├── tokenizer.py         # BPE tokenization
│   │   ├── generate.py          # Text generation (temp, top-k, top-p)
│   │   ├── checkpoint.py        # Checkpoint save/load
│   │   ├── optim.py             # Optimizer factory (AdamW, 8-bit)
│   │   ├── corpus/              # Training corpora (~27GB available)
│   │   └── download_corpora.py  # Fetch additional datasets
│   ├── fine_tuning/             # QLoRA fine-tuning experiments
│   ├── learning_rate/           # LR sweep experiment
│   ├── lora_rank/               # Rank comparison experiment
│   ├── forgetting/              # Catastrophic forgetting tests
│   ├── attention/               # Attention extraction & visualization
│   ├── probing/                 # Activation analysis
│   └── paper_reproduction/      # Bayesian geometry claims
├── api/                         # FastAPI backend
│   ├── main.py                  # App factory, health endpoint
│   ├── routers/                 # Endpoints per track
│   └── services/                # Model management
├── dashboard/                   # React/TypeScript frontend
│   └── src/
│       ├── pages/               # Per-track pages
│       ├── components/          # UI components (charts, controls)
│       ├── context/             # WebSocket state management
│       └── types/               # TypeScript type definitions
├── docs/
│   ├── concepts/                # Self-assessments and concept docs
│   └── papers/                  # Paper annotations
├── pyproject.toml               # Dependencies (managed with uv)
├── run-dashboard.sh             # One-command dashboard launcher
├── TRAINING-GUIDE.md            # Pretraining walkthrough
├── VISION.md                    # Project vision
└── QUESTIONS.md                 # Central questions log
```

## Dashboard

The dashboard provides a real-time web UI for all learning tracks, with WebSocket streaming for live metrics during training.

```bash
./run-dashboard.sh
# Frontend: http://localhost:5173
# API docs: http://localhost:8000/docs
```

Features:
- **Pretraining**: Configure and launch GPT training runs, watch loss curves and sample generations update live, compare past runs, resume from checkpoints
- **Fine-tuning**: QLoRA experiments with real-time metrics
- **Attention**: Heatmap visualization of attention patterns
- **Probing**: Activation extraction and layer analysis

## Learning Tracks

### Pretraining (Primary Track)

Train GPT models from scratch to understand training dynamics firsthand:
- **Model configs**: nano (~10M), small (~50M), medium (~124M params)
- **Corpora**: verdict (quick test), TinyStories (recommended), WikiText-2, Shakespeare, Wikipedia GA, PG-19
- **Observe**: Loss curves, learning rate schedules, sample generation quality over time

```bash
# Quick pipeline test
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3

# Meaningful training run
python -m experiments.pretraining.train --config nano --corpus tinystories --epochs 1
```

See [TRAINING-GUIDE.md](TRAINING-GUIDE.md) for detailed walkthrough and learning exercises.

### Fine-Tuning Mechanics

Understand what happens during QLoRA fine-tuning of 7B models:
- **Loss curves**: What patterns mean and how to interpret them
- **Learning rate**: Effects of different LR choices
- **LoRA rank**: Trade-offs between expressiveness and efficiency
- **Forgetting**: Measuring and mitigating capability loss

```bash
python -m experiments.fine_tuning.loss_curve_analysis
python -m experiments.learning_rate.experiment
python -m experiments.lora_rank.experiment
python -m experiments.forgetting.experiment
```

### Attention Visualization

See what attention is doing:
- Hook-based extraction for Qwen architecture
- Heatmaps, head comparisons, layer progressions
- Base vs fine-tuned attention patterns

```bash
python -m experiments.attention.compare_experiment
```

### Representation Probing

Examine the residual stream:
- Activation extraction at any layer
- FFN vs attention contribution analysis
- Token position similarity across layers

```bash
python -m experiments.probing.run_analysis
```

### Paper Reproduction

Test claims from "The Bayesian Geometry of Transformer Attention":

```bash
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

## Key Documents

| Document | Purpose |
|----------|---------|
| [GETTING-STARTED.md](GETTING-STARTED.md) | Setup and first steps |
| [TRAINING-GUIDE.md](TRAINING-GUIDE.md) | Pretraining walkthrough and exercises |
| [VISION.md](VISION.md) | Project goals and motivation |
| [QUESTIONS.md](QUESTIONS.md) | Open questions and progress |

## Hardware

**Current**: NVIDIA RTX 5060 Ti 16GB (Blackwell)

Sufficient for:
- GPT pretraining up to ~124M parameters
- QLoRA fine-tuning of 7B models (4-bit quantized)
- Attention and activation extraction
- Small-scale interpretability experiments

## License

Learning project - use freely for educational purposes.
