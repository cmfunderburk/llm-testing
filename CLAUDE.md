# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Learning Lab is a hands-on learning environment for building intuition about how LLMs work. It's a **learning vehicle, not a production pipeline**. The goal is to develop mental models deep enough to read research papers fluently and understand training dynamics.

Hardware context: NVIDIA RTX 5060 Ti 16GB (sufficient for QLoRA fine-tuning of 7B models in 4-bit).

## Running Experiments

All experiments run as Python modules from the project root with the virtual environment activated:

```bash
source .venv/bin/activate

# Track A: Fine-Tuning Mechanics
python -m experiments.fine_tuning.basic_qlora
python -m experiments.fine_tuning.loss_curve_analysis
python -m experiments.learning_rate.experiment
python -m experiments.lora_rank.experiment
python -m experiments.forgetting.experiment

# Track B: Attention Visualization
python -m experiments.attention.compare_experiment

# Track C: Representation Probing
python -m experiments.probing.run_analysis

# Track D: Paper Reproduction
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

This project uses **uv** for package management. Install dependencies with:
```bash
uv sync
```

To add new packages: `uv add <package>`

## Architecture

### Experiment Structure

Each experiment follows a standard template (`experiments/EXPERIMENT_TEMPLATE.py`):
- Docstring with HYPOTHESIS, METHODOLOGY, QUESTIONS TO ANSWER, RESULTS, LEARNINGS
- CONFIG dict for hyperparameters
- `setup()`, `run()`, `analyze()` functions
- `run_experiment()` as main entry point

Experiments are organized by learning track:
- `experiments/fine_tuning/` - QLoRA, loss curves
- `experiments/learning_rate/` - LR sweep comparisons
- `experiments/lora_rank/` - Rank trade-offs
- `experiments/forgetting/` - Catastrophic forgetting
- `experiments/attention/` - Attention extraction and visualization
- `experiments/probing/` - Residual stream activation analysis
- `experiments/paper_reproduction/` - Testing claims from papers

### Key Extraction Modules

**Attention (`experiments/attention/extract.py`):**
- `AttentionExtractor` - Hook-based extraction from transformer attention modules
- `extract_attention(model, tokenizer, text, layers)` - High-level API
- Returns `AttentionOutput` with shape `(batch, num_heads, seq_len, seq_len)`
- Supports Qwen/Llama and GPT2 architectures

**Activations (`experiments/probing/extract.py`):**
- `ActivationExtractor` - Captures residual stream at layer boundaries
- `extract_activations(model, tokenizer, text, layers, positions)`
- Positions: `pre_attn` (before attention), `post_ffn` (after FFN)
- `get_layer_diff()` computes attention vs FFN contribution

### Model Loading Pattern

Experiments use Unsloth's FastLanguageModel with specific import order:
```python
import psutil  # Must import before unsloth (CPU count caching bug)
import os
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"
from unsloth import FastLanguageModel
```

Standard model loading:
```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)
```

## Output Locations

Experiments save outputs to `outputs/<experiment_name>/`:
- `outputs/fine_tuning/` - Training logs, checkpoints
- `outputs/attention/` - Attention visualizations
- `outputs/representation_analysis/` - Activation statistics
- `outputs/paper_reproduction/` - Claim test results

## Key Documents

- `QUESTIONS.md` - Central questions log with open/answered tracking
- `docs/concepts/` - Self-assessment documents for each track
- `docs/papers/` - Paper annotations (e.g., bayesian-geometry-attention.md)
- `ralph/` - Design principles for autonomous iteration loops

## Dashboard

The unified dashboard provides a web UI for interactive experimentation across all learning tracks.

### Running the Dashboard

```bash
# Quick start (runs both API and frontend)
./run-dashboard.sh

# Or manually in separate terminals:
# Terminal 1: Start the API server
source .venv/bin/activate
uvicorn api.main:app --reload --port 8000

# Terminal 2: Start the frontend
cd dashboard && npm install && npm run dev
```

Dashboard URL: http://localhost:5173
API docs: http://localhost:8000/docs

### Dashboard Architecture

```
api/                        # Unified FastAPI backend
├── main.py                 # Application entry, mounts routers
├── config.py               # Server configuration
├── routers/
│   ├── pretraining.py      # GPT pretraining endpoints + WebSocket
│   ├── attention.py        # Attention extraction endpoints
│   └── probing.py          # Activation extraction endpoints
└── services/
    └── model_manager.py    # Lazy model loading for GPU memory

dashboard/                  # React + TypeScript frontend
├── src/
│   ├── pages/              # Track pages (Pretraining, Attention, Probing)
│   ├── components/
│   │   ├── layout/         # Header, Sidebar, Layout
│   │   ├── pretraining/    # Training controls, charts, metrics
│   │   ├── attention/      # Heatmap visualization
│   │   └── probing/        # Activation statistics
│   ├── context/            # TrainingContext for state management
│   └── hooks/              # useWebSocket for real-time metrics
```

### API Endpoints

- `GET /api/health` - Health check
- `POST /api/pretraining/start` - Start GPT training
- `POST /api/attention/extract` - Extract attention weights
- `POST /api/probing/extract` - Extract layer activations
- `WS /ws/training` - Real-time training metrics stream
