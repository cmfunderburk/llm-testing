# LLM Learning Lab

Hands-on environment for building deep intuition about LLM behavior through instrumented training, analysis, and educational code review.

This project is intentionally educational, not production infrastructure.

## Quick Start

```bash
# 1) Install Python dependencies and create .venv
uv sync

# 2) Launch unified dashboard (FastAPI + React)
./run-dashboard.sh
```

Open the frontend URL printed by the launcher (usually `http://127.0.0.1:5173`).

Or run pretraining directly from CLI:

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

## What Is New

- Pretraining has expanded into a full lab surface with:
  - Multiple model scales (`nano` through `xlarge`)
  - Broader corpus support (TinyStories, WikiText-2, Shakespeare, Wikipedia GA intros, PG-19 variants including doc-boundary and normalized forms)
  - VRAM estimation, checkpoint resume, manual save-now, run persistence, and run-to-run chart overlays in the dashboard
- A dedicated educational code review track now ships in the dashboard:
  - Canonical source: `misc/microgpt.py` (Karpathy-inspired microgpt "art project" style)
  - Companion guide: `docs/microgpt_line_by_line.md`
  - Served via `GET /api/education/microgpt` and rendered as a line-aligned reading workspace

## Learning Tracks

### MicroGPT (Canonical Review Surface)

Educational deep-read mode for the canonical artifact pair:

- `misc/microgpt.py`
- `docs/microgpt_line_by_line.md`

The dashboard pages this into guided line ranges and keeps source/doc alignment explicit.

### Pretraining (Primary Track)

Train GPT models from scratch and inspect live dynamics:

- Configs: `nano` (~10M), `small` (~50M), `medium` (~124M), `large` (~204M), `xlarge` (~355M)
- Real-time: loss, validation loss, LR, tokens/sec, generation samples, loading/tokenization progress
- Controls: optimizer selection, precision (`fp32`/`bf16`/`fp16`), attention backend (`manual`/`sdpa`), gradient checkpointing, tied embeddings, gradient accumulation
- Persistence: run metadata/history, checkpoint browser, manual checkpoint save, resume support

### Fine-Tuning

QLoRA training and adapter management for Qwen-family models with live status and checkpoint controls.

### Attention

Attention extraction and heatmap exploration for loaded models.

### Probing

Activation statistics, layer-diff views, and token activation inspection.

### Paper Reproduction and Focused Experiments

Supporting experiment tracks include learning-rate sweeps, LoRA-rank comparisons, forgetting tests, and targeted paper reproduction.

## Dashboard and API

`./run-dashboard.sh` starts:

- Frontend: `http://127.0.0.1:5173` (or next free local port)
- API docs: `http://127.0.0.1:8000/docs`

If an API is already healthy on port `8000`, the launcher attaches to it instead of starting a second backend.

Core backend routers:

- `/api/pretraining/*`
- `/api/fine-tuning/*`
- `/api/attention/*`
- `/api/probing/*`
- `/api/education/*`
- WebSockets: `/ws/training`, `/ws/fine-tuning`

## Project Structure

```text
llm-testing/
├── api/                           # Unified FastAPI backend
│   ├── main.py
│   ├── routers/                   # pretraining, fine-tuning, attention, probing, education
│   └── services/                  # shared model lifecycle management
├── dashboard/                     # React + TypeScript dashboard
│   └── src/
│       ├── pages/                 # track pages (incl. MicroGPT)
│       ├── components/            # controls, charts, visualizers
│       ├── context/               # global state + websocket handlers
│       └── types/                 # shared frontend types
├── experiments/
│   ├── pretraining/               # GPT model, training loop, corpora tools, checkpoints
│   ├── fine_tuning/               # QLoRA-focused experiments
│   ├── attention/                 # attention extraction/visualization scripts
│   ├── probing/                   # activation extraction/analysis scripts
│   ├── learning_rate/             # LR exploration experiments
│   ├── lora_rank/                 # LoRA rank comparison experiments
│   ├── forgetting/                # forgetting/capability regression tests
│   └── paper_reproduction/        # claim reproduction experiments
├── docs/
│   └── microgpt_line_by_line.md   # canonical MicroGPT companion guide
├── misc/
│   └── microgpt.py                # canonical educational source artifact
├── qwen-finetune/                 # standalone fine-tuning workflow sandbox
├── outputs/                       # generated artifacts (runs, checkpoints, adapters, analyses)
├── GETTING-STARTED.md
├── TRAINING-GUIDE.md
└── run-dashboard.sh
```

## Core Commands

```bash
# Launch dashboard
./run-dashboard.sh

# Dashboard launcher options
./run-dashboard.sh --help

# List downloadable pretraining corpora
python -m experiments.pretraining.download_corpora --list

# Download recommended dataset
python -m experiments.pretraining.download_corpora tinystories

# CLI pretraining smoke test
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

## Key Documents

- [GETTING-STARTED.md](GETTING-STARTED.md): setup + first run path
- [TRAINING-GUIDE.md](TRAINING-GUIDE.md): pretraining progression and interpretation
- [LOCAL-INFERENCE-MODEL-SIZING.md](LOCAL-INFERENCE-MODEL-SIZING.md): local model sizing notes
- [VISION.md](VISION.md): project vision
- [QUESTIONS.md](QUESTIONS.md): ongoing research/learning questions

## Hardware Note

Current reference system is an RTX 5060 Ti 16GB setup. The codebase includes VRAM estimators in both pretraining and fine-tuning surfaces to help map config choices to memory budget before launch.

## License

Educational project. Use freely for learning and experimentation.
