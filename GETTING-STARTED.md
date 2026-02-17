# Getting Started

Practical setup and first-run flow for the current LLM Learning Lab.

## Prerequisites

- Python 3.12+ (see `.python-version`)
- NVIDIA GPU (16GB VRAM recommended for the full dashboard experience)
- CUDA toolkit + working NVIDIA driver
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ (dashboard frontend)
- Disk:
  - ~20GB for typical experimentation
  - significantly more if you plan to pull full PG-19 variants

## Setup

### 1. Install dependencies

```bash
cd /home/cmf/Work/llm-testing
uv sync
```

This creates `.venv` and installs Python dependencies from `pyproject.toml`.

### 2. Verify CUDA visibility

```bash
source .venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Launch the unified dashboard

```bash
./run-dashboard.sh
```

Open the frontend URL printed by the launcher (default `http://127.0.0.1:5173`, auto-fallback if occupied).

## First 15 Minutes

### 1. Explore MicroGPT review mode (educational track)

In the sidebar, open **MicroGPT**:

- Canonical source: `misc/microgpt.py`
- Companion guide: `docs/microgpt_line_by_line.md`

Use the section toggles to review line ranges side-by-side with explanation and math context.

### 2. Run a pretraining smoke test

In **Pretraining**:

- Model: `nano`
- Corpus: `verdict`
- Epochs: `3`

Start the run and watch live loss and generation updates.

### 3. Pull a meaningful corpus

```bash
# List options
python -m experiments.pretraining.download_corpora --list

# Recommended first real dataset
python -m experiments.pretraining.download_corpora tinystories
```

Then run `nano + tinystories` in the dashboard (or CLI) for a longer signal-bearing run.

## CLI Alternative (No Dashboard)

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

## Key Commands

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Start dashboard | `./run-dashboard.sh` |
| Dashboard options | `./run-dashboard.sh --help` |
| Pretraining smoke test (CLI) | `python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3` |
| List downloadable corpora | `python -m experiments.pretraining.download_corpora --list` |
| Download TinyStories | `python -m experiments.pretraining.download_corpora tinystories` |
| Fine-tuning experiment | `python -m experiments.fine_tuning.basic_qlora` |
| Attention comparison | `python -m experiments.attention.compare_experiment` |
| Probing analysis | `python -m experiments.probing.run_analysis` |

## Output Locations

Common generated artifacts:

```text
outputs/
├── pretraining/
│   ├── runs/                       # persisted run metadata and metrics history
│   └── <config>_<corpus>_b*_ctx*/ # checkpoint directories
├── fine_tuning/
│   ├── runs/                       # fine-tuning run outputs
│   └── adapters/                   # adapter checkpoints surfaced by API
├── learning_rate_exploration/
├── lora_rank_comparison/
├── forgetting_test/
├── representation_analysis/
└── paper_reproduction/
```

## Troubleshooting

### Out of memory

- Start with `nano`
- Reduce batch size
- Reduce context length
- In dashboard pretraining controls, switch to memory-oriented settings:
  - `attention_impl=sdpa`
  - `precision=bf16` (if supported)
  - `gradient_checkpointing=true`

### CUDA not detected

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Dashboard won’t start

- Ensure `.venv` exists (`uv sync`)
- Ensure Node is installed (`node -v`)
- Re-run `./run-dashboard.sh` (it installs `dashboard/node_modules` automatically if missing)

## Next Steps

1. Follow [TRAINING-GUIDE.md](TRAINING-GUIDE.md) for the expanded pretraining workflow.
2. Use run history overlays in **Pretraining** to compare two runs directly.
3. Move to **Attention** and **Probing** once you have checkpoints to inspect.
4. Track new hypotheses in [QUESTIONS.md](QUESTIONS.md).
