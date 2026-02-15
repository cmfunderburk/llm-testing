# Getting Started

Quick guide to setting up and running your first experiments.

## Prerequisites

- Python 3.12+ (see `.python-version`)
- NVIDIA GPU with 16GB+ VRAM
- CUDA toolkit installed
- [uv](https://docs.astral.sh/uv/) package manager
- Node.js 18+ (for the dashboard)
- ~20GB disk space for models and corpora

## Setup

### 1. Install Dependencies

```bash
cd /home/cmf/Work/llm-testing
uv sync
```

This creates the `.venv` virtual environment and installs all Python dependencies from `pyproject.toml`.

### 2. Verify Installation

```bash
source .venv/bin/activate
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Download Training Corpora (Optional)

The repo includes small corpora (verdict, tiny) for quick testing. For meaningful pretraining experiments, download larger datasets:

```bash
# See what's available
python -m experiments.pretraining.download_corpora --list

# Download recommended dataset
python -m experiments.pretraining.download_corpora tinystories

# Or download everything
python -m experiments.pretraining.download_corpora --all
```

## First Experiment: Pretraining a GPT

The fastest way to see something interesting is to pretrain a small GPT model.

### Option A: Via the Dashboard (Recommended)

```bash
./run-dashboard.sh
```

Open http://localhost:5173, navigate to **Pretraining**, set config to `nano`, corpus to `verdict`, epochs to `3`, and hit Start. Watch the loss curve and sample generations update in real time.

### Option B: Via the CLI

```bash
source .venv/bin/activate
python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3
```

**What to observe**:
- Loss drops rapidly (the model is memorizing a small text)
- Generated samples go from gibberish to recognizable fragments
- Final loss should be very low (<0.5)

For a more meaningful run, try TinyStories:

```bash
python -m experiments.pretraining.train --config nano --corpus tinystories --epochs 1
```

See [TRAINING-GUIDE.md](TRAINING-GUIDE.md) for a full walkthrough with learning exercises.

## Other Experiments

### Fine-Tuning (QLoRA)

```bash
python -m experiments.fine_tuning.basic_qlora
```

Loads Qwen2.5-7B-Instruct (4-bit quantized) and fine-tunes on a small dataset.

### Attention Visualization

```bash
python -m experiments.attention.compare_experiment
```

### Representation Probing

```bash
python -m experiments.probing.run_analysis
```

### Paper Reproduction

```bash
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

## Key Commands

| Task | Command |
|------|---------|
| Install dependencies | `uv sync` |
| Launch dashboard | `./run-dashboard.sh` |
| Pretrain (CLI) | `python -m experiments.pretraining.train --config nano --corpus verdict --epochs 3` |
| Fine-tune | `python -m experiments.fine_tuning.basic_qlora` |
| LR comparison | `python -m experiments.learning_rate.experiment` |
| LoRA rank test | `python -m experiments.lora_rank.experiment` |
| Attention analysis | `python -m experiments.attention.compare_experiment` |
| Download corpora | `python -m experiments.pretraining.download_corpora --list` |

## Output Locations

Experiments save outputs to:
```
outputs/
├── pretraining/             # Checkpoints, run history
├── fine_tuning/             # Training logs, checkpoints
├── learning_rate/           # LR comparison results
├── lora_rank/               # Rank comparison results
├── attention/               # Attention visualizations
├── representation_analysis/ # Activation statistics
└── paper_reproduction/      # Claim test results
```

## Troubleshooting

### Out of Memory

- Use the `nano` config (smallest model)
- Reduce batch size in the dashboard or via CLI args
- Reduce context length

### CUDA Not Found

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors

Ensure you're in the project root with the venv activated:
```bash
cd /home/cmf/Work/llm-testing
source .venv/bin/activate
```

## Next Steps

1. Run through the [TRAINING-GUIDE.md](TRAINING-GUIDE.md) exercises
2. Explore the dashboard's run comparison features
3. Try different model configs and corpora
4. Move on to attention and probing experiments
5. Track questions in [QUESTIONS.md](QUESTIONS.md)
