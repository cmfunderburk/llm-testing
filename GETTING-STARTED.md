# Getting Started

Quick guide to setting up and running your first experiments.

## Prerequisites

- Python 3.11+
- NVIDIA GPU with 16GB+ VRAM
- CUDA toolkit installed
- ~20GB disk space for models

## Setup

### 1. Clone and Enter Directory

```bash
cd /home/cmf/Work/llm-testing
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install unsloth transformers trl datasets torch matplotlib
```

Or if a requirements.txt exists:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "from experiments.attention import extract_attention; print('Setup OK')"
```

## First Experiment

Start with a simple QLoRA fine-tuning run:

```bash
python -m experiments.fine_tuning.basic_qlora
```

This will:
1. Load Qwen2.5-7B-Instruct (4-bit quantized)
2. Fine-tune on a small dataset
3. Show training metrics

Watch for:
- Initial loss value
- Loss curve shape
- Final loss value

## Recommended Path

### Week 1: Track A (Fine-Tuning)

```bash
# 1. Basic training to see loss curves
python -m experiments.fine_tuning.basic_qlora

# 2. Compare learning rates
python -m experiments.learning_rate.experiment

# 3. Read the self-assessment
cat docs/concepts/training-dynamics-self-assessment.md
```

### Week 2: Track B (Attention)

```bash
# 1. Extract and visualize attention
python -m experiments.attention.compare_experiment

# 2. Read the self-assessment
cat docs/concepts/attention-self-assessment.md
```

### Week 3: Track C (Representations)

```bash
# 1. Analyze activations
python -m experiments.probing.run_analysis

# 2. Read the self-assessment
cat docs/concepts/representations-self-assessment.md
```

### Week 4: Track D (Paper)

```bash
# 1. Read the paper annotation
cat docs/papers/bayesian-geometry-attention.md

# 2. Run claim reproduction
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

## Key Commands

| Task | Command |
|------|---------|
| Run fine-tuning | `python -m experiments.fine_tuning.basic_qlora` |
| LR comparison | `python -m experiments.learning_rate.experiment` |
| LoRA rank test | `python -m experiments.lora_rank.experiment` |
| Forgetting test | `python -m experiments.forgetting.experiment` |
| Attention analysis | `python -m experiments.attention.compare_experiment` |
| Activation analysis | `python -m experiments.probing.run_analysis` |
| Paper reproduction | `python -m experiments.paper_reproduction.bayesian_geometry.experiment` |

## Output Locations

Experiments save outputs to:
```
outputs/
├── fine_tuning/          # Training logs, checkpoints
├── learning_rate/        # LR comparison results
├── lora_rank/            # Rank comparison results
├── attention/            # Attention visualizations
├── representation_analysis/  # Activation statistics
└── paper_reproduction/   # Claim test results
```

## Troubleshooting

### Out of Memory

Reduce batch size in experiment CONFIG:
```python
CONFIG = {
    "per_device_train_batch_size": 1,  # Reduce from 2
    ...
}
```

### CUDA Not Found

Ensure CUDA is installed and visible:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Import Errors

Ensure you're in the project root and venv is activated:
```bash
cd /home/cmf/Work/llm-testing
source .venv/bin/activate
```

## Learning Tips

1. **Read the hypothesis first**: Each experiment file has a HYPOTHESIS section. Read it before running.

2. **Predict before observing**: What do you expect to happen? Write it down.

3. **Update your understanding**: After running, compare results to predictions.

4. **Track questions**: Add new questions to QUESTIONS.md as they arise.

5. **Write self-assessments**: The docs/concepts/ files are templates. Fill them in with your own understanding.

## Next Steps

After running experiments:

1. Review outputs in `outputs/` directory
2. Update experiment results sections in the code
3. Fill in QUESTIONS.md with answers
4. Read [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md) for the full learning journey

## Need Help?

- Check experiment docstrings for detailed methodology
- Read concept docs in `docs/concepts/`
- Review the paper annotation in `docs/papers/`
