# LLM Learning Lab

A hands-on learning environment to build deep intuition about how LLMs work through instrumented experiments.

## Purpose

This is a **learning vehicle**, not a production pipeline. The goal is to build mental models deep enough to:
- Read research papers fluently
- Understand training dynamics and inference
- Make informed decisions about fine-tuning approaches
- Develop experimental intuition about LLM behavior

See [VISION.md](VISION.md) for the full vision statement.

## Structure

```
llm-testing/
├── experiments/           # Instrumented learning experiments
│   ├── fine_tuning/      # Track A: Training mechanics
│   ├── attention/        # Track B: Attention visualization
│   ├── probing/          # Track C: Representation analysis
│   └── paper_reproduction/ # Track D: Paper claim testing
├── docs/
│   └── concepts/         # Structured concept documentation
├── qwen-finetune/        # Original working code (reference)
├── QUESTIONS.md          # Central questions log
├── VISION.md             # Project vision
└── PRD-LLM-LEARNING-LAB.json  # Implementation roadmap
```

## Getting Started

### Prerequisites
- Python 3.11+
- NVIDIA GPU with 16GB+ VRAM (tested on RTX 5060 Ti)
- CUDA toolkit

### Setup
```bash
cd experiments
python -m venv .venv
source .venv/bin/activate
pip install unsloth transformers trl datasets torch
```

### Run First Experiment
```bash
# Basic QLoRA fine-tuning with instrumentation
python -m experiments.fine_tuning.basic_qlora
```

## Learning Tracks

### Track A: Fine-Tuning Mechanics
Understand what happens during training:
- Loss curves and their meaning
- Learning rate effects
- LoRA rank trade-offs
- Catastrophic forgetting

### Track B: Attention Visualization
See what attention is doing:
- Pattern visualization across layers
- How attention changes during training
- Base vs fine-tuned comparisons

### Track C: Representation Probing
Examine the residual stream:
- Activation extraction
- Layer-by-layer transformations
- Feature analysis

### Track D: Paper Reproduction
Test specific claims from papers:
- Target: "The Bayesian Geometry of Transformer Attention"
- Read, summarize, and reproduce claims

## Hardware

Current setup: NVIDIA RTX 5060 Ti 16GB (Blackwell)

Sufficient for:
- QLoRA fine-tuning of 7B models
- Small-scale interpretability experiments
- Quantized model inference

Hardware investment decision is an **output** of this learning process.

## Success Criteria

1. **Mental model formed**: Can explain transformer architecture without referencing materials
2. **Paper-reading fluency**: Can follow methodology in research papers
3. **Experimental intuition**: Can hypothesize experiment outcomes and understand surprises
