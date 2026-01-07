# LLM Learning Lab

A hands-on learning environment to build deep intuition about how LLMs work through instrumented experiments.

## Status

**All 6 phases complete** - Infrastructure ready, experiments designed, awaiting GPU execution.

| Track | Focus | Status |
|-------|-------|--------|
| A | Fine-Tuning Mechanics | Ready to run |
| B | Attention Visualization | Ready to run |
| C | Representation Probing | Ready to run |
| D | Paper Reproduction | Ready to run |

## Purpose

This is a **learning vehicle**, not a production pipeline. The goal is to build mental models deep enough to:
- Read research papers fluently
- Understand training dynamics and inference
- Make informed decisions about fine-tuning approaches
- Develop experimental intuition about LLM behavior

## Quick Start

```bash
# Setup
cd /home/cmf/Work/llm-testing
source .venv/bin/activate  # or create: python -m venv .venv && pip install -r requirements.txt

# Run an experiment
python -m experiments.fine_tuning.basic_qlora
```

See [GETTING-STARTED.md](GETTING-STARTED.md) for detailed setup instructions.

## Structure

```
llm-testing/
├── experiments/
│   ├── fine_tuning/          # Track A: Loss curves, LR, LoRA rank
│   ├── learning_rate/        # LR sweep experiment
│   ├── lora_rank/            # Rank comparison experiment
│   ├── forgetting/           # Catastrophic forgetting tests
│   ├── attention/            # Track B: Attention extraction & viz
│   ├── probing/              # Track C: Activation analysis
│   └── paper_reproduction/   # Track D: Bayesian geometry claims
├── docs/
│   ├── concepts/             # Self-assessments and concept docs
│   ├── papers/               # Paper annotations
│   ├── HARDWARE-DECISION.md  # Hardware recommendation
│   └── RETROSPECTIVE.md      # Learning journey reflection
├── QUESTIONS.md              # Central questions log
├── VISION.md                 # Project vision
└── GETTING-STARTED.md        # Setup guide
```

## Learning Tracks

### Track A: Fine-Tuning Mechanics
Understand what happens during training:
- **Loss curves**: What patterns mean and how to interpret them
- **Learning rate**: Effects of different LR choices
- **LoRA rank**: Trade-offs between expressiveness and efficiency
- **Forgetting**: Measuring and mitigating capability loss

Key experiments:
```bash
python -m experiments.fine_tuning.loss_curve_analysis
python -m experiments.learning_rate.experiment
python -m experiments.lora_rank.experiment
python -m experiments.forgetting.experiment
```

### Track B: Attention Visualization
See what attention is doing:
- Hook-based extraction for Qwen architecture
- Heatmaps, head comparisons, layer progressions
- Base vs fine-tuned attention patterns

```bash
python -m experiments.attention.compare_experiment
```

### Track C: Representation Probing
Examine the residual stream:
- Activation extraction at any layer
- FFN vs attention contribution analysis
- Token position similarity across layers

```bash
python -m experiments.probing.run_analysis
```

### Track D: Paper Reproduction
Test claims from "The Bayesian Geometry of Transformer Attention":
- Paper annotation and summary in own words
- Claim: FFN performs "posterior update" (larger contribution than attention)

```bash
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

## Key Documents

| Document | Purpose |
|----------|---------|
| [GETTING-STARTED.md](GETTING-STARTED.md) | Setup and first steps |
| [VISION.md](VISION.md) | Project goals and motivation |
| [QUESTIONS.md](QUESTIONS.md) | Open questions and progress |
| [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md) | Learning journey summary |
| [docs/HARDWARE-DECISION.md](docs/HARDWARE-DECISION.md) | Hardware recommendations |

## Self-Assessments

After each track, write what you learned:
- [Training Dynamics](docs/concepts/training-dynamics-self-assessment.md)
- [Attention Mechanics](docs/concepts/attention-self-assessment.md)
- [Representations](docs/concepts/representations-self-assessment.md)

## Hardware

**Current**: NVIDIA RTX 5060 Ti 16GB (Blackwell)

**Recommendation**: Stay with current hardware + cloud burst for occasional scale experiments. See [HARDWARE-DECISION.md](docs/HARDWARE-DECISION.md) for details.

Sufficient for:
- QLoRA fine-tuning of 7B models (4-bit quantized)
- Attention and activation extraction
- Small-scale interpretability experiments

## Success Criteria

1. **Mental model formed**: Can explain transformer architecture without referencing materials
2. **Paper-reading fluency**: Can follow methodology in research papers
3. **Experimental intuition**: Can hypothesize experiment outcomes and understand surprises

## License

Learning project - use freely for educational purposes.
