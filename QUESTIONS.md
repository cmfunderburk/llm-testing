# Questions Log

Central repository for questions that arise during learning experiments.

## Status Legend
- **OPEN**: Question not yet answered
- **EXPLORING**: Actively investigating through experiments
- **ANSWERED**: Question answered (see linked docs/experiments)
- **SUPERSEDED**: Question no longer relevant or replaced by better question

---

## Training Dynamics (Track A)

### OPEN: What does the loss curve actually tell us?
*Source: VISION.md*

Specific sub-questions:
- Does a sharp initial drop indicate easy learning or large initial error?
- What does a plateau mean for model learning?
- How does eval loss divergence relate to overfitting?

### OPEN: How do different learning rates affect what the model learns?
*Source: VISION.md*

- Is there a "critical" learning rate where behavior changes qualitatively?
- Does higher LR lead to faster forgetting of base capabilities?

### OPEN: What's happening in the optimizer state?
*Source: VISION.md*

- What do momentum and velocity values tell us about training?
- Can we visualize optimizer state to understand training progress?

### OPEN: How does batch size interact with learning rate?
*Source: VISION.md*

- Is the "linear scaling rule" valid for LoRA fine-tuning?
- What batch sizes are tractable on 16GB VRAM?

### OPEN: What's the minimum experiment that demonstrates something interesting?
*Source: VISION.md*

- How few examples/steps show measurable behavior change?
- What's the fastest experiment that teaches something real?

---

## Mechanistic Interpretability (Tracks B & C)

### OPEN: How does attention distribute across tokens and layers?
*Source: VISION.md*

- Are there consistent patterns (e.g., first/last token attention)?
- How do patterns differ between tasks?

### OPEN: What are the residual stream and how does it accumulate information?
*Source: VISION.md*

- What's in the residual stream at different layers?
- How do attention and FFN outputs add to it?

### OPEN: How do FFN layers transform representations?
*Source: VISION.md*

- What does the "gate" mechanism do?
- Are certain neurons/features interpretable?

### OPEN: Can we observe Bayesian inference mechanics?
*Source: VISION.md, target paper*

- How does "The Bayesian Geometry of Transformer Attention" relate to observable patterns?
- What experiments would test the paper's claims?

---

## Hardware & Scale

### OPEN: How much can be learned from 7B models about dynamics that transfer to larger scales?
*Source: VISION.md*

- Are training dynamics qualitatively similar at 7B vs 70B?
- What phenomena are scale-dependent?

### OPEN: Which interpretability tools work well with Qwen architecture?
*Source: VISION.md*

- Does bertviz work with Qwen?
- Are there Qwen-specific considerations?

### OPEN: What experiments would inform the hardware investment decision?
*Source: VISION.md*

- What's compute-limited on 16GB that would unlock with more?
- Is memory or compute the bottleneck?

---

## Per-Experiment Questions Convention

Each experiment file should include a "QUESTIONS TO ANSWER" section in its docstring. After running, add answered questions to the experiment's README with links to this central log.

Format in experiment files:
```python
"""
QUESTIONS TO ANSWER
-------------------
- Specific question 1? (links to QUESTIONS.md#section)
- Specific question 2?
"""
```

When a question is answered, update both:
1. The experiment's results section
2. This file (change status to ANSWERED, add link to evidence)

---

*Last updated: 2025-01-07*
