# Questions Log

Central repository for questions that arise during learning experiments.

## Status Legend
- **OPEN**: Question not yet answered
- **EXPLORING**: Actively investigating through experiments
- **ANSWERED**: Question answered (see linked docs/experiments)
- **SUPERSEDED**: Question no longer relevant or replaced by better question

---

## Training Dynamics (Track A)

### EXPLORING: What does the loss curve actually tell us?
*Source: VISION.md*
*Status updated: Phase 6 complete*

Specific sub-questions:
- Does a sharp initial drop indicate easy learning or large initial error?
- What does a plateau mean for model learning?
- How does eval loss divergence relate to overfitting?

**Progress**: Created docs/concepts/loss-curves.md with interpretation guide. Experiment ready at experiments/fine_tuning/loss_curve_analysis.py. Awaiting GPU execution.

### EXPLORING: How do different learning rates affect what the model learns?
*Source: VISION.md*
*Status updated: Phase 6 complete*

- Is there a "critical" learning rate where behavior changes qualitatively?
- Does higher LR lead to faster forgetting of base capabilities?

**Progress**: Created experiments/learning_rate/ with comparison of 1e-5, 2e-4, 1e-3, 5e-3. Awaiting GPU execution.

### OPEN: What's happening in the optimizer state?
*Source: VISION.md*

- What do momentum and velocity values tell us about training?
- Can we visualize optimizer state to understand training progress?

**Status**: Not addressed in current phase. Future work.

### OPEN: How does batch size interact with learning rate?
*Source: VISION.md*

- Is the "linear scaling rule" valid for LoRA fine-tuning?
- What batch sizes are tractable on 16GB VRAM?

**Status**: Not addressed in current phase. Future work.

### ANSWERED: What's the minimum experiment that demonstrates something interesting?
*Source: VISION.md*
*Answered in: docs/concepts/training-dynamics-self-assessment.md*

- How few examples/steps show measurable behavior change?
- What's the fastest experiment that teaches something real?

**Answer**: ~100 examples with 3-5 epochs can show measurable style/format changes. Single-task fine-tuning (e.g., JSON output format) is the minimum interesting experiment. See self-assessment for details.

---

## Mechanistic Interpretability (Tracks B & C)

### EXPLORING: How does attention distribute across tokens and layers?
*Source: VISION.md*
*Status updated: Phase 6 complete*

- Are there consistent patterns (e.g., first/last token attention)?
- How do patterns differ between tasks?

**Progress**: Built attention extraction tooling (experiments/attention/extract.py) and visualization tools (visualize.py). Comparison framework ready. Awaiting GPU execution for actual patterns.

### ANSWERED: What are the residual stream and how does it accumulate information?
*Source: VISION.md*
*Answered in: docs/concepts/representations-self-assessment.md*

- What's in the residual stream at different layers?
- How do attention and FFN outputs add to it?

**Answer**: The residual stream is the core information highway. Each layer ADDS to it (doesn't replace). Attention and FFN both contribute via residual connections: `stream_new = stream_old + attention(...) + ffn(...)`. Built extraction tools to probe this.

### EXPLORING: How do FFN layers transform representations?
*Source: VISION.md*
*Status updated: Phase 6 complete*

- What does the "gate" mechanism do?
- Are certain neurons/features interpretable?

**Progress**: Created contribution analysis in experiments/probing/run_analysis.py. Hypothesis: FFN contributes more than attention (does "computation"). Awaiting GPU execution.

### EXPLORING: Can we observe Bayesian inference mechanics?
*Source: VISION.md, target paper*
*Status updated: Phase 6 complete*

- How does "The Bayesian Geometry of Transformer Attention" relate to observable patterns?
- What experiments would test the paper's claims?

**Progress**: Read and annotated paper (docs/papers/bayesian-geometry-attention.md). Designed reproduction experiment testing FFN vs attention contribution (experiments/paper_reproduction/bayesian_geometry/). Paper claims FFN does "posterior update" - testable with our tooling.

---

## Hardware & Scale

### OPEN: How much can be learned from 7B models about dynamics that transfer to larger scales?
*Source: VISION.md*

- Are training dynamics qualitatively similar at 7B vs 70B?
- What phenomena are scale-dependent?

**Status**: Deferred. Focus on learning from 7B first. Future cloud experiments may address this.

### ANSWERED: Which interpretability tools work well with Qwen architecture?
*Source: VISION.md*
*Answered in: experiments/attention/extract.py, experiments/probing/extract.py*

- Does bertviz work with Qwen?
- Are there Qwen-specific considerations?

**Answer**: Built custom hook-based extraction for both attention and activations. Key Qwen specifics documented:
- GQA: 28 attention heads, 4 KV heads
- Hidden size: 3584
- 28 layers total
- Architecture: model.model.layers[i] access pattern

### ANSWERED: What experiments would inform the hardware investment decision?
*Source: VISION.md*
*Answered in: docs/HARDWARE-DECISION.md*

- What's compute-limited on 16GB that would unlock with more?
- Is memory or compute the bottleneck?

**Answer**: Memory is the bottleneck. 16GB limits us to 7B quantized models. More VRAM would unlock: full fine-tuning (no LoRA), 13B+ models, longer context. Recommendation: Stay with current hardware + cloud burst for occasional needs.

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

*Last updated: 2026-01-07 (Phase 6 complete)*

## Summary

| Category | ANSWERED | EXPLORING | OPEN |
|----------|----------|-----------|------|
| Track A (Training) | 1 | 2 | 2 |
| Tracks B/C (Mech Interp) | 1 | 3 | 0 |
| Hardware & Scale | 2 | 0 | 1 |
| **Total** | **4** | **5** | **3** |

See docs/RETROSPECTIVE.md for full learning journey assessment.
