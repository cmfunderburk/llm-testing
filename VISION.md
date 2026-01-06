# LLM Learning Lab: Vision

A personal learning lab for understanding how large language models work — their training dynamics, internal mechanics, and the theory behind fine-tuning.

---

## Purpose

This repo is a **learning vehicle**, not a production pipeline.

The goal is to build a mental model of LLMs deep enough to:
- Read research papers fluently (e.g., "The Bayesian Geometry of Transformer Attention")
- Understand what's happening during training and inference
- Make informed decisions about when/how fine-tuning is useful
- Decide whether investing in better hardware is worthwhile for continued exploration

Fine-tuning is one experimental approach among several — a hands-on way to probe how models learn and adapt.

---

## Core Interests

### 1. Training Dynamics
How models learn. What happens during forward/backward passes. How loss landscapes evolve. Why certain hyperparameters matter.

Questions to explore:
- What does the loss curve actually tell us?
- How do different learning rates affect what the model learns?
- What's happening in the optimizer state?
- How does batch size interact with learning rate?

### 2. Mechanistic Interpretability
What's happening inside the model. Attention patterns, residual streams, circuits, features.

Questions to explore:
- How does attention distribute across tokens and layers?
- What are the residual stream and how does it accumulate information?
- How do FFN layers transform representations?
- Can we observe Bayesian inference mechanics (per the referenced paper)?

---

## Approach: Theory-Practice Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   READ            →     EXPERIMENT       →      REFLECT         │
│   Papers,               Run training,           Document        │
│   tutorials,            probe internals,        learnings,      │
│   code                  visualize               refine model    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    Updated mental model
                              │
                              ▼
                    New questions to explore
```

Each experiment should teach something about how LLMs work, not just produce a trained artifact.

---

## Success Criteria (First Pass)

The survey phase is successful when:

1. **Mental model formed**: Can explain transformer architecture, attention mechanics, and training dynamics without referencing materials
2. **Paper-reading fluency**: Can read a paper like "The Bayesian Geometry of Transformer Attention" and follow the methodology and claims
3. **Experimental intuition**: Can hypothesize what an experiment will show before running it, and understand why when results differ

Not required for first pass:
- A production-ready fine-tuned model
- A hardware upgrade decision (this emerges from the survey)
- Mastery of all fine-tuning methods

---

## What This Is NOT

- **Not a production fine-tuning pipeline**: The existing design docs (unsloth_finetune_design_v2-refined.md) describe a practical workflow — useful as one experimental track, but not the repo's purpose
- **Not a Knowledge-OS integration project**: K-OS is a separate tool for structured learning; this repo is for hands-on experimentation
- **Not a race to train models**: Speed of iteration matters less than depth of understanding

---

## Hardware Context

**Current setup**: NVIDIA RTX 5060 Ti 16GB (Blackwell)

This is sufficient for:
- QLoRA fine-tuning of 7B models
- Small-scale interpretability experiments
- Running inference on quantized models

The hardware investment question ("should I upgrade?") is an **output** of this learning process, not an input. After surveying the landscape, I'll have better sense of what compute levels different research directions require.

---

## Artifacts

Learnings should be captured as:

1. **Working code with inline notes**: Experiments that run, with comments explaining what's being tested and what was learned
2. **Structured documentation**: Markdown files capturing concepts, decisions, and evolving understanding
3. **Questions log**: Track questions that arise, which get answered, which remain open

The code is not the point — understanding is. But working code ensures ideas are grounded in reality, not just abstraction.

---

## Potential Experiment Tracks

### Track A: Fine-Tuning Mechanics (current repo focus)
Use the existing Unsloth/QLoRA setup to:
- Observe loss curves during training
- Compare LoRA ranks and their effects
- Probe what changes in model behavior after fine-tuning
- Test catastrophic forgetting

### Track B: Attention Visualization
Tools and experiments to see what attention is doing:
- Attention pattern visualization across layers
- How patterns change during training
- Comparison between base and fine-tuned models

### Track C: Representation Probing
Examine what's in the residual stream:
- Activation extraction and analysis
- Feature visualization
- Layer-by-layer transformation tracking

### Track D: Paper Reproduction
Pick targeted claims from papers and attempt verification:
- Start with smaller-scale experiments
- Document methodology and findings
- Note discrepancies and hypotheses

---

## Reference Materials

Papers of interest (exemplars, not exhaustive):
- "The Bayesian Geometry of Transformer Attention" (arxiv:2512.22471) — mechanistic understanding of attention
- "Attention Is All You Need" — foundational architecture
- "LoRA: Low-Rank Adaptation" — parameter-efficient fine-tuning
- "QLoRA" — quantized fine-tuning approach

(Expand this list as exploration proceeds)

---

## Relationship to Existing Files

| File | Role |
|------|------|
| `VISION.md` | North star — why this repo exists |
| `unsloth_finetune_design_v2-refined.md` | Track A implementation guide — practical fine-tuning workflow |
| `completion.md` | Track A completion checklist — specific to fine-tuning goal |
| (future) `experiments/` | Per-experiment code and notes |
| (future) `docs/concepts/` | Structured concept documentation |

---

## Open Questions

- What's the minimum experiment that demonstrates something interesting about training dynamics?
- How much can be learned from 7B models about dynamics that transfer to larger scales?
- Which interpretability tools work well with Qwen architecture?
- What experiments would inform the hardware investment decision?

---

*Created 2025-01-06 via conversation clarifying project intent*
