# LLM Learning Lab: Final Retrospective

## Overview

This document reflects on the complete learning journey through the LLM Learning Lab, assessing what was learned, what worked, and where to go next.

**Duration**: Phases 1-6 completed
**Tracks Completed**: A (Fine-Tuning), B (Attention), C (Probing), D (Paper Reproduction)

---

## Self-Assessment Against Original Success Criteria

### Criterion 1: Mental Model Formation
*"Can explain transformer architecture, attention mechanics, and training dynamics without referencing materials"*

**Assessment**: PARTIALLY ACHIEVED

| Concept | Understanding Level | Evidence |
|---------|-------------------|----------|
| Transformer architecture | Strong | Can explain residual stream, layer composition |
| Attention mechanics | Strong | Self-assessment doc, extraction tooling built |
| Training dynamics | Moderate | Loss curves understood, but optimizer state less so |
| QLoRA/LoRA | Strong | Implemented experiments, understand rank tradeoffs |
| Residual stream | Strong | Built extraction tools, understand accumulation |
| FFN role | Moderate | Hypotheses formed, awaiting empirical validation |

**Gap**: Haven't run experiments on actual GPU yet. Theory is solid, empirical intuition pending.

### Criterion 2: Paper-Reading Fluency
*"Can read 'Bayesian Geometry of Transformer Attention' and follow methodology/claims"*

**Assessment**: ACHIEVED

Evidence:
- Created annotated reading notes (docs/papers/bayesian-geometry-attention.md)
- Summarized core argument in own words
- Identified testable claims
- Designed reproduction experiment
- Connected paper claims to our Track B/C findings

The paper's claims about:
- Residual stream as "belief substrate" - understood
- FFN as "posterior update" - understood, designed test
- Attention as "routing" - understood, connects to Track B

### Criterion 3: Experimental Intuition
*"Can hypothesize what an experiment will show before running it"*

**Assessment**: PARTIALLY ACHIEVED

Examples of predictions made:
- Activation norms increase through layers (residual accumulation)
- FFN contribution > attention contribution (more parameters)
- Later layers show task-specific patterns
- Low learning rates underfit, high rates destabilize

**Gap**: Predictions not yet validated. This criterion requires running experiments.

---

## Track-by-Track Retrospective

### Track A: Fine-Tuning Mechanics

**What was built**:
- Loss curve analysis experiment
- Learning rate exploration (1e-5, 2e-4, 1e-3, 5e-3)
- LoRA rank comparison (8, 16, 32, 64)
- Catastrophic forgetting test suite
- Training dynamics self-assessment

**Key learnings**:
1. QLoRA enables 7B fine-tuning on 16GB VRAM
2. Learning rate is the most sensitive hyperparameter
3. LoRA rank is about expressiveness vs efficiency tradeoff
4. Forgetting is measurable but mitigatable

**What worked**:
- Structured experiment template (hypothesis → method → results → learnings)
- Documenting predictions before running

**What didn't work**:
- Couldn't actually run experiments yet (GPU required)
- Optimizer state visualization not implemented

### Track B: Attention Visualization

**What was built**:
- Attention extraction tooling (hook-based, Qwen-compatible)
- Visualization tools (heatmaps, head comparisons, layer progressions)
- Base vs fine-tuned comparison framework

**Key learnings**:
1. Qwen uses GQA (28 heads, 4 KV heads per layer)
2. Attention weights are (batch, heads, seq, seq) tensors
3. Attention patterns reveal what the model "looks at"
4. Hook-based extraction is clean and non-invasive

**What worked**:
- Documenting tensor shapes before coding
- Lazy imports for verification without GPU

**What didn't work**:
- No actual visualizations generated yet
- Didn't integrate external viz tools (bertviz)

### Track C: Representation Probing

**What was built**:
- Activation extraction for residual stream
- Statistics analysis (norms, variance across layers)
- Contribution analysis (attention vs FFN)
- Token position analysis

**Key learnings**:
1. Residual stream is the core information pathway
2. Each layer ADDS to stream (doesn't replace)
3. Pre/post activation hooks capture flow
4. Linear probing can reveal encoded properties

**What worked**:
- Clean separation of extraction vs analysis
- Connecting to paper's mechanistic interpretation

**What didn't work**:
- Didn't implement actual linear probes
- PCA/visualization of representation space not done

### Track D: Paper Reproduction

**What was built**:
- Annotated paper notes with understanding verification
- Claim selection with rationale
- Reproduction experiment design
- Results template ready for execution

**Key learnings**:
1. Papers use controlled environments (wind tunnels) for rigor
2. "Bayesian inference" in transformers has geometric interpretation
3. Connecting paper claims to observable measurements is key
4. Reproduction ≠ replication; testing specific claims is tractable

**What worked**:
- Reading paper with our Track B/C context made it accessible
- Identifying claims that match our existing tooling

**What didn't work**:
- Didn't implement paper's synthetic tasks
- Full replication is beyond scope

---

## Infrastructure Assessment

### What We Built

| Component | Status | Quality |
|-----------|--------|---------|
| experiments/ structure | Complete | Good |
| docs/concepts/ framework | Complete | Good |
| QUESTIONS.md log | Complete | Needs updating |
| Experiment templates | Complete | Good |
| Attention extraction | Complete | Good |
| Activation extraction | Complete | Good |
| Visualization tools | Partial | Basic |
| Paper reproduction | Partial | Framework only |

### Technical Decisions Made

1. **Hook-based extraction**: Chose over TransformerLens for simplicity
2. **Lazy imports**: Enables verification without torch/GPU
3. **Matplotlib for viz**: Simple, dependency-light
4. **JSON config files**: Track experiment parameters

---

## Open Questions for Future Exploration

### High Priority

1. **Do our predictions hold?**
   - Run the experiments on GPU
   - Compare results to hypotheses
   - Update understanding based on empirical data

2. **Does FFN really contribute more than attention?**
   - Run contribution analysis
   - Test across different inputs/tasks
   - Compare to paper's claims

3. **What does fine-tuning change?**
   - Compare base vs fine-tuned attention
   - Measure representation drift
   - Quantify forgetting

### Medium Priority

4. **Can we detect "Bayesian updating" in language tasks?**
   - Design sequential reasoning tests
   - Measure belief representation across layers

5. **What's in the value vectors?**
   - PCA of value space
   - Cluster by task properties

6. **How do different prompts affect attention?**
   - Systematic prompt comparison
   - Measure attention pattern changes

### Lower Priority (Future Research)

7. **Scale comparison**: Do patterns hold for larger models?
8. **Architecture comparison**: How does Qwen compare to LLaMA?
9. **Full paper replication**: Implement bijection elimination task

---

## Hardware Decision Outcome

**Decision**: Stay with RTX 5060 Ti 16GB + cloud burst

**Rationale**: Current hardware handles all Track A-D experiments. Cloud for occasional scale experiments. Reassess after running experiments and confirming learning directions.

---

## Key Takeaways

### What This Learning Lab Provides

1. **Structured exploration framework**: Not random experimentation
2. **Connected understanding**: Tracks build on each other
3. **Paper-grounded intuition**: Theory tied to implementation
4. **Reusable tooling**: Extraction/viz tools for future work

### Mental Model Formed

```
Transformer as Bayesian Computer:
┌─────────────────────────────────────────┐
│  Input → Embedding                      │
│            ↓                            │
│  ┌─────────────────────────────────┐    │
│  │ Layer N:                        │    │
│  │   Residual Stream (beliefs)     │───→│ Prior
│  │         ↓                       │    │
│  │   Attention (route evidence)    │───→│ Evidence selection
│  │         ↓                       │    │
│  │   FFN (update beliefs)          │───→│ Posterior update
│  │         ↓                       │    │
│  │   Residual Stream (updated)     │───→│ New prior
│  └─────────────────────────────────┘    │
│            ↓ (repeat N times)           │
│  Output (final beliefs) → Logits        │
└─────────────────────────────────────────┘
```

### What's Different Now

Before: "Transformers do attention somehow"
After: "Attention routes information, FFN updates representations, residual stream accumulates beliefs"

Before: "Fine-tuning changes the model"
After: "QLoRA adds low-rank adapters to specific layers, modifying the update functions"

Before: "The paper is dense and hard to read"
After: "The paper claims attention implements geometric Bayesian inference, and I can test that"

---

## Next Steps

### Immediate (Run Experiments)

1. Execute Track A experiments on GPU
2. Generate actual visualizations
3. Run paper claim reproduction
4. Update results docs with empirical findings

### Short-term (Deepen Understanding)

1. Implement linear probes
2. Compare base vs fine-tuned representations
3. Test more paper claims

### Long-term (Research Directions)

1. Study in-context learning through Bayesian lens
2. Explore prompt engineering mechanistically
3. Consider scaling experiments (cloud)

---

## Conclusion

The LLM Learning Lab has achieved its primary goal: building structured intuition about how transformers work. The mental model is now grounded in:

- **Concrete implementations** (extraction tooling, experiments)
- **Documented hypotheses** (predictions before experiments)
- **Paper-level understanding** (can read and design reproductions)
- **Identified gaps** (what's still unknown)

The experiments are ready to run. The infrastructure is in place. The learning continues.

---

*Completed: Phase 6 of LLM Learning Lab*
*Features: 20/20*
*Status: Ready for empirical validation*
