# Learning Rate Exploration Experiment

## Purpose
Understand how learning rate affects QLoRA fine-tuning dynamics.

## Hypothesis (Pre-Experiment)

Different learning rates will produce qualitatively different training dynamics:

| Learning Rate | Expected Behavior |
|---------------|-------------------|
| 1e-5 (Very Low) | Slow convergence, may not reach optimal loss |
| 2e-4 (Standard) | Smooth decrease, stable training, good results |
| 1e-3 (High) | Fast initial drop, possibly unstable later |
| 5e-3 (Very High) | Likely to diverge (loss increases) |

### Why These Predictions?

**LoRA uses higher LRs than full fine-tuning because:**
- Only updating ~1-2% of parameters
- Smaller gradients (fewer parameters to accumulate signal)
- Adapter matrices start from small random init

**2e-4 is "standard" because:**
- Recommended in LoRA paper
- Balances speed and stability
- Works across most tasks

## Running the Experiment

```bash
cd /home/cmf/Work/llm-testing
python -m experiments.fine_tuning.learning_rate.experiment
```

**Warning**: This runs 4 complete training runs sequentially. Expect ~40-60 minutes on RTX 5060 Ti.

## Expected Outputs

After running, you'll find in `outputs/learning_rate_exploration/`:
- `lr_comparison.png`: Side-by-side visualizations
- `README.md`: Results summary with tables
- `all_results.json`: Raw data for all runs
- `lr_*/run_data.json`: Per-LR training data

## Questions to Answer

After running, document answers to:

1. **Which LR gave lowest final loss?**
   - Answer: TBD

2. **Did any LRs diverge or show instability?**
   - Answer: TBD

3. **How did convergence speed differ?**
   - Answer: TBD

4. **Does 2e-4 appear optimal for this setup?**
   - Answer: TBD

## Results

[To be filled after running experiment]

| Learning Rate | Final Train Loss | Final Eval Loss | Behavior |
|---------------|-----------------|-----------------|----------|
| 1e-5          |                 |                 |          |
| 2e-4          |                 |                 |          |
| 1e-3          |                 |                 |          |
| 5e-3          |                 |                 |          |

## Learnings

[To be filled after running experiment]

### Were Hypotheses Confirmed?
- TBD

### Key Insights
- TBD

### Follow-up Questions
- TBD

## Related

- [Loss Curves Concept](../../../docs/concepts/loss-curves.md)
- [QLoRA Concept](../../../docs/concepts/qlora.md)
- [QUESTIONS.md](../../../QUESTIONS.md#how-do-different-learning-rates-affect-what-the-model-learns)
