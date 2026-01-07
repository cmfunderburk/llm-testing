# LoRA Rank Comparison Experiment

## Purpose
Understand how LoRA rank affects adaptation capacity and training efficiency.

## Why Low-Rank Adaptation Works (Theory)

The LoRA paper (Hu et al., 2021) hypothesizes:

> "We hypothesize that the change in weights during model adaptation also has a low 'intrinsic rank'."

**Key theoretical insights:**

1. **Pre-trained models are over-parameterized**: Much capacity is unused for any specific task
2. **Fine-tuning changes are structured**: Not random noise, but coherent adjustments
3. **Low-rank matrices capture structured changes**: Rank r means r "directions" of change

**Mathematical intuition:**
- Full weight update: ΔW ∈ R^(d×k) has d×k parameters
- LoRA update: ΔW = BA where B ∈ R^(d×r), A ∈ R^(r×k) has (d+k)×r parameters
- If r << min(d,k), massive parameter reduction
- Works because meaningful changes lie in low-dimensional subspace

## Hypothesis (Pre-Experiment)

| Rank | Parameters | Expected Behavior |
|------|-----------|-------------------|
| 8    | ~4M       | Minimal capacity, may underfit complex tasks |
| 16   | ~8M       | Good for simple adaptation |
| 32   | ~17M      | Standard choice, reliable |
| 64   | ~34M      | Maximum tested, may plateau |

**Prediction:** For simple instruction-following (Alpaca), ranks 16+ will perform similarly,
suggesting the adaptation truly is low-dimensional for this task.

## Running the Experiment

```bash
cd /home/cmf/Work/llm-testing
python -m experiments.lora_rank.experiment
```

**Warning**: This runs 4 complete training runs. Expect ~40-60 minutes on RTX 5060 Ti.

## Expected Outputs

After running, you'll find in `outputs/lora_rank_comparison/`:
- `rank_comparison.png`: Visualizations comparing ranks
- `README.md`: Results summary
- `all_results.json`: Raw data
- `rank_*/run_data.json`: Per-rank data

## Questions to Answer

1. **Is there a diminishing returns point?**
   - Answer: TBD

2. **Does rank 8 work as well as rank 64?**
   - Answer: TBD (would support low-rank hypothesis)

3. **How does parameter count scale with rank?**
   - Answer: TBD (should be linear)

4. **Is training time significantly affected by rank?**
   - Answer: TBD

## Results

[To be filled after running experiment]

| Rank | Trainable Params | Final Train Loss | Final Eval Loss |
|------|-----------------|-----------------|-----------------|
| 8    |                 |                 |                 |
| 16   |                 |                 |                 |
| 32   |                 |                 |                 |
| 64   |                 |                 |                 |

## Learnings

[To be filled after running experiment]

### Were Hypotheses Confirmed?
- TBD

### Key Insights
- TBD

### Implications for Future Experiments
- TBD

## Related

- [QLoRA Concept](../../docs/concepts/qlora.md)
- [Loss Curves Concept](../../docs/concepts/loss-curves.md)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
