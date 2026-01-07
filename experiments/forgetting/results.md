# Catastrophic Forgetting Test

## Purpose
Test whether QLoRA fine-tuning degrades the model's general capabilities.

## Hypothesis (Pre-Experiment)

Fine-tuning with QLoRA should **preserve most general capabilities** because:

1. **LoRA is additive**: Base weights frozen, adapters only add to outputs
2. **Small parameter change**: Only ~1-2% of parameters modified
3. **Similar task distribution**: Alpaca is general instruction-following

### Expected Outcomes

| Category | Expectation | Rationale |
|----------|-------------|-----------|
| Math | Preserved | Pre-training includes math; LoRA shouldn't erase |
| Reasoning | Preserved | Core capability in base model |
| Code | Preserved | Alpaca includes some code tasks |
| Knowledge | Preserved | Factual knowledge in frozen base weights |

### When We'd Expect Forgetting

Forgetting might occur if:
- Learning rate too high (damages adapter target layers)
- Training distribution very different from pre-training
- LoRA alpha too high (adapter outputs dominate)
- Training for too many epochs on narrow domain

## Test Suite

### Math (4 questions)
- Arithmetic: 17 * 23
- Word problem: Speed/distance calculation
- Percentage: 15% of 200
- Algebra: 3x + 7 = 22

### Reasoning (3 questions)
- Logic: Syllogism fallacy detection
- Cognitive reflection: Bat and ball problem
- Cognitive reflection: Widget machine problem

### Code (3 questions)
- Function writing: Prime number checker
- Comprehension: List comprehension explanation
- Complexity: Binary search time complexity

### Knowledge (4 questions)
- Geography: Capital of France
- Literature: Romeo and Juliet author
- History: End of WWII
- Science: Photosynthesis explanation

## Running the Experiment

```bash
cd /home/cmf/Work/llm-testing
python -m experiments.forgetting.experiment
```

**Expected time**: ~20-30 minutes (includes one training run + evaluations)

## Results

[To be filled after running experiment]

| Category | Base Model | Fine-tuned | Change |
|----------|-----------|------------|--------|
| Math     |           |            |        |
| Reasoning|           |            |        |
| Code     |           |            |        |
| Knowledge|           |            |        |
| **Overall** |        |            |        |

## Interpretation Guide

### No Forgetting (change > -5%)
- QLoRA working as designed
- Safe to fine-tune on this task

### Minor Forgetting (-5% to -15%)
- Some degradation but acceptable
- Consider reducing learning rate or epochs

### Significant Forgetting (< -15%)
- Concerning degradation
- Investigate: LR, alpha, training data
- May need different approach

## Learnings

[To be filled after running experiment]

### Were Hypotheses Confirmed?
- TBD

### Key Insights
- TBD

### Implications
- TBD

## Related

- [QLoRA Concept](../../docs/concepts/qlora.md)
- [Loss Curves Concept](../../docs/concepts/loss-curves.md)
- [QUESTIONS.md](../../QUESTIONS.md)
