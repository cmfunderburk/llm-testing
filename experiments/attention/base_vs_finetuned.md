# Attention Comparison: Base vs Fine-tuned

## Purpose
Compare attention patterns between base model and fine-tuned model to understand what changes during training.

## Hypothesis (Pre-Experiment)

Fine-tuning will change attention patterns in specific ways:

1. **Instruction heads**: Some heads may become more sensitive to instruction tokens
2. **Response planning**: Late-layer attention may show different patterns for response generation
3. **Minimal change**: Most heads should remain similar (LoRA is additive, small)

### Expected Changes by Layer

| Layer Type | Expected Change |
|------------|----------------|
| Early (0-5) | Minimal - low-level patterns preserved |
| Middle (10-15) | Some change - task-specific patterns |
| Late (25-27) | Most change - output behavior |

### What Would Surprise Us

- **Large changes everywhere**: Would suggest training is too aggressive
- **No changes at all**: Would suggest fine-tuning isn't working
- **Changes only in certain heads**: Would suggest head specialization

## Methodology

1. Load base model (without LoRA adapters)
2. Load fine-tuned model (with trained LoRA adapters)
3. Run same prompts through both
4. Extract attention from key layers (0, 7, 14, 21, 27)
5. Generate side-by-side visualizations
6. Compute difference metrics

## Running the Experiment

```bash
cd /home/cmf/Work/llm-testing
python -m experiments.attention.compare_experiment
```

**Prerequisites**:
- Run at least one fine-tuning experiment first (to have a trained model)
- Or use a pre-trained adapter from HuggingFace

## Test Prompts

We'll use prompts that exercise different capabilities:

1. **Instruction following**: "List three benefits of exercise."
2. **Creative**: "Write a haiku about coffee."
3. **Reasoning**: "If all cats are animals, and some animals are pets, are some cats pets?"
4. **Code**: "Write a Python function to reverse a string."

## Results

[To be filled after running experiment]

### Quantitative Changes

| Layer | Avg Attention Difference | Max Head Change | Notes |
|-------|-------------------------|-----------------|-------|
| 0     |                         |                 |       |
| 7     |                         |                 |       |
| 14    |                         |                 |       |
| 21    |                         |                 |       |
| 27    |                         |                 |       |

### Qualitative Observations

#### Prompt 1: Instruction Following
- Base model attention: TBD
- Fine-tuned attention: TBD
- Key differences: TBD

#### Prompt 2: Creative
- Base model attention: TBD
- Fine-tuned attention: TBD
- Key differences: TBD

#### Prompt 3: Reasoning
- Base model attention: TBD
- Fine-tuned attention: TBD
- Key differences: TBD

#### Prompt 4: Code
- Base model attention: TBD
- Fine-tuned attention: TBD
- Key differences: TBD

## Visualizations

See `outputs/attention_comparison/` for:
- `base_layer_X.png`: Base model attention patterns
- `finetuned_layer_X.png`: Fine-tuned model attention patterns
- `diff_layer_X.png`: Difference heatmaps
- `side_by_side.png`: Combined comparison

## Analysis

### Were Hypotheses Confirmed?

1. **Instruction heads emerged?**
   - TBD

2. **Late layers changed more?**
   - TBD

3. **Changes were minimal overall?**
   - TBD

### What Changes Mean

[Fill in interpretation after seeing results]

### Implications for Fine-Tuning

[What do these changes suggest about how fine-tuning works?]

## Follow-up Questions

- Do attention changes correlate with behavior changes?
- Are the same heads modified across different prompts?
- How do attention changes scale with LoRA rank?

## Related

- [Attention Extraction](./extract.py)
- [Attention Visualization](./visualize.py)
- [docs/concepts/attention-self-assessment.md](../../docs/concepts/attention-self-assessment.md)
