# Representation Analysis Experiments

## Purpose
Examine what information is encoded in activations and how it transforms layer by layer.

## Hypothesis (Pre-Experiment)

### Layer-by-Layer Transformation

| Layer Range | Expected Behavior |
|-------------|-------------------|
| Early (0-5) | Low-level features: token identity, position, basic syntax |
| Middle (10-15) | Higher-level: semantic relationships, entity tracking |
| Late (25-27) | Task-specific: output preparation, next-token prediction |

### Activation Statistics

1. **Norm should increase**: Information accumulates through residual connections
2. **FFN contribution > Attention**: FFN has more parameters, does more "work"
3. **Variance across tokens**: Later layers may show more variance (task-dependent)

## Experiments

### Experiment 1: Activation Statistics Across Layers

**Question**: How do activation statistics change through the model?

**Method**:
1. Run text through model
2. Extract activations from all layers
3. Compute: mean, std, norm, min, max per layer
4. Plot trends

**Expected**:
- Norm increases (residual accumulation)
- Std may increase (specialization)

### Experiment 2: Layer Contribution Analysis

**Question**: How much do attention vs FFN contribute at each layer?

**Method**:
1. Extract pre_attn and post_ffn for each layer
2. Compute attention contribution = post_attn - pre_attn
3. Compute FFN contribution = post_ffn - post_attn
4. Compare norms

**Expected**:
- FFN contribution slightly larger (more parameters)
- Both contribute meaningfully throughout

### Experiment 3: Token Position Analysis

**Question**: Do activations differ by token position?

**Method**:
1. Extract activations for a sentence
2. Compare first token, middle tokens, last token
3. Look at cosine similarity between positions

**Expected**:
- Early layers: tokens more distinct
- Late layers: convergence toward output

### Experiment 4: Simple Linear Probe

**Question**: Can we predict token properties from activations?

**Method**:
1. Extract activations for many tokens
2. Label tokens (e.g., noun vs verb, capitalized vs not)
3. Train linear classifier on activations
4. Measure accuracy per layer

**Expected**:
- Some properties detectable early (capitalization)
- Some properties need later layers (part of speech)

## Running the Experiments

```bash
cd /home/cmf/Work/llm-testing
python -m experiments.probing.run_analysis
```

## Results

[To be filled after running experiments]

### Experiment 1: Activation Statistics

| Layer | Mean | Std | Norm | Notes |
|-------|------|-----|------|-------|
| 0     |      |     |      |       |
| 7     |      |     |      |       |
| 14    |      |     |      |       |
| 21    |      |     |      |       |
| 27    |      |     |      |       |

### Experiment 2: Layer Contributions

| Layer | Attention Contrib (norm) | FFN Contrib (norm) | Ratio |
|-------|--------------------------|-------------------|-------|
| 0     |                          |                   |       |
| 14    |                          |                   |       |
| 27    |                          |                   |       |

### Experiment 3: Token Position Analysis

[Visualization and observations]

### Experiment 4: Linear Probe

| Layer | Capitalization Accuracy | Part-of-Speech Accuracy |
|-------|------------------------|------------------------|
| 0     |                        |                        |
| 14    |                        |                        |
| 27    |                        |                        |

## Key Findings

[To be filled after running experiments]

1. **Activation growth pattern**: TBD
2. **Attention vs FFN roles**: TBD
3. **Layer specialization**: TBD
4. **Probing results**: TBD

## Implications

[What do these findings suggest about how the model works?]

## Related

- [Activation Extraction](./extract.py)
- [docs/concepts/representations-self-assessment.md](../../docs/concepts/representations-self-assessment.md)
