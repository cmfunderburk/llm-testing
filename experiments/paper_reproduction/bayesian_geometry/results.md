# Paper Claim Reproduction Results

## Paper
"The Bayesian Geometry of Transformer Attention" (arXiv:2512.22471)

## Claim Tested
**"Feed-forward networks perform the posterior update"**

Operationalized as: FFN contribution norm > Attention contribution norm

## Rationale for Selection
1. Directly testable with existing activation extraction tooling
2. Connects to Track C representation analysis
3. Provides mechanistic insight into model behavior
4. Does not require implementing paper's synthetic tasks

## Methodology
1. Extract activations at pre_attn, post_attn, post_ffn positions
2. Compute: attention_contribution = post_attn - pre_attn
3. Compute: ffn_contribution = post_ffn - post_attn
4. Compare L2 norms across layers and inputs

## Test Inputs
- Factual reasoning: "The capital of France is Paris..."
- Sequential reasoning: "If A is true and B is false..."
- Simple completion: "The quick brown fox..."
- Mathematical: "2 + 2 = 4. 3 + 3 = 6..."

## Results

### Status: EXPERIMENT READY TO RUN

Run the experiment:
```bash
cd /home/cmf/Work/llm-testing
python -m experiments.paper_reproduction.bayesian_geometry.experiment
```

### Expected Output Format

| Layer | Avg FFN/Attn Ratio | FFN Larger (%) | Supports Claim |
|-------|-------------------|----------------|----------------|
| 0     | TBD               | TBD            | TBD            |
| 3     | TBD               | TBD            | TBD            |
| ...   | ...               | ...            | ...            |

### Prediction (Before Running)

Based on the paper and our Track C hypothesis:
- **Prediction**: FFN/Attention ratio > 1 for most layers
- **Confidence**: Medium (paper used synthetic tasks, not language)
- **Expected support rate**: 60-80% of layers

## Interpretation

[To be filled after running]

## Comparison to Paper Claims

| Paper Claim | Our Finding | Match? |
|-------------|-------------|--------|
| FFN does posterior update | [Pending] | [Pending] |
| Attention provides routing | [Pending] | [Pending] |
| Residual = belief substrate | Not tested | N/A |

## Limitations

1. **Different task domain**: Paper uses synthetic Bayesian tasks; we use natural language
2. **Different model**: Paper uses small custom transformers; we use Qwen2.5-7B
3. **Operationalization**: "Posterior update" may not map directly to contribution norm
4. **Single metric**: L2 norm is one possible measure of contribution

## Connection to Learning Journey

This reproduction experiment connects:
- **Track B**: Attention patterns (now interpreted as "routing")
- **Track C**: FFN vs attention contributions (now interpreted as "update vs routing")
- **Paper understanding**: Concrete test of theoretical claim

The paper provides a mechanistic interpretation for what we observed in earlier tracks:
- Attention does "less" (routing) than FFN (computation)
- The residual stream accumulates "beliefs" (posterior probabilities)
- Layer-by-layer processing = iterative Bayesian updating

## Next Steps After Running

1. Compare actual results to paper's claims
2. Visualize contribution ratios across layers
3. Consider testing orthogonal key bases claim (requires Q, K matrix extraction)
4. Update paper annotation with empirical findings

## Related

- [Paper Annotation](../../../docs/papers/bayesian-geometry-attention.md)
- [Track C Experiments](../../probing/run_analysis.py)
- [Attention Self-Assessment](../../../docs/concepts/attention-self-assessment.md)
- [Representations Self-Assessment](../../../docs/concepts/representations-self-assessment.md)
