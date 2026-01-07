# The Bayesian Geometry of Transformer Attention

**Paper**: [arXiv:2512.22471](https://arxiv.org/abs/2512.22471)
**Authors**: Naman Aggarwal, Siddhartha R. Dalal, Vishal Misra
**Affiliations**: Dream Sports, Columbia University
**Date**: December 2025

## Reading Notes

### The Central Question

Do transformers actually perform Bayesian inference, or does it just look like they do?

**Why this is hard to answer**:
1. Natural language data lacks analytic posteriors (we don't know the "true" answer)
2. Large models conflate reasoning with memorization (maybe they just remember patterns)
3. No controlled environment to test rigorously

### The Key Innovation: Bayesian Wind Tunnels

The authors create controlled environments where:
- The true Bayesian posterior is known in closed form
- Memorization is provably impossible (synthetic, novel inputs)
- Success = reproducing the known posterior

**My understanding**: Like a wind tunnel for aerodynamics, this isolates the variable of interest. If the model matches the known posterior, it's doing Bayesian inference. If not, it's doing something else.

### Two Test Tasks

#### Task 1: Bijection Elimination
- Given partial observations, eliminate impossible bijections
- Each observation rules out some mappings
- True posterior is uniform over remaining possibilities
- Tests: Can the model track which bijections remain valid?

#### Task 2: Hidden Markov Model (HMM) State Tracking
- Observe emissions from unknown hidden state
- Update belief about current state given observations
- True posterior follows Bayes' rule exactly
- Tests: Can the model maintain and update a belief distribution?

### The Central Claims

#### Claim 1: Transformers Achieve Bayesian Accuracy
Small transformers reproduce posteriors with **10^-3 to 10^-4 bit accuracy**.

**What this means**: The KL divergence between model output and true posterior is incredibly small. The model is not approximately Bayesian - it's precisely Bayesian.

#### Claim 2: MLPs Fail Completely
Capacity-matched MLPs (same parameter count) fail "by orders of magnitude."

**What this means**: It's not about having enough parameters. The transformer architecture itself enables Bayesian inference. Flat architectures can't do it.

#### Claim 3: Specific Geometric Mechanism
The paper identifies HOW transformers implement Bayesian inference:

| Component | Role in Bayesian Inference |
|-----------|---------------------------|
| Residual stream | Belief substrate (stores current posterior) |
| Feed-forward networks | Posterior update (applies Bayes' rule) |
| Attention | Content-addressable routing (selects relevant information) |

### Geometric Diagnostics

The authors probe the learned representations and find:

1. **Orthogonal key bases**: Keys form orthogonal directions in representation space
2. **Progressive query-key alignment**: Queries learn to align with relevant keys
3. **Low-dimensional value manifold**: Values live on a manifold parameterized by posterior entropy
4. **Frame-precision dissociation**: During training, precision improves while attention frame stays stable

### My Interpretation

The paper suggests that attention is not just "soft selection" but implements a specific computational primitive needed for Bayesian inference. The residual stream accumulates evidence, FFN applies the update rule, and attention routes information.

This connects to my Track C understanding:
- Residual stream = information highway (we built extraction tools for this)
- FFN contribution > attention contribution (we hypothesized this)
- Layer-by-layer transformation = belief updating

### Key Equations (Conceptual)

**Bayesian update**:
```
P(hypothesis | evidence) ∝ P(evidence | hypothesis) × P(hypothesis)
```

**Transformer as Bayesian computer**:
```
residual_new = residual_old + attention(residual_old) + ffn(residual_old + attention(...))
```

The paper argues this architecture naturally implements the Bayesian update, with:
- `residual_old` = prior
- `attention(...)` = selecting relevant evidence
- `ffn(...)` = likelihood computation
- `residual_new` = posterior

## Summary in My Own Words

Transformers don't just correlate patterns - they implement Bayesian inference through their architecture. The authors prove this by creating "wind tunnel" tests where the correct Bayesian answer is known, showing transformers match it precisely while simpler architectures fail.

The geometric mechanism is:
1. **Residual stream stores beliefs** (probability distributions over hypotheses)
2. **Attention routes evidence** (selecting which observations matter)
3. **FFN computes updates** (applying Bayes' rule)

This explains why transformers generalize: they're not memorizing, they're computing. And it explains why attention matters: without it, you can't route evidence to the right computation.

## Claims Identified for Reproduction

### High Priority (Feasible with our setup)

1. **Attention pattern structure**: Do attention patterns in our models show orthogonal key bases?
   - Method: Extract Q, K matrices, measure orthogonality
   - Connection: We have attention extraction tooling (Track B)

2. **Residual stream as belief substrate**: Does information accumulate monotonically?
   - Method: Track representation similarity to output across layers
   - Connection: We have activation extraction tooling (Track C)

3. **FFN vs Attention contribution**: Does FFN contribute more to "updating" than attention?
   - Method: Compare contribution norms at each layer
   - Connection: Already in our experiments/probing/run_analysis.py

### Medium Priority (Requires more setup)

4. **Low-dimensional value manifold**: Do value vectors cluster by some task property?
   - Method: PCA/UMAP on value vectors, color by task metric

5. **MLP vs Transformer comparison**: On a simple task, does MLP fail?
   - Method: Train both architectures on bijection task

### Lower Priority (May exceed our scope)

6. **Full wind tunnel replication**: Implement bijection elimination task
   - This is a significant implementation effort

## Questions This Paper Raises

1. Does this apply to language? The paper uses synthetic tasks - does the mechanism transfer?

2. What happens during fine-tuning? Does the Bayesian structure change when we fine-tune?

3. Is this why prompting works? If attention routes evidence, does prompting provide "evidence frames"?

4. Connection to in-context learning: Is ICL = Bayesian updating on examples?

## Related to Our Learning Journey

| Track | Connection to Paper |
|-------|---------------------|
| Track A (Fine-tuning) | Does fine-tuning disrupt the Bayesian geometry? |
| Track B (Attention) | Paper gives mechanistic interpretation of attention patterns |
| Track C (Probing) | Paper gives meaning to residual stream contents |

## Sources

- [arXiv:2512.22471](https://arxiv.org/abs/2512.22471) - Original paper
- [arXiv HTML](https://arxiv.org/html/2512.22471v1) - Full text
- [EmergentMind Summary](https://www.emergentmind.com/papers/2512.22471)
