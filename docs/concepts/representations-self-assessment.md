# Representations Self-Assessment

A self-assessment on understanding internal representations in transformers.

## Core Concepts

### 1. The Residual Stream

**Question**: What is the residual stream and why is it central to transformer architecture?

**My Understanding**:
The residual stream is the main "information highway" through a transformer. Each token has its own stream that flows through all layers. The key insight is:

```
stream_0 = embedding(token)
stream_1 = stream_0 + attention_0(stream_0) + ffn_0(stream_0)
stream_2 = stream_1 + attention_1(stream_1) + ffn_1(stream_1)
...
output = unembed(stream_N)
```

Each component ADDS to the stream rather than replacing it. This means:
- Early information is preserved (with gradual modification)
- The final representation is a sum of all contributions
- We can analyze what each layer adds

**Confidence**: 8/10
**Gap**: I haven't verified empirically that early information persists. Could probe for token identity at late layers.

### 2. What Activations Represent

**Question**: What information is encoded in the activation vector for a token?

**My Understanding**:
A single activation vector (e.g., 3584 dimensions for Qwen2.5-7B) encodes:
- Token identity (what word/subword this is)
- Position information (where in sequence)
- Context (relationship to other tokens)
- Semantic meaning (building through layers)
- Syntactic role (noun, verb, etc.)
- Task-relevant features (what's needed for next prediction)

The representation is distributed - no single dimension means "noun" but the combination encodes it. Different subspaces may encode different properties.

**Confidence**: 6/10
**Gap**: Haven't verified what's recoverable via probing. Need to train linear classifiers on activations.

### 3. Layer-wise Processing

**Question**: How does information transform across layers?

**My Understanding**:
| Layer Range | Hypothesized Function |
|-------------|----------------------|
| Early (0-5) | Low-level: token identity, basic syntax, position |
| Middle (10-15) | Semantic: entity tracking, relationships, meaning |
| Late (25-27) | Task-specific: output preparation, next-token focus |

The "layer cake" view suggests progressive abstraction, but reality may be messier:
- Some information available throughout (token identity)
- Some information peaks in middle layers
- Output layers may "forget" some intermediate info

**Confidence**: 5/10
**Gap**: This is hypothesis based on papers. Need to run probing experiments to verify on our model.

### 4. Attention vs FFN Contributions

**Question**: What role does each component play?

**My Understanding**:
- **Attention**: Information routing. Moves information between token positions. "What should I pay attention to?"
- **FFN**: Information processing. Transforms the representation at each position. "What should I do with this information?"

The FFN has more parameters (~2/3 of layer parameters) and likely does more "computation." Attention does less computation but enables the crucial cross-position communication.

**Confidence**: 7/10
**Gap**: Haven't measured relative contribution norms. The experiments should reveal this.

### 5. Probing as Analysis Method

**Question**: How does linear probing reveal what's encoded?

**My Understanding**:
Linear probing trains a simple linear classifier: `label = W @ activation + b`

If a linear probe can predict a property (like part-of-speech) with high accuracy, that property is "linearly encoded" - it exists in a way the model could easily access.

Limitations:
- Probing might find information the model doesn't use
- Some information might require nonlinear decoding
- Probe accuracy ≠ how the model uses the information

**Confidence**: 7/10
**Gap**: Haven't implemented or trained probes yet. The theory is clear but practice may reveal subtleties.

## Deeper Questions

### Information Bottlenecks
Does information ever decrease through the model? In the residual stream paradigm, information accumulates. But:
- Final layers may discard intermediate computation
- Attention might "filter out" irrelevant tokens
- Some information might become "overwritten"

**Open question**: Measure mutual information between early and late layer representations.

### Superposition
Can the model store more concepts than it has dimensions? Evidence suggests yes:
- Models can represent many more features than hidden dimensions
- Features may be stored in "almost orthogonal" directions
- This creates interference but enables compression

**Open question**: Can we find evidence of superposition in our probing?

### Geometry of Representations
What structure does the representation space have?
- Are similar concepts clustered?
- Is there linear structure (king - man + woman = queen)?
- What's the intrinsic dimensionality?

**Open question**: Visualize representation space with dimensionality reduction.

## Track C Retrospective

### What Worked
1. Hook-based extraction is clean and non-invasive
2. Separating extraction from analysis enables reuse
3. Documenting tensor shapes before coding prevented confusion

### What I'd Do Differently
1. Start with simpler visualizations before complex analysis
2. Design probe experiments before extraction (know what you need)
3. Consider memory constraints earlier (full model activations are large)

### Experiments to Run

**High Priority**:
1. Activation statistics across layers (verify norm growth)
2. Attention vs FFN contribution comparison
3. Simple linear probes (capitalization, token length)

**Medium Priority**:
4. Cosine similarity between tokens across layers
5. Principal component analysis of activations
6. Comparison between base and fine-tuned model representations

**Low Priority (Future)**:
7. Nonlinear probes for complex properties
8. Causal interventions (patch activations)
9. Representation similarity analysis across models

## Knowledge Gaps to Address

| Gap | Approach | Priority |
|-----|----------|----------|
| Don't know actual norm growth pattern | Run experiment 1 | High |
| Haven't verified layer-wise abstraction | Train probes per layer | High |
| Uncertain about FFN/attention roles | Measure contributions | High |
| No intuition for representation geometry | Visualize with PCA/UMAP | Medium |
| Don't understand superposition | Read Anthropic papers | Medium |

## Summary

The residual stream view provides a powerful framework for understanding transformers: information flows and accumulates, with attention routing and FFN processing. But much of this is theoretical until verified through probing experiments.

Key insight: The representation at any point is the *sum* of all contributions before it. This enables interpretability work because we can isolate and analyze each contribution.

Next step: Run the representation analysis experiments to ground these concepts in empirical data from our actual model.

## Related

- [Attention Self-Assessment](./attention-self-assessment.md)
- [Training Dynamics Self-Assessment](./training-dynamics-self-assessment.md)
- [Activation Extraction](../../experiments/probing/extract.py)
- [Analysis Experiments](../../experiments/probing/run_analysis.py)
