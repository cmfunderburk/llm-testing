# Attention Mechanics Self-Assessment

*Track B: Attention Visualization - Final Self-Assessment*

## Status
- [ ] Can explain QKV computation
- [ ] Can explain attention scores
- [ ] Can explain multi-head attention
- [ ] Can explain positional encoding interaction
- [ ] Written Track B retrospective

---

## 1. QKV Computation

### My Understanding

**Query, Key, Value (QKV)** are three different projections of the input:

```
Input: X of shape (seq_len, hidden_dim)

Q = X @ W_q    # What am I looking for?
K = X @ W_k    # What do I contain?
V = X @ W_v    # What information do I provide?
```

**Intuition:**
- **Query (Q)**: "I'm token 5, and I want to find relevant context"
- **Key (K)**: "I'm token 2, here's what I'm about"
- **Value (V)**: "I'm token 2, here's my information to share"

The attention mechanism finds which Keys match each Query, then retrieves the corresponding Values.

**Qwen2.5 Specifics:**
- Uses Grouped Query Attention (GQA)
- 28 Query heads, but only 4 Key-Value heads
- 7 Query heads share each KV head
- Saves memory and compute while maintaining quality

### What I'm Still Unsure About
- Why does GQA work as well as full multi-head attention?
- What exactly do the projection matrices W_q, W_k, W_v learn?
- How do LoRA adapters on q_proj/k_proj/v_proj change attention?

---

## 2. Attention Scores

### My Understanding

Attention scores determine how much each position attends to every other position:

```
Scores = Q @ K.T / sqrt(d_k)     # Dot product, scaled
Weights = softmax(Scores)        # Normalize to probabilities
Output = Weights @ V             # Weighted sum of values
```

**Step by step:**

1. **Dot Product**: Q @ K.T gives similarity between query and all keys
   - High dot product = Query and Key are aligned
   - Shape: (seq_len, seq_len)

2. **Scaling**: Divide by sqrt(d_k) where d_k is key dimension
   - Prevents dot products from getting too large
   - Large values → softmax becomes very peaked
   - sqrt(128) ≈ 11.3 for Qwen's head dimension

3. **Causal Mask**: Set future positions to -infinity
   - Token 5 can't see tokens 6, 7, 8...
   - Creates triangular attention pattern

4. **Softmax**: Convert to probability distribution
   - Each row sums to 1
   - Higher scores → higher probability → more attention

5. **Value Aggregation**: Weighted sum of value vectors
   - Output for position i = sum(attention[i,j] * V[j] for all j)

### Key Insight
Attention is a "soft dictionary lookup":
- Query = what you're searching for
- Keys = index entries
- Values = content at each entry
- Softmax = fuzzy matching (not exact lookup)

### What I'm Still Unsure About
- What happens when attention is very peaked vs uniform?
- How does the model learn when to attend locally vs globally?
- Why does scaling by sqrt(d_k) specifically help?

---

## 3. Multi-Head Attention

### My Understanding

Instead of one attention computation, we do many in parallel:

```
For each head h:
    Q_h = X @ W_q_h    # Separate projections per head
    K_h = X @ W_k_h
    V_h = X @ W_v_h
    Head_h = Attention(Q_h, K_h, V_h)

Output = Concat(Head_1, ..., Head_n) @ W_o
```

**Why multiple heads?**

1. **Different patterns**: Each head can learn different attention patterns
   - Head 1: Attend to previous token (local)
   - Head 2: Attend to sentence start (global)
   - Head 3: Attend to verbs (syntactic)

2. **Subspace specialization**: Each head works in a smaller dimension
   - 28 heads × 128 dims = 3584 total (Qwen's hidden size)
   - More efficient than one 3584-dim attention

3. **Redundancy**: If one head fails, others can compensate

**The Output Projection (W_o)**:
- Combines all head outputs back to hidden dimension
- Allows heads to "communicate" their findings
- This is also a LoRA target module

### What I'm Still Unsure About
- Do heads have interpretable "roles" or is it emergent?
- How do heads coordinate (or do they just average out)?
- Why 28 heads specifically for Qwen-7B?

---

## 4. Positional Encoding Interaction

### My Understanding

Transformers have no inherent notion of position - attention treats input as a set, not a sequence. Positional encodings add position information.

**Qwen uses RoPE (Rotary Position Embeddings):**

Instead of adding position to embeddings, RoPE rotates the Q and K vectors:

```
Q_rotated = rotate(Q, position)
K_rotated = rotate(K, position)
```

**Key property**: The dot product Q·K becomes position-dependent
- Q[pos=5] · K[pos=3] depends on relative position (5-3=2)
- Naturally captures "how far apart" tokens are

**Why RoPE is clever:**
1. **Relative positions**: Attention depends on distance, not absolute position
2. **Extrapolation**: Can handle longer sequences than training
3. **Efficient**: Applied at attention time, not embedding time

**How it affects attention:**
- Nearby tokens get slightly boosted attention (rotation similarity)
- Very far tokens get slightly reduced attention
- But the model can still learn to override this with content similarity

### What I'm Still Unsure About
- Exact math of rotation matrices in RoPE
- How context length limits relate to RoPE
- Does fine-tuning change positional encoding behavior?

---

## 5. Track B Retrospective

### What Was Accomplished

Created attention analysis infrastructure:
1. **Extraction tooling**: Hook-based attention capture for Qwen
2. **Visualization**: Heatmaps, head grids, layer comparisons
3. **Comparison framework**: Base vs fine-tuned analysis setup
4. **Documentation**: Tensor shapes, interpretation guides

### What Was Learned (Theoretical)

- Attention is a soft dictionary lookup (Q→K→V)
- Multi-head allows parallel pattern capture
- RoPE encodes position through rotation
- Qwen uses GQA (grouped query attention) for efficiency

### What Needs Experimental Validation

These theoretical understandings need grounding:
- Do different heads actually capture different patterns?
- Does attention change meaningfully after fine-tuning?
- Can we see the RoPE effect in attention patterns?

### What I'd Do Differently

1. **Start with simple visualization**: Just one head, one layer, before building infrastructure
2. **Compare with known patterns**: Find examples where attention is "obviously" doing something
3. **Run experiments earlier**: Theory without observation is incomplete

### Recommendations for Track C

- Build on attention extraction to also capture activations
- Look for layer-by-layer changes in representation
- Connect probing findings to attention patterns

---

## Self-Assessment Summary

### Confidence Levels

| Topic | Confidence | Evidence |
|-------|------------|----------|
| QKV computation | Medium | Can explain steps; unsure on learning dynamics |
| Attention scores | Medium | Understand formula; unsure on behavior extremes |
| Multi-head attention | Medium | Know structure; unsure on head coordination |
| Positional encoding | Low | Conceptual grasp; can't derive RoPE math |

### What Would Increase Confidence

1. **Run visualizations on real prompts**: See actual attention patterns
2. **Compare across prompts**: See what changes
3. **Read RoPE paper**: Understand the math
4. **Interpret specific heads**: Find "named" patterns

---

*Last updated: 2025-01-07*
*Status: Infrastructure complete, experiments not yet run*
