# MicroGPT Line-by-Line Guide (Canonical Companion)

This document is the educational companion to `misc/microgpt.py`.

Canonicality policy:
- The code in `misc/microgpt.py` is the source of truth.
- This guide is indexed by line numbers in that file.
- If line numbers drift, update this guide immediately.

## Scope and Perspective

`microgpt.py` is intentionally minimal. It combines:
- dataset loading,
- tokenizer construction,
- scalar autograd,
- transformer forward pass,
- Adam optimization,
- autoregressive sampling,

in one pure-Python file.

The educational value is not engineering efficiency. It is algorithmic transparency.

## Mathematical Notation Used Here

- Vocabulary: \(\mathcal{V}\), with size \(|\mathcal{V}|\)
- Token sequence: \(x_0, x_1, \dots, x_T\)
- Embedding width: \(d\)
- Heads: \(H\), head dimension \(d_h = d/H\)
- Logits at time \(t\): \(z_t \in \mathbb{R}^{|\mathcal{V}|}\)
- Probabilities: \(p_t = \mathrm{softmax}(z_t)\)
- Per-token NLL: \(-\log p_t[x_{t+1}]\)
- Sequence loss: \(L = -\frac{1}{n}\sum_{t=0}^{n-1} \log p_t[x_{t+1}]\)

---

## Detailed Walkthrough

### Lines 1-7: Module docstring

- **1-7**: Declares the design thesis: complete GPT training/inference algorithm in one dependency-free script.

### Lines 9-12: Imports and seeding

- **9** `import os`: Used for filesystem existence checks.
- **10** `import math`: Supplies scalar elementary functions (`log`, `exp`, etc.) used by autograd primitives.
- **11** `import random`: Supplies initialization sampling and categorical sampling for generation.
- **12** `random.seed(42)`: Fixes Python RNG stream for reproducibility of initialization and document shuffling.

Reproducibility note:
- Given identical Python version and identical input file, this makes runs deterministic with respect to this script's stochastic calls.

### Lines 14-21: Dataset acquisition and loading

- **14**: Conceptual declaration: dataset is list of documents (`list[str]`).
- **15** `if not os.path.exists('input.txt'):`: Lazy bootstrap path if no local corpus exists.
- **16** `import urllib.request`: Deferred import (only needed on bootstrap).
- **17** `names_url = ...`: Default corpus source (names list).
- **18** `urllib.request.urlretrieve(...)`: Downloads corpus file to `input.txt`.
- **19** list-comprehension load pipeline:
  - read whole file,
  - split on newline,
  - strip whitespace,
  - discard empty rows.
- **20** `random.shuffle(docs)`: Randomizes document order to reduce training-order bias.
- **21** logs corpus size.

Statistical interpretation:
- Shuffling approximates i.i.d. presentation order over epochs for SGD-like optimization.

### Lines 23-27: Character tokenizer + BOS token

- **24** `uchars = sorted(set(''.join(docs)))`:
  - constructs observed character support set,
  - sorted for stable index assignment.
- **25** `BOS = len(uchars)`: Reserve one additional token id for BOS/end marker behavior.
- **26** `vocab_size = len(uchars) + 1`: Total categorical support size.
- **27** logs vocabulary size.

Formalization:
- Let \(\Sigma\) be unique observed characters.
- Token id map is bijection \(f: \Sigma \cup \{\mathrm{BOS}\} \to \{0,\dots,|\Sigma|\}\).

### Lines 29-73: Scalar autograd engine (`Value`)

#### Lines 30-37: Data model

- **30** class `Value`: Scalar node in computation graph.
- **31** `__slots__`: Memory optimization, avoids dynamic attribute dict per node.
- **33** constructor stores:
  - **34** `data`: primal scalar value,
  - **35** `grad`: adjoint (initialized to 0),
  - **36** `_children`: direct dependencies,
  - **37** `_local_grads`: local partial derivatives \(\partial v/\partial u_i\).

Interpretation:
- Each node stores exactly what reverse-mode AD needs: graph connectivity + local Jacobian entries.

#### Lines 39-57: Overloaded scalar ops

- **39-41** addition:
  - forward: \(z=x+y\),
  - local derivatives: \(\partial z/\partial x = 1\), \(\partial z/\partial y = 1\).
- **43-45** multiplication:
  - forward: \(z=xy\),
  - local derivatives: \(\partial z/\partial x = y\), \(\partial z/\partial y = x\).
- **47** power: \(z=x^a\), derivative \(a x^{a-1}\).
- **48** log: derivative \(1/x\).
- **49** exp: derivative \(e^x\).
- **50** ReLU:
  - forward: \(\max(0,x)\),
  - subgradient used: \(\mathbb{1}_{x>0}\).
- **51-57** convenience operators (`-`, `/`, reflected forms) reduced to core primitives.

Computational consequence:
- Any composed scalar expression builds a dynamic DAG of `Value` nodes.

#### Lines 59-73: Reverse-mode backprop

- **59** `backward(self)` computes gradients of output node wrt all ancestors.
- **60-68** topological ordering:
  - DFS traversal from output node,
  - append node after descendants,
  - yields postorder suitable for reverse pass.
- **69** seed output gradient to 1 (\(\partial L/\partial L = 1\)).
- **70-72** reverse accumulation:
  - for each node \(v\), propagate
  - \(\bar{u}_i += (\partial v/\partial u_i)\,\bar{v}\)
  - exactly chain rule on DAG.

AD complexity:
- Time: linear in number of edges in traced graph.
- Memory: stores full graph for current forward pass.

### Lines 74-90: Model parameter initialization

- **75-79** architecture hyperparameters:
  - embedding width \(d=16\),
  - heads \(H=4\),
  - layers \(L=1\),
  - context window \(B=16\),
  - head width \(d_h = d/H = 4\).
- **80** `matrix(...)`: helper returning matrix of `Value` scalars sampled from \(\mathcal{N}(0, 0.08^2)\).
- **81** base matrices:
  - token embedding `wte` shape \((|\mathcal{V}|, d)\),
  - positional embedding `wpe` shape \((B, d)\),
  - output projection `lm_head` shape \((|\mathcal{V}|, d)\).
- **82-88** per-layer weights:
  - attention projections `wq`, `wk`, `wv`, `wo` each \((d,d)\),
  - MLP `fc1` \((4d, d)\), `fc2` \((d,4d)\).
- **89** flatten all scalar parameters into one list.
- **90** logs parameter count.

Why flatten?
- Simplifies optimizer loop over heterogeneous matrices.

### Lines 92-107: Core primitives used by transformer

- **94-95** `linear(x, w)`:
  - computes matrix-vector multiply
  - each output row `wo` gives dot product \(\langle w_o, x\rangle\).
- **97-101** `softmax(logits)`:
  - **98** subtract max logit \(m\) for numerical stability,
  - **99** exponentiate shifted logits,
  - **100** sum normalizer,
  - **101** divide to get probability simplex point.

Stability identity:
- \(\mathrm{softmax}(z) = \mathrm{softmax}(z - m\mathbf{1})\), preserving probabilities while avoiding overflow.

- **103-106** `rmsnorm(x)`:
  - mean square \(\mathrm{ms} = \frac{1}{d}\sum_i x_i^2\),
  - scale \((\mathrm{ms}+\epsilon)^{-1/2}\),
  - normalized vector \(x_i \cdot \mathrm{scale}\).

Contrast with LayerNorm:
- RMSNorm normalizes by magnitude without centering by mean.

### Lines 108-145: GPT single-step forward pass

This function predicts next-token logits for one position, given current token id and cached keys/values.

#### Lines 109-113: Input representation

- **109** token embedding lookup.
- **110** positional embedding lookup.
- **111** elementwise sum (token + position signal).
- **112** RMS normalization before block stack.

#### Lines 114-134: Attention sub-block

- **114** iterate over layers.
- **116** save residual stream.
- **117** pre-attention RMSNorm.
- **118-120** compute Q, K, V via linear projections.
- **121-122** append current timestep K and V to per-layer caches.

Causal structure:
- At position \(t\), cache contains \(\{K_0,\dots,K_t\}\) and \(\{V_0,\dots,V_t\}\).
- No future keys exist in cache, so causality is enforced implicitly.

- **124** loop heads.
- **125** head start offset in concatenated channel axis.
- **126** current query head slice \(q_h \in \mathbb{R}^{d_h}\).
- **127-128** stacked past keys/values for head \(h\).
- **129** attention logits:
  - \(\ell_t = q_h^\top k_t / \sqrt{d_h}\)
  - for all available timesteps.
- **130** softmax over time -> attention weights.
- **131** weighted value sum -> head output vector.
- **132** append head output to concatenated multi-head vector.
- **133** output projection \(W_O\).
- **134** residual add.

#### Lines 135-141: MLP sub-block

- **136** residual capture.
- **137** pre-MLP RMSNorm.
- **138** expand to \(4d\) hidden width (`fc1`).
- **139** ReLU nonlinearity.
- **140** project back to \(d\) (`fc2`).
- **141** residual add.

#### Lines 143-144: LM head projection

- **143** map residual stream to vocabulary logits.
- **144** return logits.

Functional form:
- For token at position \(t\):
  - \(h_t = \mathrm{TransformerBlock}(e_{x_t}+p_t; K_{\le t}, V_{\le t})\)
  - \(z_t = W_{\text{lm}} h_t\)

### Lines 146-150: Adam buffers

- **147** optimizer hyperparameters:
  - base LR \(\eta=10^{-2}\),
  - \(\beta_1=0.85\), \(\beta_2=0.99\),
  - \(\epsilon=10^{-8}\).
- **148** initialize first-moment vector \(m_0=0\).
- **149** initialize second-moment vector \(v_0=0\).

### Lines 151-185: Training loop

#### Lines 152-158: Build token training example

- **152** fixed number of optimization steps.
- **153** iterate SGD steps.
- **156** choose document cyclically via modulo.
- **157** tokenize characters; wrap with BOS at both ends.
- **158** effective unroll length \(n = \min(B, \text{len(tokens)}-1)\).

Why BOS on both ends?
- Prefix BOS initializes generation context.
- Suffix BOS makes sequence termination an explicit prediction target.

#### Lines 160-169: Forward objective construction

- **161** reset per-layer KV caches for this sequence.
- **162** collect per-position losses.
- **163** iterate positions up to unroll length.
- **164** define autoregressive pair \((x_t, x_{t+1})\).
- **165** compute logits from model.
- **166** convert logits to probabilities.
- **167** negative log-likelihood term for correct next token.
- **168** append per-position loss.
- **169** average losses over positions.

Objective:
- \(L(\theta) = -\frac{1}{n}\sum_{t=0}^{n-1} \log p_\theta(x_{t+1}\mid x_{\le t})\)

Gradient fact:
- For softmax + NLL, logit gradient is
  - \(\partial L/\partial z_i = p_i - \mathbb{1}_{i=y}\)
  per position (scaled by averaging factor).

#### Lines 171-182: Backprop + Adam update

- **172** reverse-mode gradient accumulation from scalar loss through entire unrolled graph.
- **175** linearly decayed learning rate: \(\eta_t = \eta_0(1 - t/T)\).
- **176** iterate all parameters.
- **177** first moment update:
  - \(m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t\)
- **178** second moment update:
  - \(v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2\)
- **179-180** bias correction:
  - \(\hat m_t = m_t/(1-\beta_1^t)\),
  - \(\hat v_t = v_t/(1-\beta_2^t)\)
- **181** parameter update:
  - \(\theta \leftarrow \theta - \eta_t \hat m_t/(\sqrt{\hat v_t}+\epsilon)\)
- **182** zero gradient for next step.

#### Line 184: Logging

- **184** prints step index and scalar mean loss.

### Lines 186-200: Inference loop

- **187** temperature \(T\) controls sampling entropy.
- **188** header print.
- **189** draw multiple samples.
- **190** clear KV caches per sample.
- **191** start generation at BOS token.
- **192** initialize output char buffer.
- **193** rollout up to `block_size` tokens.
- **194** model logits for current token.
- **195** temperature-adjusted softmax.
- **196** categorical sample from probabilities.
- **197-198** if sampled BOS, stop sequence.
- **199** otherwise decode char and append.
- **200** print generated string.

Sampling theory:
- As \(T \downarrow 0\), distribution sharpens toward argmax.
- As \(T \uparrow\), distribution flattens, increasing diversity and error rate.

---

## Conceptual Integrity Checks

To verify understanding, confirm the following claims directly from the code:

1. **Causality enforcement is structural, not mask-based**.
- The cache only stores past keys/values; future positions are never inserted before current prediction.

2. **Autograd is scalar and exact for represented ops**.
- Every primitive registers local derivatives; reverse traversal applies chain rule exactly on built DAG.

3. **Training objective is true autoregressive MLE**.
- Loss is mean NLL over next-token targets from left-context predictions.

4. **Optimization is Adam with bias correction and LR decay**.
- Both moments and correction factors are explicitly implemented.

5. **Inference uses ancestral sampling**.
- Next token sampled from model's categorical distribution, then fed back recursively.

---

## Pedagogical Extensions (Optional)

If you want to extend this file while preserving educational minimalism:

1. Add gradient clipping before Adam update and observe stability.
2. Replace ReLU with GeLU and compare sample quality trajectories.
3. Add train/val split and track held-out NLL.
4. Add weight tying (`lm_head = wte^T`) and compare parameter count + behavior.
5. Add explicit causal mask path (no cache) and verify equivalence on short sequences.

These modifications preserve the same mathematical pipeline while exposing important design trade-offs.
