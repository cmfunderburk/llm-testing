# Training Dynamics Self-Assessment

*Track A: Fine-Tuning Mechanics - Final Self-Assessment*

## Status
- [ ] Can explain forward/backward pass
- [ ] Can explain loss landscapes
- [ ] Can explain optimizer state
- [ ] Can explain batch size/LR interaction
- [ ] Answered minimum interesting experiment question
- [ ] Written Track A retrospective

---

## 1. Forward Pass

### My Understanding

The forward pass is how the model generates predictions:

1. **Input tokenization**: Text → token IDs
2. **Embedding lookup**: Token IDs → embedding vectors
3. **Transformer layers** (repeated N times):
   - Layer normalization
   - Multi-head attention: tokens "attend" to each other
   - Residual connection: add attention output to input
   - Layer normalization
   - Feed-forward network: per-token transformation
   - Residual connection: add FFN output to input
4. **Final layer norm**
5. **Output projection**: Hidden states → vocabulary logits
6. **Softmax**: Logits → probability distribution over next token

### Key Insight
The residual connections are crucial - they let information flow directly through
the network without being forced through every transformation. This is why we can
add LoRA adapters: they contribute to the residual stream additively.

### What I'm Still Unsure About
- Exact role of layer normalization
- How attention patterns differ across layers
- What information is encoded at each layer

---

## 2. Backward Pass

### My Understanding

The backward pass computes gradients for learning:

1. **Compute loss**: Compare predictions to targets (cross-entropy for next token prediction)
2. **Backpropagate gradients**:
   - Start at loss, work backward through network
   - Chain rule: multiply local gradients together
   - Each layer receives gradient of loss w.r.t. its output
   - Each layer computes gradient w.r.t. its parameters and inputs
3. **Accumulate gradients**: Sum gradients across all examples in batch
4. **Apply gradients**: Optimizer uses gradients to update parameters

### Key Insight
With QLoRA, gradients only flow to LoRA adapter parameters. Base model weights are
frozen, so their gradients aren't computed (saving memory and compute). The adapter
gradients still depend on the full forward pass through the base model.

### What I'm Still Unsure About
- Gradient clipping mechanics
- How gradient checkpointing saves memory
- Numerical stability concerns

---

## 3. Loss Landscapes

### My Understanding

The loss landscape is a visualization of how loss changes with parameters:

- **High-dimensional**: One dimension per parameter (billions for LLMs!)
- **Non-convex**: Many local minima, saddle points
- **Shaped by data**: Different datasets create different landscapes

Key concepts:
- **Minima**: Points where gradient is zero, loss is low
- **Saddle points**: Gradient zero but not minimum (flat in some directions)
- **Valleys**: Regions where many parameter configurations give similar loss
- **Sharpness**: How quickly loss increases when moving from minimum

### Relevance to Fine-Tuning

Pre-trained models start in a "good" region of loss landscape. Fine-tuning:
- Navigates to nearby minima suited for our task
- LoRA restricts movement to low-rank subspace
- Lower LR = smaller steps = stay in same basin
- Higher LR = larger steps = might jump to different basin

### What I'm Still Unsure About
- How to visualize loss landscapes (dimensionality reduction?)
- Relationship between sharpness and generalization
- How batch normalization/layer norm affect the landscape

---

## 4. Optimizer State

### My Understanding

Optimizers maintain state beyond just the parameters:

**SGD**: No state (just uses current gradient)

**Adam** (what we use):
- **Momentum (m)**: Exponential moving average of gradients
  - Helps overcome noise, maintains direction
  - Like a ball rolling downhill with inertia
- **Velocity (v)**: Exponential moving average of squared gradients
  - Adapts learning rate per-parameter
  - Parameters with consistently large gradients get smaller updates
- **Beta values**: Control how much history to keep (β₁=0.9, β₂=0.999 typical)

**AdamW 8-bit** (what QLoRA uses):
- Same as Adam but:
  - Decoupled weight decay (more principled)
  - 8-bit quantized state (saves memory)

### Key Insight
Optimizer state can be 2-3x the size of model parameters! For a 7B model in fp16,
that's potentially 28-42GB just for optimizer. 8-bit quantization reduces this
dramatically, enabling training on consumer GPUs.

### What I'm Still Unsure About
- Exactly how 8-bit quantization affects training quality
- When to reset optimizer state
- How warmup interacts with momentum

---

## 5. Batch Size / Learning Rate Interaction

### My Understanding

Batch size and learning rate are coupled:

**Larger batch**:
- More accurate gradient estimate (average over more examples)
- Can use higher learning rate (more confident in direction)
- Linear scaling rule: if batch × 2, LR × 2
- But: diminishing returns, may need warmup

**Smaller batch**:
- Noisier gradients (fewer examples)
- Need lower learning rate (less confident)
- More regularization effect (noise helps generalization)
- More parameter updates per epoch

**Effective batch size** = batch_size × gradient_accumulation
- Gradient accumulation simulates larger batches on limited memory
- Our setup: 4 × 4 = 16 effective batch size

### QLoRA Specifics

LoRA typically uses higher LR (1e-4 to 5e-4) than full fine-tuning because:
- Fewer parameters being updated
- Starting from random init (adapters), not pre-trained weights
- Need stronger signal to move from initialization

### What I'm Still Unsure About
- Optimal batch size for quality vs speed tradeoff
- When linear scaling rule breaks down
- Effect on training stability

---

## 6. Minimum Interesting Experiment

### The Question
"What's the minimum experiment that demonstrates something interesting about training dynamics?"

### My Answer

**The minimum interesting experiment is: 50 examples, 1 epoch, compare base vs fine-tuned on 3 targeted prompts.**

### Rationale

1. **Why 50 examples?**
   - Enough to show measurable loss decrease
   - Fast enough to iterate quickly (~2-5 min)
   - Demonstrates that learning happens even with tiny data

2. **Why 1 epoch?**
   - See the full loss curve shape
   - Avoid overfitting complications
   - Single pass = cleaner signal

3. **Why 3 targeted prompts?**
   - Not trying to measure "general improvement"
   - Want to see: does the model behave differently?
   - Pick prompts related to training data style

4. **What makes it "interesting"?**
   - Shows learning happened (loss decreased)
   - Shows behavior changed (qualitative difference in outputs)
   - Raises questions: how much data for bigger changes? what's not changing?

### Alternative Minimum Experiments

- **Loss curve only**: 100 examples, just observe loss (no inference)
- **A/B comparison**: Same prompt to base and fine-tuned, side by side
- **Extreme LR test**: Same setup but 10x learning rate - see what breaks

---

## 7. Track A Retrospective

### What Was Accomplished

Created infrastructure for 4 training dynamics experiments:
1. Loss curve analysis - detailed logging and visualization
2. Learning rate exploration - compare 4 LR values
3. LoRA rank comparison - compare 4 rank values
4. Catastrophic forgetting test - benchmark preservation

### What Was Learned (So Far)

*Note: Experiments haven't been run yet - this is infrastructure-only at this point.*

Theoretical understanding:
- How QLoRA reduces memory through quantization and low-rank adapters
- Why LoRA uses higher learning rates
- Relationship between loss and learning
- Theory behind low-rank adaptation

### What Questions Were Answered

Partially answered through code/research:
- How to log and visualize loss curves ✓
- How to compare hyperparameter configurations ✓
- How to test for catastrophic forgetting ✓

### What Questions Remain Open

- What does the actual loss curve look like on our hardware?
- Is 2e-4 really optimal for our setup?
- How much does rank matter for Alpaca-style tasks?
- Does QLoRA truly prevent forgetting?

### What I'd Do Differently

- Would run quick experiments earlier to ground theory in observation
- Would create simpler "minimum experiment" first before elaborate infrastructure
- Would focus on one experiment deeply before creating all four

### Recommendations for Track B

- Start with actual experiments, not just code
- Run the minimum interesting experiment first
- Let observations drive the next experiment, rather than pre-planning everything

---

## Self-Assessment Summary

### Confidence Levels

| Topic | Confidence | Evidence |
|-------|------------|----------|
| Forward pass | Medium | Can explain steps; uncertain on details |
| Backward pass | Medium | Understand chain rule; unsure on implementation |
| Loss landscapes | Low | Conceptual understanding; can't visualize |
| Optimizer state | Medium | Know what Adam tracks; unsure on dynamics |
| Batch/LR interaction | Medium | Know the rules; haven't tested limits |
| Minimum experiment | High | Clear rationale; testable hypothesis |

### What Would Increase Confidence

1. **Run the experiments** - seeing real loss curves
2. **Visualize attention** - Track B will help
3. **Debug a training run** - understanding through failure
4. **Try extreme configurations** - see what breaks

---

*Last updated: 2025-01-07*
*Status: Infrastructure complete, experiments not yet run*
