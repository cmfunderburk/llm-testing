# Concept: Loss Curves in Fine-Tuning

## Status
- [x] Initial understanding
- [ ] Can explain without reference
- [ ] Can apply in experiments
- [ ] Can teach to others

## Core Idea
A loss curve shows how the model's prediction error changes during training. For language models, loss is typically cross-entropy: how surprised the model is by the next token. Lower loss = better predictions.

Loss curves are our primary window into training dynamics - they tell us whether learning is happening, when to stop, and whether something is going wrong.

## Key Concepts

### 1. What Loss Measures
- **Cross-entropy loss**: -log(probability assigned to correct token)
- Higher loss = model assigns low probability to correct answers
- Loss of 2.0 means model gives ~13.5% probability to correct token
- Loss of 1.0 means model gives ~36.8% probability
- Loss of 0.5 means model gives ~60.6% probability

### 2. Typical Loss Curve Shapes

**Exponential Decay (Common)**
```
Loss
 |
 |****
 |    ***
 |       ***
 |          ******
 +-------------------> Steps
```
- Sharp initial drop, then gradual decrease
- Indicates effective learning
- Most fine-tuning runs look like this

**Linear Decay**
```
Loss
 |
 |****
 |    ****
 |        ****
 |            ****
 +-------------------> Steps
```
- Steady, consistent improvement
- Less common in fine-tuning
- May indicate learning rate is too low

**Plateau**
```
Loss
 |
 |****
 |    ***
 |       **********
 |
 +-------------------> Steps
```
- Stops improving
- May need: lower learning rate, different data, more capacity

**Divergence (Bad)**
```
Loss
 |              ***
 |          ****
 |    *****
 |****
 +-------------------> Steps
```
- Loss increasing = learning rate too high
- Stop and restart with lower LR

### 3. Train vs Eval Loss

**Healthy Training**
```
      Train: ----
      Eval:  ....
Loss
 |
 |*...
 | *..
 |  *..
 |   *.....
 +-------------------> Steps
```
- Both decrease together
- Small gap is normal (train always slightly lower)

**Overfitting**
```
Loss
 |
 |*....
 | * ...
 |  *  ....
 |   *     .....
 +-------------------> Steps
```
- Train continues down, eval goes up
- Model memorizing instead of generalizing
- Stop training or add regularization

## Why This Matters
- **Know when to stop**: Stop when eval loss plateaus or starts rising
- **Detect problems early**: Divergence or erratic loss = configuration issue
- **Compare configurations**: Better configuration = lower final loss + faster convergence
- **Build intuition**: Predict what loss patterns mean for model behavior

## Related Concepts
- [[learning-rate]]: Affects loss curve shape
- [[overfitting]]: Detected via train/eval divergence
- [[qlora]]: Context for these experiments

## Experiments That Demonstrate This
- `experiments/fine_tuning/loss_curve_analysis.py`: Captures and visualizes loss curves
- `experiments/fine_tuning/basic_qlora.py`: Basic training with loss observation

## Questions
- What final loss value corresponds to "good enough" behavior?
- How sensitive is the loss curve to random seed?
- Is eval loss a reliable proxy for actual capability?
- How do loss curves differ between task types (instruction following vs QA vs code)?

## Interpreting Your Results

After running an experiment, ask:

1. **What was the loss reduction?**
   - >50% reduction: Strong learning
   - 20-50%: Moderate learning
   - <20%: May need different configuration

2. **Did it plateau?**
   - Yes: Training converged, can stop here
   - No: Could train longer

3. **Train/eval gap?**
   - <10% gap: Healthy
   - 10-30% gap: Monitor for overfitting
   - >30% gap: Likely overfitting

4. **Was curve smooth?**
   - Smooth: Good batch size / LR balance
   - Noisy: Consider larger batch or gradient accumulation

## Sources
- Empirical observation from fine-tuning experiments
- Understanding comes from experiments, not just theory
- Visualizations are essential - always plot your losses

---
*Last updated: 2025-01-07*
*Confidence: low - needs experimental validation*
