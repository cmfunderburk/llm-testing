# Hardware Decision Document

## Current Setup

**GPU**: NVIDIA RTX 5060 Ti 16GB (Blackwell architecture)
**Key Specs**:
- 16GB VRAM
- Blackwell architecture (latest generation consumer)
- Good for: 7B models with 4-bit quantization, inference, light fine-tuning

## Track-by-Track Compute Analysis

### Track A: Fine-Tuning Mechanics

| Experiment | VRAM Usage (Est.) | Compute Time | Bottleneck? |
|------------|-------------------|--------------|-------------|
| Basic QLoRA (Qwen2.5-7B, 4-bit) | ~10-12GB | Minutes/epoch | No |
| Loss curve analysis | Same as above | Minutes | No |
| Learning rate sweep (4 configs) | ~12GB each | ~1 hour total | Minor (sequential) |
| LoRA rank comparison (4 ranks) | ~10-14GB | ~1 hour total | Minor |
| Forgetting evaluation | ~10GB (inference) | Minutes | No |

**Track A Assessment**: Current hardware is **adequate**. QLoRA with 4-bit quantization fits comfortably. The main limitation is running sweeps sequentially rather than in parallel, adding hours not days.

### Track B: Attention Visualization

| Experiment | VRAM Usage (Est.) | Compute Time | Bottleneck? |
|------------|-------------------|--------------|-------------|
| Attention extraction | ~10GB | Seconds/sample | No |
| Visualization generation | CPU-bound | Seconds | No |
| Base vs fine-tuned comparison | ~10GB | Minutes | No |

**Track B Assessment**: Current hardware is **more than adequate**. Attention extraction is lightweight. No compute bottlenecks identified.

### Track C: Representation Probing

| Experiment | VRAM Usage (Est.) | Compute Time | Bottleneck? |
|------------|-------------------|--------------|-------------|
| Activation extraction | ~10-12GB | Seconds/sample | No |
| Statistics computation | CPU-bound | Seconds | No |
| Linear probe training | ~1-2GB | Minutes | No |

**Track C Assessment**: Current hardware is **adequate**. Probing experiments are lightweight.

## What Would Require More Compute?

### Experiments NOT Currently Feasible

1. **Full fine-tuning (no LoRA)**
   - Requires: ~40-60GB VRAM for 7B model
   - Current limitation: Cannot compare full fine-tuning vs LoRA

2. **Larger models (13B, 70B)**
   - 13B quantized: ~18-24GB (borderline)
   - 70B quantized: ~40GB+ (not feasible)
   - Current limitation: Cannot explore scaling effects

3. **Full precision training**
   - Requires: 4x current VRAM
   - Current limitation: Must use quantization

4. **Batch size experimentation**
   - Larger batches require more VRAM
   - Current limitation: Batch size constrained

5. **Training on longer sequences**
   - 2048+ tokens: VRAM scales quadratically with attention
   - Current limitation: Sequence length constrained

### Research Directions That Would Benefit

| Direction | Resource Need | Impact |
|-----------|--------------|--------|
| Model scaling experiments | 24-48GB GPU | High (understand scale) |
| Full fine-tune comparisons | 48GB+ GPU | Medium (academic interest) |
| Multi-GPU training | 2+ GPUs | Medium (parallelism) |
| Longer context experiments | 24GB+ GPU | Medium (context window) |
| Faster iteration | A100/H100 | Quality of life |

## Hardware Upgrade Options

### Option 1: No Upgrade
- **Cost**: $0
- **Capability**: 7B quantized models, QLoRA fine-tuning
- **Limitation**: Can't explore scaling, full fine-tuning

### Option 2: RTX 4090 / RTX 5090 (24GB)
- **Cost**: $1,500-2,000
- **Capability**: 13B quantized, slightly larger batches
- **Gain**: Marginal improvement, still consumer-tier

### Option 3: Used A100 40GB
- **Cost**: $5,000-8,000
- **Capability**: 30B quantized, full fine-tune 7B
- **Gain**: Opens most experiments, datacenter performance

### Option 4: Cloud Computing (on-demand)
- **Cost**: $2-4/hour (A100), pay-per-use
- **Capability**: Any size model
- **Gain**: Maximum flexibility, no capital investment
- **Downside**: Ongoing cost, less convenient

## Recommendation

### Primary Recommendation: **Stay with Current Hardware + Cloud Burst**

**Rationale**:

1. **Current hardware handles the learning objectives**: The RTX 5060 Ti 16GB can run all Track A-D experiments as designed. The learning goals (mental model formation, paper reading fluency, experimental intuition) don't require larger models.

2. **Diminishing returns on consumer hardware**: Upgrading to 24GB gains little. The jump from 16GB to 24GB doesn't unlock qualitatively different experiments.

3. **Cloud for occasional needs**: If an experiment requires more compute (e.g., reproducing a paper claim at scale), use cloud A100s for that specific run. Cost: $20-50 per experiment session.

4. **Defer capital investment**: After completing Track D and the full learning journey, reassess. By then, you'll know:
   - What research directions genuinely interest you
   - Whether you need compute for actual research vs. learning
   - What the landscape looks like (new GPU generations, cloud pricing)

### When to Reconsider

Upgrade to dedicated research hardware (A100+) if:
- You pursue research requiring repeated large-scale experiments
- Cloud costs exceed ~$200/month consistently
- A specific project requires always-on large model access

## Summary

| Question | Answer |
|----------|--------|
| Can I complete the LLM Learning Lab? | **Yes**, with current hardware |
| What am I missing? | Full fine-tuning, 13B+ models, long context |
| Should I upgrade now? | **No** - use cloud for occasional needs |
| When to upgrade? | After Track D, if pursuing compute-intensive research |

**Decision**: Proceed with current RTX 5060 Ti 16GB. Use cloud burst computing for experiments that exceed local capacity. Reassess after completing all learning tracks.

## Related

- [Track A: Training Dynamics Self-Assessment](./concepts/training-dynamics-self-assessment.md)
- [Track B: Attention Self-Assessment](./concepts/attention-self-assessment.md)
- [Track C: Representations Self-Assessment](./concepts/representations-self-assessment.md)
