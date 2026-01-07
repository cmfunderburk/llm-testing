# Concept: QLoRA (Quantized Low-Rank Adaptation)

## Status
- [x] Initial understanding
- [ ] Can explain without reference
- [ ] Can apply in experiments
- [ ] Can teach to others

## Core Idea
QLoRA combines two memory-saving techniques: quantizing the base model to 4-bit precision and training only small low-rank adapter matrices. This allows fine-tuning large language models on consumer GPUs that couldn't otherwise hold the full model in memory.

## Key Components
1. **4-bit Quantization**: Base model weights are stored in 4-bit NormalFloat (NF4) format, reducing memory by ~4x compared to fp16
2. **LoRA Adapters**: Small trainable matrices (rank r, typically 8-64) added to frozen base weights
3. **Double Quantization**: Further compresses quantization constants themselves
4. **Paged Optimizers**: Handles memory spikes during backprop

## How It Works
For a weight matrix W in the base model:
- W is quantized to 4-bit and kept frozen
- Two small matrices B (d×r) and A (r×d) are added as trainable adapters
- Output becomes: W_4bit × x + B × A × x
- Only B and A are trained (typically 1-2% of parameters)

The key insight: most of the model's "knowledge" is in W, which we keep. We just need to learn small adjustments (B×A) for our specific task.

## Why It Matters
- Makes 7B+ models trainable on 16GB VRAM
- Demonstrates that fine-tuning doesn't need full model gradients
- Suggests model capabilities are "mostly there" - we just adjust behavior
- Opens up LLM research to consumer hardware

## Related Concepts
- [[lora]]: The base adapter technique QLoRA builds on
- [[quantization]]: How 4-bit representation works
- [[fine-tuning]]: The broader context of model adaptation

## Experiments That Demonstrate This
- `experiments/fine_tuning/basic_qlora.py`: Shows parameter count difference and basic training

## Questions
- How much capability is lost to quantization?
- Does LoRA rank matter more for some tasks than others?
- Why these specific target modules (q,k,v,o,gate,up,down)?
- How does training dynamic differ from full fine-tuning?

## Sources
- QLoRA paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- LoRA paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- Unsloth documentation for implementation details

---
*Last updated: 2025-01-07*
*Confidence: low*
