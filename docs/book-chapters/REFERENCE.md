# Book Reference: Build a Large Language Model (From Scratch)

**Author:** Sebastian Raschka
**Source:** Manning Publications (2025)

This reference maps book chapters to their text files for easy lookup when building the LLM pretraining experiment track.

---

## Quick Reference

| File | Chapter | Key Topics | Lines |
|------|---------|------------|-------|
| `text/01-*.txt` | Ch 1: Understanding LLMs | LLM overview, transformer intro, project roadmap | 804 |
| `text/02-*.txt` | Ch 2: Working with Text Data | Tokenization, BPE, embeddings, data loaders | 1880 |
| `text/03-*.txt` | Ch 3: Coding Attention | Self-attention, causal attention, multi-head attention | 2538 |
| `text/04-*.txt` | Ch 4: Implementing GPT | Layer norm, GELU, FFN, transformer blocks, full architecture | 2077 |
| `text/05-*.txt` | Ch 5: Pretraining | Loss computation, training loop, text generation, loading weights | 2367 |
| `text/06-*.txt` | Ch 6: Fine-tuning (Classification) | Classification head, spam detection example | 2019 |
| `text/07-*.txt` | Ch 7: Fine-tuning (Instructions) | Instruction datasets, chat/assistant fine-tuning | 2663 |
| `text/appendix-a-*.txt` | Appendix A: PyTorch Intro | Tensors, autograd, neural network basics | 2105 |
| `text/appendix-d-*.txt` | Appendix D: Training Enhancements | LR warmup, cosine decay, gradient clipping | 479 |
| `text/appendix-e-*.txt` | Appendix E: LoRA | Low-rank adaptation, parameter-efficient fine-tuning | 827 |

---

## Chapter Details

### Stage 1: Building the LLM Architecture

#### Chapter 1: Understanding Large Language Models
**File:** `text/01-understanding-large-language-models.txt`

Topics covered:
- High-level LLM concepts
- Transformer architecture overview
- Decoder-only vs encoder-decoder models
- Planning the from-scratch implementation

*Use for:* Conceptual grounding, explaining "why" of architecture decisions

---

#### Chapter 2: Working with Text Data
**File:** `text/02-working-with-text-data.txt`

Topics covered:
- Text preprocessing for LLM training
- Word and subword tokenization
- Byte Pair Encoding (BPE) algorithm
- Sliding window data sampling
- Token-to-vector conversion (embeddings)
- PyTorch DataLoader setup

*Use for:* Implementing tokenizer, creating training data pipeline

---

#### Chapter 3: Coding Attention Mechanisms
**File:** `text/03-coding-attention-mechanisms.txt`

Topics covered:
- Why attention mechanisms work
- Basic self-attention implementation
- Scaled dot-product attention
- Causal (masked) attention for autoregressive generation
- Dropout in attention weights
- Multi-head attention module

*Use for:* Understanding/implementing the core attention mechanism

---

#### Chapter 4: Implementing a GPT Model
**File:** `text/04-implementing-gpt-model.txt`

Topics covered:
- GPT architecture overview
- Layer normalization (Pre-LN style)
- GELU activation function
- Feed-forward network (FFN) blocks
- Shortcut/residual connections
- Transformer block assembly
- Complete GPTModel class
- GPT-2 configuration (124M parameters)
- Parameter counting

Key configuration (GPT-2 124M):
```python
GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}
```

*Use for:* Building the complete model architecture

---

### Stage 2: Pretraining

#### Chapter 5: Pretraining on Unlabeled Data
**File:** `text/05-pretraining-on-unlabeled-data.txt`

Topics covered:
- Evaluating generative text models
- Cross-entropy loss for next-token prediction
- Training vs validation loss
- Perplexity metric
- Training loop implementation
- Text generation strategies:
  - Greedy decoding
  - Temperature scaling
  - Top-k sampling
- Saving/loading model checkpoints
- Loading OpenAI's pretrained GPT-2 weights

*Use for:* Implementing training loop, loss visualization, text generation

---

### Stage 3: Fine-tuning

#### Chapter 6: Fine-tuning for Classification
**File:** `text/06-fine-tuning-for-classification.txt`

Topics covered:
- Fine-tuning approaches overview
- Dataset preparation for classification
- Modifying pretrained LLM for classification
- Spam detection example
- Accuracy evaluation
- Using fine-tuned classifier

*Use for:* Adding classification capability to pretrained model

---

#### Chapter 7: Fine-tuning to Follow Instructions
**File:** `text/07-fine-tuning-to-follow-instructions.txt`

Topics covered:
- Instruction fine-tuning process
- Supervised instruction datasets
- Organizing instruction data in batches
- Chat/assistant model fine-tuning
- Response extraction and evaluation
- Instruction-following evaluation

*Use for:* Creating conversational/assistant models

---

## Appendices

#### Appendix A: Introduction to PyTorch
**File:** `text/appendix-a-intro-to-pytorch.txt`

Topics covered:
- PyTorch setup and GPU support
- Tensor operations
- Automatic differentiation (autograd)
- Backpropagation basics
- Neural network fundamentals

*Use for:* PyTorch refresher, debugging tensor operations

---

#### Appendix D: Training Loop Enhancements
**File:** `text/appendix-d-training-loop-bells-whistles.txt`

Topics covered:
- Learning rate warmup
- Cosine learning rate decay
- Gradient clipping
- Enhanced training function

*Use for:* Improving training stability and convergence

---

#### Appendix E: Parameter-efficient Fine-tuning with LoRA
**File:** `text/appendix-e-lora.txt`

Topics covered:
- Low-rank adaptation concept
- LoRA implementation
- Applying LoRA to linear layers
- Efficient fine-tuning

*Use for:* Parameter-efficient fine-tuning (connects to existing Track A experiments)

---

## Recommended Reading Order for Pretraining Track

1. **Ch 4** - Implement the GPT architecture
2. **Ch 2** - Build the data pipeline
3. **Ch 5** - Implement training loop and generation
4. **Appendix D** - Add training enhancements
5. **Ch 3** - Deep dive into attention (for visualization/debugging)

---

## File Paths

Text extracts location: `docs/book-chapters/text/`

Full filenames:
```
text/00-front-matter.txt
text/01-understanding-large-language-models.txt
text/02-working-with-text-data.txt
text/03-coding-attention-mechanisms.txt
text/04-implementing-gpt-model.txt
text/05-pretraining-on-unlabeled-data.txt
text/06-fine-tuning-for-classification.txt
text/07-fine-tuning-to-follow-instructions.txt
text/appendix-a-intro-to-pytorch.txt
text/appendix-b-references.txt
text/appendix-c-exercise-solutions.txt
text/appendix-d-training-loop-bells-whistles.txt
text/appendix-e-lora.txt
text/index.txt
```

---

*Generated 2026-01-12 for LLM Learning Lab pretraining track*
