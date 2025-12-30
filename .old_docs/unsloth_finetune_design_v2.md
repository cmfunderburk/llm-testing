# Unsloth QLoRA Fine-Tuning Guide v2

A beginner-friendly guide to fine-tuning Qwen2.5-7B with custom instruction styles using Unsloth on an NVIDIA 5060 Ti 16GB GPU.

---

## Table of Contents

1. [Overview](#1-overview)
   - 1.5 [Is Fine-Tuning Right For You?](#15-is-fine-tuning-right-for-your-use-case)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Quick Start: Your First Fine-Tune](#3-quick-start-your-first-fine-tune)
4. [Understanding Key Concepts](#4-understanding-key-concepts)
5. [Repository Structure](#5-repository-structure)
6. [Finding & Preparing Training Data](#6-finding--preparing-training-data)
7. [Training Configuration & Hyperparameters](#7-training-configuration--hyperparameters)
8. [Core Training Script](#8-core-training-script)
9. [Inference & Testing](#9-inference--testing)
10. [Export to Ollama](#10-export-to-ollama)
11. [Evaluation: Side-by-Side Comparison](#11-evaluation-side-by-side-comparison)
12. [Gradio Comparison UI](#12-gradio-comparison-ui)
13. [Advanced: Custom Data Strategies](#13-advanced-custom-data-strategies)
14. [Troubleshooting & Common Issues](#14-troubleshooting--common-issues)
15. [Next Steps & Resources](#15-next-steps--resources)
16. [Quick Reference](#16-quick-reference)

---

## 1. Overview

### What You'll Achieve

By following this guide, you will:
- Fine-tune Qwen2.5-7B to adopt a custom writing style or domain expertise
- Export your model to Ollama for easy local inference
- Compare your fine-tuned model against base models side-by-side

### Target Configuration

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Base Model** | Qwen2.5-7B-Instruct | Best balance of quality and VRAM for 16GB GPU |
| **Method** | QLoRA (4-bit quantization + LoRA) | 70% less VRAM, 2x faster training |
| **Framework** | Unsloth | Optimized kernels, easy API |
| **Hardware** | NVIDIA 5060 Ti 16GB | Comfortable fit with ~5GB headroom |
| **Deployment** | Ollama (GGUF export) | Simple local inference |

### Workflow Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. SETUP       │────▶│  2. TRAIN       │────▶│  3. EXPORT      │
│  Environment    │     │  Fine-tune      │     │  To GGUF/Ollama │
│  + test run     │     │  with your data │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                                               ┌─────────────────┐
                                               │  4. COMPARE     │
                                               │  Side-by-side   │
                                               │  evaluation     │
                                               └─────────────────┘
```

---

## 1.5 Is Fine-Tuning Right For Your Use Case?

Before investing time in fine-tuning, consider whether it's the right approach:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING DECISION TREE                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                    Have you tried prompting?
                                │
              ┌─────────────────┴─────────────────┐
              │ NO                                │ YES
              ▼                                   ▼
    ┌─────────────────────┐          Does prompting achieve
    │ TRY PROMPTING FIRST │          your goals?
    │                     │                       │
    │ • System prompts    │         ┌─────────────┴─────────────┐
    │ • Few-shot examples │         │ YES                       │ NO
    │ • Structured output │         ▼                           ▼
    │   instructions      │   ┌───────────────┐     Is the issue...
    └─────────────────────┘   │ DON'T         │           │
                              │ FINE-TUNE     │     ┌─────┴─────┐
                              │               │     │           │
                              │ Prompting is  │  Missing     Style/
                              │ simpler and   │  knowledge?   Format?
                              │ more flexible │     │           │
                              └───────────────┘     ▼           ▼
                                            ┌──────────┐  ┌──────────┐
                                            │ TRY RAG  │  │FINE-TUNE │
                                            │          │  │          │
                                            │ Retrieval│  │ This     │
                                            │ Augmented│  │ guide is │
                                            │ Gen.     │  │ for you! │
                                            └──────────┘  └──────────┘
```

**When fine-tuning makes sense:**
- Consistent output format that prompting can't reliably produce
- Specific writing style or tone
- Domain-specific terminology or jargon
- Reducing token usage (style baked in vs. long system prompts)

**When to avoid fine-tuning:**
- Need for factual knowledge (use RAG instead)
- One-off tasks (just prompt well)
- Rapidly changing requirements (prompts are easier to update)
- When you have <50 quality examples

---

## 2. Prerequisites & Environment Setup

### System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on 5060 Ti)
- **CUDA**: 12.4 or compatible version
- **RAM**: 32GB recommended (16GB minimum)
- **Storage**: ~50GB for models and checkpoints
- **OS**: Linux (recommended) or WSL2

### Installation

We use **uv** for fast, reliable Python package management.

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create project and virtual environment
mkdir qwen-finetune && cd qwen-finetune
uv init
uv venv --python 3.11
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.4
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install Unsloth (includes transformers, peft, trl, bitsandbytes)
uv pip install unsloth

# 5. Install additional utilities
uv pip install datasets gradio wandb httpx pyyaml

# 6. Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

python -c "import unsloth; print('Unsloth installed successfully')"
```

**Why uv?**
- 10-100x faster than pip
- Reliable dependency resolution
- Built-in virtual environment management
- Drop-in replacement for pip commands

### Install Ollama (for deployment)

```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Verify
ollama --version

# Start server (runs in background)
ollama serve &

# Pull base model for comparison
ollama pull qwen2.5:7b
```

---

## 3. Quick Start: Your First Fine-Tune

**Goal**: Run a complete fine-tuning cycle with a small public dataset to learn the workflow before using your own data.

### 3.1 Choose a Test Dataset

| Dataset | Size | Type | Best For |
|---------|------|------|----------|
| `yahma/alpaca-cleaned` | 52K | Instruction-following | General test, learning workflow |
| `teknium/OpenHermes-2.5` | 1M | Instruction-following | Diverse instructions |
| `HuggingFaceH4/ultrachat_200k` | 200K | Conversational | Multi-turn dialogue |
| `Open-Orca/SlimOrca` | 518K | Instruction-following | Reasoning tasks |

For your first test, we'll use a small subset of **alpaca-cleaned** (fast, well-understood).

### 3.2 Minimal Test Script

Create `test_finetune.py`:

```python
"""
Minimal test fine-tune script.
Uses 500 examples from Alpaca for a quick learning run (~10-15 min on 5060 Ti).
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ============================================
# 1. Load Model (4-bit quantized)
# ============================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=1024,  # Shorter for test
    load_in_4bit=True,
    dtype=None,
)

# ============================================
# 2. Add LoRA Adapters
# ============================================
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # Lower rank for faster test
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

trainable, total = model.get_nb_trainable_parameters()
print(f"Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ============================================
# 3. Setup Chat Template
# ============================================
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# ============================================
# 4. Load Test Dataset (500 examples)
# ============================================
print("Loading dataset...")
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:500]")

def format_alpaca(example):
    """Convert Alpaca format to chat format."""
    messages = []

    # Combine instruction and input
    if example.get("input", "").strip():
        user_content = f"{example['instruction']}\n\nInput: {example['input']}"
    else:
        user_content = example["instruction"]

    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

dataset = dataset.map(format_alpaca, remove_columns=dataset.column_names)

# Split for validation
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# ============================================
# 5. Training
# ============================================
training_args = TrainingArguments(
    output_dir="outputs/test_run",
    num_train_epochs=1,  # Just 1 epoch for test
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_ratio=0.1,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",  # Don't save checkpoints for test
    fp16=True,
    optim="adamw_8bit",
    seed=42,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    args=training_args,
    dataset_text_field="text",
    max_seq_length=1024,
    # NOTE: packing=True for speed in this test script.
    # For production training (src/train.py), we use packing=False with
    # DataCollatorForCompletionOnlyLM for better quality (assistant-only loss).
    packing=True,
)

print("Starting training...")
trainer.train()

# ============================================
# 6. Quick Test
# ============================================
print("\n" + "="*50)
print("Testing the model...")
print("="*50)

FastLanguageModel.for_inference(model)

test_prompts = [
    "Explain what machine learning is in simple terms.",
    "Write a haiku about programming.",
    "What are three tips for learning a new language?",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        do_sample=True,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response[:500]}...")
    print("-" * 50)

print("\nTest complete! Your environment is working correctly.")
```

### 3.3 Run the Test

```bash
python test_finetune.py
```

**Expected output**:
- Model loads in ~30 seconds
- Training runs for ~10-15 minutes
- VRAM usage: ~8-10 GB
- You'll see sample outputs at the end

**What this teaches you**:
1. How the training pipeline works
2. What VRAM usage to expect
3. How to format data for training
4. That your environment is correctly configured

---

## 4. Understanding Key Concepts

### 4.1 What is QLoRA?

**QLoRA** = **Q**uantized **Lo**w-**R**ank **A**daptation

Instead of fine-tuning all 7 billion parameters:
1. **Quantize** the base model to 4-bit (reduces VRAM from ~14GB to ~4GB)
2. **Add small trainable adapters** (LoRA) to key layers
3. **Train only the adapters** (~0.5-2% of parameters)

```
┌─────────────────────────────────────────────────────────────┐
│                    Base Model (Frozen, 4-bit)               │
│                         ~4 GB VRAM                          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ LoRA A  │  │ LoRA B  │  │ LoRA C  │  │ LoRA D  │  ...   │
│  │ (train) │  │ (train) │  │ (train) │  │ (train) │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│                    ~0.5 GB VRAM                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 LoRA Rank: What Does It Mean?

The **rank** (`r`) controls how much the model can learn:

| Rank | Parameters | Learning Capacity | Use Case |
|------|------------|-------------------|----------|
| 16 | ~40M | Low | Simple style tweaks |
| 32 | ~80M | Medium | Good for testing |
| 64 | ~160M | High | **Recommended default** |
| 128 | ~320M | Very High | Complex domain adaptation |

**Rule of thumb**: Start with `r=64`. Only increase if you have lots of data (1000+) and aren't seeing convergence.

### 4.3 Key Hyperparameters Explained

| Parameter | Default | What It Does | When to Change |
|-----------|---------|--------------|----------------|
| `learning_rate` | 2e-4 | How fast model learns | Lower (1e-4) if training is unstable |
| `num_epochs` | 1-3 | Passes through data | More epochs for small datasets |
| `batch_size` | 4 | Examples per step | Lower if OOM, higher if VRAM available |
| `gradient_accumulation` | 4 | Simulated larger batch | Increase for effective batch size |
| `max_seq_length` | 1024 | Max tokens per example | Increase for longer conversations |
| `warmup_ratio` | 0.1 | Gradual LR increase | Usually fine as-is |

**Effective batch size** = `batch_size × gradient_accumulation` = 4 × 4 = 16

### 4.4 Assistant-Only Loss (Recommended)

By default, training computes loss on the entire conversation (user + assistant tokens). This is suboptimal because:
- The model learns to "predict" user messages, which it never needs to do
- It can hurt instruction-following and format adherence

**Solution**: Use `DataCollatorForCompletionOnlyLM` to mask user tokens:

```python
from trl import DataCollatorForCompletionOnlyLM

# For Qwen2.5 ChatML format
response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)

trainer = SFTTrainer(
    ...
    data_collator=collator,
    packing=False,  # Required when using completion-only collator
)
```

**Note**: When using `DataCollatorForCompletionOnlyLM`, you must set `packing=False` as the collator is incompatible with sequence packing.

### 4.5 Packing vs. Assistant-Only Loss: When to Use Each

The `packing` configuration option controls how training examples are batched:

| Setting | How It Works | Best For |
|---------|--------------|----------|
| `packing: true` | Concatenates multiple examples into single sequences for efficiency | Large datasets, speed priority |
| `packing: false` + assistant-only loss | Preserves example boundaries, trains only on assistant responses | Style tuning, small datasets, quality priority |

**Use `packing: true` when:**
- Dataset is large (1000+ examples)
- Training speed is important
- Examples are short and uniform length
- You're doing general instruction-tuning (not style-specific)

**Use `packing: false` (default) when:**
- Dataset is small (<500 examples)
- Style consistency is the goal
- You want to debug/verify training behavior
- Examples vary significantly in length

The test script uses `packing=True` for speed, but the production training script defaults to `packing=False` with assistant-only loss for better quality on style-focused fine-tuning.

### 4.6 Multi-Turn Conversations: Which Turns to Train On

When training with multi-turn conversations, you have choices about which assistant responses to include in the loss calculation:

#### All Assistant Turns (Default)

The default `DataCollatorForCompletionOnlyLM` trains on **all** assistant responses in a conversation:

```
User: What is Python?           ← masked (no loss)
Assistant: Python is a...       ← TRAINED
User: Can you show an example?  ← masked (no loss)
Assistant: Sure, here's...      ← TRAINED
```

**Pros**: Model learns from entire conversation context
**Cons**: Earlier turns may have less context, potentially noisier signal
**Best for**: Most use cases, especially when all assistant responses are high quality

#### Final Turn Only

For some use cases, you may want to train only on the final assistant response:

```python
# Custom collator that only includes final turn
def format_final_turn_only(example):
    """Keep full conversation but only train on final assistant response."""
    convos = example["conversations"]
    # Find the last assistant turn
    for i in range(len(convos) - 1, -1, -1):
        if convos[i]["role"] == "assistant":
            # Mark earlier assistant turns to be masked
            # (Implementation depends on your data format)
            break
    return example
```

**Pros**: Cleaner signal if earlier turns are inconsistent
**Cons**: Loses learning signal from context-dependent responses
**Best for**: When earlier assistant responses vary in quality

#### Recommendation

For most style-tuning use cases, train on **all assistant turns** (the default). Only consider final-turn training if you have explicit quality issues with intermediate responses.

### 4.7 Memory Budget (5060 Ti 16GB)

| Component | VRAM Usage |
|-----------|------------|
| Qwen2.5-7B (4-bit) | ~4.0 GB |
| LoRA adapters (r=64) | ~0.5 GB |
| Optimizer states | ~1.0 GB |
| Activations (batch=4, seq=1024) | ~2.5 GB |
| Gradient checkpointing | ~1.0 GB |
| **Total** | **~9-10 GB** |
| **Headroom** | **~6 GB** |

You have room to increase `batch_size` to 8 or `max_seq_length` to 2048 if needed.

---

## 5. Repository Structure

```
qwen-finetune/
├── pyproject.toml                # Project config for proper imports
├── configs/
│   └── training_config.yaml      # Hyperparameters
├── data/
│   ├── raw/                      # Original data sources
│   ├── processed/                # Formatted for training
│   │   ├── train.jsonl
│   │   └── val.jsonl
│   └── examples/                 # Sample data for reference
├── src/
│   ├── __init__.py
│   ├── train.py                  # Main training script
│   ├── inference.py              # Generation utilities
│   ├── export.py                 # Export to GGUF
│   └── compare_ui.py             # Gradio comparison UI
├── scripts/
│   ├── test_finetune.py          # Quick test script
│   ├── prepare_dataset.py        # Dataset preparation
│   ├── data_stats.py             # Data quality analysis
│   ├── estimate_vram.py          # VRAM usage estimation
│   ├── compare_models.py         # CLI model comparison
│   ├── test_forgetting.py        # Capability regression tests
│   └── merge_lora.py             # LoRA merge utility
├── outputs/
│   ├── {run_name}/               # Per-run directories
│   │   ├── checkpoints/          # Training checkpoints
│   │   ├── final_model/          # Trained LoRA adapters
│   │   └── training_samples.json # Sample outputs per epoch
│   └── exports/                  # GGUF files
└── README.md
```

### 5.1 Project Setup (pyproject.toml)

For clean imports between scripts, create a minimal `pyproject.toml`:

```toml
[project]
name = "qwen-finetune"
version = "0.1.0"
requires-python = ">=3.11"

[tool.setuptools]
packages = ["src"]
```

Then install in development mode:

```bash
uv pip install -e .
```

This allows scripts to import from `src/` reliably (e.g., `from src.inference import generate`).

---

## 6. Finding & Preparing Training Data

### 6.1 How Much Data Do You Need?

Before collecting data, understand the quantity-quality tradeoff:

| Dataset Size | Expected Results | Recommended For |
|--------------|------------------|-----------------|
| 50-100 | Basic style adoption, may be inconsistent | Initial experiments |
| **200-500** | **Solid style consistency** | **First real fine-tune** |
| 500-1000 | Strong style adherence | Production use |
| 1000+ | Diminishing returns (unless very diverse) | Complex domain adaptation |

**Key insight**: Quality matters more than quantity. 200 well-crafted examples often outperform 2000 noisy ones.

**Signs you need more data**:
- Model frequently ignores your target style
- Outputs vary wildly in format/tone
- Model "forgets" style on certain topic types

**Signs you have enough**:
- Consistent style across diverse prompts
- Validation loss has plateaued
- Additional examples don't improve eval metrics

### 6.2 Recommended Public Datasets

#### For Instruction-Following (General)

| Dataset | Size | Quality | Link |
|---------|------|---------|------|
| `yahma/alpaca-cleaned` | 52K | Good | [HuggingFace](https://huggingface.co/datasets/yahma/alpaca-cleaned) |
| `Open-Orca/SlimOrca` | 518K | Excellent | [HuggingFace](https://huggingface.co/datasets/Open-Orca/SlimOrca) |
| `teknium/OpenHermes-2.5` | 1M | Excellent | [HuggingFace](https://huggingface.co/datasets/teknium/OpenHermes-2.5) |

#### For Creative Writing

| Dataset | Size | Focus | Link |
|---------|------|-------|------|
| `lemonilia/LimaRP` | 2.4K | Roleplay/creative | [HuggingFace](https://huggingface.co/datasets/lemonilia/LimaRP) |
| `euclaise/writingprompts` | 300K | Story writing | [HuggingFace](https://huggingface.co/datasets/euclaise/writingprompts) |
| `roneneldan/TinyStories` | 2M | Short stories | [HuggingFace](https://huggingface.co/datasets/roneneldan/TinyStories) |

#### For Domain Expertise

| Dataset | Domain | Link |
|---------|--------|------|
| `codeparrot/github-code` | Programming | [HuggingFace](https://huggingface.co/datasets/codeparrot/github-code) |
| `Open-Orca/FLAN` | Reasoning | [HuggingFace](https://huggingface.co/datasets/Open-Orca/FLAN) |
| `pubmed_qa` | Medical Q&A | [HuggingFace](https://huggingface.co/datasets/pubmed_qa) |

### 6.3 Data Format

Your training data must be in **conversation format** (JSONL):

```jsonl
{"conversations": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a high-level programming language known for its readability and versatility."}]}
{"conversations": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a hello world program."}, {"role": "assistant", "content": "Here's a hello world program in Python:\n\n```python\nprint(\"Hello, World!\")\n```"}]}
```

### 6.4 System Prompt Strategies

System prompts in training data significantly affect model behavior. Choose your strategy based on your use case:

#### Strategy 1: No System Prompt (Simplest)

Train without system prompts; add them only at inference time.

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Pros**: Maximum flexibility at inference, simpler data preparation
**Cons**: Style must be learned entirely from assistant responses
**Best for**: When you want to use different system prompts at inference time

#### Strategy 2: Fixed System Prompt (Most Consistent)

Include the same system prompt in every training example.

```jsonl
{"conversations": [{"role": "system", "content": "You are a concise technical writer."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Pros**: Strongest style adherence, consistent behavior
**Cons**: Model may underperform with different system prompts at inference
**Best for**: When you'll always use the same system prompt in production

#### Strategy 3: Varied System Prompts (Most Robust)

Mix examples: some with system prompts, some without, some with variations.

```jsonl
{"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"conversations": [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"conversations": [{"role": "system", "content": "You are a technical expert."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

**Pros**: Model learns to follow various system prompts while maintaining style
**Cons**: Requires more diverse training data, harder to prepare
**Best for**: When you need flexibility with consistent underlying style

#### Recommendation

For most style-tuning use cases, start with **Strategy 1** (no system prompt in training). If style adherence is inconsistent, try **Strategy 2** with a fixed prompt that describes your target style.

### 6.5 Data Preparation Script

Create `scripts/prepare_dataset.py`:

```python
"""
Prepare various dataset formats for training.
"""

from datasets import load_dataset
import json
from pathlib import Path


def prepare_alpaca(output_dir: str = "data/processed", num_examples: int = None):
    """Convert Alpaca format to conversation format."""
    print("Loading Alpaca dataset...")
    dataset = load_dataset("yahma/alpaca-cleaned", split="train")

    if num_examples:
        dataset = dataset.select(range(min(num_examples, len(dataset))))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Split 90/10
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    for split_name, split_data in [("train", dataset["train"]), ("val", dataset["test"])]:
        output_file = output_path / f"{split_name}.jsonl"

        with open(output_file, "w") as f:
            for example in split_data:
                conversations = []

                # Combine instruction and input
                if example.get("input", "").strip():
                    user_content = f"{example['instruction']}\n\nInput: {example['input']}"
                else:
                    user_content = example["instruction"]

                conversations.append({"role": "user", "content": user_content})
                conversations.append({"role": "assistant", "content": example["output"]})

                f.write(json.dumps({"conversations": conversations}) + "\n")

        print(f"Wrote {len(split_data)} examples to {output_file}")


def prepare_openhermes(output_dir: str = "data/processed", num_examples: int = 5000):
    """Convert OpenHermes format to conversation format."""
    print("Loading OpenHermes dataset...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split=f"train[:{num_examples}]")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    for split_name, split_data in [("train", dataset["train"]), ("val", dataset["test"])]:
        output_file = output_path / f"{split_name}.jsonl"

        with open(output_file, "w") as f:
            for example in split_data:
                conversations = example.get("conversations", [])
                if conversations:
                    f.write(json.dumps({"conversations": conversations}) + "\n")

        print(f"Wrote {len(split_data)} examples to {output_file}")


def prepare_custom_jsonl(input_file: str, output_dir: str = "data/processed"):
    """
    Validate and split a custom JSONL file.

    Expected format per line:
    {"conversations": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    """
    with open(input_file) as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(data)} examples from {input_file}")

    # Validate
    for i, item in enumerate(data):
        if "conversations" not in item:
            raise ValueError(f"Line {i+1} missing 'conversations' key")
        for msg in item["conversations"]:
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Line {i+1} has invalid message format")

    # Split
    from random import shuffle, seed
    seed(42)
    shuffle(data)

    split_idx = int(len(data) * 0.9)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_data in [("train", train_data), ("val", val_data)]:
        output_file = output_path / f"{split_name}.jsonl"
        with open(output_file, "w") as f:
            for item in split_data:
                f.write(json.dumps(item) + "\n")
        print(f"Wrote {len(split_data)} examples to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["alpaca", "openhermes", "custom"], required=True)
    parser.add_argument("--input", help="Input file for custom source")
    parser.add_argument("--output", default="data/processed")
    parser.add_argument("--num-examples", type=int, default=None)
    args = parser.parse_args()

    if args.source == "alpaca":
        prepare_alpaca(args.output, args.num_examples)
    elif args.source == "openhermes":
        prepare_openhermes(args.output, args.num_examples or 5000)
    elif args.source == "custom":
        if not args.input:
            raise ValueError("--input required for custom source")
        prepare_custom_jsonl(args.input, args.output)
```

**Usage**:

```bash
# Prepare Alpaca dataset (full)
python scripts/prepare_dataset.py --source alpaca

# Prepare 2000 examples from OpenHermes
python scripts/prepare_dataset.py --source openhermes --num-examples 2000

# Prepare your custom data
python scripts/prepare_dataset.py --source custom --input my_data.jsonl
```

### 6.6 Data Statistics Script

Before training, review your dataset statistics to catch potential issues.

Create `scripts/data_stats.py`:

```python
"""
Analyze training data and display statistics.
Helps identify potential issues before training.
"""

import json
from pathlib import Path
from collections import Counter


def analyze_dataset(data_file: str):
    """Analyze a JSONL dataset and print statistics."""

    with open(data_file) as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"\n{'='*60}")
    print(f"Dataset: {data_file}")
    print(f"{'='*60}")

    # Basic counts
    print(f"\nTotal examples: {len(data)}")

    # Message counts
    turn_counts = []
    assistant_lengths = []
    user_lengths = []
    has_system = 0

    for example in data:
        convos = example.get("conversations", [])
        turn_counts.append(len(convos))

        for msg in convos:
            if msg["role"] == "assistant":
                assistant_lengths.append(len(msg["content"]))
            elif msg["role"] == "user":
                user_lengths.append(len(msg["content"]))
            elif msg["role"] == "system":
                has_system += 1

    # Turn statistics
    print(f"\nConversation turns:")
    print(f"  Min: {min(turn_counts)}")
    print(f"  Max: {max(turn_counts)}")
    print(f"  Avg: {sum(turn_counts)/len(turn_counts):.1f}")

    # Length statistics (characters)
    print(f"\nAssistant response lengths (chars):")
    print(f"  Min: {min(assistant_lengths)}")
    print(f"  Max: {max(assistant_lengths)}")
    print(f"  Avg: {sum(assistant_lengths)/len(assistant_lengths):.1f}")
    print(f"  Median: {sorted(assistant_lengths)[len(assistant_lengths)//2]}")

    print(f"\nUser prompt lengths (chars):")
    print(f"  Min: {min(user_lengths)}")
    print(f"  Max: {max(user_lengths)}")
    print(f"  Avg: {sum(user_lengths)/len(user_lengths):.1f}")

    # System prompt usage
    print(f"\nSystem prompts: {has_system} examples ({100*has_system/len(data):.1f}%)")

    # Length distribution buckets
    print(f"\nAssistant response length distribution:")
    buckets = {"<100": 0, "100-500": 0, "500-1000": 0, "1000-2000": 0, ">2000": 0}
    for length in assistant_lengths:
        if length < 100:
            buckets["<100"] += 1
        elif length < 500:
            buckets["100-500"] += 1
        elif length < 1000:
            buckets["500-1000"] += 1
        elif length < 2000:
            buckets["1000-2000"] += 1
        else:
            buckets[">2000"] += 1

    for bucket, count in buckets.items():
        pct = 100 * count / len(assistant_lengths)
        bar = "█" * int(pct / 2)
        print(f"  {bucket:>10}: {bar} {pct:.1f}%")

    # Warnings
    print(f"\n{'='*60}")
    print("Potential issues:")
    print(f"{'='*60}")

    issues = []
    if len(data) < 50:
        issues.append("⚠️  Very small dataset (<50). Consider adding more examples.")
    if max(assistant_lengths) > 4000:
        issues.append("⚠️  Some responses are very long (>4000 chars). May be truncated.")
    if min(assistant_lengths) < 10:
        issues.append("⚠️  Some responses are very short (<10 chars). Check data quality.")
    if has_system > 0 and has_system < len(data):
        issues.append(f"⚠️  Mixed system prompt usage ({has_system}/{len(data)}). Be consistent.")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✓ No obvious issues detected.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    args = parser.parse_args()

    for data_file in [args.train, args.val]:
        if Path(data_file).exists():
            analyze_dataset(data_file)
        else:
            print(f"File not found: {data_file}")
```

**Usage**:

```bash
python scripts/data_stats.py
# Or specify files:
python scripts/data_stats.py --train data/processed/train.jsonl --val data/processed/val.jsonl
```

---

## 7. Training Configuration & Hyperparameters

### 7.1 Configuration File

Create `configs/training_config.yaml`:

```yaml
# Model Configuration
model:
  name: "unsloth/Qwen2.5-7B-Instruct"
  max_seq_length: 1024
  load_in_4bit: true

# LoRA Configuration
lora:
  r: 64                  # Rank - increase for more capacity
  alpha: 64              # Usually same as r
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

# Training Configuration
training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4  # Effective batch = 16
  learning_rate: 2.0e-4           # See "Learning Rate Scaling" section below
  warmup_ratio: 0.1
  weight_decay: 0.01
  packing: false                  # See "Packing vs Assistant-Only Loss" below

  # Evaluation
  eval_steps: 50
  logging_steps: 10

  # Checkpointing
  save_steps: 100
  save_total_limit: 3

# Data Configuration
data:
  train_file: "data/processed/train.jsonl"
  val_file: "data/processed/val.jsonl"
  val_min_samples: 20             # Always keep at least this many validation samples

# Output & Run Naming
# Run names combine description + timestamp: alpaca_r64_20241228_1430
run_name: "alpaca"  # Descriptive prefix for this experiment
output_base: "outputs"  # Base directory for all runs
# Auto-generated paths: outputs/alpaca_r64_YYYYMMDD_HHMM/checkpoints, /final_model

# Early Stopping (optional)
early_stopping:
  enabled: true
  patience: 5  # Stop if val loss doesn't improve for N eval steps
  threshold: 0.001  # Minimum improvement to count as "better"

# Experiment Tracking (optional)
wandb:
  enabled: false          # Set to true to enable Weights & Biases logging
  project: "qwen-finetune"
  entity: null            # Your W&B username or team name

# Reproducibility
seed: 42  # Fixed seed for reproducibility (affects data splits, model init, training)
```

### 7.2 Hyperparameter Selection Guide

#### How Many Epochs?

| Dataset Size | Recommended Epochs | Notes |
|--------------|-------------------|-------|
| < 200 | 5-10 | Small data needs more passes |
| 200-1000 | 3-5 | Standard range |
| 1000-5000 | 2-3 | Watch for overfitting |
| 5000+ | 1-2 | Usually sufficient |

#### Sequence Length?

| Your Data | Recommended `max_seq_length` |
|-----------|------------------------------|
| Short Q&A (< 256 tokens) | 512 |
| Typical instructions | 1024 |
| Long-form content | 2048 |
| Multi-turn conversations | 2048-4096 |

**Memory impact**: Doubling sequence length roughly doubles activation memory.

#### Learning Rate Scaling by Dataset Size

Learning rate should scale with dataset size to avoid overfitting on small datasets or underfitting on large ones:

| Dataset Size | Recommended Learning Rate | Rationale |
|--------------|--------------------------|-----------|
| < 100 examples | 5e-5 to 1e-4 | Very conservative to avoid memorization |
| 100-500 examples | 1e-4 | Standard for style tuning |
| 500-2000 examples | 2e-4 | Default, works well for most cases |
| 2000+ examples | 2e-4 to 3e-4 | Can be more aggressive with more data |

**Additional learning rate guidance:**

| Situation | Adjustment |
|-----------|------------|
| Training unstable (loss spikes) | Reduce by 50% |
| Loss not decreasing | Increase by 50% |
| Validation loss increasing early | Reduce by 50% |

#### Small Dataset Recommendations (Style Tuning)

For style-focused fine-tuning with small datasets (<500 examples), use conservative settings to avoid overfitting:

| Parameter | Standard | Small Dataset (<500) |
|-----------|----------|---------------------|
| `lora_r` | 64 | **32** |
| `learning_rate` | 2e-4 | **1e-4** |
| `num_epochs` | 3 | **2-3** |
| `packing` | True | **False** |

**Why these changes?**
- **Lower rank (r=32)**: Reduces model capacity, prevents memorizing training data
- **Lower learning rate**: Slower, more stable learning
- **Fewer epochs**: Less risk of overfitting
- **No packing**: Preserves sample boundaries, easier to diagnose format issues

**Signs of overfitting**:
- Validation loss increases while training loss decreases
- Model outputs become repetitive or "locked" to training style
- Model struggles with prompts outside training distribution

### 7.3 Understanding Training & Validation Loss

During training, you'll see two loss values. Here's how to interpret them:

```
Step 50:  train_loss=1.82  eval_loss=1.85
Step 100: train_loss=1.45  eval_loss=1.52
Step 150: train_loss=1.12  eval_loss=1.48  ← Validation plateauing
Step 200: train_loss=0.85  eval_loss=1.55  ← Validation increasing = overfitting!
```

#### What Each Pattern Means

| Train Loss | Val Loss | Interpretation | Action |
|------------|----------|----------------|--------|
| ↓ Decreasing | ↓ Decreasing | Healthy learning | Continue training |
| ↓ Decreasing | → Flat | Learning slowing | Consider stopping soon |
| ↓ Decreasing | ↑ Increasing | **Overfitting** | Stop training, use earlier checkpoint |
| → Flat | → Flat | Converged | Training complete |
| → Flat or ↑ | Any | Not learning | Check LR, data format |

#### When to Stop Training

With early stopping enabled (`early_stopping.enabled: true`), training auto-stops when validation loss hasn't improved for `patience` eval steps. Without early stopping:

1. **Watch validation loss** - Stop when it starts increasing
2. **Check qualitatively** - Generate samples every few hundred steps
3. **Use checkpoints** - `load_best_model_at_end=True` loads the best checkpoint automatically

#### Practical Tips

- **Validation loss slightly higher than training is normal** - They won't be equal
- **Small spikes are okay** - Look at the trend, not individual points
- **If val loss never decreases** - Check data format, may be corrupted
- **If val loss decreases then spikes** - Learning rate may be too high

---

## 8. Core Training Script

Create `src/train.py`:

```python
"""
Main training script for Qwen2.5-7B QLoRA fine-tuning.
"""

import yaml
from pathlib import Path
from datetime import datetime
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, EarlyStoppingCallback
import torch


def load_config(config_path: str = "configs/training_config.yaml") -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def generate_run_name(config: dict) -> str:
    """Generate run name: {description}_{lora_r}_{timestamp}"""
    run_name = config.get("run_name", "run")
    lora_r = config["lora"]["r"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{run_name}_r{lora_r}_{timestamp}"


def main(config_path: str = "configs/training_config.yaml"):
    # =========================================
    # 1. Load Configuration
    # =========================================
    config = load_config(config_path)
    print(f"Loaded config from {config_path}")

    # Generate unique run name
    run_name = generate_run_name(config)
    output_base = config.get("output_base", "outputs")
    output_dir = Path(output_base) / run_name / "checkpoints"
    final_model_dir = Path(output_base) / run_name / "final_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run name: {run_name}")

    # =========================================
    # 2. Load Model with 4-bit Quantization
    # =========================================
    print(f"Loading {config['model']['name']}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        dtype=None,  # Auto-detect (bf16 if supported)
    )

    # =========================================
    # 3. Add LoRA Adapters
    # =========================================
    lora_config = config["lora"]
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_config["r"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Print trainable parameters
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # =========================================
    # 4. Setup Chat Template
    # =========================================
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # =========================================
    # 5. Load Dataset
    # =========================================
    data_config = config["data"]

    def format_conversations(example):
        """Format conversations for training."""
        convos = example["conversations"]
        text = tokenizer.apply_chat_template(
            convos,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = load_dataset(
        "json",
        data_files={
            "train": data_config["train_file"],
            "validation": data_config["val_file"],
        },
    )

    dataset = dataset.map(format_conversations, remove_columns=["conversations"])

    print(f"Training examples: {len(dataset['train'])}")
    print(f"Validation examples: {len(dataset['validation'])}")

    # =========================================
    # 6. Training Arguments
    # =========================================
    train_config = config["training"]

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_config["num_epochs"],
        per_device_train_batch_size=train_config["batch_size"],
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        learning_rate=train_config["learning_rate"],
        warmup_ratio=train_config["warmup_ratio"],
        weight_decay=train_config["weight_decay"],
        logging_steps=train_config["logging_steps"],
        eval_strategy="steps",
        eval_steps=train_config["eval_steps"],
        save_strategy="steps",
        save_steps=train_config["save_steps"],
        save_total_limit=train_config["save_total_limit"],
        load_best_model_at_end=True,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=42,
        report_to="none",  # Change to "wandb" for tracking
    )

    # =========================================
    # 7. Create Data Collator (Assistant-Only Loss)
    # =========================================
    # Only compute loss on assistant responses, not user prompts
    # This improves instruction-following and format adherence
    response_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        tokenizer=tokenizer,
    )

    # =========================================
    # 8. Create Trainer with Optional Early Stopping
    # =========================================
    callbacks = []
    early_stop_config = config.get("early_stopping", {})
    if early_stop_config.get("enabled", False):
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=early_stop_config.get("patience", 3),
            early_stopping_threshold=early_stop_config.get("threshold", 0.01),
        ))
        print(f"Early stopping enabled: patience={early_stop_config.get('patience', 3)}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        data_collator=collator,
        dataset_text_field="text",
        max_seq_length=config["model"]["max_seq_length"],
        packing=False,  # Disable packing for style tuning (preserves sample boundaries)
        callbacks=callbacks if callbacks else None,
    )

    # =========================================
    # 9. Train
    # =========================================
    print("Starting training...")
    print(f"  Epochs: {train_config['num_epochs']}")
    print(f"  Batch size: {train_config['batch_size']}")
    print(f"  Gradient accumulation: {train_config['gradient_accumulation_steps']}")
    print(f"  Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation_steps']}")
    print(f"  Learning rate: {train_config['learning_rate']}")

    trainer.train()

    # =========================================
    # 10. Save Final Model
    # =========================================
    print(f"Saving model to {final_model_dir}...")
    final_model_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    print("Training complete!")
    print(f"\nRun directory: {output_base}/{run_name}/")
    print(f"\nNext steps:")
    print(f"  1. Test: python src/inference.py --model {final_model_dir}")
    print(f"  2. Export: python src/export.py --model {final_model_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    main(args.config)
```

### 8.2 Sample Generation During Training

To monitor training progress qualitatively, add sample generation at the end of each epoch.

Add this callback to your training script:

```python
from transformers import TrainerCallback
import json
from pathlib import Path


class SampleGenerationCallback(TrainerCallback):
    """Generate sample outputs at the end of each epoch for monitoring."""

    def __init__(self, model, tokenizer, output_dir, prompts=None):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.prompts = prompts or [
            "Explain what an API is in simple terms.",
            "Write a haiku about programming.",
            "What are three tips for better code reviews?",
        ]
        self.samples = []

    def on_epoch_end(self, args, state, control, **kwargs):
        """Generate samples at end of each epoch."""
        from unsloth import FastLanguageModel

        # Switch to inference mode
        FastLanguageModel.for_inference(self.model)

        epoch_samples = {"epoch": state.epoch, "step": state.global_step, "responses": []}

        for prompt in self.prompts:
            messages = [{"role": "user", "content": prompt}]
            inputs = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.shape[1]:], skip_special_tokens=True
            )

            epoch_samples["responses"].append({
                "prompt": prompt,
                "response": response[:500]
            })

            print(f"\n[Epoch {state.epoch}] {prompt[:50]}...")
            print(f"Response: {response[:200]}...")

        self.samples.append(epoch_samples)

        # Save all samples
        samples_file = self.output_dir / "training_samples.json"
        with open(samples_file, "w") as f:
            json.dump(self.samples, f, indent=2)

        # Switch back to training mode
        self.model.train()


# Add to trainer callbacks:
# sample_callback = SampleGenerationCallback(model, tokenizer, output_dir)
# callbacks.append(sample_callback)
```

This saves sample outputs to `training_samples.json` and prints progress to console.

**Run training**:

```bash
# Prepare data first
python scripts/prepare_dataset.py --source alpaca --num-examples 2000

# Run training
python src/train.py --config configs/training_config.yaml
```

---

## 9. Inference & Testing

Create `src/inference.py`:

```python
"""
Inference script for testing fine-tuned models.
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


def load_model(model_path: str, max_seq_length: int = 2048):
    """Load fine-tuned model for inference."""
    print(f"Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )

    FastLanguageModel.for_inference(model)
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """Generate a response."""
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
    return response


def interactive_chat(model, tokenizer, system_prompt: str = None):
    """Interactive chat loop."""
    print("\n" + "="*50)
    print("Interactive Chat (type 'quit' to exit)")
    print("="*50 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower() in ("quit", "exit", "q"):
            break

        if not user_input:
            continue

        response = generate(model, tokenizer, user_input, system_prompt)
        print(f"\nAssistant: {response}\n")


def batch_test(model, tokenizer, prompts: list[str], system_prompt: str = None):
    """Test model on multiple prompts."""
    print("\n" + "="*50)
    print("Batch Testing")
    print("="*50)

    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        response = generate(model, tokenizer, prompt, system_prompt)
        print(f"Response: {response}")
        print("-" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/final_model")
    parser.add_argument("--system", default=None, help="System prompt")
    parser.add_argument("--batch", action="store_true", help="Run batch test")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model)

    if args.batch:
        test_prompts = [
            "Explain what an API is in simple terms.",
            "Write a short poem about coding.",
            "What are three tips for better sleep?",
            "How do I make a basic omelet?",
        ]
        batch_test(model, tokenizer, test_prompts, args.system)
    else:
        interactive_chat(model, tokenizer, args.system)
```

**Usage**:

```bash
# Interactive chat
python src/inference.py --model outputs/final_model

# Batch test
python src/inference.py --model outputs/final_model --batch

# With system prompt
python src/inference.py --model outputs/final_model --system "You are a helpful coding assistant."
```

---

## 10. Export to Ollama

### 10.1 Export Prerequisites

Before exporting to GGUF format, ensure you have:

| Requirement | Details |
|-------------|---------|
| **System RAM** | 32GB recommended (export temporarily loads full model to CPU) |
| **Disk space** | ~8-15GB for GGUF file depending on quantization |
| **Unsloth version** | 2024.8+ (earlier versions may have GGUF export issues) |

**How export works**:
1. Unsloth loads your LoRA adapters
2. Merges LoRA weights into base model (happens automatically)
3. Converts merged model to GGUF format on CPU
4. Applies quantization (q4_k_m, q5_k_m, etc.)

**Common export issues**:
- **OOM during export**: Close other applications, export uses significant RAM
- **Slow export**: Normal - GGUF conversion is CPU-bound, takes 5-15 minutes
- **Missing llama.cpp**: Unsloth handles this automatically in recent versions

### 10.2 Export Script

Create `src/export.py`:

```python
"""
Export fine-tuned model to GGUF format for Ollama.
"""

from unsloth import FastLanguageModel
from pathlib import Path


def find_gguf_file(output_dir: Path) -> Path:
    """Find the GGUF file in output directory (Unsloth naming varies by version)."""
    gguf_files = list(output_dir.glob("*.gguf"))
    if not gguf_files:
        raise FileNotFoundError(f"No .gguf file found in {output_dir}")
    # Return most recently modified if multiple exist
    return max(gguf_files, key=lambda p: p.stat().st_mtime)


def export_to_gguf(
    model_path: str,
    output_path: str = "outputs/exports/gguf",
    quantization: str = "q4_k_m",
):
    """
    Export model to GGUF format.

    Quantization options:
    - q4_k_m: 4-bit, good balance (recommended)
    - q5_k_m: 5-bit, better quality
    - q8_0: 8-bit, best quality
    - f16: Full fp16, largest
    """
    print(f"Loading model from {model_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to GGUF with {quantization} quantization...")
    print("(This may take 5-15 minutes and uses significant RAM)")

    model.save_pretrained_gguf(
        str(output_dir),
        tokenizer,
        quantization_method=quantization,
    )

    # Find the actual GGUF file (naming varies by Unsloth version)
    gguf_file = find_gguf_file(output_dir)

    print(f"\nExported to: {gguf_file}")
    print(f"\nTo use with Ollama:")
    print(f"  1. Create Modelfile (see below)")
    print(f"  2. ollama create my-model -f Modelfile")
    print(f"  3. ollama run my-model")

    # Generate Modelfile with absolute path for reliability
    modelfile_content = f'''FROM {gguf_file.absolute()}

TEMPLATE """{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "<|im_end|>"
'''

    modelfile_path = output_dir / "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)

    print(f"\nModelfile created: {modelfile_path}")
    print(f"\nQuick start:")
    print(f"  cd {output_dir}")
    print(f"  ollama create my-finetuned -f Modelfile")
    print(f"  ollama run my-finetuned")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/final_model")
    parser.add_argument("--output", default="outputs/exports/gguf")
    parser.add_argument("--quantization", default="q4_k_m",
                       choices=["q4_k_m", "q5_k_m", "q8_0", "f16"])
    args = parser.parse_args()

    export_to_gguf(args.model, args.output, args.quantization)
```

**Usage**:

```bash
# Export with default quantization (q4_k_m)
python src/export.py --model outputs/final_model

# Export with higher quality
python src/export.py --model outputs/final_model --quantization q5_k_m

# Then create Ollama model
cd outputs/exports/gguf
ollama create my-finetuned -f Modelfile
ollama run my-finetuned
```

### 10.3 Tuning Inference Parameters

The Modelfile includes default inference parameters that significantly affect output quality. Adjust these based on your use case:

#### Temperature

Controls randomness/creativity:

| Temperature | Effect | Best For |
|-------------|--------|----------|
| 0.1 - 0.3 | Very deterministic, focused | Factual Q&A, code generation, structured output |
| 0.5 - 0.7 | Balanced (default: 0.7) | General conversation, instruction following |
| 0.8 - 1.2 | Creative, varied | Creative writing, brainstorming |
| 1.5+ | High randomness | Experimental, may produce incoherent output |

#### Top-p (Nucleus Sampling)

Controls diversity by limiting to top probability mass:

| Top-p | Effect | Best For |
|-------|--------|----------|
| 0.5 | Conservative, predictable | Factual tasks |
| 0.9 | Balanced (default) | Most use cases |
| 0.95 - 1.0 | Maximum diversity | Creative tasks |

#### Runtime Override

You can override Modelfile defaults at runtime:

```bash
# More creative
ollama run my-finetuned --temperature 1.0 --top-p 0.95

# More focused
ollama run my-finetuned --temperature 0.3 --top-p 0.7
```

#### Editing the Modelfile

To permanently change defaults, edit the Modelfile and recreate:

```bash
# Edit outputs/exports/gguf/Modelfile
# Change PARAMETER temperature 0.7 to your preferred value

# Recreate model
ollama rm my-finetuned
ollama create my-finetuned -f Modelfile
```

### 10.4 Manual LoRA Merge (For Non-Ollama Deployment)

The training script saves only LoRA adapters. For deployment with vanilla Transformers, vLLM, or other frameworks, you may need to merge adapters into the base model manually.

```python
"""
Merge LoRA adapters into base model for non-Ollama deployment.
"""

from unsloth import FastLanguageModel
from pathlib import Path


def merge_lora(
    adapter_path: str = "outputs/final_model",
    output_path: str = "outputs/merged_model",
    save_method: str = "merged_16bit",  # or "merged_4bit"
):
    """
    Merge LoRA adapters into base model.

    save_method options:
    - "merged_16bit": Full precision merged model (largest, most compatible)
    - "merged_4bit": 4-bit quantized merged model (smaller, still needs bitsandbytes)
    - "lora": Just save adapters (smallest, requires PEFT to load)
    """
    print(f"Loading adapter from {adapter_path}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged model to {output_dir}...")

    model.save_pretrained_merged(
        str(output_dir),
        tokenizer,
        save_method=save_method,
    )

    print(f"\nMerged model saved to: {output_dir}")
    print(f"\nTo load with Transformers:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{output_dir}')")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/final_model")
    parser.add_argument("--output", default="outputs/merged_model")
    parser.add_argument("--method", default="merged_16bit",
                       choices=["merged_16bit", "merged_4bit", "lora"])
    args = parser.parse_args()

    merge_lora(args.adapter, args.output, args.method)
```

**Usage**:

```bash
# Full precision merge (for maximum compatibility)
python scripts/merge_lora.py --adapter outputs/final_model --output outputs/merged

# 4-bit merge (smaller, requires bitsandbytes)
python scripts/merge_lora.py --adapter outputs/final_model --method merged_4bit
```

### 10.5 Alternative Deployment Targets

While this guide focuses on Ollama, your fine-tuned model can be deployed elsewhere:

| Target | Format Needed | Notes |
|--------|--------------|-------|
| **Ollama** | GGUF | Covered in this guide |
| **vLLM** | Merged HF model | Use `merge_lora.py` above, then load with vLLM |
| **Text Generation Inference** | Merged HF model | Same as vLLM |
| **Hugging Face Hub** | LoRA adapters | Push adapters for sharing/collaboration |
| **llama.cpp** | GGUF | Same file as Ollama |

For Hugging Face Hub sharing:

```bash
# Push LoRA adapters (not merged model - smaller upload)
huggingface-cli login
huggingface-cli upload your-username/qwen-finetuned outputs/final_model
```

---

## 11. Evaluation: Side-by-Side Comparison

The best way to evaluate your fine-tuned model is to compare it directly against the base model on the same prompts.

### 11.1 Comparison Strategy

1. **Prepare test prompts** covering your target use cases
2. **Generate responses** from both base and fine-tuned models
3. **Compare side-by-side** looking for style/quality differences

### 11.2 CLI Comparison Script

Create `scripts/compare_models.py`:

```python
"""
CLI tool for comparing base vs fine-tuned model responses.
"""

import json
from pathlib import Path
from src.inference import load_model, generate


def compare_responses(
    base_model_path: str,
    finetuned_model_path: str,
    prompts: list[str],
    system_prompt: str = None,
    output_file: str = None,
):
    """Compare responses from base and fine-tuned models."""

    results = []

    # Load fine-tuned model
    print("Loading fine-tuned model...")
    ft_model, ft_tokenizer = load_model(finetuned_model_path)

    # Generate fine-tuned responses
    print("\nGenerating fine-tuned responses...")
    ft_responses = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
        response = generate(ft_model, ft_tokenizer, prompt, system_prompt)
        ft_responses.append(response)

    # Unload fine-tuned model
    del ft_model, ft_tokenizer
    import torch
    torch.cuda.empty_cache()

    # Load base model
    print("\nLoading base model...")
    base_model, base_tokenizer = load_model(base_model_path)

    # Generate base responses
    print("\nGenerating base model responses...")
    base_responses = []
    for i, prompt in enumerate(prompts):
        print(f"  [{i+1}/{len(prompts)}] {prompt[:50]}...")
        response = generate(base_model, base_tokenizer, prompt, system_prompt)
        base_responses.append(response)

    # Display comparison
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)

    for i, prompt in enumerate(prompts):
        print(f"\n{'─'*80}")
        print(f"PROMPT: {prompt}")
        print(f"{'─'*80}")
        print(f"\n[BASE MODEL]:\n{base_responses[i]}")
        print(f"\n[FINE-TUNED]:\n{ft_responses[i]}")

        results.append({
            "prompt": prompt,
            "base_response": base_responses[i],
            "finetuned_response": ft_responses[i],
        })

    # Save results
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--finetuned", default="outputs/final_model")
    parser.add_argument("--prompts-file", help="JSON file with list of prompts")
    parser.add_argument("--system", default=None)
    parser.add_argument("--output", default="outputs/comparison_results.json")
    args = parser.parse_args()

    if args.prompts_file:
        with open(args.prompts_file) as f:
            prompts = json.load(f)
    else:
        prompts = [
            "Explain what Docker is and why it's useful.",
            "Write a function to reverse a string in Python.",
            "What are the main differences between SQL and NoSQL databases?",
            "Give me three tips for writing cleaner code.",
        ]

    compare_responses(
        args.base,
        args.finetuned,
        prompts,
        args.system,
        args.output,
    )
```

**Usage**:

```bash
# Compare with default prompts
python scripts/compare_models.py --finetuned outputs/final_model

# Compare with custom prompts
echo '["What is Python?", "Explain REST APIs."]' > test_prompts.json
python scripts/compare_models.py --prompts-file test_prompts.json
```

### 11.3 Catastrophic Forgetting Test Suite

Fine-tuning can sometimes degrade the model's general capabilities ("catastrophic forgetting"). Use this test suite to verify your model retains base abilities.

Create `scripts/test_forgetting.py`:

```python
"""
Test suite to detect catastrophic forgetting.
Compares base model vs fine-tuned on general capability prompts.
"""

import json
from pathlib import Path
from src.inference import load_model, generate

# General capability test prompts (not related to your fine-tuning domain)
FORGETTING_TEST_PROMPTS = {
    "math": [
        "What is 15% of 80?",
        "If a train travels at 60 mph for 2.5 hours, how far does it go?",
        "Solve: 2x + 5 = 17",
    ],
    "logic": [
        "If all cats are mammals, and all mammals are animals, are all cats animals?",
        "What comes next in the sequence: 2, 4, 8, 16, ?",
        "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
    ],
    "knowledge": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the chemical symbol for gold?",
    ],
    "code": [
        "Write a Python function that checks if a number is even.",
        "What does the 'len()' function do in Python?",
        "Fix this code: print('Hello World)",
    ],
}


def test_forgetting(
    base_model_path: str = "unsloth/Qwen2.5-7B-Instruct",
    finetuned_model_path: str = "outputs/final_model",
    output_file: str = "outputs/forgetting_test.json",
):
    """Run forgetting tests and save results."""

    results = {"base": {}, "finetuned": {}, "comparison": {}}

    # Test fine-tuned model first (then unload)
    print("Testing fine-tuned model...")
    ft_model, ft_tokenizer = load_model(finetuned_model_path)

    for category, prompts in FORGETTING_TEST_PROMPTS.items():
        results["finetuned"][category] = []
        for prompt in prompts:
            response = generate(ft_model, ft_tokenizer, prompt, temperature=0.1)
            results["finetuned"][category].append({
                "prompt": prompt,
                "response": response[:500]
            })

    del ft_model, ft_tokenizer
    import torch
    torch.cuda.empty_cache()

    # Test base model
    print("Testing base model...")
    base_model, base_tokenizer = load_model(base_model_path)

    for category, prompts in FORGETTING_TEST_PROMPTS.items():
        results["base"][category] = []
        for prompt in prompts:
            response = generate(base_model, base_tokenizer, prompt, temperature=0.1)
            results["base"][category].append({
                "prompt": prompt,
                "response": response[:500]
            })

    # Print comparison
    print("\n" + "="*80)
    print("FORGETTING TEST RESULTS")
    print("="*80)

    for category in FORGETTING_TEST_PROMPTS.keys():
        print(f"\n### {category.upper()} ###")
        for i, prompt in enumerate(FORGETTING_TEST_PROMPTS[category]):
            print(f"\nPrompt: {prompt}")
            print(f"Base: {results['base'][category][i]['response'][:200]}...")
            print(f"Fine-tuned: {results['finetuned'][category][i]['response'][:200]}...")

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_file}")

    print("\n" + "="*80)
    print("REVIEW: Compare responses manually. Signs of forgetting:")
    print("  - Incorrect math answers")
    print("  - Wrong factual information")
    print("  - Broken code syntax")
    print("  - Illogical reasoning")
    print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="unsloth/Qwen2.5-7B-Instruct")
    parser.add_argument("--finetuned", default="outputs/final_model")
    parser.add_argument("--output", default="outputs/forgetting_test.json")
    args = parser.parse_args()

    test_forgetting(args.base, args.finetuned, args.output)
```

**Usage**:

```bash
python scripts/test_forgetting.py --finetuned outputs/final_model
```

**What to look for**:
- Math answers should be correct (12, 150 miles, x=6)
- Logic should be sound (yes, 32, 5 minutes)
- Facts should be accurate (Paris, Shakespeare, Au)
- Code should be syntactically correct

If you see degradation, try:
- Reducing learning rate
- Fewer training epochs
- Lower LoRA rank
- Adding general instruction data to training mix

---

## 12. Gradio Comparison UI

For interactive side-by-side comparison, use this Gradio interface.

Create `src/compare_ui.py`:

```python
"""
Gradio UI for side-by-side model comparison via Ollama.
"""

import gradio as gr
import httpx
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Simple Ollama API client with actionable error messages."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=60.0)

    def is_available(self) -> bool:
        try:
            self.client.get(f"{self.base_url}/api/tags", timeout=2.0)
            return True
        except httpx.RequestError:
            return False

    def list_models(self) -> list[str]:
        try:
            resp = self.client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
            return [m["name"] for m in resp.json().get("models", [])]
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to Ollama. Is it running?\n"
                "  Fix: Run 'ollama serve' in another terminal"
            )
            return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    def generate(
        self,
        model: str,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = self.client.post(f"{self.base_url}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json()["response"]
        except httpx.ConnectError:
            raise RuntimeError(
                "Cannot connect to Ollama. Is it running?\n"
                "  Fix: Run 'ollama serve' in another terminal"
            )
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise RuntimeError(
                    f"Model '{model}' not found.\n"
                    f"  Fix: Run 'ollama pull {model}' or check model name"
                )
            raise RuntimeError(f"Ollama API error: {e.response.text}")
        except httpx.TimeoutException:
            raise RuntimeError(
                f"Request timed out. Model '{model}' may still be loading.\n"
                "  Fix: Wait and retry, or check GPU memory"
            )

    def unload(self, model: str):
        """Unload model from VRAM. Logs warning on failure."""
        try:
            self.client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "keep_alive": 0},
            )
        except Exception as e:
            logger.warning(f"Failed to unload model '{model}': {e}. Model may still be in VRAM.")


# Initialize client
client = OllamaClient()


def compare_models(
    prompt: str,
    system_prompt: str,
    model_a: str,
    model_b: str,
    temperature: float,
    max_tokens: int,
):
    """Generate responses sequentially (one model at a time in VRAM)."""
    if not prompt.strip():
        return "Please enter a prompt.", "Please enter a prompt."

    system = system_prompt.strip() if system_prompt.strip() else None

    # Generate with Model A, then unload
    try:
        response_a = client.generate(model_a, prompt, system, temperature, max_tokens)
    except Exception as e:
        response_a = f"Error: {e}"
    finally:
        client.unload(model_a)

    # Generate with Model B, then unload
    try:
        response_b = client.generate(model_b, prompt, system, temperature, max_tokens)
    except Exception as e:
        response_b = f"Error: {e}"
    finally:
        client.unload(model_b)

    return response_a, response_b


def create_app():
    """Create Gradio interface."""
    models = client.list_models()

    if not models:
        models = ["qwen2.5:7b", "my-finetuned"]  # Fallback

    with gr.Blocks(title="Model Comparison", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Model Comparison")
        gr.Markdown("Compare responses from different models side-by-side.")

        if not client.is_available():
            gr.Markdown("**Warning**: Ollama not running. Start with: `ollama serve`")

        with gr.Row():
            model_a = gr.Dropdown(
                choices=models,
                value=models[0] if models else None,
                label="Model A (Base)",
            )
            model_b = gr.Dropdown(
                choices=models,
                value=models[1] if len(models) > 1 else models[0] if models else None,
                label="Model B (Fine-tuned)",
            )

        system_prompt = gr.Textbox(
            label="System Prompt (optional)",
            placeholder="You are a helpful assistant...",
            lines=2,
        )

        prompt = gr.Textbox(
            label="User Prompt",
            placeholder="Enter your prompt here...",
            lines=4,
        )

        with gr.Row():
            temperature = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(64, 2048, 512, step=64, label="Max Tokens")

        compare_btn = gr.Button("Compare", variant="primary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Model A Response")
                output_a = gr.Textbox(label="", lines=15, show_copy_button=True)
            with gr.Column():
                gr.Markdown("### Model B Response")
                output_b = gr.Textbox(label="", lines=15, show_copy_button=True)

        compare_btn.click(
            fn=compare_models,
            inputs=[prompt, system_prompt, model_a, model_b, temperature, max_tokens],
            outputs=[output_a, output_b],
        )

        # Example prompts
        gr.Markdown("### Example Prompts")
        gr.Examples(
            examples=[
                ["Explain the difference between TCP and UDP."],
                ["Write a Python function to check if a number is prime."],
                ["What are the pros and cons of microservices?"],
                ["Give me three tips for learning a new programming language."],
            ],
            inputs=prompt,
        )

        # Refresh button
        def refresh_models():
            models = client.list_models()
            return gr.update(choices=models), gr.update(choices=models)

        refresh_btn = gr.Button("Refresh Model List", size="sm")
        refresh_btn.click(fn=refresh_models, outputs=[model_a, model_b])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
```

**Usage**:

```bash
# Make sure Ollama is running with your models
ollama serve &
ollama pull qwen2.5:7b

# If you've exported your fine-tuned model:
# cd outputs/exports/gguf && ollama create my-finetuned -f Modelfile

# Launch the UI
python src/compare_ui.py

# Open http://localhost:7860 in browser
```

---

## 13. Advanced: Custom Data Strategies

### 13.1 Creating High-Quality Training Data

#### Quality Checklist

Before training, verify your data:

- [ ] Every example follows your exact desired format
- [ ] Responses are the quality you want the model to produce
- [ ] Mix of short and long responses
- [ ] Coverage of different topics/question types
- [ ] No contradictory examples
- [ ] System prompt consistent (if using one)

#### Quantity Guidelines

| Dataset Size | Expected Results |
|--------------|------------------|
| 50-100 | Basic style adoption, may be inconsistent |
| 200-500 | Solid style consistency |
| 500-1000 | Strong style adherence |
| 1000+ | Diminishing returns unless very diverse |

### 13.2 Data Collection Approaches

#### Option 1: Manual Creation (Highest Quality)

Write examples yourself. Time-consuming but produces best results.

```python
# Example of hand-crafted training data
{
    "conversations": [
        {"role": "user", "content": "What is a REST API?"},
        {"role": "assistant", "content": "A REST API is an interface that allows applications to communicate over HTTP using standard methods.\n\nKey points:\n- Uses HTTP methods (GET, POST, PUT, DELETE)\n- Stateless - each request contains all needed information\n- Returns data in JSON or XML format\n- Follows URL patterns for resources (e.g., /users/123)"}
    ]
}
```

#### Option 2: Reformat Existing Data

Take Q&A data and rewrite answers in your target style.

```python
# Script to help reformat data
def reformat_to_style(original_answer: str, style_template: str) -> str:
    """
    Use an LLM to reformat an answer to your style.
    """
    prompt = f"""Reformat this answer to follow this style:

STYLE: {style_template}

ORIGINAL ANSWER:
{original_answer}

REFORMATTED ANSWER:"""

    # Call your preferred LLM API
    return llm_generate(prompt)
```

#### Option 3: LLM-Assisted Generation

Use Claude or GPT-4 to generate examples, then curate.

```python
# Example prompt for generating training data
generation_prompt = """Generate a training example for fine-tuning.
The assistant should respond in this style:
- Start with a one-sentence direct answer
- Follow with "Key points:" header
- List 3-5 bullet points
- No hedging language

Topic: {topic}

Generate the user question and assistant response in JSON format:
{{"conversations": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}
"""
```

#### Option 4: Hybrid Approach (Recommended)

1. Create 20-50 manual seed examples
2. Use LLM to expand to 200-500 examples
3. Human review and filter
4. Iterate based on model output quality

### 13.3 Defining Your Style

Create a clear style document:

```markdown
# My Custom Style Guide

## Format
1. One-sentence direct answer (no preamble)
2. "Key points:" header
3. 3-5 bullet points
4. Optional code example (if relevant)

## Tone
- Professional but approachable
- No hedging ("I think", "probably", "might")
- Active voice
- Concise sentences

## Avoid
- Unnecessary greetings ("Great question!")
- Filler phrases
- Over-qualification
- Walls of text
```

---

## 14. Troubleshooting & Common Issues

### 14.1 Out of Memory (OOM)

**Symptoms**: CUDA out of memory error during training

#### OOM Fallback Ladder (16GB VRAM)

Try these fixes **in order**. Each step recovers ~1-3GB VRAM:

| Step | Change | VRAM Saved | Impact |
|------|--------|------------|--------|
| 1 | `max_seq_length: 1024 → 512` | ~2-3 GB | Truncates long examples |
| 2 | `batch_size: 4 → 2` | ~1-2 GB | Slower training (increase `gradient_accumulation_steps` to 8) |
| 3 | `lora_r: 64 → 32` | ~0.5 GB | Reduced learning capacity |
| 4 | `packing: True → False` | ~0.5 GB | Less efficient batching |
| 5 | `batch_size: 2 → 1` | ~1 GB | Much slower (set `gradient_accumulation_steps` to 16) |

**Example: Maximum memory savings configuration**:
```yaml
model:
  max_seq_length: 512      # Down from 1024

lora:
  r: 32                    # Down from 64

training:
  batch_size: 1            # Down from 4
  gradient_accumulation_steps: 16  # Up from 4 (maintains effective batch size)
```

**Additional tips**:
```python
# Clear VRAM before training
import torch
torch.cuda.empty_cache()

# Monitor VRAM during training
import subprocess
result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
print(f"VRAM used: {result.stdout.strip()} MB")
```

**Still OOM?** Check for:
- Other processes using GPU (`nvidia-smi`)
- Jupyter notebooks with loaded models
- Multiple training runs

#### VRAM Estimation Script

Before training, estimate VRAM usage for your configuration to avoid OOM surprises.

Create `scripts/estimate_vram.py`:

```python
"""
Estimate VRAM usage for a given training configuration.
Uses formula-based estimation (no actual model loading).
"""

import yaml


def estimate_vram(config_path: str = "configs/training_config.yaml"):
    """Estimate VRAM usage based on config parameters."""

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Extract parameters
    seq_len = config["model"]["max_seq_length"]
    batch_size = config["training"]["batch_size"]
    lora_r = config["lora"]["r"]
    is_4bit = config["model"].get("load_in_4bit", True)

    print(f"\n{'='*50}")
    print("VRAM ESTIMATION")
    print(f"{'='*50}")
    print(f"\nConfiguration:")
    print(f"  Model: {config['model']['name']}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  LoRA rank: {lora_r}")
    print(f"  4-bit quantization: {is_4bit}")

    # Base model VRAM (7B parameters)
    if is_4bit:
        model_vram = 4.0  # 4-bit: ~0.5 bytes per param
    else:
        model_vram = 14.0  # fp16: ~2 bytes per param

    # LoRA adapters (approximate)
    # Each LoRA layer adds 2 * hidden_dim * r parameters
    # Qwen2.5-7B: hidden_dim = 4096, 32 layers, 7 target modules
    hidden_dim = 4096
    num_layers = 32
    num_modules = 7
    lora_params = 2 * hidden_dim * lora_r * num_layers * num_modules
    lora_vram = lora_params * 2 / 1e9  # fp16

    # Optimizer states (AdamW: 2x parameters for momentum + variance)
    optimizer_vram = lora_vram * 2

    # Activations (rough estimate)
    # Scales with batch_size * seq_len * hidden_dim * num_layers
    bytes_per_activation = 2  # fp16
    activation_multiplier = 2  # for gradients
    activations_vram = (batch_size * seq_len * hidden_dim * num_layers *
                        bytes_per_activation * activation_multiplier) / 1e9

    # Gradient checkpointing reduces activation memory ~4x
    activations_vram = activations_vram / 4

    # Total estimate
    total_vram = model_vram + lora_vram + optimizer_vram + activations_vram
    overhead = 0.5  # CUDA overhead, fragmentation

    print(f"\nEstimated VRAM breakdown:")
    print(f"  Base model:    {model_vram:.1f} GB")
    print(f"  LoRA adapters: {lora_vram:.2f} GB")
    print(f"  Optimizer:     {optimizer_vram:.2f} GB")
    print(f"  Activations:   {activations_vram:.1f} GB")
    print(f"  Overhead:      {overhead:.1f} GB")
    print(f"  ─────────────────────")
    print(f"  TOTAL:         {total_vram + overhead:.1f} GB")

    # Recommendations
    print(f"\n{'='*50}")
    print("RECOMMENDATIONS")
    print(f"{'='*50}")

    if total_vram + overhead > 16:
        print("⚠️  Estimated usage exceeds 16GB!")
        print("   Suggestions:")
        if seq_len > 512:
            print(f"   - Reduce max_seq_length from {seq_len} to 512")
        if batch_size > 2:
            print(f"   - Reduce batch_size from {batch_size} to 2")
        if lora_r > 32:
            print(f"   - Reduce lora_r from {lora_r} to 32")
    elif total_vram + overhead > 14:
        print("⚠️  Tight fit for 16GB GPU. Leave headroom for spikes.")
    else:
        print("✓  Should fit comfortably in 16GB VRAM")

    available_headroom = 16 - (total_vram + overhead)
    print(f"\n   Estimated headroom: {available_headroom:.1f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    args = parser.parse_args()

    estimate_vram(args.config)
```

**Usage**:

```bash
python scripts/estimate_vram.py --config configs/training_config.yaml
```

**Note**: This is an estimation. Actual usage may vary by ±1-2GB. Always monitor actual VRAM during initial training steps.

### 14.2 Resuming from Checkpoint

If training is interrupted (OOM, crash, power failure), you can resume from the last checkpoint.

#### Finding Your Checkpoint

Checkpoints are saved in your output directory:

```bash
ls outputs/alpaca_r64_20241228_1430/checkpoints/
# checkpoint-100/  checkpoint-200/  checkpoint-300/
```

#### Resuming Training

Add the `resume_from_checkpoint` argument to your training call:

```python
# In src/train.py, modify the train() call:
trainer.train(resume_from_checkpoint="outputs/alpaca_r64_20241228_1430/checkpoints/checkpoint-200")

# Or use "latest" to auto-detect:
trainer.train(resume_from_checkpoint=True)  # Resumes from latest checkpoint in output_dir
```

#### Command Line Resume

You can also pass it via command line by modifying train.py:

```python
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training_config.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint path or 'latest'")
    args = parser.parse_args()

    main(args.config, resume_from=args.resume)
```

Then run:

```bash
# Resume from specific checkpoint
python src/train.py --resume outputs/alpaca_r64_20241228_1430/checkpoints/checkpoint-200

# Resume from latest
python src/train.py --resume latest
```

**Note**: Resuming restores optimizer state, learning rate schedule, and RNG state, so training continues exactly where it left off.

### 14.3 Training Loss Not Decreasing

**Possible causes**:

1. **Learning rate too low**: Try `3e-4` or `5e-4`
2. **Data format issues**: Verify data is correctly formatted
3. **Too few examples**: Need at least 50-100 for meaningful training

**Debug**:
```python
# Check a sample from your dataset
from datasets import load_dataset
ds = load_dataset("json", data_files="data/processed/train.jsonl", split="train[:1]")
print(ds[0])
```

### 14.3 Model Outputs Gibberish

**Causes**:
- Wrong chat template
- Corrupted training run
- Inference settings wrong

**Fix**:
```python
# Make sure you're using the right chat template
tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

# Use reasonable temperature
response = generate(model, tokenizer, prompt, temperature=0.7)  # Not 2.0!
```

### 14.4 Ollama Model Not Loading

**Check**:

1. Ollama server running: `ollama serve`
2. Model exists: `ollama list`
3. Modelfile path is correct (use absolute path)
4. GGUF file not corrupted: check file size (~4GB for q4_k_m)

**Recreate model**:
```bash
ollama rm my-finetuned
ollama create my-finetuned -f Modelfile
```

### 14.5 Slow Training

**Optimizations**:

1. Enable `packing=True` in SFTTrainer (already enabled)
2. Use `use_gradient_checkpointing="unsloth"` (already enabled)
3. Ensure you're using 4-bit quantization
4. Check CUDA version matches PyTorch version

### 14.6 Style Not Transferring

**If the model isn't adopting your style**:

1. **Check data quality**: Are your examples consistent?
2. **Increase examples**: Try 200-500 instead of 50
3. **Increase epochs**: 5-10 for small datasets
4. **Verify format**: Print a few formatted examples to check

---

## 15. Next Steps & Resources

### After Your First Successful Fine-Tune

1. **Experiment with hyperparameters**: Try different LoRA ranks, learning rates
2. **Collect more data**: Build a larger, higher-quality dataset
3. **Try different base models**: Qwen2.5-14B, Mistral 7B
4. **Add evaluation metrics**: Track specific style criteria

### Recommended Reading

- [Unsloth Documentation](https://github.com/unslothai/unsloth)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [Hugging Face TRL](https://huggingface.co/docs/trl)

### Model Alternatives

| Model | Parameters | Notes |
|-------|------------|-------|
| Qwen2.5-7B-Instruct | 7B | **Recommended starting point** |
| Qwen2.5-14B-Instruct | 14B | Better quality, needs more VRAM optimization |
| Llama-3.2-8B-Instruct | 8B | Strong general-purpose |
| Mistral-7B-Instruct-v0.3 | 7B | Fast inference |
| Qwen3-8B | 8B | Latest generation with reasoning |

### Getting Help

- [Unsloth GitHub Issues](https://github.com/unslothai/unsloth/issues)
- [Hugging Face Forums](https://discuss.huggingface.co/)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

### Licensing Note

Before deploying commercially, verify the license of your base model:
- **Qwen2.5 models**: Apache 2.0 license (permissive, commercial use allowed)
- Check the model card on Hugging Face for specific license terms
- Your fine-tuned adapters inherit the base model's license

### Data Privacy Note

When fine-tuning with sensitive data:
- Training data can be memorized and potentially extracted from the model
- Never train on data you wouldn't want in model outputs
- Consider keeping training data and models on local storage only
- Don't upload adapters trained on private data to public hubs

### Reproducibility

The training scripts use fixed random seeds (default: 42) for reproducibility. This affects:
- Dataset shuffling and splits
- Model weight initialization
- Training sampling behavior

To reproduce a training run exactly, ensure:
- Same seed value in config
- Same package versions (pin with `uv pip freeze`)
- Same hardware (GPU type affects floating point operations)
- Same data files

---

## 16. Quick Reference

### Command Reference Table

| Task | Command |
|------|---------|
| **Setup** | |
| Install dependencies | `uv pip install torch --index-url https://download.pytorch.org/whl/cu124 && uv pip install unsloth datasets gradio httpx pyyaml` |
| Verify installation | `python -c "import unsloth; print('OK')"` |
| **Data Preparation** | |
| Prepare Alpaca data | `python scripts/prepare_dataset.py --source alpaca --num-examples 2000` |
| Prepare custom data | `python scripts/prepare_dataset.py --source custom --input my_data.jsonl` |
| Analyze dataset stats | `python scripts/data_stats.py` |
| **Training** | |
| Estimate VRAM | `python scripts/estimate_vram.py` |
| Run training | `python src/train.py --config configs/training_config.yaml` |
| Resume training | `python src/train.py --resume latest` |
| **Testing** | |
| Interactive test | `python src/inference.py --model outputs/final_model` |
| Batch test | `python src/inference.py --model outputs/final_model --batch` |
| Test for forgetting | `python scripts/test_forgetting.py --finetuned outputs/final_model` |
| **Export & Deployment** | |
| Export to GGUF | `python src/export.py --model outputs/final_model` |
| Create Ollama model | `ollama create my-finetuned -f outputs/exports/gguf/Modelfile` |
| Run in Ollama | `ollama run my-finetuned` |
| Merge LoRA adapters | `python scripts/merge_lora.py --adapter outputs/final_model` |
| **Comparison** | |
| Compare models (CLI) | `python scripts/compare_models.py --finetuned outputs/final_model` |
| Launch comparison UI | `python src/compare_ui.py` |

---

## Quick Start Summary

Minimal steps to fine-tune and deploy:

```bash
# 1. Setup
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install unsloth datasets gradio httpx pyyaml

# 2. Prepare data
python scripts/prepare_dataset.py --source alpaca --num-examples 500

# 3. Train (edit config first!)
python src/train.py

# 4. Export and run
python src/export.py --model outputs/*/final_model
cd outputs/exports/gguf && ollama create my-model -f Modelfile
ollama run my-model
```

---

## Complete Workflow Summary

Full workflow including validation and comparison:

```bash
# ═══════════════════════════════════════════════════════════════
# PHASE 1: SETUP
# ═══════════════════════════════════════════════════════════════

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create project
mkdir qwen-finetune && cd qwen-finetune
uv init && uv venv --python 3.11 && source .venv/bin/activate

# Install dependencies
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install unsloth datasets gradio httpx pyyaml

# Verify installation
python -c "import torch; import unsloth; print('Setup OK')"

# ═══════════════════════════════════════════════════════════════
# PHASE 2: DATA PREPARATION
# ═══════════════════════════════════════════════════════════════

# Prepare training data (choose one):
python scripts/prepare_dataset.py --source alpaca --num-examples 2000
# OR: python scripts/prepare_dataset.py --source custom --input my_data.jsonl

# Validate data quality
python scripts/data_stats.py

# ═══════════════════════════════════════════════════════════════
# PHASE 3: TRAINING
# ═══════════════════════════════════════════════════════════════

# Estimate VRAM usage
python scripts/estimate_vram.py

# Edit configs/training_config.yaml as needed, then train
python src/train.py

# ═══════════════════════════════════════════════════════════════
# PHASE 4: TESTING
# ═══════════════════════════════════════════════════════════════

# Test model outputs
python src/inference.py --model outputs/*/final_model --batch

# Test for capability regression
python scripts/test_forgetting.py --finetuned outputs/*/final_model

# ═══════════════════════════════════════════════════════════════
# PHASE 5: DEPLOYMENT
# ═══════════════════════════════════════════════════════════════

# Export to GGUF
python src/export.py --model outputs/*/final_model

# Create Ollama model
cd outputs/exports/gguf
ollama create my-finetuned -f Modelfile
ollama run my-finetuned

# ═══════════════════════════════════════════════════════════════
# PHASE 6: COMPARISON
# ═══════════════════════════════════════════════════════════════

# Pull base model for comparison
ollama pull qwen2.5:7b

# Launch comparison UI
python src/compare_ui.py
# Open http://localhost:7860
```

---

*Document version: 2.1*
*Target: First-time fine-tuners with NVIDIA 5060 Ti 16GB*
*Base model: Qwen2.5-7B-Instruct with Unsloth QLoRA*
