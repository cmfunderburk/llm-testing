# Unsloth QLoRA Fine-Tuning Guide v2 (Refined)

A beginner-friendly guide to fine-tuning Qwen2.5-7B with custom instruction styles and domain adaptation using Unsloth on an NVIDIA 5060 Ti 16GB GPU.

---

## Overview

This guide walks you through fine-tuning a 7B parameter language model to adopt your custom writing style and domain-specific behavior. By the end, you'll have a model deployed locally via Ollama that responds in your preferred format.

### What You'll Achieve

- Fine-tune Qwen2.5-7B to adopt a custom writing style and domain terminology
- Create high-quality training data using LLM-assisted generation
- Export your model to Ollama for local inference
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
│  1. SETUP       │────▶│  2. CREATE DATA │────▶│  3. TRAIN       │
│  Environment    │     │  LLM-assisted   │     │  Fine-tune      │
│  + test run     │     │  generation     │     │  with your data │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  5. COMPARE     │◀────│  4. EXPORT      │
                        │  Side-by-side   │     │  To GGUF/Ollama │
                        │  evaluation     │     │                 │
                        └─────────────────┘     └─────────────────┘
```

---

## Is Fine-Tuning Right For Your Use Case?

Before investing time in fine-tuning, understand what it can and cannot do.

### The Fine-Tuning vs RAG Decision

Fine-tuning and RAG (Retrieval-Augmented Generation) solve different problems:

| Problem Type | Solution | Examples |
|--------------|----------|----------|
| **Style/Format** | Fine-tuning | "Always respond with bullet points", "Use our company's tone" |
| **Terminology/Jargon** | Fine-tuning | "Use 'customer' not 'user'", "Know that 'sprint' means 2-week cycle" |
| **Behavioral Patterns** | Fine-tuning | "Be more concise", "Always ask clarifying questions" |
| **Factual Knowledge** | RAG | "Know our product catalog", "Answer questions about our documentation" |
| **Current Information** | RAG | "Today's stock prices", "Latest company announcements" |
| **Large Knowledge Base** | RAG | "Search across 10,000 support tickets" |

### Decision Framework

```
┌─────────────────────────────────────────────────────────────────────┐
│                    WHAT ARE YOU TRYING TO CHANGE?                   │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
   HOW the model            WHAT terms              WHAT facts
   responds                 it uses                 it knows
   (tone, format,           (jargon,                (data, documents,
   structure)               terminology)            current info)
        │                       │                       │
        ▼                       ▼                       ▼
   ┌──────────┐           ┌──────────┐           ┌──────────┐
   │FINE-TUNE │           │FINE-TUNE │           │   RAG    │
   │          │           │ (+ maybe │           │          │
   │ Style    │           │   RAG)   │           │ Knowledge│
   └──────────┘           └──────────┘           └──────────┘
```

**This guide covers**: Style tuning, terminology adoption, and domain adaptation (the left and middle paths).

**Not covered**: RAG implementation (right path). If you need factual knowledge injection, implement RAG separately.

### Combined Goals: Style + Domain Adaptation

If you're doing both style tuning AND domain adaptation (teaching the model your terminology while shaping how it responds), you'll need:

| Goal Type | Dataset Size | Notes |
|-----------|--------------|-------|
| Style only | 200-500 examples | Consistent format, tone, structure |
| Domain terminology only | 300-700 examples | Specialized vocabulary, jargon |
| **Style + Domain combined** | **500-1000 examples** | Need coverage of both aspects |

**Key insight**: When combining goals, ensure your training examples demonstrate both the style AND the terminology together. Don't create separate "style examples" and "terminology examples"—each example should exhibit both.

### When to Avoid Fine-Tuning

- Need for factual knowledge that changes (use RAG)
- One-off tasks (just prompt well)
- Rapidly changing requirements (prompts are easier to update)
- When you have <100 quality examples for combined goals

---

## Prerequisites & Environment Setup

### System Requirements

- **GPU**: NVIDIA GPU with 16GB+ VRAM (tested on 5060 Ti)
- **CUDA**: Compatible version for your GPU (check with `nvidia-smi`)
- **RAM**: 32GB recommended (16GB minimum)
- **Storage**: ~50GB for models and checkpoints
- **OS**: Linux (recommended) or WSL2

### CUDA Version Compatibility

Your CUDA version must match your GPU and PyTorch installation:

```bash
# Check your current CUDA version
nvidia-smi | grep "CUDA Version"

# Check PyTorch CUDA version after installation
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
```

**Matching versions**: When installing PyTorch, use the wheel URL that matches your system's CUDA version:
- CUDA 12.1: `--index-url https://download.pytorch.org/whl/cu121`
- CUDA 12.4: `--index-url https://download.pytorch.org/whl/cu124`
- CUDA 12.6+: Check [PyTorch's install page](https://pytorch.org/get-started/locally/) for latest options

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

# 3. Check your CUDA version
nvidia-smi | grep "CUDA Version"

# 4. Install PyTorch with matching CUDA version
# Replace cu124 with your CUDA version (cu121, cu124, etc.)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install Unsloth (includes transformers, peft, trl, bitsandbytes)
uv pip install unsloth

# 6. Install additional utilities
uv pip install datasets gradio wandb httpx pyyaml

# 7. Verify installation
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

python -c "import unsloth; print('Unsloth installed successfully')"
```

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

## Repository Structure

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
│   ├── generate_data.py          # LLM-assisted data generation
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

---

## Quick Start: Your First Fine-Tune

**Goal**: Run a complete fine-tuning cycle with a small public dataset to learn the workflow before creating your own data.

### Choose a Test Dataset

| Dataset | Size | Type | Best For |
|---------|------|------|----------|
| `yahma/alpaca-cleaned` | 52K | Instruction-following | General test, learning workflow |
| `teknium/OpenHermes-2.5` | 1M | Instruction-following | Diverse instructions |
| `HuggingFaceH4/ultrachat_200k` | 200K | Conversational | Multi-turn dialogue |

For your first test, use a small subset of **alpaca-cleaned** (fast, well-understood).

### Minimal Test Script

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
    packing=True,  # For speed in this test script
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

### Run the Test

```bash
python test_finetune.py
```

**Expected output**:
- Model loads in ~30 seconds
- Training runs for ~10-15 minutes
- VRAM usage: ~8-10 GB
- You'll see sample outputs at the end

---

## Creating Training Data

Since you need to create your own training data, this section covers LLM-assisted data generation workflows.

### Dataset Size Guidelines

| Goal | Minimum | Recommended | Notes |
|------|---------|-------------|-------|
| Style only | 200 | 400-500 | Consistent format/tone |
| Domain terminology | 300 | 500-700 | Specialized vocabulary |
| **Style + Domain** | **500** | **700-1000** | Each example shows both |

**Quality over quantity**: 500 well-crafted examples outperform 2000 inconsistent ones.

### Data Format

Your training data must be in **conversation format** (JSONL):

```jsonl
{"conversations": [{"role": "user", "content": "What is Python?"}, {"role": "assistant", "content": "Python is a high-level programming language known for its readability and versatility."}]}
{"conversations": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a hello world program."}, {"role": "assistant", "content": "Here's a hello world program in Python:\n\n```python\nprint(\"Hello, World!\")\n```"}]}
```

### Step 1: Define Your Style Guide

Before generating data, document your target style:

```markdown
# My Style Guide

## Format Requirements
- Start with a direct one-sentence answer (no preamble)
- Use "Key points:" header for lists
- 3-5 bullet points maximum
- Include code examples when relevant

## Terminology
- Use "customer" not "user" or "client"
- Use "sprint" to mean 2-week development cycle
- Use "ship" to mean deploy/release

## Tone
- Professional but approachable
- No hedging ("I think", "probably", "might")
- Active voice
- Concise sentences

## Avoid
- Greetings like "Great question!"
- Filler phrases ("In order to", "It's important to note")
- Overly long paragraphs
```

### Step 2: Create Seed Examples (Manual)

Create 20-50 high-quality examples by hand. These define the gold standard:

Create `data/raw/seed_examples.jsonl`:

```jsonl
{"conversations": [{"role": "user", "content": "What is a REST API?"}, {"role": "assistant", "content": "A REST API is an interface that allows applications to communicate over HTTP using standard methods.\n\nKey points:\n- Uses HTTP methods (GET, POST, PUT, DELETE)\n- Stateless - each request contains all needed information\n- Returns data in JSON or XML format\n- Follows URL patterns for resources (e.g., /customers/123)"}]}
{"conversations": [{"role": "user", "content": "How do I handle errors in Python?"}, {"role": "assistant", "content": "Use try/except blocks to catch and handle exceptions gracefully.\n\nKey points:\n- Wrap risky code in `try` blocks\n- Catch specific exceptions before generic ones\n- Use `finally` for cleanup that must always run\n- Raise custom exceptions for domain-specific errors\n\n```python\ntry:\n    result = process_data(input)\nexcept ValueError as e:\n    log_error(e)\n    return default_value\nfinally:\n    cleanup()\n```"}]}
```

### Step 3: LLM-Assisted Data Generation

Use this script to expand your seed examples:

Create `scripts/generate_data.py`:

```python
"""
LLM-assisted training data generation.
Uses Claude or GPT-4 to generate examples matching your style.
"""

import json
import os
from pathlib import Path
import httpx

# Configure your API
API_KEY = os.environ.get("ANTHROPIC_API_KEY")  # or OPENAI_API_KEY
API_URL = "https://api.anthropic.com/v1/messages"  # or OpenAI endpoint


def load_style_guide(path: str = "data/raw/style_guide.md") -> str:
    """Load your style guide."""
    with open(path) as f:
        return f.read()


def load_seed_examples(path: str = "data/raw/seed_examples.jsonl", n: int = 5) -> list:
    """Load a few seed examples for few-shot prompting."""
    with open(path) as f:
        examples = [json.loads(line) for line in f if line.strip()]
    return examples[:n]


def generate_example(
    topic: str,
    style_guide: str,
    seed_examples: list,
    api_key: str = API_KEY,
) -> dict:
    """Generate a single training example using an LLM."""

    # Format seed examples for the prompt
    examples_text = ""
    for i, ex in enumerate(seed_examples, 1):
        convos = ex["conversations"]
        user_msg = next(c["content"] for c in convos if c["role"] == "user")
        asst_msg = next(c["content"] for c in convos if c["role"] == "assistant")
        examples_text += f"\nExample {i}:\nUser: {user_msg}\nAssistant: {asst_msg}\n"

    prompt = f"""Generate a training example for fine-tuning an LLM.

STYLE GUIDE:
{style_guide}

EXAMPLE OUTPUTS (match this style exactly):
{examples_text}

TOPIC FOR NEW EXAMPLE:
{topic}

Generate a realistic user question about this topic and an assistant response that perfectly matches the style guide. The response should demonstrate both the format/structure AND any domain-specific terminology from the style guide.

Output as JSON:
{{"conversations": [{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

JSON only, no other text:"""

    # Call API (example for Anthropic)
    headers = {
        "x-api-key": api_key,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01",
    }

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }

    response = httpx.post(API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()

    content = response.json()["content"][0]["text"]

    # Parse JSON from response
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from: {content}")


def generate_dataset(
    topics: list[str],
    output_path: str = "data/raw/generated.jsonl",
    style_guide_path: str = "data/raw/style_guide.md",
    seed_path: str = "data/raw/seed_examples.jsonl",
):
    """Generate a dataset from a list of topics."""

    style_guide = load_style_guide(style_guide_path)
    seed_examples = load_seed_examples(seed_path)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    generated = []
    errors = []

    for i, topic in enumerate(topics):
        print(f"[{i+1}/{len(topics)}] Generating: {topic[:50]}...")
        try:
            example = generate_example(topic, style_guide, seed_examples)
            generated.append(example)

            # Save incrementally
            with open(output_path, "a") as f:
                f.write(json.dumps(example) + "\n")

        except Exception as e:
            print(f"  Error: {e}")
            errors.append({"topic": topic, "error": str(e)})

    print(f"\nGenerated {len(generated)} examples, {len(errors)} errors")
    print(f"Saved to {output_path}")

    if errors:
        error_path = output_path.replace(".jsonl", "_errors.json")
        with open(error_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {error_path}")


# Example topic list for generation
EXAMPLE_TOPICS = [
    "What is Docker and why use it?",
    "How do microservices communicate?",
    "Explain database indexing",
    "What is CI/CD?",
    "How does caching work?",
    "What are environment variables?",
    "Explain API rate limiting",
    "What is a load balancer?",
    "How do webhooks work?",
    "What is OAuth?",
    # Add more topics for your domain...
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--topics-file", help="File with topics (one per line)")
    parser.add_argument("--output", default="data/raw/generated.jsonl")
    parser.add_argument("--style-guide", default="data/raw/style_guide.md")
    parser.add_argument("--seeds", default="data/raw/seed_examples.jsonl")
    args = parser.parse_args()

    if args.topics_file:
        with open(args.topics_file) as f:
            topics = [line.strip() for line in f if line.strip()]
    else:
        topics = EXAMPLE_TOPICS
        print("Using example topics. Create a topics file for your domain.")

    generate_dataset(
        topics=topics,
        output_path=args.output,
        style_guide_path=args.style_guide,
        seed_path=args.seeds,
    )
```

### Step 4: Review and Curate

**Critical step**: LLM-generated data requires human review.

```bash
# Generate initial batch
python scripts/generate_data.py --topics-file data/raw/my_topics.txt

# Review the output
# - Delete examples that don't match your style
# - Edit examples that are close but not perfect
# - Note patterns of errors to improve your style guide

# Check quality statistics
python scripts/data_stats.py --train data/raw/generated.jsonl
```

### Step 5: Prepare Final Dataset

```bash
# Combine seed + generated data and split into train/val
python scripts/prepare_dataset.py --source custom --input data/raw/all_examples.jsonl
```

### Topic Generation Tips

For combined style + domain goals, ensure topics cover:

1. **Core domain concepts** (50% of topics)
   - Fundamental terminology
   - Common questions in your domain

2. **Edge cases** (25% of topics)
   - Complex scenarios
   - Multi-part questions

3. **Format-testing topics** (25% of topics)
   - Questions that require lists
   - Questions that need code examples
   - Questions that need step-by-step instructions

---

## Understanding Key Concepts

### What is QLoRA?

**QLoRA** = **Q**uantized **Lo**w-**R**ank **A**daptation

Instead of fine-tuning all 7 billion parameters:
1. **Quantize** the base model to 4-bit (reduces VRAM from ~14GB to ~4GB)
2. **Add small trainable adapters** (LoRA) to key layers
3. **Train only the adapters** (~0.5-2% of parameters)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Base Model (Frozen, 4-bit)                   │
│                         ~4 GB VRAM                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │ LoRA A  │  │ LoRA B  │  │ LoRA C  │  │ LoRA D  │  ...       │
│  │ (train) │  │ (train) │  │ (train) │  │ (train) │            │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘            │
│                    ~0.5 GB VRAM                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LoRA Rank Selection

| Rank | Parameters | Use Case |
|------|------------|----------|
| 32 | ~80M | Small datasets (<300 examples), style-only |
| **64** | **~160M** | **Recommended default** |
| 128 | ~320M | Large datasets (1000+), complex domain |

### Key Hyperparameters

| Parameter | Default | Impact |
|-----------|---------|--------|
| `learning_rate` | 2e-4 | Lower (1e-4) for small datasets |
| `num_epochs` | 3 | More for small datasets (5-10) |
| `batch_size` | 4 | Lower if OOM |
| `max_seq_length` | 1024 | Increase for long content |

### Assistant-Only Loss

By default, loss is computed on the entire conversation. Better results come from training only on assistant responses:

```python
from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
)
```

This is enabled by default in the production training script.

### Packing vs. No Packing

| Setting | Best For |
|---------|----------|
| `packing: true` | Large datasets, speed priority |
| `packing: false` | Small datasets, style tuning, quality priority |

The training script defaults to `packing: false` for better style transfer.

---

## Training Configuration

### Configuration File

Create `configs/training_config.yaml`:

```yaml
# Model Configuration
model:
  name: "unsloth/Qwen2.5-7B-Instruct"
  max_seq_length: 1024
  load_in_4bit: true

# LoRA Configuration
lora:
  r: 64                  # Rank - adjust based on dataset size
  alpha: 64
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
  learning_rate: 2.0e-4
  warmup_ratio: 0.1
  weight_decay: 0.01
  packing: false                  # Better for style tuning

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

# Run Naming
run_name: "mystyle"
output_base: "outputs"

# Early Stopping
early_stopping:
  enabled: true
  patience: 5
  threshold: 0.001

# Reproducibility
seed: 42
```

### Settings for Combined Style + Domain

For style + domain adaptation with 500-1000 examples:

```yaml
lora:
  r: 64                    # Keep at 64 for domain learning capacity

training:
  num_epochs: 3            # 3 epochs usually sufficient
  learning_rate: 1.5e-4    # Slightly lower than default
  packing: false           # Preserve example boundaries
```

---

## Core Training Script

Create `src/train.py` (see original document for full implementation).

Key differences from test script:
- Uses `DataCollatorForCompletionOnlyLM` for assistant-only loss
- `packing=False` for better style preservation
- Early stopping to prevent overfitting
- Proper checkpoint management

### Run Training

```bash
# Prepare your data
python scripts/prepare_dataset.py --source custom --input data/raw/all_examples.jsonl

# Estimate VRAM before training
python scripts/estimate_vram.py

# Run training
python src/train.py --config configs/training_config.yaml
```

---

## Inference & Testing

### Test Your Model

```bash
# Interactive chat
python src/inference.py --model outputs/mystyle_r64_*/final_model

# Batch test
python src/inference.py --model outputs/mystyle_r64_*/final_model --batch
```

### Test for Catastrophic Forgetting

```bash
python scripts/test_forgetting.py --finetuned outputs/mystyle_r64_*/final_model
```

Review outputs for:
- Correct math answers
- Accurate factual information
- Valid code syntax
- Sound logical reasoning

---

## Export to Ollama

### Export Script

```bash
# Export with default quantization (q4_k_m)
python src/export.py --model outputs/mystyle_r64_*/final_model

# Create Ollama model
cd outputs/exports/gguf
ollama create my-finetuned -f Modelfile
ollama run my-finetuned
```

### Tuning Inference Parameters

Adjust `temperature` and `top_p` in the Modelfile:

| Use Case | Temperature | Top-p |
|----------|-------------|-------|
| Consistent output | 0.3 | 0.7 |
| Balanced (default) | 0.7 | 0.9 |
| Creative variation | 1.0 | 0.95 |

---

## Evaluation: Side-by-Side Comparison

### Compare with Base Model

```bash
python scripts/compare_models.py --finetuned outputs/mystyle_r64_*/final_model
```

### Gradio Comparison UI

```bash
# Start Ollama
ollama serve &

# Launch UI
python src/compare_ui.py

# Open http://localhost:7860
```

---

## Troubleshooting

### Out of Memory (OOM)

Try these fixes in order:

| Step | Change | VRAM Saved |
|------|--------|------------|
| 1 | `max_seq_length: 1024 → 512` | ~2-3 GB |
| 2 | `batch_size: 4 → 2` | ~1-2 GB |
| 3 | `lora_r: 64 → 32` | ~0.5 GB |
| 4 | `batch_size: 2 → 1` | ~1 GB |

### Style Not Transferring

1. **Check data quality**: Are examples consistent?
2. **Increase examples**: 500+ for combined goals
3. **Verify format**: Print formatted examples to check
4. **Lower learning rate**: Try 1e-4

### Training Loss Not Decreasing

1. Check data format is valid JSONL
2. Verify conversations have correct structure
3. Try higher learning rate (3e-4)

---

## Maintenance & Iteration

### After Your First Fine-Tune

1. **Evaluate systematically**: Test on prompts covering your use cases
2. **Identify gaps**: Note where style/terminology isn't consistent
3. **Expand dataset**: Add examples addressing gaps
4. **Retrain**: Use same config, updated data

### Updating Your Model

When to retrain:
- Style requirements change
- New terminology to incorporate
- Base model updates to newer version

Retraining workflow:
```bash
# Add new examples to your dataset
cat new_examples.jsonl >> data/raw/all_examples.jsonl

# Re-prepare dataset
python scripts/prepare_dataset.py --source custom --input data/raw/all_examples.jsonl

# Train with same config
python src/train.py

# Compare old vs new fine-tuned model
```

### Migrating to Newer Base Models

When Qwen3 or other improved base models release:
1. Update `model.name` in config
2. Verify chat template compatibility
3. Retrain with your existing data
4. Compare performance before switching production

---

## Quick Reference

| Task | Command |
|------|---------|
| **Setup** | |
| Install dependencies | See Installation section (match your CUDA version) |
| Verify installation | `python -c "import unsloth; print('OK')"` |
| **Data Creation** | |
| Generate data | `python scripts/generate_data.py --topics-file topics.txt` |
| Analyze dataset | `python scripts/data_stats.py` |
| Prepare for training | `python scripts/prepare_dataset.py --source custom --input data.jsonl` |
| **Training** | |
| Estimate VRAM | `python scripts/estimate_vram.py` |
| Run training | `python src/train.py` |
| **Testing** | |
| Interactive test | `python src/inference.py --model outputs/*/final_model` |
| Forgetting test | `python scripts/test_forgetting.py` |
| **Deployment** | |
| Export to GGUF | `python src/export.py --model outputs/*/final_model` |
| Create Ollama model | `ollama create my-model -f Modelfile` |
| **Comparison** | |
| CLI comparison | `python scripts/compare_models.py` |
| Launch UI | `python src/compare_ui.py` |

---

## Open Items

| Item | Context | Impact |
|------|---------|--------|
| 5060 Ti availability | Hardware section | GPU may not be available/stable at time of use |
| CUDA version for Blackwell | Setup section | May need different CUDA version than 12.x |
| Anthropic API rate limits | Data generation script | May affect bulk data generation speed |

---

*Refined via /refine on 2025-12-29*
