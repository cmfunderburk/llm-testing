"""
Minimal test fine-tune script.
Uses 500 examples from Alpaca for a quick learning run (~10-15 min on 5060 Ti).
"""

import psutil  # Must import before unsloth to avoid cache bug
import os
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"  # Try to disable trainer patching

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
    bf16=True,  # Use bf16 for modern GPUs (RTX 30xx+)
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
    dataset_num_proc=4,  # Explicitly set to avoid psutil bug in unsloth
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
