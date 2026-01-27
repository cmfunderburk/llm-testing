"""
Experiment: Basic QLoRA Fine-Tuning
===================================

HYPOTHESIS
----------
Fine-tuning a quantized LLM with LoRA adapters will:
1. Modify only a small fraction of parameters (~1-2%)
2. Show decreasing loss over training steps
3. Change model behavior on the training task domain

METHODOLOGY
-----------
- Use Unsloth's FastLanguageModel with 4-bit quantization
- Apply LoRA adapters to attention and FFN projection layers
- Train on 500 Alpaca examples (general instruction-following)
- Log loss at each step for analysis

WHAT WE'RE LEARNING
-------------------
This experiment establishes baseline understanding of:
- How QLoRA reduces memory requirements (4-bit base + fp16 adapters)
- What "trainable parameters" means in the LoRA context
- How loss curves behave during fine-tuning
- Whether the model changes behavior observably

QUESTIONS TO ANSWER
-------------------
- What does the loss curve look like? Sharp initial drop? Gradual descent?
- How does eval loss compare to train loss? (overfitting signal)
- Can we observe behavior change in model outputs?

RESULTS
-------
[To be filled after running experiment]

Observations:
- TBD

Data:
- Training loss: TBD
- Eval loss: TBD
- Trainable parameters: TBD

LEARNINGS
---------
[To be filled after running experiment]

What did we learn?
- TBD

Were hypotheses confirmed or refuted?
- TBD

What new questions arose?
- TBD
"""

# ============================================
# SETUP: Import order matters for Unsloth
# ============================================
# Unsloth has a known issue where psutil must be imported first
# to avoid a CPU count caching bug. This is a quirk of the library.
import psutil  # Must import before unsloth
import os

# Disable trainer patching to use standard HuggingFace trainer behavior
# This gives us more predictable/documented behavior for learning
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments


def load_model(model_name: str = "unsloth/Qwen2.5-7B-Instruct", max_seq_length: int = 1024):
    """
    Load a 4-bit quantized model.

    LEARNING NOTES:
    - 4-bit quantization reduces memory by ~4x vs fp16
    - Qwen2.5-7B needs ~14GB VRAM in fp16, ~4GB in 4-bit
    - Unsloth's FastLanguageModel handles quantization automatically
    - max_seq_length affects memory linearly (attention is O(n^2) but...)

    Returns:
        model, tokenizer tuple
    """
    print(f"Loading model: {model_name}")
    print(f"  - Using 4-bit quantization (reduces VRAM ~4x)")
    print(f"  - Max sequence length: {max_seq_length}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # 4-bit quantization
        dtype=None,  # Auto-detect best dtype for compute
    )

    return model, tokenizer


def add_lora_adapters(model, r: int = 32, lora_alpha: int = 32, lora_dropout: float = 0.05):
    """
    Add LoRA (Low-Rank Adaptation) adapters to the model.

    LEARNING NOTES:
    - LoRA adds small trainable matrices to frozen base weights
    - For weight W, we compute W + BA where B is (d x r) and A is (r x d)
    - r (rank) controls adapter capacity: higher = more parameters, more expressive
    - lora_alpha scales the adapter output: effective_lr = lr * (alpha / r)
    - Target modules are the projection layers in attention and FFN blocks

    WHY THESE TARGET MODULES?
    - q_proj, k_proj, v_proj, o_proj: Attention projections (where "thinking" happens)
    - gate_proj, up_proj, down_proj: FFN layers (where "processing" happens)
    - These are the core compute pathways in a transformer

    Returns:
        model with LoRA adapters attached
    """
    print(f"\nAdding LoRA adapters:")
    print(f"  - Rank (r): {r} - controls adapter capacity")
    print(f"  - Alpha: {lora_alpha} - scaling factor")
    print(f"  - Dropout: {lora_dropout} - regularization")

    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            # Attention projections - where the model "attends" to context
            "q_proj",  # Query projection
            "k_proj",  # Key projection
            "v_proj",  # Value projection
            "o_proj",  # Output projection
            # FFN projections - where the model "processes" information
            "gate_proj",  # Gating mechanism
            "up_proj",    # Up-projection (expand dimension)
            "down_proj",  # Down-projection (compress dimension)
        ],
        bias="none",  # Don't train bias terms (minimal impact, saves memory)
        use_gradient_checkpointing="unsloth",  # Memory optimization
        random_state=42,  # Reproducibility
    )

    # Report parameter counts - this is key for understanding LoRA's efficiency
    trainable, total = model.get_nb_trainable_parameters()
    percentage = 100 * trainable / total
    print(f"\n  Parameter counts:")
    print(f"    Trainable: {trainable:,}")
    print(f"    Total:     {total:,}")
    print(f"    Ratio:     {percentage:.2f}%")
    print(f"\n  INSIGHT: We're only training {percentage:.2f}% of the model!")

    return model


def prepare_dataset(tokenizer, n_examples: int = 500):
    """
    Load and format training data.

    LEARNING NOTES:
    - Using Alpaca dataset: instruction-following examples
    - Format matters: model expects specific chat template
    - We convert to the model's native chat format for consistency

    Returns:
        Train/test split datasets
    """
    print(f"\nLoading dataset: {n_examples} examples from Alpaca")

    dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{n_examples}]")

    # Apply chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def format_example(example):
        """Convert Alpaca format to chat format."""
        messages = []

        # Combine instruction and input if both present
        if example.get("input", "").strip():
            user_content = f"{example['instruction']}\n\nInput: {example['input']}"
        else:
            user_content = example["instruction"]

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": example["output"]})

        # Apply tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    # Split for validation (90/10)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"  Train examples: {len(dataset['train'])}")
    print(f"  Eval examples:  {len(dataset['test'])}")

    return dataset, tokenizer


def train(model, tokenizer, dataset, output_dir: str = "outputs/basic_qlora"):
    """
    Run training and observe loss dynamics.

    LEARNING NOTES:
    - Learning rate 2e-4 is typical for LoRA (higher than full fine-tuning)
    - Warmup helps prevent early instability
    - Gradient accumulation simulates larger batch sizes
    - bf16 uses bfloat16 for compute (good for modern GPUs)
    - adamw_8bit saves memory on optimizer state

    WHAT TO OBSERVE:
    - Loss at each logging step
    - Train vs eval loss divergence (overfitting indicator)
    - Whether loss plateaus or keeps decreasing
    """
    print("\nConfiguring training:")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        learning_rate=2e-4,  # Typical for LoRA
        warmup_ratio=0.1,  # 10% of steps for warmup
        logging_steps=10,  # Log every 10 steps
        eval_strategy="steps",
        eval_steps=50,  # Evaluate every 50 steps
        save_strategy="no",  # Don't save checkpoints (learning experiment)
        bf16=True,  # bfloat16 for compute
        optim="adamw_8bit",  # 8-bit optimizer to save memory
        seed=42,
        report_to="none",  # No external logging (we'll observe directly)
    )

    print(f"  - Learning rate: {training_args.learning_rate}")
    print(f"  - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - Warmup ratio: {training_args.warmup_ratio}")
    print(f"  - Logging every {training_args.logging_steps} steps")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        dataset_text_field="text",
        max_seq_length=1024,
        packing=True,  # Pack multiple examples per sequence for efficiency
        dataset_num_proc=4,
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING - Observe loss values below")
    print("=" * 60)

    result = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Final training loss: {result.training_loss:.4f}")

    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"  Adapter saved to: {adapter_dir}")

    return trainer, result


def test_model(model, tokenizer, prompts: list[str] = None):
    """
    Test model behavior after training.

    LEARNING NOTES:
    - Switch to inference mode (disables dropout, etc.)
    - Compare outputs to baseline intuition
    - Look for signs of training influence
    """
    if prompts is None:
        prompts = [
            "Explain what machine learning is in simple terms.",
            "Write a haiku about programming.",
            "What are three tips for learning a new language?",
        ]

    print("\n" + "=" * 60)
    print("TESTING MODEL BEHAVIOR")
    print("=" * 60)

    FastLanguageModel.for_inference(model)

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
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
        print("-" * 60)


def run_experiment():
    """
    Main experiment runner.

    LEARNINGS TO CAPTURE:
    - What was the loss curve shape?
    - Did eval loss diverge from train loss?
    - Was behavior change observable?
    """
    print("=" * 60)
    print("EXPERIMENT: Basic QLoRA Fine-Tuning")
    print("=" * 60)

    # Step 1: Load quantized model
    model, tokenizer = load_model()

    # Step 2: Add LoRA adapters
    model = add_lora_adapters(model)

    # Step 3: Prepare data
    dataset, tokenizer = prepare_dataset(tokenizer)

    # Step 4: Train and observe
    trainer, result = train(model, tokenizer, dataset)

    # Step 5: Test behavior
    test_model(model, tokenizer)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print("\nREFLECTION PROMPTS:")
    print("1. What shape was the loss curve? (Sharp drop? Gradual?)")
    print("2. Did eval loss stay close to train loss?")
    print("3. Did model outputs seem different from base model?")
    print("4. What would you change for next experiment?")

    return trainer, result


if __name__ == "__main__":
    run_experiment()
