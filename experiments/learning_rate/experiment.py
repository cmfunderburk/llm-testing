"""
Experiment: Learning Rate Exploration
=====================================

HYPOTHESIS
----------
Different learning rates will produce qualitatively different training dynamics:

1. **Too low (1e-5)**: Slow convergence, may not reach good loss in limited steps
2. **Optimal (2e-4)**: Smooth decrease, good final loss, stable training
3. **Too high (1e-3)**: Unstable training, possibly divergent, or quick initial drop then plateau
4. **Very high (5e-3)**: Likely to diverge (loss increases)

For LoRA fine-tuning, higher learning rates are typically needed than full fine-tuning
because we're only updating a small fraction of parameters.

METHODOLOGY
-----------
- Run 4 training configurations with different learning rates: 1e-5, 2e-4, 1e-3, 5e-3
- Keep all other hyperparameters constant
- Train on same 500 Alpaca examples
- Compare loss curves, final losses, and training stability
- Generate side-by-side visualizations

WHAT WE'RE LEARNING
-------------------
- How learning rate affects convergence speed
- What "too high" and "too low" look like in practice
- Why LoRA uses different LR ranges than full fine-tuning
- How to choose learning rate for new tasks

QUESTIONS TO ANSWER
-------------------
- What's the optimal learning rate range for QLoRA on 7B models?
- Is there a "cliff" where LR becomes too high?
- How does warmup interact with initial LR?
- Does optimal LR depend on LoRA rank?

RESULTS
-------
[To be filled after running experiment]

| Learning Rate | Final Train Loss | Final Eval Loss | Converged? | Notes |
|---------------|-----------------|-----------------|------------|-------|
| 1e-5          | TBD             | TBD             | TBD        | TBD   |
| 2e-4          | TBD             | TBD             | TBD        | TBD   |
| 1e-3          | TBD             | TBD             | TBD        | TBD   |
| 5e-3          | TBD             | TBD             | TBD        | TBD   |

LEARNINGS
---------
[To be filled after running experiment]

What did we learn about learning rates?
- TBD

Were hypotheses confirmed?
- TBD
"""

import psutil  # Must import before unsloth
import os
import json
from datetime import datetime
from pathlib import Path

os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
import matplotlib.pyplot as plt


# ============================================
# CONFIGURATION
# ============================================
BASE_CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 1024,
    "n_examples": 500,
    "lora_r": 32,
    "lora_alpha": 32,
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "logging_steps": 5,
    "eval_steps": 25,
    "seed": 42,
}

# Learning rates to compare
LEARNING_RATES = [1e-5, 2e-4, 1e-3, 5e-3]

# Human-readable names for plots
LR_NAMES = {
    1e-5: "Very Low (1e-5)",
    2e-4: "Standard (2e-4)",
    1e-3: "High (1e-3)",
    5e-3: "Very High (5e-3)",
}


# ============================================
# TRACKING CALLBACK
# ============================================
class LRExperimentCallback(TrainerCallback):
    """Track losses for learning rate comparison."""

    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)

        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)


# ============================================
# EXPERIMENT CODE
# ============================================
def load_model():
    """Load model (fresh for each LR run)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_CONFIG["model_name"],
        max_seq_length=BASE_CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer


def add_lora_adapters(model):
    """Add LoRA adapters."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=BASE_CONFIG["lora_r"],
        lora_alpha=BASE_CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=BASE_CONFIG["seed"],
    )
    return model


def prepare_dataset(tokenizer):
    """Load and format training data."""
    dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{BASE_CONFIG['n_examples']}]")
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    def format_example(example):
        messages = []
        if example.get("input", "").strip():
            user_content = f"{example['instruction']}\n\nInput: {example['input']}"
        else:
            user_content = example["instruction"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": example["output"]})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.1, seed=BASE_CONFIG["seed"])
    return dataset, tokenizer


def run_single_lr(learning_rate, dataset, output_base):
    """Run training with a specific learning rate."""
    print(f"\n{'='*60}")
    print(f"Training with LR = {learning_rate}")
    print(f"{'='*60}")

    # Fresh model for each run
    model, tokenizer = load_model()
    model = add_lora_adapters(model)

    callback = LRExperimentCallback()
    output_dir = output_base / f"lr_{learning_rate}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=BASE_CONFIG["num_epochs"],
        per_device_train_batch_size=BASE_CONFIG["batch_size"],
        gradient_accumulation_steps=BASE_CONFIG["gradient_accumulation"],
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=BASE_CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=BASE_CONFIG["eval_steps"],
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        seed=BASE_CONFIG["seed"],
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        dataset_text_field="text",
        max_seq_length=BASE_CONFIG["max_seq_length"],
        packing=True,
        dataset_num_proc=4,
        callbacks=[callback],
    )

    result = trainer.train()

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  Adapter saved to: {adapter_dir}")

    # Save this run's data
    run_data = {
        "learning_rate": learning_rate,
        "train_losses": callback.train_losses,
        "eval_losses": callback.eval_losses,
        "steps": callback.steps,
        "eval_steps": callback.eval_steps,
        "final_train_loss": callback.train_losses[-1] if callback.train_losses else None,
        "final_eval_loss": callback.eval_losses[-1] if callback.eval_losses else None,
    }

    with open(output_dir / "run_data.json", "w") as f:
        json.dump(run_data, f, indent=2)

    # Clean up to free VRAM
    del model, trainer
    import torch
    torch.cuda.empty_cache()

    return run_data


def generate_comparison_plots(all_results, output_dir):
    """Generate comparison visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = ['blue', 'green', 'orange', 'red']

    # Plot 1: All training losses
    ax1 = axes[0, 0]
    for (lr, data), color in zip(all_results.items(), colors):
        if data["train_losses"]:
            ax1.plot(data["steps"], data["train_losses"],
                    color=color, linewidth=1.5, label=LR_NAMES.get(lr, str(lr)))
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs Learning Rate')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: All eval losses
    ax2 = axes[0, 1]
    for (lr, data), color in zip(all_results.items(), colors):
        if data["eval_losses"]:
            ax2.plot(data["eval_steps"], data["eval_losses"],
                    color=color, linewidth=1.5, marker='o', markersize=4,
                    label=LR_NAMES.get(lr, str(lr)))
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Evaluation Loss')
    ax2.set_title('Evaluation Loss vs Learning Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final losses bar chart
    ax3 = axes[1, 0]
    lrs = list(all_results.keys())
    train_finals = [all_results[lr]["final_train_loss"] or 0 for lr in lrs]
    eval_finals = [all_results[lr]["final_eval_loss"] or 0 for lr in lrs]

    x = range(len(lrs))
    width = 0.35
    ax3.bar([i - width/2 for i in x], train_finals, width, label='Train', color='steelblue')
    ax3.bar([i + width/2 for i in x], eval_finals, width, label='Eval', color='coral')
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Convergence speed (steps to reach 80% of loss reduction)
    ax4 = axes[1, 1]
    convergence_steps = []
    for lr, data in all_results.items():
        if data["train_losses"] and len(data["train_losses"]) > 1:
            initial = data["train_losses"][0]
            final = data["train_losses"][-1]
            target = initial - 0.8 * (initial - final)

            # Find step where we first reach 80% of reduction
            step = None
            for i, loss in enumerate(data["train_losses"]):
                if loss <= target:
                    step = data["steps"][i]
                    break
            convergence_steps.append(step if step else data["steps"][-1])
        else:
            convergence_steps.append(0)

    ax4.bar(x, convergence_steps, color=colors[:len(x)])
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Steps to 80% Reduction')
    ax4.set_title('Convergence Speed')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f"{lr:.0e}" for lr in lrs])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "lr_comparison.png", dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {output_dir / 'lr_comparison.png'}")


def generate_summary(all_results, output_dir):
    """Generate markdown summary."""
    summary_path = output_dir / "README.md"

    summary = f"""# Learning Rate Exploration Results

Generated: {datetime.now().isoformat()}

## Hypothesis

Different learning rates will produce qualitatively different training dynamics:
- Too low (1e-5): Slow convergence
- Optimal (2e-4): Smooth decrease, good final loss
- Too high (1e-3): Possibly unstable
- Very high (5e-3): Likely to diverge

## Results

| Learning Rate | Final Train Loss | Final Eval Loss | Notes |
|---------------|-----------------|-----------------|-------|
"""

    for lr, data in all_results.items():
        train = f"{data['final_train_loss']:.4f}" if data['final_train_loss'] else "N/A"
        eval_l = f"{data['final_eval_loss']:.4f}" if data['final_eval_loss'] else "N/A"

        # Analyze behavior
        if data['train_losses']:
            initial = data['train_losses'][0]
            final = data['train_losses'][-1]
            if final > initial:
                notes = "DIVERGED"
            elif (initial - final) / initial < 0.1:
                notes = "Minimal learning"
            else:
                notes = "Converged"
        else:
            notes = "No data"

        summary += f"| {lr:.0e} | {train} | {eval_l} | {notes} |\n"

    summary += """
## Observations

[Fill in after reviewing results]

1. Which learning rate produced the lowest final loss?
2. Did any learning rates diverge?
3. What was the convergence speed difference?
4. Did hypothesis match observations?

## Visualizations

See `lr_comparison.png` for:
- Training loss curves for all LRs
- Evaluation loss curves
- Final loss comparison
- Convergence speed comparison

## Conclusions

[Fill in after analysis]

Recommended learning rate for this setup: TBD

## Next Steps

Based on these results:
1. TBD
2. TBD
"""

    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


def run_experiment():
    """Run the full learning rate comparison experiment."""
    print("=" * 60)
    print("EXPERIMENT: Learning Rate Exploration")
    print("=" * 60)

    output_dir = Path("outputs/learning_rate_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save base config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "base_config": BASE_CONFIG,
            "learning_rates": LEARNING_RATES,
        }, f, indent=2)

    # Load dataset once (shared across runs)
    print("\nLoading dataset...")
    model, tokenizer = load_model()
    dataset, tokenizer = prepare_dataset(tokenizer)
    del model  # Free memory
    import torch
    torch.cuda.empty_cache()

    # Run each learning rate
    all_results = {}
    for lr in LEARNING_RATES:
        try:
            data = run_single_lr(lr, dataset, output_dir)
            all_results[lr] = data
        except Exception as e:
            print(f"Error with LR {lr}: {e}")
            all_results[lr] = {
                "learning_rate": lr,
                "train_losses": [],
                "eval_losses": [],
                "steps": [],
                "eval_steps": [],
                "final_train_loss": None,
                "final_eval_loss": None,
                "error": str(e),
            }

    # Generate comparison outputs
    print("\nGenerating comparison visualizations...")
    generate_comparison_plots(all_results, output_dir)
    generate_summary(all_results, output_dir)

    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump({str(k): v for k, v in all_results.items()}, f, indent=2)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nREFLECTION PROMPTS:")
    print("1. Which LR gave the best final loss?")
    print("2. Were any LRs clearly too high or too low?")
    print("3. Does the 'standard' 2e-4 seem optimal?")
    print("4. What would you try next?")


if __name__ == "__main__":
    run_experiment()
