"""
Experiment: LoRA Rank Comparison
================================

HYPOTHESIS
----------
Different LoRA ranks will produce different tradeoffs:

| Rank | Parameters | Expected Behavior |
|------|-----------|-------------------|
| 8    | ~4M       | Minimal adaptation, may underfit complex tasks |
| 16   | ~8M       | Good balance for simple tasks |
| 32   | ~17M      | Standard choice, good for most tasks |
| 64   | ~34M      | Maximum capacity, may not improve much over 32 |

**Why low-rank works (theory):**
The LoRA paper hypothesizes that fine-tuning happens in a low-dimensional subspace.
The model's pre-trained weights already encode most necessary knowledge - we only
need small adjustments to adapt to new tasks. These adjustments can be captured
by low-rank matrices because:
1. Task-specific changes are sparse/structured
2. Most of the model's capacity is already useful
3. Over-parameterization is unnecessary for adaptation

**Key insight:** If rank 8 works nearly as well as rank 64, it suggests the
adaptation truly is low-dimensional. If higher ranks consistently help,
the adaptation may need more capacity.

METHODOLOGY
-----------
- Compare ranks: 8, 16, 32, 64
- Keep learning rate constant at 2e-4
- Train on 500 Alpaca examples
- Measure: parameter count, final loss, training speed

WHAT WE'RE LEARNING
-------------------
- How rank affects model capacity
- Diminishing returns of higher ranks
- Whether our task needs high or low capacity adapters
- Empirical validation of the "low-rank hypothesis"

QUESTIONS TO ANSWER
-------------------
- Is there a "cliff" where higher rank stops helping?
- How does parameter count scale with rank?
- Does training time increase with rank?
- For simple instruction following, what's the minimum viable rank?

RESULTS
-------
[To be filled after running experiment]

| Rank | Parameters | Final Train Loss | Final Eval Loss | Time (s) |
|------|-----------|-----------------|-----------------|----------|
| 8    |           |                 |                 |          |
| 16   |           |                 |                 |          |
| 32   |           |                 |                 |          |
| 64   |           |                 |                 |          |

LEARNINGS
---------
[To be filled after running experiment]

Were hypotheses confirmed?
- TBD
"""

import psutil  # Must import before unsloth
import os
import json
import time
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
    "lora_alpha": 32,  # Keep constant
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "logging_steps": 10,
    "eval_steps": 50,
    "seed": 42,
}

# Ranks to compare
LORA_RANKS = [8, 16, 32, 64]


# ============================================
# TRACKING CALLBACK
# ============================================
class RankExperimentCallback(TrainerCallback):
    """Track losses for rank comparison."""

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
    """Load model (fresh for each rank run)."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_CONFIG["model_name"],
        max_seq_length=BASE_CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer


def add_lora_adapters(model, rank):
    """Add LoRA adapters with specified rank."""
    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        lora_alpha=BASE_CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=BASE_CONFIG["seed"],
    )

    trainable, total = model.get_nb_trainable_parameters()
    return model, trainable, total


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


def run_single_rank(rank, dataset, output_base):
    """Run training with a specific LoRA rank."""
    print(f"\n{'='*60}")
    print(f"Training with LoRA rank = {rank}")
    print(f"{'='*60}")

    # Fresh model for each run
    model, tokenizer = load_model()
    model, trainable_params, total_params = add_lora_adapters(model, rank)

    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")

    callback = RankExperimentCallback()
    output_dir = output_base / f"rank_{rank}"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=BASE_CONFIG["num_epochs"],
        per_device_train_batch_size=BASE_CONFIG["batch_size"],
        gradient_accumulation_steps=BASE_CONFIG["gradient_accumulation"],
        learning_rate=BASE_CONFIG["learning_rate"],
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

    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time

    # Save this run's data
    run_data = {
        "rank": rank,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_ratio": trainable_params / total_params,
        "train_losses": callback.train_losses,
        "eval_losses": callback.eval_losses,
        "steps": callback.steps,
        "eval_steps": callback.eval_steps,
        "final_train_loss": callback.train_losses[-1] if callback.train_losses else None,
        "final_eval_loss": callback.eval_losses[-1] if callback.eval_losses else None,
        "training_time_seconds": training_time,
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
    ranks = list(all_results.keys())

    # Plot 1: All training losses
    ax1 = axes[0, 0]
    for (rank, data), color in zip(all_results.items(), colors):
        if data["train_losses"]:
            ax1.plot(data["steps"], data["train_losses"],
                    color=color, linewidth=1.5, label=f'Rank {rank}')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss vs LoRA Rank')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Parameter count vs final loss
    ax2 = axes[0, 1]
    params = [all_results[r]["trainable_params"] / 1e6 for r in ranks]
    losses = [all_results[r]["final_train_loss"] or 0 for r in ranks]
    ax2.scatter(params, losses, c=colors[:len(ranks)], s=100)
    for i, rank in enumerate(ranks):
        ax2.annotate(f'r={rank}', (params[i], losses[i]),
                    xytext=(5, 5), textcoords='offset points')
    ax2.set_xlabel('Trainable Parameters (M)')
    ax2.set_ylabel('Final Training Loss')
    ax2.set_title('Parameter Count vs Performance')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Final losses bar chart
    ax3 = axes[1, 0]
    train_finals = [all_results[r]["final_train_loss"] or 0 for r in ranks]
    eval_finals = [all_results[r]["final_eval_loss"] or 0 for r in ranks]

    x = range(len(ranks))
    width = 0.35
    ax3.bar([i - width/2 for i in x], train_finals, width, label='Train', color='steelblue')
    ax3.bar([i + width/2 for i in x], eval_finals, width, label='Eval', color='coral')
    ax3.set_xlabel('LoRA Rank')
    ax3.set_ylabel('Final Loss')
    ax3.set_title('Final Loss by Rank')
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(r) for r in ranks])
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Training time
    ax4 = axes[1, 1]
    times = [all_results[r]["training_time_seconds"] for r in ranks]
    ax4.bar(x, times, color=colors[:len(x)])
    ax4.set_xlabel('LoRA Rank')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.set_title('Training Time by Rank')
    ax4.set_xticks(x)
    ax4.set_xticklabels([str(r) for r in ranks])
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "rank_comparison.png", dpi=150)
    plt.close()
    print(f"\nComparison plot saved to {output_dir / 'rank_comparison.png'}")


def generate_summary(all_results, output_dir):
    """Generate markdown summary."""
    summary_path = output_dir / "README.md"

    summary = f"""# LoRA Rank Comparison Results

Generated: {datetime.now().isoformat()}

## Why Low-Rank Works (Theory)

LoRA hypothesizes that fine-tuning happens in a low-dimensional subspace. The key insights:

1. **Pre-trained knowledge is mostly sufficient**: The model already knows language, reasoning, etc.
2. **Adaptation is sparse**: Task-specific changes don't need full model capacity
3. **Low-rank matrices can capture structured changes**: If rank 8 works as well as 64,
   the adaptation truly is low-dimensional

## Results

| Rank | Params (M) | Train Ratio | Final Train | Final Eval | Time (s) |
|------|-----------|-------------|-------------|------------|----------|
"""

    for rank, data in all_results.items():
        params_m = data['trainable_params'] / 1e6
        ratio = data['trainable_ratio'] * 100
        train = f"{data['final_train_loss']:.4f}" if data['final_train_loss'] else "N/A"
        eval_l = f"{data['final_eval_loss']:.4f}" if data['final_eval_loss'] else "N/A"
        time_s = f"{data['training_time_seconds']:.1f}"
        summary += f"| {rank} | {params_m:.1f} | {ratio:.2f}% | {train} | {eval_l} | {time_s} |\n"

    summary += """
## Observations

[Fill in after reviewing results]

1. **Did higher ranks significantly improve loss?**
   - TBD

2. **Is there a point of diminishing returns?**
   - TBD

3. **Parameter efficiency (loss per million params)?**
   - TBD

4. **Training time tradeoff?**
   - TBD

## Implications for Low-Rank Hypothesis

[Fill in after analysis]

If rank 8 ≈ rank 64:
- Supports the low-rank adaptation hypothesis
- Can use small ranks for efficiency

If rank 64 >> rank 8:
- Task may need more adaptation capacity
- Consider task complexity

## Visualizations

See `rank_comparison.png` for:
- Training loss curves by rank
- Parameter count vs performance
- Final loss comparison
- Training time comparison

## Recommended Rank

Based on these results: TBD

Consider:
- If all ranks similar: use lower rank (faster, less memory)
- If higher ranks clearly better: worth the cost for this task
"""

    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Summary saved to {summary_path}")


def run_experiment():
    """Run the full LoRA rank comparison experiment."""
    print("=" * 60)
    print("EXPERIMENT: LoRA Rank Comparison")
    print("=" * 60)

    output_dir = Path("outputs/lora_rank_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump({
            "base_config": BASE_CONFIG,
            "lora_ranks": LORA_RANKS,
        }, f, indent=2)

    # Load dataset once
    print("\nLoading dataset...")
    model, tokenizer = load_model()
    dataset, tokenizer = prepare_dataset(tokenizer)
    del model
    import torch
    torch.cuda.empty_cache()

    # Run each rank
    all_results = {}
    for rank in LORA_RANKS:
        try:
            data = run_single_rank(rank, dataset, output_dir)
            all_results[rank] = data
        except Exception as e:
            print(f"Error with rank {rank}: {e}")
            all_results[rank] = {
                "rank": rank,
                "trainable_params": 0,
                "total_params": 0,
                "trainable_ratio": 0,
                "train_losses": [],
                "eval_losses": [],
                "steps": [],
                "eval_steps": [],
                "final_train_loss": None,
                "final_eval_loss": None,
                "training_time_seconds": 0,
                "error": str(e),
            }

    # Generate outputs
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
    print("1. Did higher ranks give better loss?")
    print("2. Was there a point of diminishing returns?")
    print("3. Does this support the low-rank hypothesis?")
    print("4. What rank would you recommend for this task?")


if __name__ == "__main__":
    run_experiment()
