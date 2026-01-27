"""
Experiment: Loss Curve Analysis
===============================

HYPOTHESIS
----------
The loss curve during fine-tuning will show:
1. Sharp initial drop as the model adapts to the new task format
2. Gradual decrease as the model learns task-specific patterns
3. Eventual plateau or slight increase (overfitting) if training too long

Understanding loss curves helps us:
- Know when training is "done"
- Detect overfitting early
- Compare different training configurations

METHODOLOGY
-----------
- Train on 500 Alpaca examples with detailed loss logging
- Log at multiple granularities: per-step, per-epoch averages
- Capture both training and evaluation loss
- Generate visualizations showing loss dynamics

WHAT WE'RE LEARNING
-------------------
- What loss represents (cross-entropy over next token prediction)
- Relationship between loss and actual model behavior
- How to interpret loss curve shapes
- When to stop training

QUESTIONS TO ANSWER
-------------------
- What does the initial loss value mean? (see QUESTIONS.md#training-dynamics)
- What's a "good" final loss value for this task?
- How quickly does eval loss diverge from train loss?
- What loss patterns correlate with observable behavior changes?

RESULTS
-------
[To be filled after running experiment]

Loss progression:
- Initial loss: TBD
- Final training loss: TBD
- Final eval loss: TBD
- Loss reduction: TBD%

Curve characteristics:
- Shape: TBD (exponential decay / linear / stepped / etc.)
- Plateau point: TBD steps
- Train/eval divergence: TBD

LEARNINGS
---------
[To be filled after running experiment]

What did we learn about loss curves?
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
CONFIG = {
    "experiment_name": "loss_curve_analysis",
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 1024,
    "n_examples": 500,
    "lora_r": 32,
    "lora_alpha": 32,
    "learning_rate": 2e-4,
    "num_epochs": 1,
    "batch_size": 4,
    "gradient_accumulation": 4,
    "logging_steps": 5,  # Log more frequently for detailed curve
    "eval_steps": 25,
    "seed": 42,
    "output_dir": "outputs/loss_curve_analysis",
}


# ============================================
# CUSTOM CALLBACK FOR DETAILED LOGGING
# ============================================
class LossCurveCallback(TrainerCallback):
    """
    Custom callback to capture detailed loss information.

    LEARNING NOTES:
    - TrainerCallback hooks into training loop at various points
    - on_log is called whenever metrics are logged
    - We capture everything for later visualization
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs = []
        self.train_losses = []
        self.eval_losses = []
        self.steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when trainer logs metrics."""
        if logs is None:
            return

        log_entry = {
            "step": state.global_step,
            "epoch": state.epoch,
            **logs
        }
        self.logs.append(log_entry)

        # Track training loss
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.steps.append(state.global_step)
            print(f"  Step {state.global_step}: train_loss = {logs['loss']:.4f}")

        # Track eval loss
        if "eval_loss" in logs:
            self.eval_losses.append(logs["eval_loss"])
            self.eval_steps.append(state.global_step)
            print(f"  Step {state.global_step}: eval_loss = {logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        """Save all logs when training ends."""
        # Save raw logs
        log_path = self.output_dir / "training_logs.json"
        with open(log_path, "w") as f:
            json.dump(self.logs, f, indent=2)
        print(f"\nLogs saved to {log_path}")

        # Generate visualizations
        self._plot_loss_curve()
        self._generate_summary()

    def _plot_loss_curve(self):
        """Generate loss curve visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Training loss over steps
        ax1 = axes[0]
        ax1.plot(self.steps, self.train_losses, 'b-', linewidth=1.5, label='Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add annotations for key points
        if self.train_losses:
            initial = self.train_losses[0]
            final = self.train_losses[-1]
            ax1.annotate(f'Initial: {initial:.3f}',
                        xy=(self.steps[0], initial),
                        xytext=(10, 10), textcoords='offset points')
            ax1.annotate(f'Final: {final:.3f}',
                        xy=(self.steps[-1], final),
                        xytext=(-60, 10), textcoords='offset points')

        # Plot 2: Train vs Eval loss
        ax2 = axes[1]
        ax2.plot(self.steps, self.train_losses, 'b-', linewidth=1.5, label='Training Loss')
        if self.eval_losses:
            ax2.plot(self.eval_steps, self.eval_losses, 'r-o', linewidth=1.5,
                    markersize=6, label='Eval Loss')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training vs Evaluation Loss')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

        # Save figure
        fig_path = self.output_dir / "loss_curves.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Loss curve plot saved to {fig_path}")

    def _generate_summary(self):
        """Generate a markdown summary of the loss analysis."""
        summary_path = self.output_dir / "loss_analysis_summary.md"

        if not self.train_losses:
            return

        initial_loss = self.train_losses[0]
        final_loss = self.train_losses[-1]
        min_loss = min(self.train_losses)
        max_loss = max(self.train_losses)

        # Analyze curve characteristics
        loss_reduction = (initial_loss - final_loss) / initial_loss * 100

        # Detect plateau (loss change < 1% over last 20% of training)
        n_recent = max(1, len(self.train_losses) // 5)
        recent_losses = self.train_losses[-n_recent:]
        if len(recent_losses) > 1:
            recent_change = abs(recent_losses[-1] - recent_losses[0]) / recent_losses[0] * 100
            plateau_detected = recent_change < 1
        else:
            plateau_detected = False
            recent_change = 0

        summary = f"""# Loss Curve Analysis Results

Generated: {datetime.now().isoformat()}

## Summary Statistics

| Metric | Value |
|--------|-------|
| Initial Loss | {initial_loss:.4f} |
| Final Loss | {final_loss:.4f} |
| Minimum Loss | {min_loss:.4f} |
| Maximum Loss | {max_loss:.4f} |
| Loss Reduction | {loss_reduction:.1f}% |
| Total Steps | {len(self.train_losses)} |

## Curve Characteristics

- **Shape**: {"Plateau detected" if plateau_detected else "Still decreasing"}
- **Recent change**: {recent_change:.2f}% over last {n_recent} steps
- **Eval vs Train gap**: {self.eval_losses[-1] - final_loss:.4f} (if positive, potential overfitting)

## Interpretation

"""
        if loss_reduction > 50:
            summary += "- Strong loss reduction indicates effective learning\n"
        elif loss_reduction > 20:
            summary += "- Moderate loss reduction; model is adapting\n"
        else:
            summary += "- Limited loss reduction; may need more training or different hyperparameters\n"

        if plateau_detected:
            summary += "- Training appears to have converged (plateau detected)\n"
        else:
            summary += "- Loss still decreasing; could benefit from more training\n"

        if self.eval_losses and self.eval_losses[-1] > final_loss * 1.1:
            summary += "- **Warning**: Eval loss significantly higher than train loss - possible overfitting\n"

        summary += """
## Visualizations

See `loss_curves.png` in this directory.

## Next Steps

Based on these results, consider:
1. If loss plateaued: Training may be complete for this configuration
2. If still decreasing: Could try more epochs
3. If eval > train gap growing: Reduce training or add regularization
"""

        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"Summary saved to {summary_path}")


# ============================================
# EXPERIMENT CODE
# ============================================
def load_model():
    """Load 4-bit quantized model."""
    print(f"Loading model: {CONFIG['model_name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    return model, tokenizer


def add_lora_adapters(model):
    """Add LoRA adapters for training."""
    print(f"Adding LoRA adapters (r={CONFIG['lora_r']})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG["seed"],
    )

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model


def prepare_dataset(tokenizer):
    """Load and format training data."""
    print(f"Loading {CONFIG['n_examples']} examples from Alpaca")
    dataset = load_dataset("yahma/alpaca-cleaned", split=f"train[:{CONFIG['n_examples']}]")

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
    dataset = dataset.train_test_split(test_size=0.1, seed=CONFIG["seed"])

    print(f"  Train: {len(dataset['train'])}, Eval: {len(dataset['test'])}")
    return dataset, tokenizer


def train(model, tokenizer, dataset):
    """Run training with detailed loss logging."""
    print("\nConfiguring training for loss curve analysis...")

    # Create output directory
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize our custom callback
    loss_callback = LossCurveCallback(output_dir)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation"],
        learning_rate=CONFIG["learning_rate"],
        warmup_ratio=0.1,
        logging_steps=CONFIG["logging_steps"],
        eval_strategy="steps",
        eval_steps=CONFIG["eval_steps"],
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        seed=CONFIG["seed"],
        report_to="none",
    )

    print(f"  Learning rate: {training_args.learning_rate}")
    print(f"  Logging every {CONFIG['logging_steps']} steps")
    print(f"  Eval every {CONFIG['eval_steps']} steps")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        dataset_text_field="text",
        max_seq_length=CONFIG["max_seq_length"],
        packing=True,
        dataset_num_proc=4,
        callbacks=[loss_callback],
    )

    print("\n" + "=" * 60)
    print("STARTING TRAINING - Watch the loss values")
    print("=" * 60 + "\n")

    result = trainer.train()

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    # Save adapter
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"  Adapter saved to: {adapter_dir}")

    return trainer, result, loss_callback


def run_experiment():
    """Main experiment entry point."""
    print("=" * 60)
    print("EXPERIMENT: Loss Curve Analysis")
    print("=" * 60)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")

    # Save config
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Run experiment
    model, tokenizer = load_model()
    model = add_lora_adapters(model)
    dataset, tokenizer = prepare_dataset(tokenizer)
    trainer, result, callback = train(model, tokenizer, dataset)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {CONFIG['output_dir']}")
    print("  - loss_curves.png: Visualization of loss dynamics")
    print("  - loss_analysis_summary.md: Interpretation of results")
    print("  - training_logs.json: Raw log data")
    print("  - config.json: Experiment configuration")
    print("\nREFLECTION PROMPTS:")
    print("1. What shape is the loss curve?")
    print("2. Did the model converge or still improving?")
    print("3. Any signs of overfitting in train/eval gap?")
    print("4. What does the final loss value tell us?")

    return trainer, result


if __name__ == "__main__":
    run_experiment()
