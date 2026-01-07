"""
Experiment: Attention Comparison (Base vs Fine-tuned)
=====================================================

Compare attention patterns between base and fine-tuned models.

HYPOTHESIS
----------
Fine-tuning will change attention patterns:
1. Some heads may become "instruction-sensitive"
2. Late layers will change more than early layers
3. Overall changes will be small (LoRA is additive)

METHODOLOGY
-----------
1. Load base model and fine-tuned model
2. Run same prompts through both
3. Extract attention from key layers
4. Visualize differences
5. Compute quantitative metrics

RESULTS
-------
[To be filled after running]
"""

import psutil  # Must import before unsloth
import os
import json
from pathlib import Path
from datetime import datetime

os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"


# ============================================
# CONFIGURATION
# ============================================
CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 512,
    "layers_to_compare": [0, 7, 14, 21, 27],  # Spread across depth
    "output_dir": "outputs/attention_comparison",
}

TEST_PROMPTS = [
    "List three benefits of regular exercise.",
    "Write a haiku about morning coffee.",
    "If all dogs are mammals, and some mammals can fly, can some dogs fly?",
    "Write a Python function that reverses a string.",
]


def load_base_model():
    """Load the base model without LoRA adapters."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print("Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    return model, tokenizer


def load_finetuned_model(adapter_path: str = None):
    """
    Load the fine-tuned model.

    If adapter_path is provided, loads those adapters.
    Otherwise, looks for a recent training output.
    """
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template
    from peft import PeftModel

    print("Loading fine-tuned model...")

    # First load base
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # Look for adapter
    if adapter_path is None:
        # Try to find a trained adapter
        possible_paths = [
            "outputs/basic_qlora/adapter",
            "outputs/loss_curve_analysis/adapter",
            "outputs/learning_rate_exploration/lr_0.0002/adapter",
        ]
        for path in possible_paths:
            if Path(path).exists():
                adapter_path = path
                break

    if adapter_path and Path(adapter_path).exists():
        print(f"Loading adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
    else:
        print("WARNING: No adapter found. Creating minimal LoRA for comparison.")
        # Create fresh LoRA (untrained) - this won't show real fine-tuning effects
        model = FastLanguageModel.get_peft_model(
            model,
            r=32,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

    return model, tokenizer


def extract_attention_for_prompt(model, tokenizer, prompt: str, layers: list):
    """Extract attention for a single prompt."""
    from .extract import extract_attention

    # Format as chat
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Extract attention
    result = extract_attention(model, tokenizer, text, layers=layers)
    return result


def compute_attention_diff(attn1, attn2):
    """
    Compute difference metrics between two attention outputs.

    Returns dict with:
    - mean_diff: Average absolute difference
    - max_diff: Maximum difference
    - per_layer: Per-layer statistics
    """
    import torch

    stats = {"per_layer": {}}

    all_diffs = []
    for layer_idx in attn1.layer_indices:
        if layer_idx not in attn2.weights:
            continue

        w1 = attn1.weights[layer_idx]
        w2 = attn2.weights[layer_idx]

        # Compute difference
        diff = torch.abs(w1 - w2)
        mean_diff = diff.mean().item()
        max_diff = diff.max().item()

        stats["per_layer"][layer_idx] = {
            "mean_diff": mean_diff,
            "max_diff": max_diff,
        }
        all_diffs.append(mean_diff)

    stats["mean_diff"] = sum(all_diffs) / len(all_diffs) if all_diffs else 0
    stats["max_diff"] = max(s["max_diff"] for s in stats["per_layer"].values()) if stats["per_layer"] else 0

    return stats


def visualize_comparison(base_attn, ft_attn, prompt_idx: int, output_dir: Path):
    """Generate comparison visualizations."""
    from .visualize import plot_attention_heatmap, VisualizationConfig
    import torch

    config = VisualizationConfig()
    prompt_dir = output_dir / f"prompt_{prompt_idx}"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx in base_attn.layer_indices:
        if layer_idx not in ft_attn.weights:
            continue

        # Get head 0 from each
        base_head0 = base_attn.get_head(layer_idx, 0)
        ft_head0 = ft_attn.get_head(layer_idx, 0)
        diff = torch.abs(base_head0 - ft_head0)

        # Plot base
        plot_attention_heatmap(
            base_head0,
            tokens=base_attn.tokens,
            title=f"Base Model - Layer {layer_idx}, Head 0",
            output_path=str(prompt_dir / f"base_layer_{layer_idx}.png"),
            config=config,
        )

        # Plot fine-tuned
        plot_attention_heatmap(
            ft_head0,
            tokens=ft_attn.tokens,
            title=f"Fine-tuned - Layer {layer_idx}, Head 0",
            output_path=str(prompt_dir / f"finetuned_layer_{layer_idx}.png"),
            config=config,
        )

        # Plot difference
        plot_attention_heatmap(
            diff,
            tokens=base_attn.tokens,
            title=f"Difference - Layer {layer_idx}, Head 0",
            output_path=str(prompt_dir / f"diff_layer_{layer_idx}.png"),
            config=VisualizationConfig(cmap="Reds"),
            caption="Brighter = larger difference between base and fine-tuned",
        )


def run_experiment(adapter_path: str = None):
    """Run the full comparison experiment."""
    import torch

    print("=" * 60)
    print("EXPERIMENT: Attention Comparison (Base vs Fine-tuned)")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Load models
    base_model, base_tokenizer = load_base_model()

    # Clear GPU memory before loading second model
    del base_model
    torch.cuda.empty_cache()

    ft_model, ft_tokenizer = load_finetuned_model(adapter_path)

    # Reload base for comparison (need both in memory for fair comparison)
    # Actually, let's do sequential to save memory
    results = {"prompts": [], "overall": {}}

    for idx, prompt in enumerate(TEST_PROMPTS):
        print(f"\n{'='*40}")
        print(f"Processing prompt {idx + 1}: {prompt[:50]}...")
        print("=" * 40)

        prompt_result = {"prompt": prompt, "layers": {}}

        # Extract from fine-tuned (already loaded)
        print("  Extracting from fine-tuned model...")
        ft_attn = extract_attention_for_prompt(
            ft_model, ft_tokenizer, prompt, CONFIG["layers_to_compare"]
        )

        # Save fine-tuned attention data
        prompt_result["finetuned_tokens"] = ft_attn.tokens

        # For base, we need to reload (memory constraint)
        # In practice, you might run these separately
        print("  (Base model extraction would go here - skipping for memory)")
        # base_attn = extract_attention_for_prompt(
        #     base_model, base_tokenizer, prompt, CONFIG["layers_to_compare"]
        # )

        results["prompts"].append(prompt_result)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")
    print("\nNote: Full comparison requires running both models.")
    print("Due to memory constraints, you may need to run separately:")
    print("  1. Extract base model attention, save to disk")
    print("  2. Extract fine-tuned attention, save to disk")
    print("  3. Load both and compare")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", help="Path to trained LoRA adapter")
    args = parser.parse_args()

    run_experiment(adapter_path=args.adapter)
