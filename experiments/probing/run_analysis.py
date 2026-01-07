"""
Representation Analysis Experiments
===================================

Analyze what information is encoded in activations.

HYPOTHESIS
----------
1. Activation norms increase through layers (residual accumulation)
2. FFN contributes more than attention (more parameters)
3. Later layers show more task-specific patterns

EXPERIMENTS
-----------
1. Activation statistics across layers
2. Attention vs FFN contribution analysis
3. Token position analysis
4. Simple linear probe

RESULTS
-------
[To be filled after running]
"""

import psutil
import os
import json
from pathlib import Path
from datetime import datetime

os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"


CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 512,
    "layers_to_analyze": [0, 3, 7, 10, 14, 17, 21, 24, 27],
    "output_dir": "outputs/representation_analysis",
}

TEST_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models can process natural language.",
    "In 1969, humans first landed on the Moon.",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
]


def load_model():
    """Load model for analysis."""
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG["model_name"],
        max_seq_length=CONFIG["max_seq_length"],
        load_in_4bit=True,
        dtype=None,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
    return model, tokenizer


def run_statistics_analysis(model, tokenizer, output_dir: Path):
    """
    Experiment 1: Compute activation statistics across layers.
    """
    from .extract import extract_activations, compute_activation_stats
    import matplotlib.pyplot as plt

    print("\n" + "=" * 50)
    print("Experiment 1: Activation Statistics")
    print("=" * 50)

    all_stats = []

    for text in TEST_TEXTS:
        print(f"  Processing: {text[:40]}...")

        result = extract_activations(
            model, tokenizer, text,
            layers=CONFIG["layers_to_analyze"]
        )

        stats = compute_activation_stats(result)
        all_stats.append(stats)

    # Aggregate stats
    layers = CONFIG["layers_to_analyze"]
    avg_norms = []
    avg_stds = []

    for layer in layers:
        norms = [s["layers"][layer]["post_ffn"]["norm"]
                for s in all_stats if layer in s["layers"]]
        stds = [s["layers"][layer]["post_ffn"]["std"]
               for s in all_stats if layer in s["layers"]]

        avg_norms.append(sum(norms) / len(norms) if norms else 0)
        avg_stds.append(sum(stds) / len(stds) if stds else 0)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(layers, avg_norms, 'b-o', linewidth=2)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Average Activation Norm')
    axes[0].set_title('Activation Norm Across Layers')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layers, avg_stds, 'r-o', linewidth=2)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Average Activation Std')
    axes[1].set_title('Activation Variability Across Layers')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "activation_statistics.png", dpi=150)
    plt.close()

    print(f"  Saved: {output_dir / 'activation_statistics.png'}")

    return {
        "layers": layers,
        "avg_norms": avg_norms,
        "avg_stds": avg_stds,
    }


def run_contribution_analysis(model, tokenizer, output_dir: Path):
    """
    Experiment 2: Analyze attention vs FFN contribution per layer.
    """
    from .extract import extract_activations
    import matplotlib.pyplot as plt
    import torch

    print("\n" + "=" * 50)
    print("Experiment 2: Layer Contribution Analysis")
    print("=" * 50)

    text = TEST_TEXTS[0]  # Use first text

    result = extract_activations(
        model, tokenizer, text,
        layers=CONFIG["layers_to_analyze"],
        positions=["pre_attn", "post_ffn"]
    )

    layers = CONFIG["layers_to_analyze"]
    attn_contribs = []
    ffn_contribs = []

    for layer in layers:
        try:
            diff = result.get_layer_diff(layer)
            attn_norm = diff["attention_contribution"].norm(dim=-1).mean().item()
            ffn_norm = diff["ffn_contribution"].norm(dim=-1).mean().item()
            attn_contribs.append(attn_norm)
            ffn_contribs.append(ffn_norm)
        except KeyError:
            attn_contribs.append(0)
            ffn_contribs.append(0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(layers))
    width = 0.35

    ax.bar([i - width/2 for i in x], attn_contribs, width, label='Attention', color='steelblue')
    ax.bar([i + width/2 for i in x], ffn_contribs, width, label='FFN', color='coral')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Contribution (L2 Norm)')
    ax.set_title('Attention vs FFN Contribution per Layer')
    ax.set_xticks(x)
    ax.set_xticklabels([str(l) for l in layers])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "layer_contributions.png", dpi=150)
    plt.close()

    print(f"  Saved: {output_dir / 'layer_contributions.png'}")

    return {
        "layers": layers,
        "attention_contributions": attn_contribs,
        "ffn_contributions": ffn_contribs,
    }


def run_token_position_analysis(model, tokenizer, output_dir: Path):
    """
    Experiment 3: Analyze how activations differ by token position.
    """
    from .extract import extract_activations
    import matplotlib.pyplot as plt
    import torch

    print("\n" + "=" * 50)
    print("Experiment 3: Token Position Analysis")
    print("=" * 50)

    text = TEST_TEXTS[0]

    result = extract_activations(
        model, tokenizer, text,
        layers=CONFIG["layers_to_analyze"]
    )

    seq_len = result.seq_len
    layers = result.layer_indices

    # Compute cosine similarity between first and last token at each layer
    first_last_sim = []
    for layer in layers:
        act = result.get(layer, "post_ffn")
        first_token = act[0, 0]  # Shape: (hidden_size,)
        last_token = act[0, -1]

        cos_sim = torch.nn.functional.cosine_similarity(
            first_token.unsqueeze(0),
            last_token.unsqueeze(0)
        ).item()
        first_last_sim.append(cos_sim)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(layers, first_last_sim, 'g-o', linewidth=2)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity (First vs Last Token)')
    ax.set_title('Token Position Similarity Across Layers')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / "token_position_similarity.png", dpi=150)
    plt.close()

    print(f"  Saved: {output_dir / 'token_position_similarity.png'}")

    return {
        "layers": layers,
        "first_last_similarity": first_last_sim,
    }


def generate_report(results: dict, output_dir: Path):
    """Generate markdown report."""
    report = f"""# Representation Analysis Results

Generated: {datetime.now().isoformat()}

## Experiment 1: Activation Statistics

![Activation Statistics](activation_statistics.png)

### Observations

| Layer | Avg Norm | Avg Std |
|-------|----------|---------|
"""

    if "statistics" in results:
        stats = results["statistics"]
        for i, layer in enumerate(stats["layers"]):
            report += f"| {layer} | {stats['avg_norms'][i]:.2f} | {stats['avg_stds'][i]:.4f} |\n"

    report += """
**Interpretation**: [Fill in observations about norm growth and variability]

## Experiment 2: Layer Contributions

![Layer Contributions](layer_contributions.png)

### Observations

| Layer | Attention Contrib | FFN Contrib | Ratio (FFN/Attn) |
|-------|------------------|-------------|------------------|
"""

    if "contributions" in results:
        contrib = results["contributions"]
        for i, layer in enumerate(contrib["layers"]):
            attn = contrib["attention_contributions"][i]
            ffn = contrib["ffn_contributions"][i]
            ratio = ffn / attn if attn > 0 else float('inf')
            report += f"| {layer} | {attn:.2f} | {ffn:.2f} | {ratio:.2f} |\n"

    report += """
**Interpretation**: [Fill in observations about attention vs FFN]

## Experiment 3: Token Position Analysis

![Token Position](token_position_similarity.png)

### Observations

[Fill in: How does first/last token similarity change across layers?]

## Key Findings

1. **Activation growth**: [Summary]
2. **Attention vs FFN**: [Summary]
3. **Position encoding**: [Summary]

## Implications

[What do these findings suggest about how the model processes information?]
"""

    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    print(f"\nSaved report: {output_dir / 'report.md'}")


def run_experiment():
    """Run all representation analysis experiments."""
    print("=" * 60)
    print("Representation Analysis Experiments")
    print("=" * 60)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    model, tokenizer = load_model()

    results = {}

    # Run experiments
    results["statistics"] = run_statistics_analysis(model, tokenizer, output_dir)
    results["contributions"] = run_contribution_analysis(model, tokenizer, output_dir)
    results["positions"] = run_token_position_analysis(model, tokenizer, output_dir)

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate report
    generate_report(results, output_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    run_experiment()
