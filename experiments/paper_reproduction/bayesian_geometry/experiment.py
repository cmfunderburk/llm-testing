"""
Paper Claim Reproduction: FFN vs Attention Contribution
========================================================

PAPER CLAIM
-----------
From "The Bayesian Geometry of Transformer Attention" (arXiv:2512.22471):

"residual streams serve as the belief substrate, feed-forward networks
perform the posterior update, and attention provides content-addressable routing"

This implies:
1. FFN should contribute MORE to representation changes (it does the "update")
2. Attention should contribute to routing/selection (more structural role)
3. Information should accumulate in residual stream (monotonic growth)

SELECTED CLAIM FOR REPRODUCTION
-------------------------------
We test: "FFN performs the posterior update"

Operationalized as: FFN contribution norm > Attention contribution norm at most layers

RATIONALE FOR SELECTION
-----------------------
1. Directly testable with our existing tooling (experiments/probing/extract.py)
2. Connects to Track C work on layer contributions
3. Doesn't require implementing the paper's synthetic tasks
4. Provides insight into our actual model's behavior

METHODOLOGY
-----------
1. Extract activations at pre_attn, post_attn, post_ffn positions
2. Compute attention contribution = post_attn - pre_attn
3. Compute FFN contribution = post_ffn - post_attn
4. Compare norms across layers
5. Test on multiple inputs to ensure consistency

HYPOTHESIS
----------
Based on the paper:
- FFN contribution norm > Attention contribution norm (at most layers)
- The ratio should be consistent across different inputs
- Later layers may show more "computation" (larger contributions)

EXPECTED RESULTS
----------------
If the paper's claim generalizes to language models:
- FFN/Attention ratio > 1 for most layers
- Ratio may increase in later layers (more belief updating needed)

If the claim is specific to their synthetic tasks:
- Pattern may be different or inconsistent
"""

import os
os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

from pathlib import Path
import json

CONFIG = {
    "model_name": "unsloth/Qwen2.5-7B-Instruct",
    "max_seq_length": 512,
    "layers_to_analyze": list(range(0, 28, 3)),  # Every 3rd layer
    "output_dir": "outputs/paper_reproduction/bayesian_geometry",
}

# Test inputs spanning different types
TEST_INPUTS = [
    # Factual reasoning (should involve belief updating)
    "The capital of France is Paris. The capital of Germany is Berlin. What is the capital of France?",

    # Sequential reasoning
    "If A is true and B is false, and C requires A and B, then C is",

    # Simple completion
    "The quick brown fox jumps over the",

    # Mathematical
    "2 + 2 = 4. 3 + 3 = 6. 4 + 4 =",
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


def run_contribution_analysis(model, tokenizer):
    """
    Test the paper's claim about FFN vs Attention contributions.
    """
    from experiments.probing.extract import extract_activations
    import torch

    print("\n" + "=" * 60)
    print("PAPER CLAIM REPRODUCTION")
    print("Claim: FFN performs posterior update (FFN contrib > Attention)")
    print("=" * 60)

    all_results = []

    for text in TEST_INPUTS:
        print(f"\nProcessing: {text[:50]}...")

        result = extract_activations(
            model, tokenizer, text,
            layers=CONFIG["layers_to_analyze"],
            positions=["pre_attn", "post_attn", "post_ffn"]
        )

        layer_contributions = {}

        for layer in CONFIG["layers_to_analyze"]:
            try:
                # Get activations at different positions
                pre_attn = result.get(layer, "pre_attn")
                post_attn = result.get(layer, "post_attn")
                post_ffn = result.get(layer, "post_ffn")

                # Compute contributions
                attn_contrib = post_attn - pre_attn
                ffn_contrib = post_ffn - post_attn

                # Compute norms (average over sequence)
                attn_norm = attn_contrib.norm(dim=-1).mean().item()
                ffn_norm = ffn_contrib.norm(dim=-1).mean().item()

                # Compute ratio
                ratio = ffn_norm / attn_norm if attn_norm > 0 else float('inf')

                layer_contributions[layer] = {
                    "attention_norm": attn_norm,
                    "ffn_norm": ffn_norm,
                    "ratio_ffn_attn": ratio,
                    "ffn_larger": ffn_norm > attn_norm,
                }

            except KeyError as e:
                print(f"  Warning: Missing data for layer {layer}: {e}")
                continue

        all_results.append({
            "text": text[:50],
            "layers": layer_contributions,
        })

    return all_results


def analyze_results(results):
    """Analyze if results support the paper's claim."""
    print("\n" + "=" * 60)
    print("RESULTS ANALYSIS")
    print("=" * 60)

    # Aggregate across all inputs
    layer_stats = {}

    for result in results:
        for layer, data in result["layers"].items():
            if layer not in layer_stats:
                layer_stats[layer] = {
                    "ratios": [],
                    "ffn_larger_count": 0,
                    "total": 0,
                }
            layer_stats[layer]["ratios"].append(data["ratio_ffn_attn"])
            layer_stats[layer]["ffn_larger_count"] += int(data["ffn_larger"])
            layer_stats[layer]["total"] += 1

    # Print summary table
    print("\n| Layer | Avg FFN/Attn Ratio | FFN > Attn | Support Claim? |")
    print("|-------|-------------------|------------|----------------|")

    claim_supported = 0
    total_layers = 0

    for layer in sorted(layer_stats.keys()):
        stats = layer_stats[layer]
        avg_ratio = sum(stats["ratios"]) / len(stats["ratios"])
        ffn_larger_pct = stats["ffn_larger_count"] / stats["total"] * 100
        supports = "YES" if ffn_larger_pct > 50 else "NO"

        if ffn_larger_pct > 50:
            claim_supported += 1
        total_layers += 1

        print(f"| {layer:5} | {avg_ratio:17.2f} | {ffn_larger_pct:9.0f}% | {supports:14} |")

    # Overall assessment
    support_rate = claim_supported / total_layers * 100

    print(f"\nOverall: {claim_supported}/{total_layers} layers ({support_rate:.0f}%) support the claim")

    if support_rate > 75:
        verdict = "STRONGLY SUPPORTED"
    elif support_rate > 50:
        verdict = "MODERATELY SUPPORTED"
    else:
        verdict = "NOT SUPPORTED"

    print(f"Verdict: Paper claim is {verdict}")

    return {
        "layer_stats": {k: {"avg_ratio": sum(v["ratios"])/len(v["ratios"]),
                           "ffn_larger_pct": v["ffn_larger_count"]/v["total"]*100}
                       for k, v in layer_stats.items()},
        "support_rate": support_rate,
        "verdict": verdict,
    }


def generate_results_doc(results, analysis, output_dir: Path):
    """Generate results documentation."""

    doc = f"""# Paper Claim Reproduction Results

## Paper
"The Bayesian Geometry of Transformer Attention" (arXiv:2512.22471)

## Claim Tested
**"Feed-forward networks perform the posterior update"**

Operationalized as: FFN contribution norm > Attention contribution norm

## Rationale for Selection
1. Directly testable with existing activation extraction tooling
2. Connects to Track C representation analysis
3. Provides mechanistic insight into model behavior
4. Does not require implementing paper's synthetic tasks

## Methodology
1. Extracted activations at pre_attn, post_attn, post_ffn positions
2. Computed: attention_contribution = post_attn - pre_attn
3. Computed: ffn_contribution = post_ffn - post_attn
4. Compared L2 norms across layers and inputs

## Test Inputs
- Factual reasoning: "The capital of France is Paris..."
- Sequential reasoning: "If A is true and B is false..."
- Simple completion: "The quick brown fox..."
- Mathematical: "2 + 2 = 4. 3 + 3 = 6..."

## Results

### Per-Layer Analysis

| Layer | Avg FFN/Attn Ratio | FFN Larger (%) | Supports Claim |
|-------|-------------------|----------------|----------------|
"""

    for layer in sorted(analysis["layer_stats"].keys()):
        stats = analysis["layer_stats"][layer]
        supports = "Yes" if stats["ffn_larger_pct"] > 50 else "No"
        doc += f"| {layer} | {stats['avg_ratio']:.2f} | {stats['ffn_larger_pct']:.0f}% | {supports} |\n"

    doc += f"""
### Overall Assessment

- **Support Rate**: {analysis['support_rate']:.0f}% of layers support the claim
- **Verdict**: {analysis['verdict']}

## Interpretation

"""

    if analysis['support_rate'] > 75:
        doc += """The results **strongly support** the paper's claim that FFN performs the
"posterior update" operation. Across most layers, FFN contributes more to
representation changes than attention does.

This aligns with the paper's mechanistic interpretation:
- Attention routes information (smaller contribution)
- FFN processes/updates (larger contribution)
"""
    elif analysis['support_rate'] > 50:
        doc += """The results **moderately support** the paper's claim. While FFN tends
to contribute more than attention in most layers, the pattern is not
universal.

Possible explanations for partial support:
1. Natural language tasks differ from synthetic Bayesian tasks
2. Our model (Qwen2.5-7B) may have different layer dynamics
3. The claim may be most applicable to specific layers
"""
    else:
        doc += """The results **do not support** the paper's claim in our setting.
Attention contributes comparably to or more than FFN in many layers.

This could indicate:
1. The mechanism is specific to the paper's synthetic tasks
2. Language modeling involves different computation patterns
3. Our operationalization doesn't capture the paper's intended meaning
"""

    doc += """
## Comparison to Paper Claims

| Paper Claim | Our Finding | Match? |
|-------------|-------------|--------|
| FFN does posterior update | FFN has larger contribution | TBD after run |
| Attention provides routing | Attention has smaller contribution | TBD after run |
| Residual = belief substrate | Info accumulates in residual | Not tested |

## Limitations

1. **Different task domain**: Paper uses synthetic Bayesian tasks; we use natural language
2. **Different model**: Paper uses small custom transformers; we use Qwen2.5-7B
3. **Operationalization**: "Posterior update" may not map directly to contribution norm
4. **Single metric**: L2 norm is one possible measure of contribution

## Conclusions

[To be filled after running experiment]

## Connection to Learning Journey

This reproduction experiment connects:
- **Track B**: Attention patterns (now interpreted as "routing")
- **Track C**: FFN vs attention contributions (now interpreted as "update vs routing")
- **Paper understanding**: Concrete test of theoretical claim

## Next Steps

1. Run experiment with GPU to get actual numbers
2. Visualize contribution ratios across layers
3. Consider testing other claims (orthogonal key bases, value manifold)

## Related

- [Paper Annotation](../../../docs/papers/bayesian-geometry-attention.md)
- [Track C Experiments](../../probing/run_analysis.py)
- [Attention Self-Assessment](../../../docs/concepts/attention-self-assessment.md)
"""

    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "results.md", "w") as f:
        f.write(doc)

    print(f"\nSaved: {output_dir / 'results.md'}")


def run_experiment():
    """Run the paper claim reproduction experiment."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)

    # Load model
    model, tokenizer = load_model()

    # Run analysis
    results = run_contribution_analysis(model, tokenizer)

    # Analyze results
    analysis = analyze_results(results)

    # Save raw results
    with open(output_dir / "raw_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    with open(output_dir / "analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Generate documentation
    generate_results_doc(results, analysis, output_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_experiment()
