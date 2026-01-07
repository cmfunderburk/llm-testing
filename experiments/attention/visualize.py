"""
Attention Visualization
=======================

Tools for visualizing attention patterns as heatmaps and comparisons.

LEARNING NOTES
--------------
Attention heatmaps show:
- X-axis: Key positions (what tokens are being attended TO)
- Y-axis: Query positions (what tokens are DOING the attending)
- Color intensity: Attention weight (0=no attention, 1=full attention)

Causal attention shows a triangular pattern:
- Tokens can only attend to previous tokens (and themselves)
- Upper right is masked (future positions)

What to look for:
- Diagonal patterns: Tokens attending to themselves
- Vertical stripes: All tokens attending to specific key positions
- Early layers: Often more local patterns
- Late layers: Often more abstract/distributed patterns
"""

from pathlib import Path
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import json


# Lazy imports for optional dependencies
def _import_plt():
    import matplotlib.pyplot as plt
    return plt

def _import_numpy():
    import numpy as np
    return np


@dataclass
class VisualizationConfig:
    """Configuration for attention visualizations."""
    figsize: Tuple[int, int] = (10, 8)
    cmap: str = "Blues"  # Good for attention (white=0, blue=high)
    dpi: int = 150
    show_colorbar: bool = True
    annotate_values: bool = False  # Show numbers in cells
    max_tokens_display: int = 50  # Truncate long sequences


def plot_attention_heatmap(
    attention: "torch.Tensor",
    tokens: Optional[List[str]] = None,
    title: str = "Attention Pattern",
    output_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    caption: Optional[str] = None,
) -> None:
    """
    Plot a single attention heatmap.

    Args:
        attention: 2D tensor of shape (seq_len, seq_len)
        tokens: Token labels for axes
        title: Plot title
        output_path: Where to save (if None, displays)
        config: Visualization settings
        caption: Interpretive caption to add below plot

    LEARNING NOTES:
    - Rows = query positions (which token is looking)
    - Cols = key positions (what token is being looked at)
    - attention[i,j] = how much token i attends to token j
    """
    plt = _import_plt()
    np = _import_numpy()

    config = config or VisualizationConfig()

    # Convert to numpy if tensor
    if hasattr(attention, 'numpy'):
        attention = attention.numpy()

    seq_len = attention.shape[0]

    # Truncate if too long
    if seq_len > config.max_tokens_display:
        attention = attention[:config.max_tokens_display, :config.max_tokens_display]
        if tokens:
            tokens = tokens[:config.max_tokens_display]
        title += f" (truncated to {config.max_tokens_display} tokens)"

    fig, ax = plt.subplots(figsize=config.figsize)

    # Plot heatmap
    im = ax.imshow(attention, cmap=config.cmap, aspect='auto', vmin=0, vmax=1)

    # Add colorbar
    if config.show_colorbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight')

    # Add token labels
    if tokens:
        # Escape special characters for display
        display_tokens = [t.replace('▁', '_') for t in tokens]
        ax.set_xticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(display_tokens)))
        ax.set_yticklabels(display_tokens, fontsize=8)

    ax.set_xlabel('Key Position (attended to)')
    ax.set_ylabel('Query Position (attending from)')
    ax.set_title(title)

    # Add caption
    if caption:
        plt.figtext(0.5, 0.02, caption, ha='center', fontsize=9, style='italic',
                   wrap=True)
        plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_attention_heads_grid(
    attention: "torch.Tensor",
    tokens: Optional[List[str]] = None,
    title: str = "Attention Heads",
    output_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
    max_heads: int = 16,
) -> None:
    """
    Plot multiple attention heads in a grid.

    Args:
        attention: 3D tensor of shape (num_heads, seq_len, seq_len)
        tokens: Token labels
        title: Overall title
        output_path: Where to save
        config: Visualization settings
        max_heads: Maximum number of heads to display

    LEARNING NOTES:
    - Different heads often capture different patterns
    - Some heads may be "positional" (attend to nearby tokens)
    - Some heads may be "semantic" (attend to related tokens)
    - Some heads may be nearly unused (low variance)
    """
    plt = _import_plt()
    np = _import_numpy()

    config = config or VisualizationConfig()

    if hasattr(attention, 'numpy'):
        attention = attention.numpy()

    num_heads = min(attention.shape[0], max_heads)

    # Calculate grid dimensions
    cols = min(4, num_heads)
    rows = (num_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_heads == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx in range(num_heads):
        row, col = idx // cols, idx % cols
        ax = axes[row][col]

        head_attn = attention[idx]

        # Truncate if needed
        if head_attn.shape[0] > config.max_tokens_display:
            head_attn = head_attn[:config.max_tokens_display, :config.max_tokens_display]

        im = ax.imshow(head_attn, cmap=config.cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Head {idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_heads, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def plot_layer_comparison(
    attention_data: "AttentionOutput",
    head_idx: int = 0,
    output_path: Optional[str] = None,
    config: Optional[VisualizationConfig] = None,
) -> None:
    """
    Compare attention patterns across layers for a single head.

    Args:
        attention_data: AttentionOutput from extraction
        head_idx: Which head to visualize
        output_path: Where to save
        config: Visualization settings

    LEARNING NOTES:
    - Early layers often show more local patterns
    - Middle layers may show syntactic patterns
    - Late layers often show more semantic/abstract patterns
    - This is a hypothesis - your observations may differ!
    """
    plt = _import_plt()
    np = _import_numpy()

    config = config or VisualizationConfig()

    layers = attention_data.layer_indices
    num_layers = len(layers)

    cols = min(4, num_layers)
    rows = (num_layers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_layers == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, layer_idx in enumerate(layers):
        row, col = idx // cols, idx % cols
        ax = axes[row][col]

        # Get attention for this layer and head
        layer_attn = attention_data.weights[layer_idx]
        head_attn = layer_attn[0, head_idx].numpy()  # First batch, specified head

        # Truncate if needed
        if head_attn.shape[0] > config.max_tokens_display:
            head_attn = head_attn[:config.max_tokens_display, :config.max_tokens_display]

        im = ax.imshow(head_attn, cmap=config.cmap, aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Layer {layer_idx}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty subplots
    for idx in range(num_layers, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row][col].axis('off')

    plt.suptitle(f'Attention Pattern Across Layers (Head {head_idx})', fontsize=12)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=config.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_path}")
    else:
        plt.show()


def generate_attention_report(
    attention_data: "AttentionOutput",
    output_dir: str,
    text: str,
) -> None:
    """
    Generate a complete attention analysis report with visualizations.

    Args:
        attention_data: AttentionOutput from extraction
        output_dir: Directory to save outputs
        text: Original input text

    Creates:
        - visualizations/ folder with heatmaps
        - report.md with analysis and captions
    """
    plt = _import_plt()
    np = _import_numpy()

    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    config = VisualizationConfig()
    tokens = attention_data.tokens

    # Track what we generate
    generated = []

    # 1. Generate head grid for each captured layer
    for layer_idx in attention_data.layer_indices:
        layer_attn = attention_data.weights[layer_idx][0]  # First batch

        output_path = viz_dir / f"layer_{layer_idx}_heads.png"
        plot_attention_heads_grid(
            layer_attn,
            tokens=tokens,
            title=f"Layer {layer_idx} - All Attention Heads",
            output_path=str(output_path),
            config=config,
        )
        generated.append({
            "type": "heads_grid",
            "layer": layer_idx,
            "path": f"visualizations/layer_{layer_idx}_heads.png",
        })

    # 2. Generate layer comparison for head 0
    if len(attention_data.layer_indices) > 1:
        output_path = viz_dir / "layer_comparison_head0.png"
        plot_layer_comparison(
            attention_data,
            head_idx=0,
            output_path=str(output_path),
            config=config,
        )
        generated.append({
            "type": "layer_comparison",
            "head": 0,
            "path": "visualizations/layer_comparison_head0.png",
        })

    # 3. Generate detailed heatmap for first layer, first head
    if attention_data.layer_indices:
        first_layer = attention_data.layer_indices[0]
        first_head_attn = attention_data.get_head(first_layer, 0)

        output_path = viz_dir / f"layer_{first_layer}_head_0_detailed.png"
        plot_attention_heatmap(
            first_head_attn,
            tokens=tokens,
            title=f"Layer {first_layer}, Head 0 - Detailed",
            output_path=str(output_path),
            config=config,
            caption="Each row shows what a token attends to. High values (dark blue) indicate strong attention.",
        )
        generated.append({
            "type": "detailed_heatmap",
            "layer": first_layer,
            "head": 0,
            "path": f"visualizations/layer_{first_layer}_head_0_detailed.png",
        })

    # 4. Generate report markdown
    report = f"""# Attention Analysis Report

## Input
```
{text}
```

## Tokens
{tokens}

## Model Configuration
- Architecture: {attention_data.model_config.get('architecture', 'unknown')}
- Layers captured: {attention_data.layer_indices}
- Number of heads: {attention_data.num_heads}
- Sequence length: {attention_data.seq_len}

## Visualizations

"""

    for item in generated:
        if item["type"] == "heads_grid":
            report += f"""### Layer {item["layer"]} - All Heads
![Layer {item["layer"]} Heads]({item["path"]})

**Observations**: [Fill in what patterns you notice]

"""
        elif item["type"] == "layer_comparison":
            report += f"""### Layer Comparison (Head {item["head"]})
![Layer Comparison]({item["path"]})

**Observations**: [How do patterns change across layers?]

"""
        elif item["type"] == "detailed_heatmap":
            report += f"""### Detailed: Layer {item["layer"]}, Head {item["head"]}
![Detailed Heatmap]({item["path"]})

**Observations**: [What tokens are being attended to strongly?]

"""

    report += """## Interpretation Guide

### What to Look For
1. **Diagonal patterns**: Tokens attending to themselves
2. **Vertical stripes**: Many tokens attending to specific positions
3. **Horizontal stripes**: Single tokens attending to many positions
4. **Local patterns**: Attention concentrated near diagonal (nearby tokens)
5. **Distributed patterns**: Attention spread across sequence

### Questions to Answer
- Do early layers show more local patterns?
- Do late layers show more distributed patterns?
- Are there "special" positions that get lots of attention?
- Do different heads capture different patterns?

## Conclusions
[Fill in your conclusions after analyzing the visualizations]
"""

    report_path = output_dir / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")

    # Save metadata
    metadata = {
        "text": text,
        "tokens": tokens,
        "layers_captured": attention_data.layer_indices,
        "model_config": attention_data.model_config,
        "visualizations": generated,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ============================================
# QUICK TEST
# ============================================
if __name__ == "__main__":
    print("Attention Visualization Module")
    print("=" * 40)
    print("\nThis module provides visualization tools for attention patterns.")
    print("\nKey functions:")
    print("  - plot_attention_heatmap(): Single head visualization")
    print("  - plot_attention_heads_grid(): Compare heads in one layer")
    print("  - plot_layer_comparison(): Compare layers for one head")
    print("  - generate_attention_report(): Full analysis with report")
