"""
Activation Extraction Tooling
=============================

Extract intermediate activations (residual stream values) from transformer models.

THE RESIDUAL STREAM
-------------------
The residual stream is the core information pathway in transformers:

```
Input Embeddings
      |
      v
  [Layer 0]
      |-- Attention reads from stream, writes back
      |-- FFN reads from stream, writes back
      v
  [Layer 1]
      |-- Attention reads from stream, writes back
      |-- FFN reads from stream, writes back
      v
    ...
      v
  [Layer N]
      |
      v
Output (logits)
```

Key insight: Information accumulates through residual connections.
- Each layer ADDS to the stream (doesn't replace)
- Final representation is sum of all contributions
- We can probe what's in the stream at any point

ACTIVATION SHAPES (Qwen2.5-7B)
-----------------------------
- Hidden size: 3584
- Each position has a 3584-dimensional activation vector
- Shape: (batch, seq_len, hidden_size) = (B, S, 3584)

EXTRACTION POINTS
-----------------
For each layer, we can extract:
1. Pre-attention: Input to attention (residual stream state)
2. Post-attention: After attention + residual connection
3. Post-FFN: After FFN + residual connection (= input to next layer)

The difference between these shows what each component contributes.
"""

import torch
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ActivationOutput:
    """
    Container for extracted activation data.

    Attributes:
        activations: Dict mapping (layer_idx, position) to tensor
                    position is 'pre_attn', 'post_attn', or 'post_ffn'
        tokens: List of token strings
        layer_indices: Which layers were captured
        model_config: Relevant model configuration
    """
    activations: Dict[tuple, torch.Tensor]
    tokens: List[str]
    layer_indices: List[int]
    model_config: Dict[str, Any]

    def get(self, layer_idx: int, position: str = "post_ffn") -> torch.Tensor:
        """
        Get activations for a specific layer and position.

        Args:
            layer_idx: Layer index
            position: 'pre_attn', 'post_attn', or 'post_ffn'

        Returns:
            Tensor of shape (batch, seq_len, hidden_size)
        """
        key = (layer_idx, position)
        if key not in self.activations:
            available = [k for k in self.activations.keys() if k[0] == layer_idx]
            raise KeyError(f"Position '{position}' not found for layer {layer_idx}. "
                          f"Available: {available}")
        return self.activations[key]

    def get_token_activation(self, layer_idx: int, token_idx: int,
                            position: str = "post_ffn") -> torch.Tensor:
        """
        Get activation for a specific token.

        Returns:
            Tensor of shape (hidden_size,)
        """
        full = self.get(layer_idx, position)
        return full[0, token_idx]  # First batch, specified token

    def get_layer_diff(self, layer_idx: int) -> Dict[str, torch.Tensor]:
        """
        Compute what attention and FFN contribute to this layer.

        Returns:
            Dict with 'attention_contribution' and 'ffn_contribution'
        """
        pre = self.get(layer_idx, "pre_attn")
        post_attn = self.get(layer_idx, "post_attn")
        post_ffn = self.get(layer_idx, "post_ffn")

        return {
            "attention_contribution": post_attn - pre,
            "ffn_contribution": post_ffn - post_attn,
        }

    @property
    def hidden_size(self) -> int:
        if not self.activations:
            return 0
        first = next(iter(self.activations.values()))
        return first.shape[-1]

    @property
    def seq_len(self) -> int:
        if not self.activations:
            return 0
        first = next(iter(self.activations.values()))
        return first.shape[1]


class ActivationExtractor:
    """
    Extract activations from transformer layers using hooks.

    LEARNING NOTES:
    - We hook into layer inputs and outputs
    - Pre-attention = layer input
    - Post-attention = after attention sublayer (with residual)
    - Post-FFN = layer output = next layer input

    Usage:
        extractor = ActivationExtractor(model)
        with extractor.capture(layers=[0, 14, 27]):
            outputs = model(input_ids)
        activations = extractor.get_activations()
    """

    def __init__(self, model, tokenizer=None):
        """
        Initialize extractor for a model.

        Args:
            model: HuggingFace transformer model
            tokenizer: Optional tokenizer for decoding tokens
        """
        self.model = model
        self.tokenizer = tokenizer
        self._hooks = []
        self._activation_cache = {}
        self._input_tokens = []

        self._detect_architecture()

    def _detect_architecture(self):
        """Detect model architecture and locate layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self._layers = self.model.model.layers
            self._arch = "qwen"
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self._layers = self.model.transformer.h
            self._arch = "gpt2"
        else:
            raise ValueError("Could not detect model architecture")

        self.num_layers = len(self._layers)

        config = self.model.config
        self._config = {
            "num_layers": self.num_layers,
            "hidden_size": getattr(config, 'hidden_size', None),
            "architecture": self._arch,
        }

    def _create_pre_attn_hook(self, layer_idx: int):
        """Hook to capture input to attention (residual stream state)."""
        def hook(module, input, output):
            # Input is a tuple, first element is hidden states
            if isinstance(input, tuple) and len(input) > 0:
                hidden_states = input[0]
                self._activation_cache[(layer_idx, "pre_attn")] = hidden_states.detach().cpu()
        return hook

    def _create_post_ffn_hook(self, layer_idx: int):
        """Hook to capture output of layer (after FFN + residual)."""
        def hook(module, input, output):
            # Output is hidden states (possibly in a tuple)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self._activation_cache[(layer_idx, "post_ffn")] = hidden_states.detach().cpu()
        return hook

    @contextmanager
    def capture(self, layers: Optional[List[int]] = None,
               positions: List[str] = ["pre_attn", "post_ffn"]):
        """
        Context manager to capture activations during forward pass.

        Args:
            layers: List of layer indices. None = all layers.
            positions: Which positions to capture. Options:
                      'pre_attn' - before attention
                      'post_ffn' - after FFN (layer output)

        Usage:
            with extractor.capture(layers=[0, 14, 27]):
                model(input_ids)
        """
        if layers is None:
            layers = list(range(self.num_layers))

        # Clear cache
        self._activation_cache = {}
        self._hooks = []

        try:
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= self.num_layers:
                    raise ValueError(f"Layer {layer_idx} out of range")

                layer = self._layers[layer_idx]

                # Pre-attention hook (layer input)
                if "pre_attn" in positions:
                    hook = layer.register_forward_pre_hook(
                        self._create_pre_attn_hook(layer_idx)
                    )
                    self._hooks.append(hook)

                # Post-FFN hook (layer output)
                if "post_ffn" in positions:
                    hook = layer.register_forward_hook(
                        self._create_post_ffn_hook(layer_idx)
                    )
                    self._hooks.append(hook)

            yield self

        finally:
            for hook in self._hooks:
                hook.remove()
            self._hooks = []

    def get_activations(self, tokens: Optional[List[str]] = None) -> ActivationOutput:
        """Get captured activation data."""
        return ActivationOutput(
            activations=self._activation_cache.copy(),
            tokens=tokens or self._input_tokens,
            layer_indices=sorted(set(k[0] for k in self._activation_cache.keys())),
            model_config=self._config,
        )


def extract_activations(
    model,
    tokenizer,
    text: str,
    layers: Optional[List[int]] = None,
    positions: List[str] = ["pre_attn", "post_ffn"],
) -> ActivationOutput:
    """
    High-level function to extract activations for text input.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer for the model
        text: Input text to process
        layers: Which layers to capture (None = all)
        positions: Which positions in each layer

    Returns:
        ActivationOutput with activations and tokens

    Example:
        >>> result = extract_activations(model, tokenizer, "Hello world", layers=[0, 27])
        >>> print(result.hidden_size)
        3584
        >>> layer0_output = result.get(0, "post_ffn")
        >>> print(layer0_output.shape)  # (1, seq_len, 3584)
    """
    extractor = ActivationExtractor(model, tokenizer)

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Extract
    with extractor.capture(layers=layers, positions=positions):
        with torch.no_grad():
            _ = model(**inputs)

    return extractor.get_activations(tokens=tokens)


# ============================================
# ANALYSIS UTILITIES
# ============================================

def compute_activation_stats(activations: ActivationOutput) -> Dict[str, Any]:
    """
    Compute statistics about activations.

    Returns:
        Dict with per-layer statistics
    """
    stats = {"layers": {}}

    for layer_idx in activations.layer_indices:
        layer_stats = {}

        for position in ["pre_attn", "post_ffn"]:
            try:
                act = activations.get(layer_idx, position)
                layer_stats[position] = {
                    "mean": act.mean().item(),
                    "std": act.std().item(),
                    "min": act.min().item(),
                    "max": act.max().item(),
                    "norm": act.norm(dim=-1).mean().item(),
                }
            except KeyError:
                continue

        # Compute contributions if both available
        try:
            diff = activations.get_layer_diff(layer_idx)
            layer_stats["attention_contrib_norm"] = diff["attention_contribution"].norm(dim=-1).mean().item()
            layer_stats["ffn_contrib_norm"] = diff["ffn_contribution"].norm(dim=-1).mean().item()
        except KeyError:
            pass

        stats["layers"][layer_idx] = layer_stats

    return stats


# ============================================
# QUICK TEST
# ============================================
if __name__ == "__main__":
    print("Activation Extraction Module")
    print("=" * 40)
    print("\nThe Residual Stream:")
    print("  - Core information pathway in transformers")
    print("  - Each layer reads from and writes to stream")
    print("  - Activations = stream state at each point")
    print("\nExtraction points per layer:")
    print("  - pre_attn: Before attention (input to layer)")
    print("  - post_ffn: After FFN (output of layer)")
    print("\nUsage:")
    print("""
    from experiments.probing import extract_activations

    result = extract_activations(
        model, tokenizer, "The quick brown fox",
        layers=[0, 14, 27]
    )

    # Get layer 0 output
    layer0 = result.get(0, "post_ffn")
    print(layer0.shape)  # (1, seq_len, hidden_size)

    # See what attention/FFN contribute
    diff = result.get_layer_diff(14)
    print(diff["attention_contribution"].norm())
    """)
