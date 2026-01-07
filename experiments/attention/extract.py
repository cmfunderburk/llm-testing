"""
Attention Extraction Tooling
============================

Extract attention weights from transformer models for visualization and analysis.

ATTENTION TENSOR SHAPES (for Qwen2.5-7B)
----------------------------------------

Qwen2.5-7B architecture:
- 28 layers
- 28 attention heads per layer
- 128 dimensions per head (hidden_size=3584, 3584/28=128)
- Grouped Query Attention (GQA): 4 KV heads shared across 28 Q heads

Tensor shapes during forward pass:
- query: (batch, seq_len, num_heads, head_dim) = (B, S, 28, 128)
- key:   (batch, seq_len, num_kv_heads, head_dim) = (B, S, 4, 128)
- value: (batch, seq_len, num_kv_heads, head_dim) = (B, S, 4, 128)

Attention weights (after softmax):
- Shape: (batch, num_heads, seq_len, seq_len) = (B, 28, S, S)
- attention[b, h, i, j] = how much token i attends to token j in head h

LEARNING NOTES
--------------
- GQA means Q heads share KV heads (7 Q heads per KV head)
- Causal attention: upper triangular mask (can't attend to future)
- Softmax rows sum to 1 (attention is a probability distribution)
"""

import torch
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class AttentionOutput:
    """
    Container for extracted attention data.

    Attributes:
        weights: Dict mapping layer index to attention tensor
                 Shape per layer: (batch, num_heads, seq_len, seq_len)
        tokens: List of token strings (for visualization labels)
        layer_indices: Which layers were captured
        model_config: Relevant model configuration
    """
    weights: Dict[int, torch.Tensor]
    tokens: List[str]
    layer_indices: List[int]
    model_config: Dict[str, Any]

    def get_layer(self, layer_idx: int) -> torch.Tensor:
        """Get attention weights for a specific layer."""
        if layer_idx not in self.weights:
            raise KeyError(f"Layer {layer_idx} not captured. Available: {list(self.weights.keys())}")
        return self.weights[layer_idx]

    def get_head(self, layer_idx: int, head_idx: int) -> torch.Tensor:
        """
        Get attention weights for a specific head.

        Returns:
            Tensor of shape (seq_len, seq_len) for first batch item
        """
        layer_attn = self.get_layer(layer_idx)
        return layer_attn[0, head_idx]  # First batch, specified head

    @property
    def num_layers(self) -> int:
        return len(self.weights)

    @property
    def num_heads(self) -> int:
        if not self.weights:
            return 0
        first_layer = next(iter(self.weights.values()))
        return first_layer.shape[1]

    @property
    def seq_len(self) -> int:
        if not self.weights:
            return 0
        first_layer = next(iter(self.weights.values()))
        return first_layer.shape[2]


class AttentionExtractor:
    """
    Extract attention weights from a transformer model using hooks.

    LEARNING NOTES:
    - PyTorch hooks let us intercept intermediate values during forward pass
    - register_forward_hook runs after a module's forward() completes
    - We attach hooks to attention modules to capture attention weights

    Usage:
        extractor = AttentionExtractor(model)
        with extractor.capture(layers=[0, 5, 10, 27]):
            outputs = model.generate(...)
        attention_data = extractor.get_attention()
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
        self._attention_cache = {}
        self._input_tokens = []

        # Detect model architecture
        self._detect_architecture()

    def _detect_architecture(self):
        """
        Detect model architecture and locate attention modules.

        LEARNING NOTES:
        Different models organize their layers differently:
        - Qwen: model.model.layers[i].self_attn
        - Llama: model.model.layers[i].self_attn
        - GPT2: model.transformer.h[i].attn
        """
        # Try to find the layers
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Qwen/Llama style
            self._layers = self.model.model.layers
            self._arch = "qwen"
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT2 style
            self._layers = self.model.transformer.h
            self._arch = "gpt2"
        else:
            raise ValueError("Could not detect model architecture. Supported: Qwen, Llama, GPT2")

        self.num_layers = len(self._layers)

        # Get model config
        config = self.model.config
        self._config = {
            "num_layers": self.num_layers,
            "num_heads": getattr(config, 'num_attention_heads', None),
            "num_kv_heads": getattr(config, 'num_key_value_heads', None),
            "hidden_size": getattr(config, 'hidden_size', None),
            "architecture": self._arch,
        }

    def _get_attention_module(self, layer_idx: int):
        """Get the attention module for a specific layer."""
        layer = self._layers[layer_idx]
        if self._arch == "qwen":
            return layer.self_attn
        elif self._arch == "gpt2":
            return layer.attn
        else:
            raise ValueError(f"Unknown architecture: {self._arch}")

    def _create_hook(self, layer_idx: int):
        """
        Create a hook function that captures attention weights.

        LEARNING NOTES:
        - The hook receives (module, input, output)
        - For attention modules, we need to modify forward to return attention weights
        - Alternative: patch the attention forward to output_attentions=True
        """
        def hook(module, input, output):
            # Output structure depends on whether attention weights are returned
            # With output_attentions=True: (hidden_states, attention_weights, ...)
            if isinstance(output, tuple) and len(output) >= 2:
                attn_weights = output[1]  # Second element is attention weights
                if attn_weights is not None:
                    # Detach and move to CPU to save GPU memory
                    self._attention_cache[layer_idx] = attn_weights.detach().cpu()
        return hook

    @contextmanager
    def capture(self, layers: Optional[List[int]] = None):
        """
        Context manager to capture attention during forward pass.

        Args:
            layers: List of layer indices to capture. None = all layers.

        Usage:
            with extractor.capture(layers=[0, 10, 27]):
                model.generate(...)
        """
        if layers is None:
            layers = list(range(self.num_layers))

        # Clear previous cache
        self._attention_cache = {}
        self._hooks = []

        try:
            # Enable attention output in model config
            original_output_attentions = self.model.config.output_attentions
            self.model.config.output_attentions = True

            # Register hooks for specified layers
            for layer_idx in layers:
                if layer_idx < 0 or layer_idx >= self.num_layers:
                    raise ValueError(f"Layer {layer_idx} out of range [0, {self.num_layers})")

                attn_module = self._get_attention_module(layer_idx)
                hook = attn_module.register_forward_hook(self._create_hook(layer_idx))
                self._hooks.append(hook)

            yield self

        finally:
            # Remove hooks
            for hook in self._hooks:
                hook.remove()
            self._hooks = []

            # Restore original config
            self.model.config.output_attentions = original_output_attentions

    def get_attention(self, tokens: Optional[List[str]] = None) -> AttentionOutput:
        """
        Get captured attention data.

        Args:
            tokens: Optional token strings for labels

        Returns:
            AttentionOutput containing weights and metadata
        """
        return AttentionOutput(
            weights=self._attention_cache.copy(),
            tokens=tokens or self._input_tokens,
            layer_indices=sorted(self._attention_cache.keys()),
            model_config=self._config,
        )


def extract_attention(
    model,
    tokenizer,
    text: str,
    layers: Optional[List[int]] = None,
) -> AttentionOutput:
    """
    High-level function to extract attention for a text input.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer for the model
        text: Input text to process
        layers: Which layers to capture (None = all)

    Returns:
        AttentionOutput with attention weights and tokens

    LEARNING NOTES:
    - We tokenize the text and run a forward pass
    - Hooks capture attention weights during the pass
    - Returns attention for the full sequence (including special tokens)

    Example:
        >>> model, tokenizer = load_model()
        >>> result = extract_attention(model, tokenizer, "Hello world")
        >>> print(result.num_layers, result.num_heads, result.seq_len)
        28 28 4
        >>> layer_0_head_0 = result.get_head(0, 0)
        >>> print(layer_0_head_0.shape)  # (seq_len, seq_len)
    """
    extractor = AttentionExtractor(model, tokenizer)

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Get token strings for labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Extract attention
    with extractor.capture(layers=layers):
        with torch.no_grad():
            _ = model(**inputs)

    return extractor.get_attention(tokens=tokens)


# ============================================
# QUICK TEST
# ============================================
if __name__ == "__main__":
    print("Attention Extraction Module")
    print("=" * 40)
    print("\nThis module provides tools for extracting attention weights.")
    print("\nKey classes:")
    print("  - AttentionExtractor: Hook-based extraction")
    print("  - AttentionOutput: Container for results")
    print("\nKey functions:")
    print("  - extract_attention(): High-level extraction")
    print("\nUsage example:")
    print("""
    from experiments.attention import extract_attention

    # Load your model and tokenizer
    model, tokenizer = load_model()

    # Extract attention for some text
    result = extract_attention(
        model, tokenizer,
        "The quick brown fox",
        layers=[0, 14, 27]  # First, middle, last
    )

    # Analyze
    print(f"Captured {result.num_layers} layers")
    print(f"Tokens: {result.tokens}")

    # Get specific attention pattern
    layer0_head0 = result.get_head(0, 0)
    print(f"Attention shape: {layer0_head0.shape}")
    """)
