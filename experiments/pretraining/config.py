"""
GPT Model Configurations

This module defines model configurations for different GPT sizes.
Following the PRD specification:
- nano (~10M params): For rapid iteration and testing
- small (~50M params): For more serious experiments
- medium (~124M params): Matches GPT-2 small architecture
- large (~204M params): Wider medium model for higher capacity
- xlarge (~355M params): GPT-2 medium class model

Each configuration is a dataclass that can be serialized to/from YAML.

Reference configurations are based on:
- GPT-2: "Language Models Are Unsupervised Multitask Learners" (Radford et al., 2019)
- GPT-3: "Language Models Are Few-Shot Learners" (Brown et al., 2020)
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Literal
import yaml


@dataclass
class GPTConfig:
    """
    Configuration for a GPT model.

    Attributes:
        vocab_size: Size of the vocabulary (number of unique tokens)
        context_length: Maximum sequence length the model can process
        emb_dim: Embedding dimension (size of token/position embeddings)
        n_heads: Number of attention heads
        n_layers: Number of transformer blocks
        drop_rate: Dropout probability (0.0 = no dropout)
        qkv_bias: Whether to use bias in Q/K/V projections

    Design Notes:
    - emb_dim should be divisible by n_heads (each head gets emb_dim // n_heads)
    - Larger models typically use larger emb_dim and more layers
    - drop_rate is typically 0.0-0.2 for training, 0.0 for inference
    """
    vocab_size: int = 50257  # GPT-2 BPE vocabulary size
    context_length: int = 1024  # Maximum context window
    emb_dim: int = 768  # Embedding dimension
    n_heads: int = 12  # Number of attention heads
    n_layers: int = 12  # Number of transformer blocks
    drop_rate: float = 0.1  # Dropout rate
    qkv_bias: bool = False  # No bias in QKV projections (GPT-2 style)
    attention_impl: Literal["manual", "sdpa"] = "manual"  # Attention backend
    tie_embeddings: bool = False  # Share token embedding and output projection
    gradient_checkpointing: bool = False  # Recompute block activations in backward
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"  # Intended train precision

    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.emb_dim % self.n_heads == 0, \
            f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.drop_rate >= 0.0 and self.drop_rate <= 1.0, \
            f"drop_rate must be between 0 and 1, got {self.drop_rate}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GPTConfig':
        """Create config from dictionary."""
        return cls(**d)

    def save_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path: str) -> 'GPTConfig':
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def estimate_params(self, tie_embeddings: bool = False) -> int:
        """
        Estimate total parameter count.

        This is an approximation based on the architecture.
        Actual count may differ slightly due to biases, etc.
        """
        # Token embeddings: vocab_size * emb_dim
        tok_emb = self.vocab_size * self.emb_dim

        # Position embeddings: context_length * emb_dim
        pos_emb = self.context_length * self.emb_dim

        # Per transformer block:
        # - QKV projections: 3 * emb_dim * emb_dim
        # - Output projection: emb_dim * emb_dim
        # - FFN: emb_dim * (4 * emb_dim) + (4 * emb_dim) * emb_dim = 8 * emb_dim^2
        # - LayerNorm: 2 * (2 * emb_dim) = 4 * emb_dim
        attn = 4 * self.emb_dim * self.emb_dim  # QKV + output proj
        ffn = 8 * self.emb_dim * self.emb_dim  # Two linear layers
        ln = 4 * self.emb_dim  # Two layer norms
        per_block = attn + ffn + ln
        total_blocks = per_block * self.n_layers

        # Final layer norm: 2 * emb_dim
        final_ln = 2 * self.emb_dim

        # Output head: vocab_size * emb_dim (often weight-tied with tok_emb)
        out_head = 0 if tie_embeddings else self.vocab_size * self.emb_dim

        return tok_emb + pos_emb + total_blocks + final_ln + out_head

    def estimate_vram_mb(
        self,
        batch_size: int,
        precision: Literal["fp32", "bf16", "fp16"] = "fp32",
        optimizer: Literal["adamw", "adamw_8bit", "paged_adamw_8bit"] = "adamw",
        attention_impl: Literal["manual", "sdpa"] = "manual",
        gradient_checkpointing: bool = False,
        tie_embeddings: bool = False,
    ) -> dict:
        """
        Estimate VRAM requirements for training in MB.

        Returns a breakdown of memory usage:
        - model: Model parameters
        - optimizer: Optimizer states
        - gradients: Gradient storage
        - activations: Forward pass activations (approximate)
        - overhead: CUDA context, fragmentation, temp buffers
        - total: Sum of all components

        Note: This is an approximation. Actual usage may vary due to:
        - CUDA memory fragmentation
        - PyTorch memory allocator overhead
        - Temporary buffers during computation
        """
        params = self.estimate_params(tie_embeddings=tie_embeddings)

        # In the current pretraining implementation, AMP affects activation dtype,
        # while model weights, gradients, and AdamW states remain fp32.
        weights_bytes_per_param = 4
        activation_bytes_per_value = 4 if precision == "fp32" else 2
        optimizer_factor = {
            "adamw": 2.0,              # two fp32 moment states
            "adamw_8bit": 0.5,         # approximate 8-bit moment storage
            "paged_adamw_8bit": 0.35,  # additional VRAM relief via paging
        }[optimizer]

        # Model weights
        model_bytes = params * weights_bytes_per_param

        # Optimizer states
        optimizer_bytes = int(params * weights_bytes_per_param * optimizer_factor)

        # Gradients
        gradient_bytes = params * weights_bytes_per_param

        # Activations (rough estimate for training)
        # During training, PyTorch stores activations for the backward pass.
        # Per layer: attention scores (batch * heads * seq * seq * bytes)
        #            + attention output (batch * seq * emb_dim * bytes)
        #            + FFN intermediates (batch * seq * 4*emb_dim * bytes)
        #            + residual connections, layer norms, etc.
        #
        # Empirically calibrated factor based on observed 16GB-class runs.
        # Target anchor: medium (124M), batch=4, ctx=1024, fp32, manual attention
        # should estimate near ~14-15GB instead of >20GB.
        activation_factor = 50
        if gradient_checkpointing:
            activation_factor = int(activation_factor * 0.65)

        activation_bytes = (
            batch_size *
            self.context_length *
            self.emb_dim *
            self.n_layers *
            activation_factor *
            activation_bytes_per_value
        )

        # Attention score matrices: batch * n_heads * seq_len^2 * bytes * n_layers
        manual_attention_scores_bytes = (
            batch_size *
            self.n_heads *
            self.context_length *
            self.context_length *
            activation_bytes_per_value *
            self.n_layers
        )
        attention_scores_bytes = manual_attention_scores_bytes
        if attention_impl == "sdpa":
            # SDPA-backed kernels avoid the full score matrix materialization path.
            attention_scores_bytes = int(manual_attention_scores_bytes * 0.06)

        # Fixed overhead: CUDA context (~1GB) + PyTorch allocator cache + misc
        overhead_bytes = 1100 * 1024 * 1024

        # Total activations including attention scores
        total_activation_bytes = activation_bytes + attention_scores_bytes

        # Safety multiplier for fragmentation, temp buffers, PyTorch caching
        subtotal = model_bytes + optimizer_bytes + gradient_bytes + total_activation_bytes
        fragmentation = int(subtotal * 0.15)  # ~15% fragmentation overhead

        total_bytes = subtotal + overhead_bytes + fragmentation

        # Convert to MB
        mb = 1024 * 1024
        return {
            "model_mb": round(model_bytes / mb, 1),
            "optimizer_mb": round(optimizer_bytes / mb, 1),
            "gradients_mb": round(gradient_bytes / mb, 1),
            "activations_mb": round(total_activation_bytes / mb, 1),
            "overhead_mb": round((overhead_bytes + fragmentation) / mb, 1),
            "total_mb": round(total_bytes / mb, 1),
            "total_gb": round(total_bytes / (1024 * mb), 2),
            "params": params,
            "batch_size": batch_size,
            "context_length": self.context_length,
            "assumptions": {
                "precision": precision,
                "optimizer": optimizer,
                "attention_impl": attention_impl,
                "gradient_checkpointing": gradient_checkpointing,
                "tie_embeddings": tie_embeddings,
            },
        }


# =============================================================================
# Predefined Configurations
# =============================================================================

# Nano model (~10M parameters)
# Good for quick testing and development
GPT_CONFIG_NANO = GPTConfig(
    vocab_size=50257,
    context_length=256,  # Shorter context for faster training
    emb_dim=256,
    n_heads=8,
    n_layers=6,
    drop_rate=0.1,
    qkv_bias=False,
)

# Small model (~50M parameters)
# Good for moderate experiments
GPT_CONFIG_SMALL = GPTConfig(
    vocab_size=50257,
    context_length=512,
    emb_dim=512,
    n_heads=8,
    n_layers=8,
    drop_rate=0.1,
    qkv_bias=False,
)

# Medium model (~124M parameters)
# Matches GPT-2 small architecture
GPT_CONFIG_MEDIUM = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

# Large model (~204M parameters)
# Wider medium model with same context window
GPT_CONFIG_LARGE = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=12,
    drop_rate=0.1,
    qkv_bias=False,
)

# XLarge model (~355M parameters)
# Similar scale to GPT-2 medium
GPT_CONFIG_XLARGE = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=1024,
    n_heads=16,
    n_layers=24,
    drop_rate=0.1,
    qkv_bias=False,
)

# GPT-2 124M (official configuration)
# For loading pretrained weights
GPT_CONFIG_124M = GPTConfig(
    vocab_size=50257,
    context_length=1024,
    emb_dim=768,
    n_heads=12,
    n_layers=12,
    drop_rate=0.0,  # No dropout for pretrained model
    qkv_bias=True,  # GPT-2 uses bias in QKV
)


# Dictionary of all configurations
GPT_CONFIGS: Dict[str, GPTConfig] = {
    'nano': GPT_CONFIG_NANO,
    'small': GPT_CONFIG_SMALL,
    'medium': GPT_CONFIG_MEDIUM,
    'large': GPT_CONFIG_LARGE,
    'xlarge': GPT_CONFIG_XLARGE,
    'gpt2-124m': GPT_CONFIG_124M,
}


def get_config(name: str) -> GPTConfig:
    """
    Get a configuration by name.

    Args:
        name: Configuration name ('nano', 'small', 'medium', 'large', 'xlarge', 'gpt2-124m')

    Returns:
        GPTConfig object

    Raises:
        ValueError: If configuration name is not found
    """
    if name not in GPT_CONFIGS:
        available = ', '.join(GPT_CONFIGS.keys())
        raise ValueError(f"Unknown config '{name}'. Available: {available}")
    return GPT_CONFIGS[name]


def print_config_comparison():
    """Print a comparison of all available configurations."""
    print("=" * 70)
    print("GPT Model Configurations")
    print("=" * 70)
    print(f"{'Config':<12} {'Params':>12} {'Layers':>8} {'Heads':>8} {'Dim':>8} {'Context':>10}")
    print("-" * 70)

    for name, cfg in GPT_CONFIGS.items():
        params = cfg.estimate_params()
        params_str = f"{params / 1e6:.1f}M"
        print(f"{name:<12} {params_str:>12} {cfg.n_layers:>8} {cfg.n_heads:>8} "
              f"{cfg.emb_dim:>8} {cfg.context_length:>10}")

    print("=" * 70)


if __name__ == '__main__':
    print_config_comparison()

    # Test serialization
    print("\nTesting YAML serialization...")
    cfg = GPT_CONFIGS['nano']
    cfg.save_yaml('/tmp/test_config.yaml')
    loaded = GPTConfig.load_yaml('/tmp/test_config.yaml')
    assert cfg.to_dict() == loaded.to_dict()
    print("YAML serialization works!")
