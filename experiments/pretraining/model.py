"""
GPT Model Implementation from Scratch

This module implements a GPT (Generative Pre-trained Transformer) model following
Sebastian Raschka's "Build a Large Language Model (From Scratch)" book (Chapters 3-4).

The architecture follows the decoder-only transformer design used in GPT-2/3, featuring:
- Multi-head causal self-attention
- Position-wise feed-forward networks
- Layer normalization (pre-norm variant)
- Residual connections

Educational comments are included throughout to explain the purpose and mechanics
of each component.

Reference: Vaswani et al., "Attention Is All You Need" (2017)
Reference: Radford et al., "Language Models Are Unsupervised Multitask Learners" (2019)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Dict, Any

from .config import GPT_CONFIGS, GPTConfig


# =============================================================================
# Layer Normalization
# =============================================================================

class LayerNorm(nn.Module):
    """
    Layer Normalization for stabilizing transformer training.

    Unlike batch normalization (which normalizes across the batch dimension),
    layer normalization normalizes across the feature dimension. This makes it
    more suitable for transformers because:

    1. It's independent of batch size, allowing flexible batch sizes during
       training and inference
    2. It normalizes each sample independently, which is crucial for
       autoregressive generation where future tokens aren't available

    The normalization ensures activations have mean=0 and variance=1, then
    applies learnable scale (gamma) and shift (beta) parameters.

    Formula: output = gamma * (x - mean) / sqrt(var + eps) + beta
    """

    def __init__(self, emb_dim: int, eps: float = 1e-5):
        """
        Args:
            emb_dim: Embedding dimension (size of the last dimension to normalize)
            eps: Small constant for numerical stability to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter (initialized to 1)
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # Learnable shift parameter (initialized to 0)
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim)

        Returns:
            Normalized tensor of same shape
        """
        # Compute mean across the embedding dimension (last dim)
        mean = x.mean(dim=-1, keepdim=True)
        # Compute variance (using biased estimator for compatibility with GPT-2)
        # Note: unbiased=False matches the original GPT-2 implementation
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # Normalize
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable affine transformation
        return self.scale * norm_x + self.shift


# =============================================================================
# GELU Activation Function
# =============================================================================

class GELU(nn.Module):
    """
    Gaussian Error Linear Unit (GELU) activation function.

    GELU is used instead of ReLU in GPT models because:

    1. Smoothness: Unlike ReLU's sharp corner at 0, GELU is smooth everywhere,
       which can lead to better optimization dynamics

    2. Probabilistic interpretation: GELU can be thought of as stochastically
       zeroing inputs based on their value (inputs more likely to be zeroed
       if they're small)

    3. Better performance: Empirically shown to work better for transformers

    The exact formula is: GELU(x) = x * Phi(x), where Phi(x) is the CDF of
    the standard normal distribution.

    We use the approximation from the original GPT-2 implementation:
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of any shape

        Returns:
            GELU-activated tensor of same shape
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# =============================================================================
# Feed-Forward Network
# =============================================================================

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    This is applied independently to each position (token) in the sequence.
    It consists of two linear transformations with a GELU activation in between.

    The FFN serves as the "processing" step after attention has gathered
    information from different positions. While attention determines WHAT
    information to combine, the FFN processes HOW to transform that information.

    Architecture:
        Input (emb_dim) -> Linear (emb_dim * 4) -> GELU -> Linear (emb_dim) -> Output

    The expansion factor of 4x is a design choice from the original transformer.
    This creates a "bottleneck" that forces the model to learn compressed
    representations.

    Why FFN follows attention:
    - Attention mixes information across positions (horizontal mixing)
    - FFN processes each position independently (vertical processing)
    - Together they provide both cross-position and within-position transformations
    """

    def __init__(self, cfg: GPTConfig):
        """
        Args:
            cfg: Configuration object with emb_dim attribute
        """
        super().__init__()
        self.layers = nn.Sequential(
            # Expand from emb_dim to 4*emb_dim
            nn.Linear(cfg.emb_dim, 4 * cfg.emb_dim),
            # Non-linear activation
            GELU(),
            # Project back to emb_dim
            nn.Linear(4 * cfg.emb_dim, cfg.emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim)

        Returns:
            Output tensor of same shape (batch, seq_len, emb_dim)
        """
        return self.layers(x)


# =============================================================================
# Multi-Head Attention
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    This is the core mechanism that allows the model to "attend" to different
    parts of the input sequence when processing each token.

    Key Concepts:

    1. Self-Attention: Each token creates a query ("what am I looking for?")
       and provides keys ("what do I contain?") and values ("what information
       do I provide?"). Attention scores are computed as query-key dot products.

    2. Causal Masking: For autoregressive generation, each position can only
       attend to itself and previous positions (not future tokens). This is
       achieved by masking attention scores with -inf before softmax.

    3. Multi-Head: Instead of one attention mechanism, we use multiple "heads"
       that can each learn different attention patterns. For example, one head
       might learn syntactic relationships, another semantic relationships.

    4. Scaled Dot-Product: Attention scores are scaled by sqrt(head_dim) to
       prevent extremely large values that would make softmax saturate.

    Query, Key, Value intuition:
    - Query (Q): "What am I looking for?" - the question each token asks
    - Key (K): "What do I contain?" - how each token describes itself
    - Value (V): "What information do I provide?" - the actual content

    The attention formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 num_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        """
        Args:
            d_in: Input embedding dimension
            d_out: Output embedding dimension (must be divisible by num_heads)
            context_length: Maximum sequence length (for causal mask)
            num_heads: Number of attention heads
            dropout: Dropout probability for attention weights
            qkv_bias: Whether to use bias in Q/K/V projections
        """
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Each head operates on a slice of the embedding
        self.head_dim = d_out // num_heads

        # Linear projections for Q, K, V
        # These learn what to query, what keys to produce, and what values to return
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Output projection to combine information from all heads
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Register causal mask as a buffer (not a parameter, but should move with model)
        # Upper triangular matrix of 1s - these positions will be masked out
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, num_tokens, d_in)

        Returns:
            Context vectors of shape (batch, num_tokens, d_out)
        """
        b, num_tokens, d_in = x.shape

        # Project input to queries, keys, and values
        # Shape: (batch, num_tokens, d_out)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape to separate heads
        # (batch, num_tokens, d_out) -> (batch, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose for batch matrix multiplication
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute attention scores: Q @ K^T
        # (batch, num_heads, num_tokens, head_dim) @ (batch, num_heads, head_dim, num_tokens)
        # -> (batch, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Apply causal mask: set future positions to -inf so softmax makes them 0
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Scale by sqrt(head_dim) to prevent large values that saturate softmax
        # This is the "scaled" in "scaled dot-product attention"
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Softmax to get attention weights (probabilities that sum to 1)
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Apply dropout to attention weights for regularization
        attn_weights = self.dropout(attn_weights)

        # Compute context vectors: weighted sum of values
        # (batch, num_heads, num_tokens, num_tokens) @ (batch, num_heads, num_tokens, head_dim)
        # -> (batch, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # Transpose back and reshape to combine heads
        # (batch, num_heads, num_tokens, head_dim) -> (batch, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)
        # (batch, num_tokens, num_heads, head_dim) -> (batch, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Final projection
        context_vec = self.out_proj(context_vec)

        return context_vec


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """
    A single Transformer block combining attention and feed-forward layers.

    Architecture (Pre-LayerNorm variant used in GPT-2):
        x -> LayerNorm -> MultiHeadAttention -> Dropout -> + (residual) ->
        -> LayerNorm -> FeedForward -> Dropout -> + (residual) -> output

    Key Design Decisions:

    1. Pre-LayerNorm: We apply layer normalization BEFORE the attention/FFN,
       not after. This is different from the original transformer but provides
       better training dynamics (more stable gradients).

    2. Residual Connections (shortcuts): The input is added directly to the
       output of each sub-layer. This allows gradients to flow directly through
       the network during backpropagation, mitigating the vanishing gradient
       problem in deep networks.

    3. Dropout: Applied after attention and FFN for regularization.

    Why residual connections matter:
    - In deep networks, gradients can become vanishingly small
    - Residual connections create "shortcut paths" for gradients
    - This allows training very deep networks (GPT-2 has 12-48 blocks)
    - The network can learn to "skip" layers that aren't useful
    """

    def __init__(self, cfg: GPTConfig):
        """
        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()

        # Multi-head attention module
        self.att = MultiHeadAttention(
            d_in=cfg.emb_dim,
            d_out=cfg.emb_dim,
            context_length=cfg.context_length,
            num_heads=cfg.n_heads,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias
        )

        # Feed-forward network
        self.ff = FeedForward(cfg)

        # Layer normalization (one for each sub-layer)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)

        # Dropout for residual connections
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, emb_dim)

        Returns:
            Output tensor of same shape
        """
        # Attention block with residual connection
        shortcut = x  # Save input for residual
        x = self.norm1(x)  # Pre-LayerNorm
        x = self.att(x)  # Multi-head attention
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # Residual connection

        # Feed-forward block with residual connection
        shortcut = x  # Save for residual
        x = self.norm2(x)  # Pre-LayerNorm
        x = self.ff(x)  # Feed-forward
        x = self.drop_shortcut(x)  # Dropout
        x = x + shortcut  # Residual connection

        return x


# =============================================================================
# GPT Model
# =============================================================================

class GPTModel(nn.Module):
    """
    Complete GPT (Generative Pre-trained Transformer) Model.

    This implements the full GPT architecture used for language modeling.
    The model takes token IDs as input and outputs logits over the vocabulary
    for next-token prediction.

    Architecture Overview:
        1. Token Embedding: Convert token IDs to dense vectors
        2. Position Embedding: Add position information to embeddings
        3. Dropout: Regularization on embeddings
        4. N x Transformer Blocks: The main processing layers
        5. Final LayerNorm: Normalize before output projection
        6. Output Head: Project to vocabulary size for predictions

    The model is trained with cross-entropy loss on next-token prediction.
    During inference, we sample from the output distribution to generate text.

    Parameter Count Breakdown (for 124M model):
    - Token embeddings: 50,257 * 768 = ~39M
    - Position embeddings: 1,024 * 768 = ~0.8M
    - Transformer blocks (12): ~73M
    - Output head: 50,257 * 768 = ~39M (often tied with token embeddings)
    """

    def __init__(self, cfg: GPTConfig):
        """
        Args:
            cfg: Configuration object with model hyperparameters
        """
        super().__init__()
        self.cfg = cfg

        # Token embedding: learns a dense vector for each token in vocabulary
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)

        # Position embedding: learns a dense vector for each position
        # This allows the model to understand token order since attention
        # is permutation-invariant
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)

        # Dropout on embeddings
        self.drop_emb = nn.Dropout(cfg.drop_rate)

        # Stack of transformer blocks - this is where the magic happens
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )

        # Final layer normalization
        self.final_norm = LayerNorm(cfg.emb_dim)

        # Output projection to vocabulary size
        # Note: GPT-2 uses weight tying (shares weights with token embedding)
        # but we keep them separate here for clarity
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for language modeling.

        Args:
            in_idx: Token IDs of shape (batch, seq_len)

        Returns:
            Logits of shape (batch, seq_len, vocab_size)

        Each position's logits represent the model's prediction for what
        the NEXT token should be. For training, we shift the targets by 1.
        """
        batch_size, seq_len = in_idx.shape

        # Get token embeddings
        tok_embeds = self.tok_emb(in_idx)  # (batch, seq_len, emb_dim)

        # Get position embeddings
        # Creates position indices [0, 1, 2, ..., seq_len-1] on the same device
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )  # (seq_len, emb_dim)

        # Combine token and position embeddings
        # Position embeddings are broadcast across the batch
        x = tok_embeds + pos_embeds  # (batch, seq_len, emb_dim)

        # Apply dropout
        x = self.drop_emb(x)

        # Pass through transformer blocks
        x = self.trf_blocks(x)  # (batch, seq_len, emb_dim)

        # Final normalization
        x = self.final_norm(x)

        # Project to vocabulary size
        logits = self.out_head(x)  # (batch, seq_len, vocab_size)

        return logits

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> Dict[str, int]:
        """Get parameter count breakdown by component."""
        breakdown = {
            'token_embedding': self.tok_emb.weight.numel(),
            'position_embedding': self.pos_emb.weight.numel(),
            'transformer_blocks': sum(
                p.numel() for p in self.trf_blocks.parameters()
            ),
            'final_norm': sum(p.numel() for p in self.final_norm.parameters()),
            'output_head': self.out_head.weight.numel(),
        }
        breakdown['total'] = sum(breakdown.values())
        return breakdown


# =============================================================================
# Text Generation
# =============================================================================

def generate_text_simple(
    model: GPTModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int
) -> torch.Tensor:
    """
    Generate text using greedy decoding.

    This is the simplest generation strategy - always pick the most likely
    next token. More sophisticated methods include:
    - Temperature scaling (softens/sharpens distribution)
    - Top-k sampling (sample from top k tokens)
    - Top-p (nucleus) sampling (sample from tokens with cumulative prob p)

    Args:
        model: The GPT model
        idx: Starting token IDs of shape (batch, seq_len)
        max_new_tokens: Number of new tokens to generate
        context_size: Maximum context length the model supports

    Returns:
        Token IDs including generated tokens, shape (batch, seq_len + max_new_tokens)
    """
    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab_size)

        # Focus on last position (next token prediction)
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Convert to probabilities
        probas = torch.softmax(logits, dim=-1)

        # Greedy: pick the most probable token
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

        # Append to running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, seq_len + 1)

    return idx


# =============================================================================
# Convenience Functions
# =============================================================================

def create_model(config_name: str = 'nano') -> GPTModel:
    """
    Create a GPT model with the specified configuration.

    Args:
        config_name: One of 'nano', 'small', 'medium', or 'gpt2-124m'

    Returns:
        Initialized GPT model
    """
    if config_name not in GPT_CONFIGS:
        available = ', '.join(GPT_CONFIGS.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    cfg = GPT_CONFIGS[config_name]
    return GPTModel(cfg)


if __name__ == '__main__':
    # Quick test of the model
    print("Testing GPT Model Components...")

    # Test with nano config
    cfg = GPT_CONFIGS['nano']
    model = GPTModel(cfg)

    # Print parameter breakdown
    breakdown = model.get_parameter_breakdown()
    print(f"\nModel: nano")
    print(f"Total parameters: {breakdown['total']:,}")
    for name, count in breakdown.items():
        if name != 'total':
            print(f"  {name}: {count:,}")

    # Test forward pass
    batch_size = 2
    seq_len = 64
    x = torch.randint(0, cfg.vocab_size, (batch_size, seq_len))

    logits = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print("Model forward pass successful!")
