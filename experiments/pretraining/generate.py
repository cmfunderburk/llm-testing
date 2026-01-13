"""
Text Generation for GPT Models

This module provides text generation functionality with various decoding strategies:
- Greedy decoding (temperature=0, deterministic)
- Temperature scaling (controls randomness)
- Top-k sampling (limits to k most likely tokens)
- Top-p (nucleus) sampling (limits to tokens with cumulative probability p)

Decoding Strategies Explained:
=============================

1. Greedy Decoding (temperature=0):
   - Always picks the most probable next token
   - Deterministic but often repetitive
   - Good for testing, not for creative text

2. Temperature Scaling:
   - Temperature < 1.0: Sharper distribution, more confident/repetitive
   - Temperature = 1.0: Original distribution
   - Temperature > 1.0: Flatter distribution, more random/creative
   - Formula: p_i = softmax(logits_i / T)

3. Top-k Sampling:
   - Only consider the k tokens with highest probability
   - Set probability of all other tokens to 0
   - Then sample from this truncated distribution
   - k=1 is equivalent to greedy decoding

4. Top-p (Nucleus) Sampling:
   - Select smallest set of tokens whose cumulative probability >= p
   - More dynamic than top-k (adjusts number of candidates based on distribution)

Reference: Chapter 5 of Raschka's "Build a Large Language Model (From Scratch)"
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union
import argparse

from .tokenizer import Tokenizer


def generate(
    model: torch.nn.Module,
    idx: torch.Tensor,
    max_new_tokens: int,
    context_size: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate text from a GPT model with configurable decoding strategy.

    This is the main generation function that supports various decoding
    strategies through temperature scaling and top-k/top-p sampling.

    Args:
        model: GPT model to generate from
        idx: Starting token IDs of shape (batch, seq_len)
        max_new_tokens: Maximum number of new tokens to generate
        context_size: Maximum context length the model supports
        temperature: Sampling temperature (0 = greedy, >1 = more random)
        top_k: If set, only sample from top k tokens
        top_p: If set, only sample from tokens with cumulative probability >= p
        eos_id: End of sequence token ID (stops generation if encountered)

    Returns:
        Token IDs including generated tokens, shape (batch, seq_len + generated_len)

    Example:
        >>> model.eval()
        >>> tokens = tokenizer.encode("Once upon a")
        >>> idx = torch.tensor([tokens])
        >>> generated = generate(model, idx, max_new_tokens=50, context_size=256)
        >>> print(tokenizer.decode(generated[0].tolist()))
    """
    model.eval()

    for _ in range(max_new_tokens):
        # Crop context if it exceeds the model's context length
        idx_cond = idx[:, -context_size:]

        # Get predictions
        with torch.no_grad():
            logits = model(idx_cond)  # (batch, seq_len, vocab_size)

        # Focus on last position (next token prediction)
        logits = logits[:, -1, :]  # (batch, vocab_size)

        # Apply temperature scaling
        # Temperature > 1 flattens the distribution (more random)
        # Temperature < 1 sharpens it (more deterministic)
        # Temperature = 0 is handled specially (greedy)
        if temperature == 0.0:
            # Greedy decoding: pick the most probable token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            # Scale logits by temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                logits = top_k_filtering(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                logits = top_p_filtering(logits, top_p)

            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

        # Append to running sequence
        idx = torch.cat((idx, idx_next), dim=1)

        # Check for end of sequence
        if eos_id is not None and (idx_next == eos_id).any():
            break

    return idx


def top_k_filtering(logits: torch.Tensor, k: int) -> torch.Tensor:
    """
    Filter logits to only keep the top k values.

    Top-k sampling restricts the vocabulary to the k most likely tokens.
    This prevents the model from sampling unlikely tokens that could
    lead to incoherent text.

    Args:
        logits: Logits of shape (batch, vocab_size)
        k: Number of top tokens to keep

    Returns:
        Filtered logits with non-top-k positions set to -inf
    """
    if k <= 0:
        return logits

    # Find the k-th largest value
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    min_top_k = top_k_values[:, -1].unsqueeze(-1)

    # Mask out tokens below the k-th largest
    filtered_logits = torch.where(
        logits >= min_top_k,
        logits,
        torch.full_like(logits, float('-inf'))
    )

    return filtered_logits


def top_p_filtering(logits: torch.Tensor, p: float) -> torch.Tensor:
    """
    Filter logits using nucleus (top-p) sampling.

    Top-p sampling dynamically selects the number of tokens to consider
    based on their cumulative probability. This is more flexible than
    top-k because it adapts to the shape of the distribution.

    Args:
        logits: Logits of shape (batch, vocab_size)
        p: Cumulative probability threshold (0.0 to 1.0)

    Returns:
        Filtered logits with low-probability positions set to -inf
    """
    if p >= 1.0:
        return logits

    # Sort by descending probability
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find cutoff: first position where cumulative prob exceeds p
    sorted_mask = cumulative_probs > p

    # Shift mask right to include the token that pushed us over p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    # Set filtered positions to -inf
    sorted_logits = sorted_logits.masked_fill(sorted_mask, float('-inf'))

    # Unsort back to original order
    filtered_logits = torch.zeros_like(logits)
    filtered_logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return filtered_logits


def generate_text(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Optional[torch.device] = None,
) -> str:
    """
    High-level text generation function.

    Convenience wrapper that handles tokenization and decoding.

    Args:
        model: GPT model
        tokenizer: Tokenizer instance
        prompt: Text prompt to continue from
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_k: Top-k filtering parameter
        top_p: Top-p (nucleus) filtering parameter
        device: Device to run generation on

    Returns:
        Generated text (including the prompt)
    """
    if device is None:
        device = next(model.parameters()).device

    # Get context size from model
    context_size = model.cfg.context_length

    # Encode prompt
    token_ids = tokenizer.encode(prompt)
    idx = torch.tensor([token_ids], device=device)

    # Generate
    generated_idx = generate(
        model=model,
        idx=idx,
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=tokenizer.eot_token,
    )

    # Decode
    return tokenizer.decode(generated_idx[0].tolist())


def main():
    """CLI entry point for text generation."""
    parser = argparse.ArgumentParser(
        description="Generate text from a trained GPT model"
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        default="The meaning of life is",
        help='Text prompt to continue from'
    )
    parser.add_argument(
        '--max-tokens', '-n',
        type=int,
        default=100,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature', '-t',
        type=float,
        default=1.0,
        help='Sampling temperature (0=greedy, >1=more random)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=None,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--top-p',
        type=float,
        default=None,
        help='Top-p (nucleus) sampling parameter'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to run on (cuda/cpu)'
    )

    args = parser.parse_args()

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Load model
    from .checkpoint import load_checkpoint
    model, _, metadata = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    print(f"Loaded checkpoint from step {metadata['step']}")
    print(f"Config: {metadata['config_name']}")
    print()

    # Create tokenizer
    tokenizer = Tokenizer()

    # Generate text
    print(f"Prompt: {args.prompt}")
    print("-" * 40)

    generated = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(generated)


if __name__ == '__main__':
    main()
