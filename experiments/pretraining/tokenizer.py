"""
BPE Tokenizer Wrapper

This module provides a tokenizer for the GPT pretraining lab using tiktoken,
the same BPE (Byte Pair Encoding) tokenizer used by GPT-2/3/4.

BPE Algorithm Overview:
=======================

Byte Pair Encoding is a subword tokenization algorithm that:
1. Starts with a base vocabulary of individual characters/bytes
2. Iteratively merges the most frequent adjacent pairs
3. Continues until reaching the desired vocabulary size

Why BPE over word-level tokenization?
- Handles unknown words gracefully (breaks into subwords)
- Balances between character-level (tiny vocab, long sequences) and
  word-level (huge vocab, short sequences)
- Naturally handles morphology ("running" -> "run" + "ning")

GPT-2 BPE Specifics:
- Vocabulary size: 50,257 tokens
- Includes special handling for spaces (represented as "Ġ" in the vocab)
- Works at the byte level, so can encode ANY Unicode text

Reference:
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (2016)
- Radford et al., "Language Models Are Unsupervised Multitask Learners" (2019)
"""

import tiktoken
from typing import List, Union, Optional
import re


class Tokenizer:
    """
    Wrapper around tiktoken for GPT-2 style tokenization.

    This tokenizer uses BPE (Byte Pair Encoding) to convert text to token IDs
    and back. The vocabulary is fixed at 50,257 tokens, matching GPT-2.

    Key properties:
    - Deterministic: Same text always produces same tokens
    - Lossless: decode(encode(text)) == text
    - Handles all Unicode characters (byte-level BPE)

    Example:
        >>> tokenizer = Tokenizer()
        >>> ids = tokenizer.encode("Hello, world!")
        >>> print(ids)
        [15496, 11, 995, 0]
        >>> text = tokenizer.decode(ids)
        >>> print(text)
        'Hello, world!'
    """

    def __init__(self, encoding_name: str = "gpt2"):
        """
        Initialize the tokenizer.

        Args:
            encoding_name: Name of the tiktoken encoding to use.
                          'gpt2' gives us the GPT-2 vocabulary (50,257 tokens).
                          Other options include 'cl100k_base' (GPT-4 style, 100k tokens).
        """
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.encoding_name = encoding_name

    @property
    def vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.encoding.n_vocab

    @property
    def eot_token(self) -> int:
        """Return the end-of-text token ID."""
        return self.encoding.eot_token

    def encode(
        self,
        text: str,
        allowed_special: Union[str, set] = set(),
        disallowed_special: Union[str, set] = "all"
    ) -> List[int]:
        """
        Encode text to a list of token IDs.

        BPE tokenization works by:
        1. Pre-tokenizing text into chunks (roughly words + punctuation)
        2. For each chunk, finding the optimal BPE merge sequence
        3. Converting subword units to token IDs

        Args:
            text: Input text string to tokenize
            allowed_special: Special tokens to allow (e.g., {"<|endoftext|>"})
            disallowed_special: Special tokens to disallow (raises error if found)

        Returns:
            List of integer token IDs

        Example:
            >>> t = Tokenizer()
            >>> t.encode("Hello world")
            [15496, 995]
        """
        return self.encoding.encode(
            text,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special
        )

    def decode(self, token_ids: List[int]) -> str:
        """
        Decode a list of token IDs back to text.

        This is the inverse of encode() and is lossless:
        decode(encode(text)) == text

        Args:
            token_ids: List of integer token IDs

        Returns:
            Decoded text string

        Example:
            >>> t = Tokenizer()
            >>> t.decode([15496, 995])
            'Hello world'
        """
        return self.encoding.decode(token_ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """
        Encode multiple texts efficiently.

        Args:
            texts: List of text strings

        Returns:
            List of token ID lists
        """
        return [self.encode(text) for text in texts]

    def decode_batch(self, token_ids_batch: List[List[int]]) -> List[str]:
        """
        Decode multiple token ID sequences.

        Args:
            token_ids_batch: List of token ID lists

        Returns:
            List of decoded text strings
        """
        return [self.decode(ids) for ids in token_ids_batch]

    def get_token_str(self, token_id: int) -> str:
        """
        Get the string representation of a single token.

        Useful for debugging and visualization.

        Args:
            token_id: Integer token ID

        Returns:
            String representation of the token
        """
        return self.decode([token_id])

    def tokenize(self, text: str) -> List[str]:
        """
        Split text into token strings (not IDs).

        This is useful for visualizing how text is tokenized.

        Args:
            text: Input text

        Returns:
            List of token strings

        Example:
            >>> t = Tokenizer()
            >>> t.tokenize("Hello, world!")
            ['Hello', ',', ' world', '!']
        """
        token_ids = self.encode(text)
        return [self.get_token_str(tid) for tid in token_ids]

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text.

        Args:
            text: Input text

        Returns:
            Number of tokens
        """
        return len(self.encode(text))

    def truncate_to_max_tokens(
        self,
        text: str,
        max_tokens: int,
        truncate_from: str = "end"
    ) -> str:
        """
        Truncate text to a maximum number of tokens.

        Args:
            text: Input text
            max_tokens: Maximum number of tokens
            truncate_from: "end" to truncate from the end,
                          "start" to truncate from the beginning

        Returns:
            Truncated text
        """
        token_ids = self.encode(text)
        if len(token_ids) <= max_tokens:
            return text

        if truncate_from == "end":
            truncated_ids = token_ids[:max_tokens]
        elif truncate_from == "start":
            truncated_ids = token_ids[-max_tokens:]
        else:
            raise ValueError(f"truncate_from must be 'start' or 'end', got {truncate_from}")

        return self.decode(truncated_ids)

    def __repr__(self) -> str:
        return f"Tokenizer(encoding='{self.encoding_name}', vocab_size={self.vocab_size})"


def demonstrate_bpe():
    """
    Demonstrate BPE tokenization with examples.

    This function shows how the tokenizer breaks text into subwords
    and how it handles different types of input.
    """
    tokenizer = Tokenizer()

    print("=" * 60)
    print("BPE Tokenization Demonstration")
    print("=" * 60)

    examples = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-2 uses BPE tokenization.",
        "Antidisestablishmentarianism",  # Long word splits into subwords
        "TensorFlow",  # Camel case
        "12345",  # Numbers
        "你好世界",  # Chinese characters
        "🎉",  # Emoji
        "   spaces   ",  # Whitespace handling
    ]

    for text in examples:
        token_ids = tokenizer.encode(text)
        tokens = tokenizer.tokenize(text)
        decoded = tokenizer.decode(token_ids)

        print(f"\nOriginal: {repr(text)}")
        print(f"Tokens:   {tokens}")
        print(f"IDs:      {token_ids}")
        print(f"Decoded:  {repr(decoded)}")
        print(f"Lossless: {text == decoded}")

    print("\n" + "=" * 60)
    print(f"Tokenizer: {tokenizer}")
    print(f"EOT token: {tokenizer.eot_token}")
    print("=" * 60)


if __name__ == '__main__':
    demonstrate_bpe()
