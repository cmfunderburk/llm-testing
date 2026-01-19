"""
Data Pipeline for GPT Pretraining

This module implements the data loading and preprocessing pipeline for GPT training.
It handles:
- Loading text corpora from files
- Tokenization using BPE
- Creating input/target pairs for next-token prediction
- Batching with configurable context length

Next-Token Prediction Setup:
============================

For language modeling, we train the model to predict the next token given
the previous tokens. Given a sequence of tokens [t1, t2, t3, t4, t5]:

    Input:  [t1, t2, t3, t4]
    Target: [t2, t3, t4, t5]

The model's output at position i predicts what token should be at position i+1.
This is achieved by shifting the targets by 1 position.

Sliding Window:
===============

For long texts, we use a sliding window approach:
- Cut the tokenized text into chunks of `context_length + 1` tokens
- Use first `context_length` tokens as input, last `context_length` as target
- Optionally use a stride smaller than context_length for overlapping windows

Reference: Chapter 2 and 5 of Raschka's "Build a Large Language Model (From Scratch)"
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
import os

from .tokenizer import Tokenizer


# =============================================================================
# Dataset Classes
# =============================================================================

class GPTDataset(Dataset):
    """
    PyTorch Dataset for GPT pretraining.

    This dataset takes tokenized text and creates input/target pairs
    using a sliding window approach.

    The sliding window creates overlapping sequences, which:
    1. Maximizes use of training data
    2. Allows the model to see tokens in different contexts
    3. Can be controlled via the `stride` parameter

    Example with context_length=4 and stride=2:
        Text tokens: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        Sample 0: input=[1,2,3,4], target=[2,3,4,5]
        Sample 1: input=[3,4,5,6], target=[4,5,6,7]
        Sample 2: input=[5,6,7,8], target=[6,7,8,9]
        ...
    """

    def __init__(
        self,
        token_ids: List[int],
        context_length: int,
        stride: int = 1
    ):
        """
        Initialize the dataset.

        Args:
            token_ids: List of token IDs from the tokenized text
            context_length: Number of tokens in each training sequence
            stride: Step size for sliding window (1 = maximum overlap)
        """
        self.token_ids = torch.tensor(token_ids, dtype=torch.long)
        self.context_length = context_length
        self.stride = stride

        # Calculate number of samples
        # We need context_length + 1 tokens for each sample (input + 1 target)
        total_len = len(token_ids)
        self.n_samples = max(0, (total_len - context_length - 1) // stride + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'input_ids' and 'labels' tensors, both of shape (context_length,)

        The relationship between input and label:
            input_ids:  [t0, t1, t2, ..., t_{n-1}]
            labels:     [t1, t2, t3, ..., t_n]

        So input_ids[i] should predict labels[i] (which is the next token).
        """
        start_idx = idx * self.stride
        end_idx = start_idx + self.context_length + 1

        # Get context_length + 1 tokens
        tokens = self.token_ids[start_idx:end_idx]

        # Input is first context_length tokens
        input_ids = tokens[:-1]
        # Target is last context_length tokens (shifted by 1)
        labels = tokens[1:]

        return {
            'input_ids': input_ids,
            'labels': labels
        }


class TextFileDataset(Dataset):
    """
    Dataset that loads and tokenizes a text file.

    This is a convenience wrapper that combines tokenization
    and the GPT dataset creation.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: Tokenizer,
        context_length: int,
        stride: Optional[int] = None
    ):
        """
        Initialize the dataset from a text file.

        Args:
            file_path: Path to the text file
            tokenizer: Tokenizer instance for encoding text
            context_length: Number of tokens in each training sequence
            stride: Step size for sliding window (default: context_length)
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        # Load and tokenize the text
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        self.token_ids = tokenizer.encode(text)
        self.n_tokens = len(self.token_ids)

        # Create the underlying dataset
        self._dataset = GPTDataset(
            self.token_ids,
            context_length,
            self.stride
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._dataset[idx]

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the dataset."""
        return {
            'file': self.file_path,
            'total_tokens': self.n_tokens,
            'num_samples': len(self),
            'context_length': self.context_length,
            'stride': self.stride,
        }


# =============================================================================
# Corpus Registry
# =============================================================================

# Directory where corpus files are stored
CORPUS_DIR = Path(__file__).parent / "corpus"

# Registry of available corpora
# Built-in (small, for testing):
#   - verdict: Short story by Edith Wharton (~8KB)
#   - tiny: A few sentences (~350 bytes) - too small for nano config
# Downloadable (run: python -m experiments.pretraining.download_corpora --all):
#   - tinystories: 2.1M synthetic stories for training small LMs (~500MB)
#   - wikitext2: Wikipedia articles (~12MB)
#   - shakespeare: Complete works of Shakespeare (~1MB)
CORPUS_REGISTRY = {
    'verdict': 'verdict.txt',
    'tiny': 'tiny.txt',
    'tinystories': 'tinystories.txt',
    'wikitext2': 'wikitext2.txt',
    'shakespeare': 'shakespeare.txt',
}


def get_corpus_path(corpus_name: str) -> Path:
    """
    Get the path to a corpus file.

    Args:
        corpus_name: Name of the corpus ('verdict', 'tiny', etc.)
                    or a direct file path

    Returns:
        Path to the corpus file
    """
    # Check if it's a direct path
    if os.path.exists(corpus_name):
        return Path(corpus_name)

    # Check the registry
    if corpus_name in CORPUS_REGISTRY:
        path = CORPUS_DIR / CORPUS_REGISTRY[corpus_name]
        if path.exists():
            return path
        raise FileNotFoundError(
            f"Corpus '{corpus_name}' is registered but file not found at {path}. "
            f"Please create the corpus file."
        )

    # Try as filename in corpus directory
    path = CORPUS_DIR / corpus_name
    if path.exists():
        return path

    raise ValueError(
        f"Unknown corpus '{corpus_name}'. "
        f"Available: {list(CORPUS_REGISTRY.keys())} or provide a file path."
    )


def list_corpora() -> Dict[str, Dict[str, Union[str, bool]]]:
    """
    List all available corpora.

    Returns:
        Dictionary mapping corpus names to their status and path.
    """
    result = {}
    for name, filename in CORPUS_REGISTRY.items():
        path = CORPUS_DIR / filename
        result[name] = {
            'file': filename,
            'exists': path.exists(),
            'path': str(path),
        }
    return result


# =============================================================================
# DataLoader Factory
# =============================================================================

def get_dataloader(
    corpus: str,
    batch_size: int = 4,
    context_length: int = 256,
    stride: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    tokenizer: Optional[Tokenizer] = None,
) -> DataLoader:
    """
    Create a DataLoader for GPT pretraining.

    This is the main entry point for getting training data.

    Args:
        corpus: Name of corpus ('verdict', 'tiny') or path to text file
        batch_size: Number of samples per batch
        context_length: Number of tokens per sequence
        stride: Sliding window stride (default: context_length, no overlap)
        shuffle: Whether to shuffle the data
        num_workers: Number of data loading workers
        tokenizer: Tokenizer to use (default: GPT-2 tokenizer)

    Returns:
        PyTorch DataLoader

    Example:
        >>> dl = get_dataloader('verdict', batch_size=8, context_length=128)
        >>> batch = next(iter(dl))
        >>> print(batch['input_ids'].shape)  # (8, 128)
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    corpus_path = get_corpus_path(corpus)

    dataset = TextFileDataset(
        file_path=str(corpus_path),
        tokenizer=tokenizer,
        context_length=context_length,
        stride=stride
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,  # Drop incomplete batches for consistent batch size
    )


def create_train_val_dataloaders(
    corpus: str,
    batch_size: int = 4,
    context_length: int = 256,
    stride: Optional[int] = None,
    val_split: float = 0.1,
    num_workers: int = 0,
    tokenizer: Optional[Tokenizer] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    Splits the corpus into training and validation sets based on position
    (not random split, to avoid leakage in sequential data).

    Args:
        corpus: Name of corpus or path to text file
        batch_size: Number of samples per batch
        context_length: Number of tokens per sequence
        stride: Sliding window stride
        val_split: Fraction of data to use for validation (0.1 = 10%)
        num_workers: Number of data loading workers
        tokenizer: Tokenizer to use

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    corpus_path = get_corpus_path(corpus)

    # Load and tokenize the full text
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()

    token_ids = tokenizer.encode(text)

    # Split into train and validation
    split_idx = int(len(token_ids) * (1 - val_split))
    train_tokens = token_ids[:split_idx]
    val_tokens = token_ids[split_idx:]

    # Create datasets
    stride_val = stride if stride is not None else context_length
    train_dataset = GPTDataset(train_tokens, context_length, stride_val)
    val_dataset = GPTDataset(val_tokens, context_length, stride_val)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )

    return train_loader, val_loader


# =============================================================================
# Utility Functions
# =============================================================================

def inspect_batch(batch: Dict[str, torch.Tensor], tokenizer: Optional[Tokenizer] = None):
    """
    Pretty-print a batch for debugging.

    Args:
        batch: Batch dictionary from dataloader
        tokenizer: Tokenizer for decoding (optional)
    """
    input_ids = batch['input_ids']
    labels = batch['labels']

    print(f"Batch size: {input_ids.shape[0]}")
    print(f"Sequence length: {input_ids.shape[1]}")
    print(f"Input IDs shape: {input_ids.shape}")
    print(f"Labels shape: {labels.shape}")

    if tokenizer is not None:
        print("\nFirst sample:")
        print(f"  Input:  {tokenizer.decode(input_ids[0].tolist()[:50])}...")
        print(f"  Target: {tokenizer.decode(labels[0].tolist()[:50])}...")


# =============================================================================
# Corpus Creation Utilities
# =============================================================================

def ensure_corpus_dir():
    """Create the corpus directory if it doesn't exist."""
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)


def create_sample_corpus():
    """
    Create sample corpus files for testing.

    This creates small test corpora that can be used for quick iterations.
    """
    ensure_corpus_dir()

    # Tiny corpus - just a few sentences
    tiny_text = """The quick brown fox jumps over the lazy dog.
Machine learning is transforming the world of artificial intelligence.
Large language models can generate human-like text.
Training neural networks requires large amounts of data.
Transformers revolutionized natural language processing.
Attention mechanisms allow models to focus on relevant parts of the input.
"""

    tiny_path = CORPUS_DIR / "tiny.txt"
    with open(tiny_path, 'w', encoding='utf-8') as f:
        f.write(tiny_text)

    # Verdict corpus - a longer sample text
    verdict_text = """The Project Gutenberg eBook of The Verdict, by Edith Wharton

This eBook is for the use of anyone anywhere in the United States and
most other parts of the world at no cost and with almost no restrictions
whatsoever. You may copy it, give it away or re-use it under the terms
of the Project Gutenberg License included with this eBook or online at
www.gutenberg.org. If you are not located in the United States, you
will have to check the laws of the country where you are located before
using this eBook.

Title: The Verdict

Author: Edith Wharton

Release Date: October 1, 2004 [eBook #13561]
[Most recently updated: July 17, 2021]

Language: English

*** START OF THE PROJECT GUTENBERG EBOOK THE VERDICT ***

THE VERDICT


I

Jack Dodd, the art critic, on his way to Dodd's New York studio, had
met Mark Dodd's servant on the ground floor of the building and found
the man somewhat pale.

"I'm glad you've come, Mr. Dodd," the servant said. "Mr. Dodd has
been very much upset by a letter."

"A letter? What kind of letter?" The critic's curiosity was instantly
alert.

The servant hesitated. "I can't say, sir; but it seems to have disturbed
him."

Jack Dodd went up the stairs thoughtfully. His brother had not been
well of late, and the critic was easily anxious about him.

The studio door was open, and Jack walked in without knocking. The big
lofty room, with its north light and its crowded canvases, was empty;
but from the inner room, which served as a combination of bed-room
and breakfast-room, came the sound of agitated walking.

"Hello, Mark!" Jack called out.

The walking stopped, and Mark Dodd appeared in the doorway. He was a
tall thin man with a fine head and nervous hands.

"Hello, Jack! I'm glad you've come. I've got something to show you."

"So I understand. Peters tells me you've had a disturbing letter."

"A letter? Oh yes, that. But something else has happened since then.
Come in here."

Jack followed him into the inner room, and Mark pointed to the wall
above the fireplace. On it hung a small picture—a portrait apparently
—which Jack did not remember to have seen there before.

"When did that come?" he asked.

"This morning. It was delivered just after the letter. And that's what
I want you to see."

Jack walked up to the picture and examined it curiously. It represented
a young woman, in the simple dress of a bygone day, seated in a high
carved chair against a background of dark drapery.

"A pleasant little thing," he said. "French, I should say, about 1820.
Who is she?"

Mark Dodd laughed nervously. "That's just the point. I don't know.
I've never seen her before. And yet there's something about her that
seems familiar."

"Familiar? In what way?"

Mark hesitated. "I can't tell you. It's just an impression—a feeling.
But when I look at that face I seem to be trying to remember something."

Jack scrutinized the portrait more closely. The face was gentle and
appealing, with soft dark eyes and a sensitive mouth.

"I don't see anything remarkable about it," he said at length. "A
pretty girl of the period—one of a hundred. Who sent it to you?"

"That's the odd part. There's no indication. It came by express, with
my name and address, but no sender's card. And the letter that came
before it was signed only with initials that I don't recognize."

"What did the letter say?"

"Very little. Just a few lines asking me to examine this portrait
carefully and tell the writer what I think of it. I'm to address my
answer to a post office box."

Jack smiled. "Some crank who wants a free opinion."

"Perhaps. But there's something about the whole business that makes
me uneasy. Why this mysterious approach? Why send the picture without
any explanation?"

"Publicity scheme, perhaps. Or someone trying to sell a picture and
thinking this is a clever way to arouse your interest."

"Maybe. But I wish I could remember where I've seen that face before."

The brothers stood side by side, gazing at the portrait. The painted
eyes looked back at them with their soft inscrutable expression.

"Well," said Jack at length, "what are you going to do about it?"

"I'm going to study it. I'm going to try to find out who she was, and
why her face haunts me. There's a mystery here, and I mean to solve it."
"""

    verdict_path = CORPUS_DIR / "verdict.txt"
    with open(verdict_path, 'w', encoding='utf-8') as f:
        f.write(verdict_text)

    print(f"Created corpus files in {CORPUS_DIR}")
    print(f"  - tiny.txt: {len(tiny_text)} chars")
    print(f"  - verdict.txt: {len(verdict_text)} chars")


if __name__ == '__main__':
    # Create sample corpora
    create_sample_corpus()

    # Test the data pipeline
    print("\n" + "=" * 60)
    print("Testing Data Pipeline")
    print("=" * 60)

    tokenizer = Tokenizer()

    # Test with tiny corpus
    try:
        dl = get_dataloader(
            'tiny',
            batch_size=2,
            context_length=32,
            stride=16
        )

        print(f"\nDataLoader created with {len(dl)} batches")

        batch = next(iter(dl))
        print("\nBatch contents:")
        inspect_batch(batch, tokenizer)

    except Exception as e:
        print(f"Error: {e}")

    # List available corpora
    print("\n" + "=" * 60)
    print("Available Corpora")
    print("=" * 60)
    for name, info in list_corpora().items():
        status = "OK" if info['exists'] else "MISSING"
        print(f"  {name}: {status} ({info['file']})")
