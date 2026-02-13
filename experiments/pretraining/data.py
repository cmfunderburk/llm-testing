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
import numpy as np

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
        token_ids: Union[List[int], np.ndarray],
        context_length: int,
        stride: int = 1
    ):
        """
        Initialize the dataset.

        Args:
            token_ids: Token IDs as numpy array (preferred) or list
            context_length: Number of tokens in each training sequence
            stride: Step size for sliding window (1 = maximum overlap)
        """
        # Convert to tensor efficiently - avoid Python list intermediate
        if isinstance(token_ids, np.ndarray):
            # Keep as int32 to avoid doubling memory (2.75B tokens = 11GB vs 22GB)
            # PyTorch embedding layers work fine with int32 inputs
            if token_ids.dtype != np.int32:
                token_ids = token_ids.astype(np.int32, copy=False)
            self.token_ids = torch.from_numpy(np.ascontiguousarray(token_ids))
        else:
            # Fallback for lists (legacy) - use int32 for consistency
            self.token_ids = torch.tensor(token_ids, dtype=torch.int32)

        self.context_length = context_length
        self.stride = stride

        # Calculate number of samples
        # We need context_length + 1 tokens for each sample (input + 1 target)
        total_len = len(self.token_ids)
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
        # Convert to int64 (long) only for the small batch slice, not full dataset
        input_ids = tokens[:-1].long()
        # Target is last context_length tokens (shifted by 1)
        labels = tokens[1:].long()

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

        # Convert to numpy array for memory efficiency
        self.token_ids = np.array(tokenizer.encode(text), dtype=np.int32)
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
# Corpus Registry and Caching
# =============================================================================

# Directory where corpus files are stored
CORPUS_DIR = Path(__file__).parent / "corpus"
CACHE_DIR = CORPUS_DIR / ".cache"

# Registry of available corpora
# Built-in (small, for testing):
#   - verdict: Short story by Edith Wharton (~8KB)
#   - tiny: A few sentences (~350 bytes) - too small for nano config
# Downloadable (run: python -m experiments.pretraining.download_corpora --all):
#   - tinystories: 2.1M synthetic stories for training small LMs (~500MB)
#   - wikitext2: Wikipedia articles (~12MB)
#   - shakespeare: Complete works of Shakespeare (~1MB)
#   - pg19_train/pg19_validation/pg19_test: Official PG-19 splits (~11GB total)
#   - pg19_*_small: Small subset (100 books per split) for testing
# Prepared locally (run: python experiments/pretraining/prepare_wikipedia_ga.py):
#   - wikipedia_ga_intros: Introductions from 50K+ Good Article+ Wikipedia articles (~65MB)
CORPUS_REGISTRY = {
    'verdict': 'verdict.txt',
    'tiny': 'tiny.txt',
    'wikipedia_ga_intros': 'wikipedia-ga-intros.txt',
    'tinystories': 'tinystories.txt',
    'wikitext2': 'wikitext2.txt',
    'shakespeare': 'shakespeare.txt',
    'pg19_train': 'pg19_train.txt',
    'pg19_validation': 'pg19_validation.txt',
    'pg19_test': 'pg19_test.txt',
    'pg19_train_small': 'pg19_train_small.txt',
    'pg19_validation_small': 'pg19_validation_small.txt',
    'pg19_test_small': 'pg19_test_small.txt',
}


def get_cache_path(corpus_name: str, encoding_name: str = "gpt2") -> Path:
    """Get the cache file path for a tokenized corpus."""
    # Normalize corpus name for cache key
    safe_name = Path(corpus_name).stem if os.path.exists(corpus_name) else corpus_name
    return CACHE_DIR / f"{safe_name}.{encoding_name}.tokens.pt"


def load_cached_tokens(corpus_path: Path, encoding_name: str = "gpt2") -> Optional[np.ndarray]:
    """
    Load cached tokenized corpus if available and valid.

    Returns None if cache doesn't exist or is stale (corpus file was modified).
    Returns numpy array (int32) for memory efficiency.
    """
    corpus_name = corpus_path.stem
    cache_path = get_cache_path(corpus_name, encoding_name)

    if not cache_path.exists():
        return None

    # Check if cache is stale (corpus modified after cache was created)
    corpus_mtime = corpus_path.stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime

    if corpus_mtime > cache_mtime:
        print(f"Cache stale for {corpus_name}, will re-tokenize")
        return None

    try:
        # Load as tensor then convert to numpy - avoids Python list intermediate
        token_tensor = torch.load(cache_path, weights_only=True)
        token_ids = token_tensor.numpy()
        del token_tensor  # Free tensor memory immediately
        print(f"Loaded {len(token_ids):,} tokens from cache for {corpus_name}")
        return token_ids
    except Exception as e:
        print(f"Failed to load cache: {e}")
        return None


# =============================================================================
# Progress Callback Type
# =============================================================================

from typing import Callable

# Progress callback signature: (phase, bytes_read, total_bytes, tokens_so_far) -> None
ProgressCallback = Callable[[str, int, int, int], None]


def tokenize_corpus_chunked(
    corpus_path: Path,
    tokenizer: "Tokenizer",
    chunk_size_mb: int = 50,
    progress_callback: Optional[ProgressCallback] = None,
) -> np.ndarray:
    """
    Tokenize a corpus file in chunks to avoid memory explosion.

    For large files (e.g., 11GB PG-19), loading the entire file into memory
    and tokenizing at once can require 30-50GB+ of RAM. This function:
    1. Reads the file in chunks (default 50MB)
    2. Tokenizes each chunk
    3. Accumulates tokens in a numpy array (much more memory-efficient than Python lists)

    Args:
        corpus_path: Path to the corpus file
        tokenizer: Tokenizer instance
        chunk_size_mb: Size of each chunk in megabytes
        progress_callback: Optional callback for progress updates
                          Called with (phase, bytes_read, total_bytes, tokens_so_far)

    Returns:
        Numpy array of token IDs (int32)
    """
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    total_size = corpus_path.stat().st_size

    # Pre-allocate numpy array (estimate ~4 chars per token for English text)
    estimated_tokens = total_size // 4
    tokens_array = np.zeros(estimated_tokens, dtype=np.int32)
    token_count = 0

    bytes_read = 0

    with open(corpus_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size_bytes)
            if not chunk:
                break

            bytes_read += len(chunk.encode('utf-8'))

            # Tokenize this chunk
            chunk_tokens = tokenizer.encode(chunk)

            # Ensure array is large enough
            needed_size = token_count + len(chunk_tokens)
            if needed_size > len(tokens_array):
                # Grow array by 50%
                new_size = max(needed_size, int(len(tokens_array) * 1.5))
                new_array = np.zeros(new_size, dtype=np.int32)
                new_array[:token_count] = tokens_array[:token_count]
                tokens_array = new_array

            # Append tokens
            tokens_array[token_count:token_count + len(chunk_tokens)] = chunk_tokens
            token_count += len(chunk_tokens)

            if progress_callback:
                progress_callback("tokenizing", bytes_read, total_size, token_count)

    # Trim to actual size and return contiguous array
    return np.ascontiguousarray(tokens_array[:token_count])


def save_cached_tokens(corpus_name: str, token_ids: Union[List[int], np.ndarray], encoding_name: str = "gpt2"):
    """Save tokenized corpus to cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = get_cache_path(corpus_name, encoding_name)

    # Convert to tensor efficiently for storage
    if isinstance(token_ids, np.ndarray):
        # Zero-copy conversion from numpy
        token_tensor = torch.from_numpy(token_ids.astype(np.int32, copy=False))
    else:
        # Fallback for lists (legacy)
        token_tensor = torch.tensor(token_ids, dtype=torch.int32)

    torch.save(token_tensor, cache_path)

    size_mb = cache_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(token_ids):,} tokens to cache ({size_mb:.1f} MB)")


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


def _load_corpus_tokens(
    corpus: str,
    tokenizer,
    encoding_name: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> np.ndarray:
    """
    Load and tokenize a corpus, using cache if available.

    Args:
        corpus: Corpus name or path
        tokenizer: Tokenizer instance
        encoding_name: Name of the encoding (for cache key)
        progress_callback: Optional callback for progress updates

    Returns:
        Numpy array of token IDs (int32) for memory efficiency
    """
    corpus_path = get_corpus_path(corpus)

    # Try to load from cache first
    if progress_callback:
        progress_callback("checking_cache", 0, 0, 0)

    token_ids = load_cached_tokens(corpus_path, encoding_name)

    if token_ids is not None:
        if progress_callback:
            progress_callback("loaded_from_cache", 0, 0, len(token_ids))
        return token_ids

    # Cache miss - tokenize the file
    file_size = corpus_path.stat().st_size
    file_size_mb = file_size / (1024 * 1024)

    # Use chunked tokenization for files > 100MB to avoid memory issues
    if file_size_mb > 100:
        print(f"Tokenizing {corpus_path.name} ({file_size_mb:.1f} MB) in chunks...")
        token_ids = tokenize_corpus_chunked(
            corpus_path,
            tokenizer,
            chunk_size_mb=50,
            progress_callback=progress_callback,
        )
    else:
        # Small file - load all at once (faster for small files)
        print(f"Tokenizing {corpus_path.name}...")
        if progress_callback:
            progress_callback("tokenizing", 0, file_size, 0)

        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Convert list to numpy array immediately
        token_ids = np.array(tokenizer.encode(text), dtype=np.int32)

        if progress_callback:
            progress_callback("tokenizing", file_size, file_size, len(token_ids))

    print(f"Tokenized into {len(token_ids):,} tokens")

    # Save to cache for next time
    if progress_callback:
        progress_callback("saving_cache", 0, 0, len(token_ids))

    save_cached_tokens(corpus_path.stem, token_ids, encoding_name)

    if progress_callback:
        progress_callback("complete", 0, 0, len(token_ids))

    return token_ids


def create_train_val_dataloaders(
    corpus: str,
    batch_size: int = 4,
    context_length: int = 256,
    stride: Optional[int] = None,
    val_split: float = 0.1,
    val_corpus: Optional[str] = None,
    num_workers: int = 0,
    tokenizer: Optional[Tokenizer] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and validation DataLoaders.

    By default, splits the corpus into training and validation sets based on
    position (not random split, to avoid leakage in sequential data).

    If val_corpus is provided, uses that as a separate validation set instead
    of splitting. This is useful for datasets with official splits like PG-19.

    Args:
        corpus: Name of corpus or path to text file (used for training)
        batch_size: Number of samples per batch
        context_length: Number of tokens per sequence
        stride: Sliding window stride
        val_split: Fraction of data to use for validation (0.1 = 10%)
                   Ignored if val_corpus is provided.
        val_corpus: Optional separate corpus for validation (e.g., 'pg19_validation')
        num_workers: Number of data loading workers
        tokenizer: Tokenizer to use
        progress_callback: Optional callback for tokenization progress updates
                          Called with (phase, bytes_read, total_bytes, tokens_so_far)

    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    if tokenizer is None:
        tokenizer = Tokenizer()

    encoding_name = getattr(tokenizer, 'encoding_name', 'gpt2')

    import gc

    if val_corpus is not None:
        # Use separate validation corpus
        train_tokens = _load_corpus_tokens(corpus, tokenizer, encoding_name, progress_callback)
        val_tokens = _load_corpus_tokens(val_corpus, tokenizer, encoding_name, progress_callback)
    else:
        # Split single corpus into train/val
        token_ids = _load_corpus_tokens(corpus, tokenizer, encoding_name, progress_callback)
        split_idx = int(len(token_ids) * (1 - val_split))
        # Copy to ensure contiguous arrays (numpy slices share memory with original)
        train_tokens = np.ascontiguousarray(token_ids[:split_idx])
        val_tokens = np.ascontiguousarray(token_ids[split_idx:])
        # Free the original array
        del token_ids
        gc.collect()

    # Create datasets
    stride_val = stride if stride is not None else context_length
    train_dataset = GPTDataset(train_tokens, context_length, stride_val)
    # Free numpy array after tensor is created
    del train_tokens
    gc.collect()

    val_dataset = GPTDataset(val_tokens, context_length, stride_val)
    # Free numpy array after tensor is created
    del val_tokens
    gc.collect()

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
