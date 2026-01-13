"""
LLM Pretraining Lab

A hands-on platform for understanding LLM pretraining from scratch,
featuring a GPT implementation with real-time visualization.

This is a LEARNING PROJECT - the goal is mental model formation,
not producing a capable language model.

Reference: Raschka, "Build a Large Language Model (From Scratch)"
See: docs/book-chapters/text/04-implementing-gpt-model.txt
     docs/book-chapters/text/05-pretraining-on-unlabeled-data.txt

Usage:
    # Training (CLI)
    python -m experiments.pretraining.train --config nano --epochs 10

    # Start API server
    python -m experiments.pretraining.api.main

    # Frontend (from experiments/pretraining/frontend/)
    npm run dev

Quick Start:
    >>> from experiments.pretraining import GPTModel, Tokenizer, get_dataloader
    >>> from experiments.pretraining.config import GPT_CONFIGS
    >>>
    >>> # Create a nano model (~10M params)
    >>> model = GPTModel(GPT_CONFIGS['nano'])
    >>> tokenizer = Tokenizer()
    >>>
    >>> # Get data loader
    >>> dl = get_dataloader('verdict', batch_size=4, context_length=128)
"""

# Core model components
from .model import (
    GPTModel,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    LayerNorm,
    GELU,
    generate_text_simple,
    create_model,
)

# Configuration
from .config import (
    GPTConfig,
    GPT_CONFIGS,
    get_config,
)

# Tokenization
from .tokenizer import Tokenizer

# Data pipeline
from .data import (
    GPTDataset,
    TextFileDataset,
    get_dataloader,
    create_train_val_dataloaders,
    create_sample_corpus,
)

# Training
from .train import (
    train_model,
    calc_loss_batch,
    calc_loss_loader,
    get_lr_scheduler,
)

# Checkpointing
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    list_checkpoints,
    get_latest_checkpoint,
)

# Generation
from .generate import (
    generate,
    generate_text,
    top_k_filtering,
    top_p_filtering,
)

__all__ = [
    # Model
    'GPTModel',
    'TransformerBlock',
    'MultiHeadAttention',
    'FeedForward',
    'LayerNorm',
    'GELU',
    'generate_text_simple',
    'create_model',
    # Config
    'GPTConfig',
    'GPT_CONFIGS',
    'get_config',
    # Tokenizer
    'Tokenizer',
    # Data
    'GPTDataset',
    'TextFileDataset',
    'get_dataloader',
    'create_train_val_dataloaders',
    'create_sample_corpus',
    # Training
    'train_model',
    'calc_loss_batch',
    'calc_loss_loader',
    'get_lr_scheduler',
    # Checkpointing
    'save_checkpoint',
    'load_checkpoint',
    'list_checkpoints',
    'get_latest_checkpoint',
    # Generation
    'generate',
    'generate_text',
    'top_k_filtering',
    'top_p_filtering',
]

__version__ = '0.1.0'
