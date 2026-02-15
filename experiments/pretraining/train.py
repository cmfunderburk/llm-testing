"""
Training Module for GPT Pretraining

This module implements the training loop for pretraining GPT models.
Following Chapter 5 of Raschka's "Build a Large Language Model (From Scratch)".

Training Loop Components:
========================

1. Loss Computation:
   - Cross-entropy loss between predicted logits and target token IDs
   - Computed as negative log probability of correct tokens
   - Perplexity = exp(loss) gives interpretable metric

2. Optimization:
   - AdamW optimizer (Adam with decoupled weight decay)
   - Learning rate scheduling with warmup and cosine decay
   - Gradient clipping to prevent exploding gradients

3. Evaluation:
   - Periodic evaluation on validation set
   - Sample generation to qualitatively assess model

4. Checkpointing:
   - Save model and optimizer state periodically
   - OFF by default (per PRD requirement)

CLI Usage:
    python -m experiments.pretraining.train --config nano --epochs 10 --corpus verdict
    python -m experiments.pretraining.train --help

Reference: Raschka, "Build a Large Language Model (From Scratch)", Chapter 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
import math
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime

from .model import GPTModel
from .config import GPTConfig, GPT_CONFIGS, get_config
from .tokenizer import Tokenizer
from .data import get_dataloader, create_train_val_dataloaders
from .checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from .generate import generate_text


# =============================================================================
# Loss Computation
# =============================================================================

def calc_loss_batch(
    input_batch: torch.Tensor,
    target_batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute cross-entropy loss for a single batch.

    Cross-entropy loss measures how well the model's predicted probability
    distribution matches the target distribution. Lower is better.

    For language modeling:
    - Target is the "correct" next token at each position
    - Model outputs logits over vocabulary
    - Loss = -log(probability of correct token)

    Args:
        input_batch: Input token IDs, shape (batch, seq_len)
        target_batch: Target token IDs, shape (batch, seq_len)
        model: GPT model
        device: Device to compute on

    Returns:
        Scalar loss tensor
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)

    # Forward pass: get logits
    logits = model(input_batch)  # (batch, seq_len, vocab_size)

    # Flatten for cross_entropy: (batch * seq_len, vocab_size)
    logits_flat = logits.flatten(0, 1)
    targets_flat = target_batch.flatten()

    # Cross-entropy loss
    # This computes: -log(softmax(logits)[target])
    loss = F.cross_entropy(logits_flat, targets_flat)

    return loss


def calc_loss_loader(
    data_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    num_batches: Optional[int] = None,
) -> float:
    """
    Compute average loss over a data loader.

    Args:
        data_loader: DataLoader to evaluate on
        model: GPT model
        device: Device to compute on
        num_batches: Max batches to evaluate (None = all)

    Returns:
        Average loss value
    """
    total_loss = 0.0

    if len(data_loader) == 0:
        return float('nan')

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break

            input_ids = batch['input_ids']
            labels = batch['labels']
            loss = calc_loss_batch(input_ids, labels, model, device)
            total_loss += loss.item()

    return total_loss / num_batches


# =============================================================================
# Learning Rate Scheduling
# =============================================================================

def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    Learning Rate Schedule:
    1. Warmup phase: LR increases linearly from 0 to peak
    2. Decay phase: LR decreases following cosine curve to min_lr

    Why warmup helps:
    - At the start, model weights are random
    - Large gradients could cause instability
    - Warmup allows gradients to stabilize before full learning

    Why cosine decay:
    - Smooth decay prevents sudden changes
    - Model can make large updates early, fine adjustments later
    - Empirically works well for transformers

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of peak (default 0.1)

    Returns:
        LR scheduler
    """
    def lr_lambda(current_step: int) -> float:
        # Warmup phase: linear increase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Decay phase: cosine decay to min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale to [min_lr_ratio, 1.0]
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# Training Loop
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    num_epochs: int,
    eval_freq: int = 100,
    eval_iters: int = 5,
    save_checkpoints: bool = False,
    checkpoint_freq: int = 1000,
    config: Optional[GPTConfig] = None,
    config_name: str = 'unknown',
    tokenizer: Optional[Tokenizer] = None,
    sample_prompt: str = "Every effort moves you",
    grad_clip: Optional[float] = 1.0,
    callback: Optional[callable] = None,
) -> Dict[str, List]:
    """
    Main training loop for GPT pretraining.

    Training Flow (per step):
    1. Zero gradients from previous step
    2. Forward pass: compute logits
    3. Compute loss (cross-entropy)
    4. Backward pass: compute gradients
    5. Clip gradients (optional, prevents explosion)
    6. Optimizer step: update weights
    7. Scheduler step: update learning rate
    8. Periodic evaluation and checkpointing

    Args:
        model: GPT model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (e.g., AdamW)
        scheduler: LR scheduler (optional)
        device: Device to train on
        num_epochs: Number of training epochs
        eval_freq: Evaluate every N steps
        eval_iters: Number of batches for evaluation
        save_checkpoints: Whether to save checkpoints
        checkpoint_freq: Save checkpoint every N steps
        config: Model config (for checkpointing)
        config_name: Config name (for checkpointing)
        tokenizer: Tokenizer for sample generation
        sample_prompt: Prompt for sample generation
        grad_clip: Max gradient norm (None = no clipping)
        callback: Optional callback(step, metrics) for external logging

    Returns:
        Dictionary with training history:
        - 'train_losses': List of training losses
        - 'val_losses': List of validation losses
        - 'steps': List of step numbers
        - 'tokens_seen': List of total tokens seen
        - 'learning_rates': List of learning rates
    """
    # Training history
    history = {
        'train_losses': [],
        'val_losses': [],
        'steps': [],
        'tokens_seen': [],
        'learning_rates': [],
    }

    # Counters
    global_step = 0
    tokens_seen = 0
    start_time = time.time()

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        for batch in train_loader:
            # Get batch data
            input_ids = batch['input_ids']
            labels = batch['labels']

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass and compute loss
            loss = calc_loss_batch(input_ids, labels, model, device)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # Optimizer step
            optimizer.step()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Update counters
            tokens_seen += input_ids.numel()
            global_step += 1

            # Periodic evaluation
            if global_step % eval_freq == 0:
                train_loss = calc_loss_loader(
                    train_loader, model, device, num_batches=eval_iters
                )
                val_loss = calc_loss_loader(
                    val_loader, model, device, num_batches=eval_iters
                )

                # Get current LR
                current_lr = optimizer.param_groups[0]['lr']

                # Record history
                history['train_losses'].append(train_loss)
                history['val_losses'].append(val_loss)
                history['steps'].append(global_step)
                history['tokens_seen'].append(tokens_seen)
                history['learning_rates'].append(current_lr)

                # Print progress
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_seen / elapsed
                perplexity = math.exp(min(train_loss, 100))  # Cap for display

                print(f"Step {global_step:6d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"PPL: {perplexity:.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Tok/s: {tokens_per_sec:.0f}")

                # Callback for external logging
                if callback is not None:
                    callback(global_step, {
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': current_lr,
                        'tokens_seen': tokens_seen,
                        'perplexity': perplexity,
                    })

                model.train()  # Back to training mode

            # Periodic checkpointing
            if save_checkpoints and global_step % checkpoint_freq == 0:
                train_loss = history['train_losses'][-1] if history['train_losses'] else 0
                val_loss = history['val_losses'][-1] if history['val_losses'] else None

                checkpoint_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    config_name=config_name,
                    optimizer_name="adamw",
                    step=global_step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                )
                print(f"  Saved checkpoint: {checkpoint_path}")

        # End of epoch: generate sample text
        epoch_time = time.time() - epoch_start_time
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} complete ({epoch_time:.1f}s) ---")

        if tokenizer is not None:
            model.eval()
            sample = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=sample_prompt,
                max_new_tokens=50,
                temperature=0.8,
                top_k=40,
                device=device,
            )
            print(f"Sample: {sample[:200]}...")
            model.train()

        print()

    # Final evaluation
    final_train_loss = calc_loss_loader(train_loader, model, device)
    final_val_loss = calc_loss_loader(val_loader, model, device)
    print(f"\nTraining complete!")
    print(f"Final Train Loss: {final_train_loss:.4f}")
    print(f"Final Val Loss: {final_val_loss:.4f}")
    print(f"Total tokens seen: {tokens_seen:,}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    # Save final checkpoint
    if save_checkpoints and config is not None:
        checkpoint_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            config=config,
            config_name=config_name,
            optimizer_name="adamw",
            step=global_step,
            epoch=num_epochs,
            train_loss=final_train_loss,
            val_loss=final_val_loss,
            filename="checkpoint_final.pt",
        )
        print(f"Saved final checkpoint: {checkpoint_path}")

    return history


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(
        description="Train a GPT model from scratch",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='nano',
        choices=list(GPT_CONFIGS.keys()),
        help='Model configuration preset'
    )

    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=4,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr', '--learning-rate',
        type=float,
        default=3e-4,
        help='Peak learning rate'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='Weight decay for AdamW'
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=100,
        help='Number of warmup steps'
    )
    parser.add_argument(
        '--grad-clip',
        type=float,
        default=1.0,
        help='Gradient clipping max norm (0 to disable)'
    )

    # Data
    parser.add_argument(
        '--corpus',
        type=str,
        default='verdict',
        help='Training corpus name or path'
    )
    parser.add_argument(
        '--val-split',
        type=float,
        default=0.1,
        help='Fraction of data for validation'
    )

    # Checkpointing
    parser.add_argument(
        '--save', '-s',
        action='store_true',
        help='Enable checkpoint saving (OFF by default)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Explicitly disable checkpoint saving'
    )
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=500,
        help='Save checkpoint every N steps'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume from checkpoint path'
    )

    # Evaluation
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=50,
        help='Evaluate every N steps'
    )
    parser.add_argument(
        '--eval-iters',
        type=int,
        default=5,
        help='Number of batches for evaluation'
    )

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to train on (cuda/cpu/mps)'
    )

    # Misc
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    print(f"Using device: {device}")

    # Load config
    config = get_config(args.config)
    print(f"Model config: {args.config}")
    print(f"  - Layers: {config.n_layers}")
    print(f"  - Heads: {config.n_heads}")
    print(f"  - Embedding dim: {config.emb_dim}")
    print(f"  - Context length: {config.context_length}")
    print(f"  - Estimated params: {config.estimate_params() / 1e6:.1f}M")

    # Create tokenizer
    tokenizer = Tokenizer()

    # Create data loaders
    print(f"\nLoading corpus: {args.corpus}")
    train_loader, val_loader = create_train_val_dataloaders(
        corpus=args.corpus,
        batch_size=args.batch_size,
        context_length=config.context_length,
        val_split=args.val_split,
        tokenizer=tokenizer,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Calculate total training steps
    total_steps = len(train_loader) * args.epochs
    print(f"Total training steps: {total_steps}")

    # Create model
    model = GPTModel(config)
    model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        model, optimizer, metadata = load_checkpoint(
            args.resume, model, optimizer, device
        )
        start_epoch = metadata.get('epoch', 0)
        print(f"Resumed from step {metadata.get('step', 0)}, epoch {start_epoch}")

    # Determine checkpoint saving
    save_checkpoints = args.save and not args.no_save

    print(f"\nCheckpoint saving: {'ON' if save_checkpoints else 'OFF'}")
    print(f"Gradient clipping: {args.grad_clip if args.grad_clip > 0 else 'disabled'}")

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        eval_freq=args.eval_freq,
        eval_iters=args.eval_iters,
        save_checkpoints=save_checkpoints,
        checkpoint_freq=args.checkpoint_freq,
        config=config,
        config_name=args.config,
        tokenizer=tokenizer,
        sample_prompt="Every effort moves you",
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
    )

    return history


if __name__ == '__main__':
    main()
