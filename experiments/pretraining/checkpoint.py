"""
Checkpoint Management for GPT Pretraining

This module handles saving and loading model checkpoints during training.

Checkpoints save:
- Model weights (state_dict)
- Optimizer state (optional, for resuming training)
- Training metadata (step, epoch, loss, config)

Design Decisions:
- Checkpoint storage is OFF by default (per PRD requirement)
- Checkpoints are saved to outputs/pretraining/<config_name>/
- Supports both full checkpoints (with optimizer) and model-only saves

Reference: Chapter 5 of Raschka's "Build a Large Language Model (From Scratch)"
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
from datetime import datetime

from .config import GPTConfig, GPT_CONFIGS


# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "outputs" / "pretraining"


def get_checkpoint_dir(
    config_name: str,
    corpus: Optional[str] = None,
    batch_size: Optional[int] = None,
    context_length: Optional[int] = None,
    base_dir: Optional[Path] = None,
) -> Path:
    """
    Get the checkpoint directory for a given configuration.

    Args:
        config_name: Name of the model config ('nano', 'small', 'medium')
        corpus: Corpus name (e.g., 'verdict', 'tinystories')
        batch_size: Batch size used for training
        context_length: Context length used for training
        base_dir: Optional base directory (default: outputs/pretraining/)

    Returns:
        Path to checkpoint directory
    """
    if base_dir is None:
        base_dir = DEFAULT_CHECKPOINT_DIR

    # If training params provided, use descriptive directory name
    if corpus and batch_size and context_length:
        dir_name = f"{config_name}_{corpus}_b{batch_size}_ctx{context_length}"
        return base_dir / dir_name

    # Legacy: just config_name
    return base_dir / config_name


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    config: GPTConfig,
    config_name: str,
    step: int,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    checkpoint_dir: Optional[Path] = None,
    filename: Optional[str] = None,
    save_optimizer: bool = True,
    corpus: Optional[str] = None,
    batch_size: Optional[int] = None,
    context_length: Optional[int] = None,
) -> Path:
    """
    Save a training checkpoint.

    Checkpoints contain everything needed to resume training or load
    a trained model for inference.

    What gets saved and why:
    - model.state_dict(): All learned weights - required for inference
    - optimizer.state_dict(): Optimizer momentum/velocity - needed to resume training
    - config: Model architecture - required to reconstruct model
    - metadata: Training progress - useful for logging/analysis

    Args:
        model: GPT model to save
        optimizer: Optimizer (optional, but required to resume training)
        config: Model configuration
        config_name: Name of config preset used
        step: Current training step
        epoch: Current epoch
        train_loss: Current training loss
        val_loss: Current validation loss (optional)
        checkpoint_dir: Directory to save checkpoints
        filename: Custom filename (default: checkpoint_step_{step}.pt)
        save_optimizer: Whether to save optimizer state
        corpus: Corpus name used for training
        batch_size: Batch size used for training
        context_length: Context length used for training

    Returns:
        Path to saved checkpoint file
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(
            config_name,
            corpus=corpus,
            batch_size=batch_size,
            context_length=context_length,
        )

    # Create directory if it doesn't exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Build checkpoint filename
    if filename is None:
        filename = f"checkpoint_step_{step:06d}.pt"

    checkpoint_path = checkpoint_dir / filename

    # Build checkpoint dictionary
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config.to_dict(),
        'config_name': config_name,
        'step': step,
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'timestamp': datetime.now().isoformat(),
        # Training parameters for identification
        'corpus': corpus,
        'batch_size': batch_size,
        'context_length': context_length,
    }

    # Optionally save optimizer state
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Also save a "latest" symlink/copy for easy access
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Dict[str, Any]]:
    """
    Load a checkpoint.

    This function can either:
    1. Load weights into an existing model (if model is provided)
    2. Create a new model from the checkpoint config (if model is None)

    Args:
        checkpoint_path: Path to checkpoint file
        model: Existing model to load weights into (optional)
        optimizer: Existing optimizer to load state into (optional)
        device: Device to load model to (default: CPU)

    Returns:
        Tuple of (model, optimizer, metadata)
        - model: Loaded model (either the provided one or newly created)
        - optimizer: Optimizer with loaded state (or None if not in checkpoint)
        - metadata: Dictionary with training metadata (step, epoch, loss, etc.)
    """
    if device is None:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create model if not provided
    if model is None:
        from .model import GPTModel
        config = GPTConfig.from_dict(checkpoint['config'])
        model = GPTModel(config)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Load optimizer state if available and optimizer provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Extract metadata
    metadata = {
        'config': checkpoint.get('config'),
        'config_name': checkpoint.get('config_name'),
        'step': checkpoint.get('step', 0),
        'epoch': checkpoint.get('epoch', 0),
        'train_loss': checkpoint.get('train_loss'),
        'val_loss': checkpoint.get('val_loss'),
        'timestamp': checkpoint.get('timestamp'),
        'corpus': checkpoint.get('corpus'),
        'batch_size': checkpoint.get('batch_size'),
        'context_length': checkpoint.get('context_length'),
    }

    return model, optimizer, metadata


def list_checkpoints(
    config_name: str,
    checkpoint_dir: Optional[Path] = None,
) -> list:
    """
    List all available checkpoints for a configuration.

    Searches both legacy directories (just config_name) and new directories
    (config_name_corpus_bN_ctxN format).

    Args:
        config_name: Name of model config
        checkpoint_dir: Optional checkpoint directory

    Returns:
        List of checkpoint info dictionaries, sorted by step
    """
    base_dir = checkpoint_dir if checkpoint_dir else DEFAULT_CHECKPOINT_DIR

    if not base_dir.exists():
        return []

    checkpoints = []

    # Find all directories matching this config_name (both legacy and new format)
    # Legacy: outputs/pretraining/nano/
    # New: outputs/pretraining/nano_verdict_b4_ctx256/
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        # Match if directory name starts with config_name (either exact or with _)
        if subdir.name != config_name and not subdir.name.startswith(f"{config_name}_"):
            continue

        # Search for all checkpoint patterns:
        # - checkpoint_step_*.pt (auto-saves at 25%, 50%, 75%)
        # - checkpoint_manual_*.pt (manual saves)
        # - checkpoint_final.pt (completion checkpoint)
        patterns = ["checkpoint_step_*.pt", "checkpoint_manual_*.pt", "checkpoint_final.pt"]
        seen_paths = set()

        for pattern in patterns:
            for path in subdir.glob(pattern):
                if path in seen_paths:
                    continue
                seen_paths.add(path)
                try:
                    # Load just the metadata (not the full model)
                    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                    checkpoints.append({
                        'path': str(path),
                        'step': checkpoint.get('step', 0),
                        'epoch': checkpoint.get('epoch', 0),
                        'train_loss': checkpoint.get('train_loss'),
                        'val_loss': checkpoint.get('val_loss'),
                        'timestamp': checkpoint.get('timestamp'),
                        'corpus': checkpoint.get('corpus'),
                        'batch_size': checkpoint.get('batch_size'),
                        'context_length': checkpoint.get('context_length'),
                    })
                except Exception as e:
                    print(f"Warning: Could not load checkpoint {path}: {e}")

    # Sort by step
    checkpoints.sort(key=lambda x: x['step'])
    return checkpoints


def get_latest_checkpoint(
    config_name: str,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[str]:
    """
    Get path to the latest checkpoint for a configuration.

    Searches across all checkpoint directories for this config_name.

    Args:
        config_name: Name of model config
        checkpoint_dir: Optional base checkpoint directory

    Returns:
        Path to latest checkpoint, or None if no checkpoints exist
    """
    # Use list_checkpoints which now searches all matching directories
    checkpoints = list_checkpoints(config_name, checkpoint_dir)
    if checkpoints:
        return checkpoints[-1]['path']

    return None


def delete_old_checkpoints(
    config_name: str,
    keep_last: int = 5,
    checkpoint_dir: Optional[Path] = None,
) -> int:
    """
    Delete old checkpoints, keeping only the most recent ones.

    Args:
        config_name: Name of model config
        keep_last: Number of checkpoints to keep
        checkpoint_dir: Optional checkpoint directory

    Returns:
        Number of checkpoints deleted
    """
    if checkpoint_dir is None:
        checkpoint_dir = get_checkpoint_dir(config_name)

    checkpoints = list_checkpoints(config_name, checkpoint_dir)

    if len(checkpoints) <= keep_last:
        return 0

    # Delete oldest checkpoints
    deleted = 0
    for ckpt in checkpoints[:-keep_last]:
        path = Path(ckpt['path'])
        if path.exists():
            path.unlink()
            deleted += 1

    return deleted


if __name__ == '__main__':
    print("Checkpoint functions available:")
    print("  - save_checkpoint(model, optimizer, config, ...)")
    print("  - load_checkpoint(path, model=None, optimizer=None, device=None)")
    print("  - list_checkpoints(config_name)")
    print("  - get_latest_checkpoint(config_name)")
    print("  - delete_old_checkpoints(config_name, keep_last=5)")
