"""
Optimizer utilities for pretraining.

Provides a single factory for selecting between standard AdamW and
bitsandbytes 8-bit variants used for memory-efficient training.
"""

from typing import Iterable, Literal

import torch


PretrainingOptimizerName = Literal["adamw", "adamw_8bit", "paged_adamw_8bit"]


def create_pretraining_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    optimizer_name: PretrainingOptimizerName,
    lr: float,
    weight_decay: float = 0.1,
    device: torch.device | None = None,
) -> torch.optim.Optimizer:
    """
    Create the configured optimizer for pretraining.

    Raises:
        RuntimeError: If an 8-bit optimizer is requested without CUDA or when
            bitsandbytes is unavailable.
        ValueError: If optimizer_name is not recognized.
    """
    if optimizer_name == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)

    effective_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if effective_device.type != "cuda":
        raise RuntimeError(
            "8-bit optimizers require CUDA. Use optimizer='adamw' when training on CPU."
        )

    try:
        import bitsandbytes as bnb
    except Exception as exc:  # pragma: no cover - depends on CUDA runtime
        raise RuntimeError(
            "bitsandbytes is required for 8-bit optimizers. "
            "Install/repair bitsandbytes or switch optimizer to 'adamw'."
        ) from exc

    if optimizer_name == "adamw_8bit":
        return bnb.optim.AdamW8bit(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "paged_adamw_8bit":
        return bnb.optim.PagedAdamW8bit(parameters, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")
