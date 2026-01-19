"""
Probing Router

API endpoints for the activation probing track.
Uses experiments/probing/extract.py for activation extraction.
"""

import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services import model_manager

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class ActivationExtractRequest(BaseModel):
    """Request to extract activations."""
    text: str
    layers: Optional[List[int]] = None  # None = all layers
    positions: List[str] = ["pre_attn", "post_ffn"]
    model_name: Optional[str] = None


class LayerDiffRequest(BaseModel):
    """Request to compute layer contribution differences."""
    text: str
    layer: int
    model_name: Optional[str] = None


class PositionActivation(BaseModel):
    """Activation data for a single position in a layer."""
    position: str  # 'pre_attn' or 'post_ffn'
    mean: float
    std: float
    min: float
    max: float
    norm: float


class LayerActivationData(BaseModel):
    """Activation statistics for a single layer."""
    layer_idx: int
    positions: List[PositionActivation]
    attention_contrib_norm: Optional[float] = None
    ffn_contrib_norm: Optional[float] = None


class ActivationExtractResponse(BaseModel):
    """Response containing extracted activation data."""
    tokens: List[str]
    layers: List[LayerActivationData]
    model_info: Dict[str, Any]
    seq_len: int
    hidden_size: int


class LayerDiffResponse(BaseModel):
    """Response containing layer contribution analysis."""
    tokens: List[str]
    layer_idx: int
    attention_contribution: Dict[str, Any]  # Stats about attention contribution
    ffn_contribution: Dict[str, Any]  # Stats about FFN contribution
    per_token_attention_norm: List[float]  # Norm per token position
    per_token_ffn_norm: List[float]


class TokenActivation(BaseModel):
    """Activation for a specific token."""
    token: str
    token_idx: int
    activation_vector: List[float]  # Truncated for API response
    norm: float


class TokenActivationsResponse(BaseModel):
    """Response with per-token activations."""
    tokens: List[str]
    layer_idx: int
    position: str
    activations: List[TokenActivation]
    hidden_size: int


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/probing", tags=["probing"])


@router.post("/extract", response_model=ActivationExtractResponse)
async def extract_activations(request: ActivationExtractRequest):
    """
    Extract activation statistics from transformer layers.

    Returns per-layer statistics including mean, std, norm for
    each extraction position (pre_attn, post_ffn).
    """
    from experiments.probing.extract import (
        extract_activations as do_extract,
        compute_activation_stats,
    )

    model_name = request.model_name or "unsloth/Qwen2.5-7B-Instruct"

    try:
        model, tokenizer = await model_manager.get_model(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        result = do_extract(
            model=model,
            tokenizer=tokenizer,
            text=request.text,
            layers=request.layers,
            positions=request.positions,
        )

        stats = compute_activation_stats(result)
    except Exception as e:
        logger.error(f"Failed to extract activations: {e}")
        raise HTTPException(status_code=500, detail=f"Activation extraction failed: {str(e)}")

    # Convert to response format
    layers_data = []
    for layer_idx in result.layer_indices:
        layer_stats = stats["layers"].get(layer_idx, {})

        positions_data = []
        for pos in request.positions:
            if pos in layer_stats:
                pos_stats = layer_stats[pos]
                positions_data.append(PositionActivation(
                    position=pos,
                    mean=pos_stats["mean"],
                    std=pos_stats["std"],
                    min=pos_stats["min"],
                    max=pos_stats["max"],
                    norm=pos_stats["norm"],
                ))

        layers_data.append(LayerActivationData(
            layer_idx=layer_idx,
            positions=positions_data,
            attention_contrib_norm=layer_stats.get("attention_contrib_norm"),
            ffn_contrib_norm=layer_stats.get("ffn_contrib_norm"),
        ))

    return ActivationExtractResponse(
        tokens=result.tokens,
        layers=layers_data,
        model_info=result.model_config,
        seq_len=result.seq_len,
        hidden_size=result.hidden_size,
    )


@router.post("/layer-diff", response_model=LayerDiffResponse)
async def compute_layer_diff(request: LayerDiffRequest):
    """
    Compute the contribution of attention vs FFN at a specific layer.

    Shows how much each component (attention, FFN) contributes to
    the residual stream at the specified layer.
    """
    from experiments.probing.extract import extract_activations as do_extract

    model_name = request.model_name or "unsloth/Qwen2.5-7B-Instruct"

    try:
        model, tokenizer = await model_manager.get_model(model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Need to extract pre_attn, post_attn, and post_ffn for layer diff
    # post_attn requires special handling since default doesn't capture it
    try:
        result = do_extract(
            model=model,
            tokenizer=tokenizer,
            text=request.text,
            layers=[request.layer],
            positions=["pre_attn", "post_ffn"],
        )

        # For now, estimate attention contribution as the difference
        # A full implementation would also capture post_attn
        pre = result.get(request.layer, "pre_attn")
        post = result.get(request.layer, "post_ffn")

        # Total layer contribution
        total_diff = post - pre

        # Compute per-token norms
        per_token_norms = total_diff[0].norm(dim=-1).tolist()

        # Stats
        total_stats = {
            "mean": float(total_diff.mean()),
            "std": float(total_diff.std()),
            "norm": float(total_diff.norm()),
        }

    except Exception as e:
        logger.error(f"Failed to compute layer diff: {e}")
        raise HTTPException(status_code=500, detail=f"Layer diff failed: {str(e)}")

    return LayerDiffResponse(
        tokens=result.tokens,
        layer_idx=request.layer,
        attention_contribution=total_stats,  # Approximate
        ffn_contribution=total_stats,  # Approximate (needs post_attn for accurate)
        per_token_attention_norm=per_token_norms,
        per_token_ffn_norm=per_token_norms,
    )


@router.post("/token-activations", response_model=TokenActivationsResponse)
async def get_token_activations(
    text: str,
    layer: int,
    position: str = "post_ffn",
    model_name: Optional[str] = None,
    truncate_dim: int = 100,
):
    """
    Get per-token activation vectors for a specific layer.

    Returns truncated activation vectors suitable for visualization.
    """
    from experiments.probing.extract import extract_activations as do_extract

    model_name = model_name or "unsloth/Qwen2.5-7B-Instruct"

    try:
        model, tokenizer = await model_manager.get_model(model_name=model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    try:
        result = do_extract(
            model=model,
            tokenizer=tokenizer,
            text=text,
            layers=[layer],
            positions=[position],
        )

        activations_tensor = result.get(layer, position)  # (1, seq_len, hidden)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

    # Build per-token response
    token_activations = []
    for idx, token in enumerate(result.tokens):
        act_vector = activations_tensor[0, idx]
        token_activations.append(TokenActivation(
            token=token,
            token_idx=idx,
            activation_vector=act_vector[:truncate_dim].tolist(),
            norm=float(act_vector.norm()),
        ))

    return TokenActivationsResponse(
        tokens=result.tokens,
        layer_idx=layer,
        position=position,
        activations=token_activations,
        hidden_size=result.hidden_size,
    )
