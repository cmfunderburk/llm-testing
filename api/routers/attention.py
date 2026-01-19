"""
Attention Router

API endpoints for the attention visualization track.
Uses experiments/attention/extract.py for attention extraction.
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

class AttentionExtractRequest(BaseModel):
    """Request to extract attention weights."""
    text: str
    layers: Optional[List[int]] = None  # None = all layers
    model_name: Optional[str] = None  # Use default if not specified


class AttentionHeadData(BaseModel):
    """Attention data for a single head."""
    head_idx: int
    weights: List[List[float]]  # (seq_len, seq_len)


class AttentionLayerData(BaseModel):
    """Attention data for a single layer."""
    layer_idx: int
    num_heads: int
    heads: List[AttentionHeadData]
    average_weights: List[List[float]]  # Head-averaged attention


class AttentionExtractResponse(BaseModel):
    """Response containing extracted attention data."""
    tokens: List[str]
    layers: List[AttentionLayerData]
    model_info: Dict[str, Any]
    seq_len: int
    num_layers_captured: int


class ModelInfo(BaseModel):
    """Information about an available model."""
    name: str
    is_loaded: bool
    description: Optional[str] = None


class ModelStatusResponse(BaseModel):
    """Status of the model manager."""
    model_loaded: bool
    model_name: Optional[str]
    loading: bool
    gpu_memory_allocated: Optional[float] = None
    gpu_memory_reserved: Optional[float] = None


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/attention", tags=["attention"])


@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models for attention extraction."""
    # Currently we support these models for attention extraction
    available_models = [
        ModelInfo(
            name="unsloth/Qwen2.5-7B-Instruct",
            is_loaded=model_manager.current_model_name == "unsloth/Qwen2.5-7B-Instruct",
            description="Qwen2.5 7B Instruct (4-bit quantized)",
        ),
        ModelInfo(
            name="unsloth/Qwen2.5-1.5B-Instruct",
            is_loaded=model_manager.current_model_name == "unsloth/Qwen2.5-1.5B-Instruct",
            description="Qwen2.5 1.5B Instruct (4-bit quantized)",
        ),
    ]
    return available_models


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get current model manager status."""
    status = model_manager.get_status()
    return ModelStatusResponse(**status)


@router.post("/extract", response_model=AttentionExtractResponse)
async def extract_attention(request: AttentionExtractRequest):
    """
    Extract attention weights from a transformer model.

    Returns attention patterns for each requested layer, including
    per-head weights and head-averaged weights.
    """
    from experiments.attention.extract import extract_attention as do_extract

    # Get model (loads if not already loaded)
    model_name = request.model_name or "unsloth/Qwen2.5-7B-Instruct"

    try:
        model, tokenizer = await model_manager.get_model(model_name=model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

    # Extract attention
    try:
        result = do_extract(
            model=model,
            tokenizer=tokenizer,
            text=request.text,
            layers=request.layers,
        )
    except Exception as e:
        logger.error(f"Failed to extract attention: {e}")
        raise HTTPException(status_code=500, detail=f"Attention extraction failed: {str(e)}")

    # Convert to response format
    layers_data = []
    for layer_idx in result.layer_indices:
        layer_weights = result.get_layer(layer_idx)  # (batch, heads, seq, seq)
        num_heads = layer_weights.shape[1]

        # Extract per-head data
        heads_data = []
        for head_idx in range(num_heads):
            head_weights = result.get_head(layer_idx, head_idx)  # (seq, seq)
            heads_data.append(AttentionHeadData(
                head_idx=head_idx,
                weights=head_weights.tolist(),
            ))

        # Compute head-averaged attention
        avg_weights = layer_weights[0].mean(dim=0)  # Average over heads

        layers_data.append(AttentionLayerData(
            layer_idx=layer_idx,
            num_heads=num_heads,
            heads=heads_data,
            average_weights=avg_weights.tolist(),
        ))

    return AttentionExtractResponse(
        tokens=result.tokens,
        layers=layers_data,
        model_info=result.model_config,
        seq_len=result.seq_len,
        num_layers_captured=result.num_layers,
    )


@router.post("/load-model")
async def load_model(model_name: str = "unsloth/Qwen2.5-7B-Instruct"):
    """Pre-load a model for faster subsequent requests."""
    try:
        await model_manager.get_model(model_name=model_name)
        return {"status": "ok", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unload-model")
async def unload_model():
    """Unload the current model to free GPU memory."""
    model_manager.unload_model()
    return {"status": "ok", "message": "Model unloaded"}
