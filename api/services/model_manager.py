"""
Model Manager Service

Centralized lazy loading and caching of LLM models for the API.
Manages model lifecycle to optimize GPU memory usage.
"""

import logging
from typing import Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Container for a loaded model and its tokenizer."""
    model: Any
    tokenizer: Any
    model_name: str
    is_ready: bool = False


class ModelManager:
    """
    Manages lazy loading of transformer models.

    Key features:
    - Lazy loading: Models loaded on first use
    - Caching: Keeps model in memory for subsequent requests
    - Memory management: Can unload models to free GPU memory

    Usage:
        manager = ModelManager()
        model, tokenizer = await manager.get_model()
    """

    def __init__(self):
        self._loaded_model: Optional[LoadedModel] = None
        self._loading = False

    @property
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self._loaded_model is not None and self._loaded_model.is_ready

    @property
    def current_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        if self._loaded_model:
            return self._loaded_model.model_name
        return None

    def _load_model_sync(
        self,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct",
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
    ) -> Tuple[Any, Any]:
        """
        Synchronously load a model using Unsloth.

        This is the actual loading logic, called from get_model.
        """
        # Import order matters for Unsloth
        import psutil  # noqa: F401 - Must import before unsloth
        import os
        os.environ.setdefault("UNSLOTH_DISABLE_TRAINER_PATCHING", "1")

        from unsloth import FastLanguageModel

        logger.info(f"Loading model: {model_name}")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_in_4bit,
        )

        logger.info(f"Model loaded successfully: {model_name}")
        return model, tokenizer

    async def get_model(
        self,
        model_name: str = "unsloth/Qwen2.5-7B-Instruct",
        max_seq_length: int = 1024,
        load_in_4bit: bool = True,
        force_reload: bool = False,
    ) -> Tuple[Any, Any]:
        """
        Get a model, loading it if necessary.

        Args:
            model_name: HuggingFace model identifier
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to quantize to 4-bit
            force_reload: Force reload even if already loaded

        Returns:
            Tuple of (model, tokenizer)
        """
        # Check if we already have the right model loaded
        if (
            not force_reload
            and self._loaded_model
            and self._loaded_model.is_ready
            and self._loaded_model.model_name == model_name
        ):
            return self._loaded_model.model, self._loaded_model.tokenizer

        # Need to load or reload
        if self._loading:
            raise RuntimeError("Model is already being loaded")

        try:
            self._loading = True

            # Unload existing model first
            if self._loaded_model:
                self.unload_model()

            # Load new model (synchronous, but we're in async context)
            import asyncio
            loop = asyncio.get_event_loop()
            model, tokenizer = await loop.run_in_executor(
                None,
                lambda: self._load_model_sync(model_name, max_seq_length, load_in_4bit)
            )

            self._loaded_model = LoadedModel(
                model=model,
                tokenizer=tokenizer,
                model_name=model_name,
                is_ready=True,
            )

            return model, tokenizer

        finally:
            self._loading = False

    def unload_model(self):
        """Unload the current model to free GPU memory."""
        if self._loaded_model:
            logger.info(f"Unloading model: {self._loaded_model.model_name}")

            # Clear references
            del self._loaded_model.model
            del self._loaded_model.tokenizer
            self._loaded_model = None

            # Try to free GPU memory
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

            logger.info("Model unloaded")

    def get_status(self) -> dict:
        """Get current model manager status."""
        status = {
            "model_loaded": self.is_model_loaded,
            "model_name": self.current_model_name,
            "loading": self._loading,
        }

        # Add GPU memory info if available
        try:
            import torch
            if torch.cuda.is_available():
                status["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
                status["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        except ImportError:
            pass

        return status


# Global model manager instance
model_manager = ModelManager()
