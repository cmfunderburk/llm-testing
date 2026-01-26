"""
Pretraining Router

API endpoints for the GPT pretraining track.
Extracted from experiments/pretraining/api/main.py
"""

import asyncio
import logging
import math
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _sanitize_float(value: Optional[float]) -> Optional[float]:
    """Convert NaN/Inf float values to None for JSON serialization."""
    if value is None:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


# =============================================================================
# Pydantic Models
# =============================================================================

class TrainingConfig(BaseModel):
    config_name: str = "nano"
    corpus: str = "verdict"
    val_corpus: Optional[str] = None  # Separate validation corpus (e.g., 'pg19_validation')
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    save_checkpoints: bool = False
    context_length: Optional[int] = None  # Optional override for model's default
    resume_from: Optional[str] = None  # checkpoint_id to resume from


class TrainingStatus(BaseModel):
    state: str  # "idle", "running", "paused", "completed", "error"
    current_step: int = 0
    current_epoch: int = 0
    total_steps: int = 0
    train_loss: Optional[float] = None
    val_loss: Optional[float] = None
    tokens_seen: int = 0
    elapsed_time: float = 0.0
    config: Optional[TrainingConfig] = None


class CheckpointInfo(BaseModel):
    id: str
    path: str
    step: int
    epoch: int
    train_loss: Optional[float]
    val_loss: Optional[float]
    timestamp: Optional[str]
    corpus: Optional[str] = None
    batch_size: Optional[int] = None
    context_length: Optional[int] = None


class GenerateRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    prompt: str = "The meaning of life is"
    max_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    prompt: str


class AttentionRequest(BaseModel):
    checkpoint_id: Optional[str] = None
    text: str
    layer: int = 0
    head: Optional[int] = None


class AttentionResponse(BaseModel):
    tokens: List[str]
    attention_weights: List[List[float]]
    layer: int
    head: Optional[int]


class VRAMEstimate(BaseModel):
    model_mb: float
    optimizer_mb: float
    gradients_mb: float
    activations_mb: float
    total_mb: float
    total_gb: float
    params: int
    batch_size: int
    context_length: int
    warning: Optional[str] = None


class SaveNowResponse(BaseModel):
    success: bool
    message: str
    step: int


# =============================================================================
# Training Manager
# =============================================================================

class TrainingManager:
    """Manages training state and background tasks."""

    def __init__(self):
        self.status = TrainingStatus(state="idle")
        self.training_task: Optional[asyncio.Task] = None
        self.should_stop = False
        self.should_pause = False
        self.should_save_now = False  # Flag for manual checkpoint save
        self.websocket_clients: List[WebSocket] = []
        self.metrics_history: List[Dict[str, Any]] = []
        # Current training state for save-now endpoint
        self._model = None
        self._optimizer = None
        self._model_config = None
        self._training_config: Optional[TrainingConfig] = None
        self._current_epoch = 0
        self._effective_context_length = 0
        # Progress tracking for data loading (shared with executor thread)
        self._loading_progress: Dict[str, Any] = {}

    async def broadcast_metrics(self, metrics: Dict[str, Any]):
        """Send metrics to all connected WebSocket clients."""
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_json(metrics)
            except Exception:
                disconnected.append(ws)

        for ws in disconnected:
            if ws in self.websocket_clients:
                self.websocket_clients.remove(ws)

    async def start_training(self, config: TrainingConfig):
        """Start training in background."""
        if self.status.state in ("running", "loading"):
            raise HTTPException(status_code=400, detail="Training already running")

        # Clear GPU memory from previous runs
        import torch
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.should_stop = False
        self.should_pause = False
        self.should_save_now = False
        self.status = TrainingStatus(
            state="loading",  # Start in loading state until training loop begins
            config=config,
        )
        self.metrics_history = []
        self._training_config = config

        self.training_task = asyncio.create_task(self._run_training(config))

    async def _run_training(self, config: TrainingConfig):
        """Background training coroutine."""
        import torch
        from experiments.pretraining.model import GPTModel
        from experiments.pretraining.config import get_config
        from experiments.pretraining.tokenizer import Tokenizer
        from experiments.pretraining.data import create_train_val_dataloaders
        from experiments.pretraining.train import calc_loss_batch, calc_loss_loader, get_lr_scheduler
        from experiments.pretraining.generate import generate_text
        from experiments.pretraining.checkpoint import save_checkpoint, load_checkpoint, list_checkpoints
        import time

        try:
            # Send loading status updates
            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": "Initializing...",
            })

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_config = get_config(config.config_name)
            tokenizer = Tokenizer()

            # Use provided context_length or fall back to model default
            effective_context_length = config.context_length if config.context_length else model_config.context_length
            self._effective_context_length = effective_context_length

            # Handle resume from checkpoint
            start_step = 0
            start_epoch = 0
            resumed_optimizer_state = None

            if config.resume_from:
                await self.broadcast_metrics({
                    "type": "status",
                    "state": "loading",
                    "message": f"Loading checkpoint {config.resume_from}...",
                })

                # Parse checkpoint_id to find the checkpoint path
                checkpoint_config_name, checkpoint_step = _parse_checkpoint_id(config.resume_from)
                checkpoints = list_checkpoints(checkpoint_config_name)
                checkpoint_path = None
                checkpoint_metadata = None

                for ckpt in checkpoints:
                    if ckpt['step'] == checkpoint_step:
                        checkpoint_path = ckpt['path']
                        checkpoint_metadata = ckpt
                        break

                if not checkpoint_path:
                    raise ValueError(f"Checkpoint not found: {config.resume_from}")

                # Load the checkpoint
                model, _, metadata = load_checkpoint(checkpoint_path, device=device)

                # Store optimizer state to load after optimizer creation
                ckpt_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
                if 'optimizer_state_dict' in ckpt_data:
                    resumed_optimizer_state = ckpt_data['optimizer_state_dict']

                start_step = metadata.get('step', 0)
                start_epoch = metadata.get('epoch', 0)

                # Warn if training params differ
                warnings = []
                if checkpoint_metadata:
                    if checkpoint_metadata.get('corpus') and checkpoint_metadata['corpus'] != config.corpus:
                        warnings.append(f"corpus: {checkpoint_metadata['corpus']} → {config.corpus}")
                    if checkpoint_metadata.get('batch_size') and checkpoint_metadata['batch_size'] != config.batch_size:
                        warnings.append(f"batch_size: {checkpoint_metadata['batch_size']} → {config.batch_size}")
                    if checkpoint_metadata.get('context_length') and checkpoint_metadata['context_length'] != effective_context_length:
                        warnings.append(f"context_length: {checkpoint_metadata['context_length']} → {effective_context_length}")

                if warnings:
                    await self.broadcast_metrics({
                        "type": "warning",
                        "message": f"Config differs from checkpoint: {', '.join(warnings)}",
                    })

                await self.broadcast_metrics({
                    "type": "status",
                    "state": "loading",
                    "message": f"Resumed from step {start_step}, epoch {start_epoch}",
                })
            else:
                # Create fresh model
                await self.broadcast_metrics({
                    "type": "status",
                    "state": "loading",
                    "message": f"Creating {config.config_name} model...",
                })

                loop = asyncio.get_event_loop()
                def create_model():
                    m = GPTModel(model_config)
                    m.to(device)
                    return m

                model = await loop.run_in_executor(None, create_model)

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": f"Loading corpus '{config.corpus}'...",
            })

            # Progress callback for tokenization (called from executor thread)
            def on_tokenization_progress(phase: str, bytes_read: int, total_bytes: int, tokens_so_far: int):
                self._loading_progress = {
                    "phase": phase,
                    "bytes_read": bytes_read,
                    "total_bytes": total_bytes,
                    "tokens_so_far": tokens_so_far,
                    "corpus": config.corpus,
                }

            # Background task to broadcast loading progress
            async def broadcast_loading_progress():
                last_broadcast = {}
                while self.status.state == "loading":
                    progress = self._loading_progress.copy()
                    if progress and progress != last_broadcast:
                        phase = progress.get("phase", "")
                        bytes_read = progress.get("bytes_read", 0)
                        total_bytes = progress.get("total_bytes", 0)
                        tokens = progress.get("tokens_so_far", 0)
                        corpus = progress.get("corpus", "")

                        if phase == "checking_cache":
                            message = f"Checking cache for '{corpus}'..."
                        elif phase == "loaded_from_cache":
                            message = f"Loaded {tokens:,} tokens from cache"
                        elif phase == "tokenizing" and total_bytes > 0:
                            pct = (bytes_read / total_bytes) * 100
                            mb_read = bytes_read / (1024 * 1024)
                            mb_total = total_bytes / (1024 * 1024)
                            message = f"Tokenizing '{corpus}': {mb_read:.0f}/{mb_total:.0f} MB ({pct:.1f}%) - {tokens:,} tokens"
                        elif phase == "saving_cache":
                            message = f"Saving {tokens:,} tokens to cache..."
                        elif phase == "complete":
                            message = f"Tokenization complete: {tokens:,} tokens"
                        else:
                            message = f"Loading '{corpus}'..."

                        await self.broadcast_metrics({
                            "type": "loading_progress",
                            "phase": phase,
                            "bytes_read": bytes_read,
                            "total_bytes": total_bytes,
                            "tokens": tokens,
                            "message": message,
                            "percent": (bytes_read / total_bytes * 100) if total_bytes > 0 else 0,
                        })
                        last_broadcast = progress

                    await asyncio.sleep(0.5)  # Update every 500ms

            # Start progress broadcaster
            loop = asyncio.get_event_loop()
            progress_task = asyncio.create_task(broadcast_loading_progress())

            try:
                # Run blocking data loading in thread pool to not block the event loop
                train_loader, val_loader = await loop.run_in_executor(
                    None,
                    lambda: create_train_val_dataloaders(
                        corpus=config.corpus,
                        val_corpus=config.val_corpus,
                        batch_size=config.batch_size,
                        context_length=effective_context_length,
                        tokenizer=tokenizer,
                        progress_callback=on_tokenization_progress,
                    )
                )
            finally:
                # Stop progress broadcaster
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass
                self._loading_progress = {}

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": "Model ready. Starting training...",
            })

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=0.1,
            )

            # Restore optimizer state if resuming
            if resumed_optimizer_state is not None:
                optimizer.load_state_dict(resumed_optimizer_state)

            total_steps = len(train_loader) * config.epochs
            scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

            # If resuming, advance scheduler to current step
            if start_step > 0:
                for _ in range(start_step):
                    scheduler.step()
                # Apply user's learning rate AFTER scheduler advancement
                # This allows overriding the LR when resuming
                # Update both optimizer and scheduler's base_lrs so future steps use new LR
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.learning_rate
                scheduler.base_lrs = [config.learning_rate for _ in scheduler.base_lrs]

            self.status.total_steps = total_steps

            # Store references for save-now endpoint
            self._model = model
            self._optimizer = optimizer
            self._model_config = model_config

            # Calculate checkpoint steps at 25%, 50%, 75% of training
            # (100% is handled by final checkpoint save)
            checkpoint_percentages = [0.25, 0.50, 0.75]
            checkpoint_steps = set(int(total_steps * pct) for pct in checkpoint_percentages)
            # Remove 0 in case total_steps is very small
            checkpoint_steps.discard(0)
            saved_checkpoints = set()

            global_step = start_step
            tokens_seen = 0
            start_time = time.time()

            # Now actually starting training - update state
            self.status.state = "running"
            await self.broadcast_metrics({
                "type": "status",
                "state": "running",
                "total_steps": total_steps,
                "resumed_from_step": start_step if start_step > 0 else None,
            })

            model.train()
            for epoch in range(config.epochs):
                # Skip epochs that were completed before resume
                if epoch < start_epoch:
                    continue

                self.status.current_epoch = epoch + 1
                self._current_epoch = epoch + 1

                steps_in_epoch = 0
                for batch in train_loader:
                    # If resuming mid-epoch, skip batches until we reach start_step
                    if config.resume_from and epoch == start_epoch:
                        steps_in_epoch += 1
                        # Calculate which batch we should resume from
                        batches_per_epoch = len(train_loader)
                        start_batch_in_epoch = start_step - (start_epoch * batches_per_epoch)
                        if steps_in_epoch <= start_batch_in_epoch:
                            continue

                    if self.should_stop:
                        self.status.state = "completed"
                        return

                    while self.should_pause:
                        await asyncio.sleep(0.1)
                        if self.should_stop:
                            self.status.state = "completed"
                            return

                    # Handle manual checkpoint save
                    if self.should_save_now:
                        self.should_save_now = False
                        manual_path = save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            config=model_config,
                            config_name=config.config_name,
                            step=global_step,
                            epoch=epoch + 1,
                            train_loss=self.status.train_loss,
                            val_loss=self.status.val_loss,
                            filename=f"checkpoint_manual_{global_step:06d}.pt",
                            corpus=config.corpus,
                            batch_size=config.batch_size,
                            context_length=effective_context_length,
                        )
                        await self.broadcast_metrics({
                            "type": "checkpoint",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "path": str(manual_path),
                            "manual": True,
                        })

                    input_ids = batch['input_ids']
                    labels = batch['labels']

                    optimizer.zero_grad()
                    loss = calc_loss_batch(input_ids, labels, model, device)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    global_step += 1
                    tokens_seen += input_ids.numel()
                    elapsed = time.time() - start_time

                    self.status.current_step = global_step
                    self.status.tokens_seen = tokens_seen
                    self.status.elapsed_time = elapsed
                    self.status.train_loss = loss.item()

                    if global_step % 50 == 0:
                        metrics = {
                            "type": "metrics",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "train_loss": loss.item(),
                            "learning_rate": optimizer.param_groups[0]['lr'],
                            "tokens_seen": tokens_seen,
                            "tokens_per_sec": tokens_seen / elapsed if elapsed > 0 else 0,
                            "elapsed_time": elapsed,
                        }
                        self.metrics_history.append(metrics)
                        await self.broadcast_metrics(metrics)

                    # Generate sample every 2000 steps for progress visibility
                    if global_step % 2000 == 0 and global_step > 0:
                        model.eval()
                        sample = generate_text(
                            model=model,
                            tokenizer=tokenizer,
                            prompt="Once upon a time",
                            max_new_tokens=50,
                            temperature=0.8,
                            device=device,
                        )
                        model.train()

                        generation_msg = {
                            "type": "generation",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "text": sample[:300],
                        }
                        await self.broadcast_metrics(generation_msg)

                    # Save checkpoint at 25%, 50%, 75% milestones (if enabled)
                    if config.save_checkpoints and global_step in checkpoint_steps and global_step not in saved_checkpoints:
                        saved_checkpoints.add(global_step)
                        pct = int(100 * global_step / total_steps)
                        checkpoint_path = save_checkpoint(
                            model=model,
                            optimizer=optimizer,
                            config=model_config,
                            config_name=config.config_name,
                            step=global_step,
                            epoch=epoch + 1,
                            train_loss=self.status.train_loss,
                            val_loss=self.status.val_loss,
                            corpus=config.corpus,
                            batch_size=config.batch_size,
                            context_length=effective_context_length,
                        )
                        await self.broadcast_metrics({
                            "type": "checkpoint",
                            "step": global_step,
                            "epoch": epoch + 1,
                            "path": str(checkpoint_path),
                            "percentage": pct,
                        })

                    await asyncio.sleep(0)

                model.eval()
                sample = generate_text(
                    model=model,
                    tokenizer=tokenizer,
                    prompt="Every effort moves you",
                    max_new_tokens=30,
                    temperature=0.8,
                    device=device,
                )
                model.train()

                generation_msg = {
                    "type": "generation",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "text": sample[:200],
                }
                await self.broadcast_metrics(generation_msg)

                # Compute validation loss at end of epoch
                model.eval()
                with torch.no_grad():
                    # Limit to 50 batches for speed on large validation sets
                    val_loss = await loop.run_in_executor(
                        None,
                        lambda: calc_loss_loader(val_loader, model, device, num_batches=50)
                    )
                self.status.val_loss = val_loss
                model.train()

                val_metrics = {
                    "type": "validation",
                    "step": global_step,
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "train_loss": self.status.train_loss,
                }
                await self.broadcast_metrics(val_metrics)

            # Always save final model on completion (100% checkpoint)
            final_checkpoint_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                config=model_config,
                config_name=config.config_name,
                step=global_step,
                epoch=config.epochs,
                train_loss=self.status.train_loss,
                val_loss=self.status.val_loss,
                filename="checkpoint_final.pt",
                corpus=config.corpus,
                batch_size=config.batch_size,
                context_length=effective_context_length,
            )

            self.status.state = "completed"
            await self.broadcast_metrics({
                "type": "complete",
                "final_step": global_step,
                "final_train_loss": self.status.train_loss,
                "checkpoint_path": str(final_checkpoint_path),
            })

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.status.state = "error"
            await self.broadcast_metrics({
                "type": "error",
                "message": str(e),
            })
        finally:
            # Clean up references
            self._model = None
            self._optimizer = None
            self._model_config = None
            # Clean up GPU memory
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU memory cleaned up after training")

    def pause_training(self):
        """Pause training."""
        if self.status.state != "running":
            raise HTTPException(status_code=400, detail="Training not running")
        self.should_pause = True
        self.status.state = "paused"

    def resume_training(self):
        """Resume training."""
        if self.status.state != "paused":
            raise HTTPException(status_code=400, detail="Training not paused")
        self.should_pause = False
        self.status.state = "running"

    def stop_training(self):
        """Stop training."""
        if self.status.state not in ("running", "paused"):
            raise HTTPException(status_code=400, detail="Training not active")
        self.should_stop = True
        self.should_pause = False
        self.status.state = "completed"

    def trigger_save_now(self):
        """Trigger immediate checkpoint save."""
        if self.status.state != "running":
            raise HTTPException(status_code=400, detail="Training not running")
        self.should_save_now = True


# Global training manager instance
training_manager = TrainingManager()


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/pretraining", tags=["pretraining"])


@router.get("/status", response_model=TrainingStatus)
async def get_training_status():
    """Get current training status."""
    return training_manager.status


@router.get("/estimate-vram", response_model=VRAMEstimate)
async def estimate_vram(config_name: str = "nano", batch_size: int = 4, context_length: Optional[int] = None):
    """Estimate VRAM requirements for a given configuration."""
    from experiments.pretraining.config import get_config

    try:
        model_config = get_config(config_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Use provided context_length or fall back to model default
    effective_context_length = context_length if context_length else model_config.context_length

    # Create a modified config if context_length differs from default
    if effective_context_length != model_config.context_length:
        # Create a copy with the new context_length for estimation
        from dataclasses import replace
        model_config = replace(model_config, context_length=effective_context_length)

    estimate = model_config.estimate_vram_mb(batch_size)

    # Add warning if estimated VRAM is high
    warning = None
    if estimate["total_gb"] > 14:
        warning = "May exceed 16GB VRAM. Consider reducing batch size."
    elif estimate["total_gb"] > 10:
        warning = "High VRAM usage. Monitor for OOM errors."

    return VRAMEstimate(**estimate, warning=warning)


@router.post("/start", response_model=TrainingStatus)
async def start_training(config: TrainingConfig):
    """Start a new training run."""
    await training_manager.start_training(config)
    return training_manager.status


@router.post("/pause", response_model=TrainingStatus)
async def pause_training():
    """Pause the current training run."""
    training_manager.pause_training()
    return training_manager.status


@router.post("/resume", response_model=TrainingStatus)
async def resume_training():
    """Resume a paused training run."""
    training_manager.resume_training()
    return training_manager.status


@router.post("/stop", response_model=TrainingStatus)
async def stop_training():
    """Stop the current training run."""
    training_manager.stop_training()
    return training_manager.status


@router.post("/checkpoint/save-now", response_model=SaveNowResponse)
async def save_checkpoint_now():
    """Trigger immediate checkpoint save during active training."""
    training_manager.trigger_save_now()
    return SaveNowResponse(
        success=True,
        message="Checkpoint save requested. Will be saved at next training step.",
        step=training_manager.status.current_step,
    )


@router.get("/checkpoints", response_model=List[CheckpointInfo])
async def list_checkpoints_endpoint(config_name: str = "nano"):
    """List available checkpoints."""
    from experiments.pretraining.checkpoint import list_checkpoints

    checkpoints = list_checkpoints(config_name)
    result = []
    for ckpt in checkpoints:
        # Build descriptive ID: nano_verdict_b4_ctx256_step1000
        corpus = ckpt.get('corpus')
        batch_size = ckpt.get('batch_size')
        context_length = ckpt.get('context_length')

        if corpus and batch_size and context_length:
            ckpt_id = f"{config_name}_{corpus}_b{batch_size}_ctx{context_length}_step{ckpt['step']}"
        else:
            # Legacy checkpoint without training params
            ckpt_id = f"{config_name}_step{ckpt['step']}"

        result.append(CheckpointInfo(
            id=ckpt_id,
            path=ckpt['path'],
            step=ckpt['step'],
            epoch=ckpt['epoch'],
            train_loss=_sanitize_float(ckpt['train_loss']),
            val_loss=_sanitize_float(ckpt['val_loss']),
            timestamp=ckpt['timestamp'],
            corpus=corpus,
            batch_size=batch_size,
            context_length=context_length,
        ))
    return result


def _parse_checkpoint_id(checkpoint_id: str) -> tuple:
    """
    Parse checkpoint ID to extract config_name and step.

    Handles both formats:
    - New: nano_verdict_b4_ctx256_step1000
    - Legacy: nano_step1000 or nano_1000
    """
    import re

    # Try new format first: {config}_{corpus}_b{batch}_ctx{ctx}_step{step}
    new_format = re.match(r'^(\w+)_\w+_b\d+_ctx\d+_step(\d+)$', checkpoint_id)
    if new_format:
        return new_format.group(1), int(new_format.group(2))

    # Try format with _step suffix: nano_step1000
    step_format = re.match(r'^(\w+)_step(\d+)$', checkpoint_id)
    if step_format:
        return step_format.group(1), int(step_format.group(2))

    # Legacy format: nano_1000
    parts = checkpoint_id.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass

    raise ValueError(f"Invalid checkpoint ID format: {checkpoint_id}")


@router.get("/checkpoints/{checkpoint_id}", response_model=CheckpointInfo)
async def get_checkpoint(checkpoint_id: str):
    """Get details for a specific checkpoint."""
    from experiments.pretraining.checkpoint import list_checkpoints

    try:
        config_name, step = _parse_checkpoint_id(checkpoint_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    checkpoints = list_checkpoints(config_name)
    for ckpt in checkpoints:
        if ckpt['step'] == step:
            return CheckpointInfo(
                id=checkpoint_id,
                path=ckpt['path'],
                step=ckpt['step'],
                epoch=ckpt['epoch'],
                train_loss=_sanitize_float(ckpt['train_loss']),
                val_loss=_sanitize_float(ckpt['val_loss']),
                timestamp=ckpt['timestamp'],
                corpus=ckpt.get('corpus'),
                batch_size=ckpt.get('batch_size'),
                context_length=ckpt.get('context_length'),
            )

    raise HTTPException(status_code=404, detail="Checkpoint not found")


class DeleteCheckpointsRequest(BaseModel):
    config_name: str = "nano"
    keep_latest: bool = True  # Keep checkpoint_latest.pt and checkpoint_final.pt


class DeleteCheckpointsResponse(BaseModel):
    deleted_count: int
    deleted_files: List[str]


@router.delete("/checkpoints", response_model=DeleteCheckpointsResponse)
async def delete_checkpoints(config_name: str = "nano", keep_latest: bool = True):
    """
    Delete checkpoints for a given configuration.

    Args:
        config_name: Name of model config ('nano', 'small', 'medium')
        keep_latest: If True, keeps checkpoint_latest.pt and checkpoint_final.pt
    """
    from pathlib import Path
    from experiments.pretraining.checkpoint import DEFAULT_CHECKPOINT_DIR

    base_dir = DEFAULT_CHECKPOINT_DIR
    if not base_dir.exists():
        return DeleteCheckpointsResponse(deleted_count=0, deleted_files=[])

    deleted_files = []

    # Find all directories matching this config_name
    for subdir in base_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name != config_name and not subdir.name.startswith(f"{config_name}_"):
            continue

        # Delete checkpoint files in this directory
        for checkpoint_file in subdir.glob("checkpoint_*.pt"):
            filename = checkpoint_file.name
            # Skip latest/final if keep_latest is True
            if keep_latest and filename in ("checkpoint_latest.pt", "checkpoint_final.pt"):
                continue

            checkpoint_file.unlink()
            deleted_files.append(str(checkpoint_file))

        # If directory is now empty (or only has latest/final), optionally clean up
        remaining = list(subdir.glob("*.pt"))
        if not remaining:
            subdir.rmdir()

    return DeleteCheckpointsResponse(
        deleted_count=len(deleted_files),
        deleted_files=deleted_files,
    )


@router.delete("/checkpoints/all", response_model=DeleteCheckpointsResponse)
async def delete_all_checkpoints():
    """Delete ALL checkpoints across all configurations."""
    from pathlib import Path
    import shutil
    from experiments.pretraining.checkpoint import DEFAULT_CHECKPOINT_DIR

    base_dir = DEFAULT_CHECKPOINT_DIR
    if not base_dir.exists():
        return DeleteCheckpointsResponse(deleted_count=0, deleted_files=[])

    deleted_files = []

    # Count files before deletion
    for pt_file in base_dir.rglob("*.pt"):
        deleted_files.append(str(pt_file))

    # Remove the entire pretraining outputs directory
    shutil.rmtree(base_dir)

    return DeleteCheckpointsResponse(
        deleted_count=len(deleted_files),
        deleted_files=deleted_files,
    )


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a model checkpoint."""
    import torch
    from experiments.pretraining.model import GPTModel
    from experiments.pretraining.config import get_config
    from experiments.pretraining.tokenizer import Tokenizer
    from experiments.pretraining.generate import generate_text
    from experiments.pretraining.checkpoint import load_checkpoint, list_checkpoints

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    if request.checkpoint_id:
        try:
            config_name, step = _parse_checkpoint_id(request.checkpoint_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Find the specific checkpoint by step
        checkpoints = list_checkpoints(config_name)
        checkpoint_path = None
        for ckpt in checkpoints:
            if ckpt['step'] == step:
                checkpoint_path = ckpt['path']
                break

        if not checkpoint_path:
            raise HTTPException(status_code=404, detail="Checkpoint not found")

        model, _, _ = load_checkpoint(checkpoint_path, device=device)
    else:
        config = get_config("nano")
        model = GPTModel(config)
        model.to(device)

    model.eval()

    text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=request.prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        device=device,
    )

    prompt_tokens = len(tokenizer.encode(request.prompt))
    total_tokens = len(tokenizer.encode(text))

    return GenerateResponse(
        text=text,
        tokens_generated=total_tokens - prompt_tokens,
        prompt=request.prompt,
    )


@router.post("/analyze/attention", response_model=AttentionResponse)
async def analyze_attention(request: AttentionRequest):
    """Get attention patterns for input text (simplified mock implementation)."""
    from experiments.pretraining.tokenizer import Tokenizer

    tokenizer = Tokenizer()
    tokens = tokenizer.tokenize(request.text)

    n_tokens = len(tokens)
    mock_weights = [[1.0/n_tokens] * n_tokens for _ in range(n_tokens)]

    return AttentionResponse(
        tokens=tokens,
        attention_weights=mock_weights,
        layer=request.layer,
        head=request.head,
    )


# =============================================================================
# WebSocket - mounted separately in main.py
# =============================================================================

async def websocket_training(websocket: WebSocket):
    """WebSocket endpoint for real-time training metrics."""
    await websocket.accept()
    training_manager.websocket_clients.append(websocket)
    logger.info(f"WebSocket client connected. Total: {len(training_manager.websocket_clients)}")

    try:
        await websocket.send_json({
            "type": "status",
            "state": training_manager.status.state,
            "current_step": training_manager.status.current_step,
        })

        for metrics in training_manager.metrics_history[-100:]:
            await websocket.send_json(metrics)

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in training_manager.websocket_clients:
            training_manager.websocket_clients.remove(websocket)
