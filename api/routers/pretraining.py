"""
Pretraining Router

API endpoints for the GPT pretraining track.
Extracted from experiments/pretraining/api/main.py
"""

import asyncio
import logging
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class TrainingConfig(BaseModel):
    config_name: str = "nano"
    corpus: str = "verdict"
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 3e-4
    warmup_steps: int = 100
    save_checkpoints: bool = False
    context_length: Optional[int] = None  # Optional override for model's default


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
        self.websocket_clients: List[WebSocket] = []
        self.metrics_history: List[Dict[str, Any]] = []

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
        self.status = TrainingStatus(
            state="loading",  # Start in loading state until training loop begins
            config=config,
        )
        self.metrics_history = []

        self.training_task = asyncio.create_task(self._run_training(config))

    async def _run_training(self, config: TrainingConfig):
        """Background training coroutine."""
        import torch
        from experiments.pretraining.model import GPTModel
        from experiments.pretraining.config import get_config
        from experiments.pretraining.tokenizer import Tokenizer
        from experiments.pretraining.data import create_train_val_dataloaders
        from experiments.pretraining.train import calc_loss_batch, get_lr_scheduler
        from experiments.pretraining.generate import generate_text
        from experiments.pretraining.checkpoint import save_checkpoint
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

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": f"Loading corpus '{config.corpus}'... (this may take a minute for large datasets)",
            })

            # Use provided context_length or fall back to model default
            effective_context_length = config.context_length if config.context_length else model_config.context_length

            # Run blocking data loading in thread pool to not block the event loop
            loop = asyncio.get_event_loop()
            train_loader, val_loader = await loop.run_in_executor(
                None,
                lambda: create_train_val_dataloaders(
                    corpus=config.corpus,
                    batch_size=config.batch_size,
                    context_length=effective_context_length,
                    tokenizer=tokenizer,
                )
            )

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": f"Creating {config.config_name} model...",
            })

            # Run model creation in thread pool as well
            def create_model():
                m = GPTModel(model_config)
                m.to(device)
                return m

            model = await loop.run_in_executor(None, create_model)

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

            total_steps = len(train_loader) * config.epochs
            scheduler = get_lr_scheduler(optimizer, config.warmup_steps, total_steps)

            self.status.total_steps = total_steps

            global_step = 0
            tokens_seen = 0
            start_time = time.time()

            # Now actually starting training - update state
            self.status.state = "running"
            await self.broadcast_metrics({
                "type": "status",
                "state": "running",
            })

            model.train()
            for epoch in range(config.epochs):
                self.status.current_epoch = epoch + 1

                for batch in train_loader:
                    if self.should_stop:
                        self.status.state = "completed"
                        return

                    while self.should_pause:
                        await asyncio.sleep(0.1)
                        if self.should_stop:
                            self.status.state = "completed"
                            return

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

                # Save checkpoint at end of each epoch (if enabled)
                if config.save_checkpoints:
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
                    })

            # Always save final model on completion
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
            train_loss=ckpt['train_loss'],
            val_loss=ckpt['val_loss'],
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
                train_loss=ckpt['train_loss'],
                val_loss=ckpt['val_loss'],
                timestamp=ckpt['timestamp'],
                corpus=ckpt.get('corpus'),
                batch_size=ckpt.get('batch_size'),
                context_length=ckpt.get('context_length'),
            )

    raise HTTPException(status_code=404, detail="Checkpoint not found")


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
