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
            self.websocket_clients.remove(ws)

    async def start_training(self, config: TrainingConfig):
        """Start training in background."""
        if self.status.state == "running":
            raise HTTPException(status_code=400, detail="Training already running")

        self.should_stop = False
        self.should_pause = False
        self.status = TrainingStatus(
            state="running",
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
        import time

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_config = get_config(config.config_name)
            tokenizer = Tokenizer()

            train_loader, val_loader = create_train_val_dataloaders(
                corpus=config.corpus,
                batch_size=config.batch_size,
                context_length=model_config.context_length,
                tokenizer=tokenizer,
            )

            model = GPTModel(model_config)
            model.to(device)

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

                    if global_step % 10 == 0:
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

            self.status.state = "completed"
            await self.broadcast_metrics({
                "type": "complete",
                "final_step": global_step,
                "final_train_loss": self.status.train_loss,
            })

        except Exception as e:
            logger.error(f"Training error: {e}")
            self.status.state = "error"
            await self.broadcast_metrics({
                "type": "error",
                "message": str(e),
            })

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
async def list_checkpoints(config_name: str = "nano"):
    """List available checkpoints."""
    from experiments.pretraining.checkpoint import list_checkpoints

    checkpoints = list_checkpoints(config_name)
    return [
        CheckpointInfo(
            id=f"{config_name}_{ckpt['step']}",
            path=ckpt['path'],
            step=ckpt['step'],
            epoch=ckpt['epoch'],
            train_loss=ckpt['train_loss'],
            val_loss=ckpt['val_loss'],
            timestamp=ckpt['timestamp'],
        )
        for ckpt in checkpoints
    ]


@router.get("/checkpoints/{checkpoint_id}", response_model=CheckpointInfo)
async def get_checkpoint(checkpoint_id: str):
    """Get details for a specific checkpoint."""
    from experiments.pretraining.checkpoint import list_checkpoints

    parts = checkpoint_id.rsplit('_', 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="Invalid checkpoint ID format")

    config_name, step_str = parts
    try:
        step = int(step_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid step number")

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
    from experiments.pretraining.checkpoint import load_checkpoint, get_latest_checkpoint

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer()

    if request.checkpoint_id:
        parts = request.checkpoint_id.rsplit('_', 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid checkpoint ID")

        config_name = parts[0]
        checkpoint_path = get_latest_checkpoint(config_name)
        if not checkpoint_path:
            raise HTTPException(status_code=404, detail="No checkpoint found")

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
