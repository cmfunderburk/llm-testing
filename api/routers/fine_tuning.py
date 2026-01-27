"""
Fine-Tuning Router

API endpoints for the QLoRA fine-tuning track.
Uses SFTTrainer run in a thread with a TrainerCallback to bridge
metrics back to the async event loop for WebSocket broadcasting.
"""

import asyncio
import gc
import json
import logging
import math
import queue
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ADAPTER_BASE_DIR = Path("outputs/fine_tuning/adapters")


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

class FineTuningConfig(BaseModel):
    model_name: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 1024
    n_examples: int = 500
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 1
    batch_size: int = 4
    gradient_accumulation: int = 4
    warmup_ratio: float = 0.1
    logging_steps: int = 10
    eval_steps: int = 50
    save_adapter: bool = True
    resume_from: Optional[str] = None  # adapter checkpoint path


class FineTuningStatus(BaseModel):
    state: str = "idle"  # idle, loading, running, paused, completed, error
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    elapsed_time: float = 0.0
    config: Optional[FineTuningConfig] = None
    trainable_params: Optional[int] = None
    total_params: Optional[int] = None


class AdapterCheckpointInfo(BaseModel):
    id: str
    path: str
    step: int
    train_loss: Optional[float] = None
    eval_loss: Optional[float] = None
    timestamp: Optional[str] = None


class GenerateRequest(BaseModel):
    adapter_path: Optional[str] = None
    prompt: str = "Explain what machine learning is in simple terms."
    max_tokens: int = 256
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int
    prompt: str
    adapter_path: Optional[str] = None


# =============================================================================
# Control Flags (thread-safe via GIL for simple bool reads/writes)
# =============================================================================

class ControlFlags:
    def __init__(self):
        self.should_stop = False
        self.should_pause = False
        self.should_save_now = False

    def reset(self):
        self.should_stop = False
        self.should_pause = False
        self.should_save_now = False


# =============================================================================
# Trainer Callback — runs in worker thread
# =============================================================================

class DashboardCallback:
    """
    HuggingFace TrainerCallback that bridges metrics from the worker thread
    to the async event loop via a queue.
    """

    def __init__(self, metrics_queue: queue.Queue, control_flags: ControlFlags,
                 model=None, tokenizer=None, save_dir: Optional[Path] = None):
        self.metrics_queue = metrics_queue
        self.control_flags = control_flags
        self.model = model
        self.tokenizer = tokenizer
        self.save_dir = save_dir

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        entry: Dict[str, Any] = {"step": state.global_step, "epoch": state.epoch}

        if "loss" in logs:
            entry["type"] = "metrics"
            entry["train_loss"] = logs["loss"]
            entry["learning_rate"] = logs.get("learning_rate")
            self.metrics_queue.put(entry)

        if "eval_loss" in logs:
            self.metrics_queue.put({
                "type": "validation",
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": logs["eval_loss"],
            })

    def on_step_end(self, args, state, control, **kwargs):
        # Check stop
        if self.control_flags.should_stop:
            control.should_training_stop = True
            return

        # Check pause — spin-wait in worker thread
        while self.control_flags.should_pause:
            if self.control_flags.should_stop:
                control.should_training_stop = True
                return
            time.sleep(0.1)

        # Check manual save
        if self.control_flags.should_save_now:
            self.control_flags.should_save_now = False
            if self.model and self.save_dir:
                save_path = self.save_dir / f"checkpoint-{state.global_step}"
                save_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(str(save_path))
                if self.tokenizer:
                    self.tokenizer.save_pretrained(str(save_path))
                # Save metadata
                metadata = {
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(save_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                self.metrics_queue.put({
                    "type": "checkpoint",
                    "step": state.global_step,
                    "path": str(save_path),
                    "manual": True,
                })

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            self.metrics_queue.put({
                "type": "validation",
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": metrics["eval_loss"],
            })


# =============================================================================
# Fine-Tuning Manager
# =============================================================================

class FineTuningManager:
    """Manages the fine-tuning training lifecycle."""

    def __init__(self):
        self.status = FineTuningStatus()
        self.training_task: Optional[asyncio.Task] = None
        self.control_flags = ControlFlags()
        self.websocket_clients: List[WebSocket] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self._metrics_queue: queue.Queue = queue.Queue()
        self._poller_task: Optional[asyncio.Task] = None
        # Cache for generation model
        self._gen_model = None
        self._gen_tokenizer = None
        self._gen_adapter_path: Optional[str] = None

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

    async def start_training(self, config: FineTuningConfig):
        """Start fine-tuning in background."""
        if self.status.state in ("running", "loading"):
            raise HTTPException(status_code=400, detail="Fine-tuning already running")

        # GPU contention guard: check if pretraining is running
        try:
            from .pretraining import training_manager as pt_manager
            if pt_manager.status.state in ("running", "loading", "paused"):
                raise HTTPException(
                    status_code=409,
                    detail="Pretraining is currently active. Stop it before starting fine-tuning."
                )
        except ImportError:
            pass

        # Clear GPU memory
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self.control_flags.reset()
        self.status = FineTuningStatus(state="loading", config=config)
        self.metrics_history = []
        self._metrics_queue = queue.Queue()

        # Clear generation cache
        self._gen_model = None
        self._gen_tokenizer = None
        self._gen_adapter_path = None

        self.training_task = asyncio.create_task(self._run_training(config))

    async def _poll_metrics_queue(self):
        """Async task that drains the metrics queue and broadcasts to WebSocket clients."""
        while True:
            try:
                while True:
                    try:
                        msg = self._metrics_queue.get_nowait()
                    except queue.Empty:
                        break

                    msg_type = msg.get("type")

                    if msg_type == "metrics":
                        self.status.current_step = msg["step"]
                        self.status.current_epoch = msg.get("epoch", 0)
                        self.status.train_loss = _sanitize_float(msg.get("train_loss"))
                        self.status.learning_rate = _sanitize_float(msg.get("learning_rate"))
                        self.metrics_history.append(msg)
                        await self.broadcast_metrics(msg)

                    elif msg_type == "validation":
                        self.status.eval_loss = _sanitize_float(msg.get("eval_loss"))
                        self.metrics_history.append(msg)
                        await self.broadcast_metrics(msg)

                    elif msg_type == "checkpoint":
                        await self.broadcast_metrics(msg)

                    else:
                        await self.broadcast_metrics(msg)

                await asyncio.sleep(0.25)

            except asyncio.CancelledError:
                break

    async def _run_training(self, config: FineTuningConfig):
        """Background training coroutine."""
        import torch

        try:
            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": "Loading model...",
            })

            loop = asyncio.get_event_loop()

            # Load model + LoRA + dataset in thread
            def _setup():
                import psutil  # noqa: F401 - must import before unsloth
                import os
                os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

                from unsloth import FastLanguageModel
                from unsloth.chat_templates import get_chat_template
                from datasets import load_dataset
                from trl import SFTTrainer
                from transformers import TrainingArguments, TrainerCallback

                # Load model
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.model_name,
                    max_seq_length=config.max_seq_length,
                    load_in_4bit=True,
                    dtype=None,
                )

                # Add LoRA adapters
                model = FastLanguageModel.get_peft_model(
                    model,
                    r=config.lora_r,
                    lora_alpha=config.lora_alpha,
                    lora_dropout=config.lora_dropout,
                    target_modules=[
                        "q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",
                    ],
                    bias="none",
                    use_gradient_checkpointing="unsloth",
                    random_state=42,
                )

                trainable, total = model.get_nb_trainable_parameters()

                # Prepare dataset
                tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

                dataset = load_dataset(
                    "yahma/alpaca-cleaned",
                    split=f"train[:{config.n_examples}]",
                )

                def format_example(example):
                    messages = []
                    if example.get("input", "").strip():
                        user_content = f"{example['instruction']}\n\nInput: {example['input']}"
                    else:
                        user_content = example["instruction"]
                    messages.append({"role": "user", "content": user_content})
                    messages.append({"role": "assistant", "content": example["output"]})
                    text = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    return {"text": text}

                dataset = dataset.map(format_example, remove_columns=dataset.column_names)
                dataset = dataset.train_test_split(test_size=0.1, seed=42)

                return model, tokenizer, dataset, trainable, total, SFTTrainer, TrainingArguments, TrainerCallback

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": f"Loading {config.model_name} with LoRA r={config.lora_r}...",
            })

            (model, tokenizer, dataset, trainable_params, total_params,
             SFTTrainer, TrainingArguments, TrainerCallbackBase) = await loop.run_in_executor(None, _setup)

            self.status.trainable_params = trainable_params
            self.status.total_params = total_params

            await self.broadcast_metrics({
                "type": "status",
                "state": "loading",
                "message": f"Model loaded. {trainable_params:,} trainable params ({100*trainable_params/total_params:.2f}%). Preparing trainer...",
            })

            # Determine save directory
            adapter_name = f"{config.model_name.split('/')[-1]}_r{config.lora_r}_n{config.n_examples}"
            save_dir = ADAPTER_BASE_DIR / adapter_name
            save_dir.mkdir(parents=True, exist_ok=True)

            # Create the callback (subclass TrainerCallback from the imported module)
            class _Callback(TrainerCallbackBase):
                pass

            cb = DashboardCallback(
                metrics_queue=self._metrics_queue,
                control_flags=self.control_flags,
                model=model,
                tokenizer=tokenizer,
                save_dir=save_dir,
            )

            # Wrap DashboardCallback methods onto the TrainerCallback subclass
            callback_instance = _Callback()
            callback_instance.on_log = cb.on_log
            callback_instance.on_step_end = cb.on_step_end
            callback_instance.on_evaluate = cb.on_evaluate

            # Configure trainer
            output_dir = f"outputs/fine_tuning/runs/{adapter_name}"

            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=config.num_epochs,
                per_device_train_batch_size=config.batch_size,
                gradient_accumulation_steps=config.gradient_accumulation,
                learning_rate=config.learning_rate,
                warmup_ratio=config.warmup_ratio,
                logging_steps=config.logging_steps,
                eval_strategy="steps",
                eval_steps=config.eval_steps,
                save_strategy="no",
                bf16=True,
                optim="adamw_8bit",
                seed=42,
                report_to="none",
            )

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"],
                args=training_args,
                dataset_text_field="text",
                max_seq_length=config.max_seq_length,
                packing=True,
                dataset_num_proc=4,
                callbacks=[callback_instance],
            )

            # Calculate total steps
            total_steps = len(trainer.get_train_dataloader()) * config.num_epochs
            self.status.total_steps = total_steps

            # Start metrics poller
            self._poller_task = asyncio.create_task(self._poll_metrics_queue())

            # Transition to running
            self.status.state = "running"
            start_time = time.time()

            await self.broadcast_metrics({
                "type": "status",
                "state": "running",
                "total_steps": total_steps,
                "trainable_params": trainable_params,
                "total_params": total_params,
            })

            # Run trainer.train() in thread
            def _train():
                if config.resume_from:
                    trainer.train(resume_from_checkpoint=config.resume_from)
                else:
                    trainer.train()

            await loop.run_in_executor(None, _train)

            elapsed = time.time() - start_time
            self.status.elapsed_time = elapsed

            # Save final adapter if configured
            if config.save_adapter:
                final_path = save_dir / f"checkpoint-final"
                final_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(final_path))
                tokenizer.save_pretrained(str(final_path))

                metadata = {
                    "step": self.status.current_step,
                    "train_loss": _sanitize_float(self.status.train_loss),
                    "eval_loss": _sanitize_float(self.status.eval_loss),
                    "config": config.model_dump(),
                    "timestamp": datetime.now().isoformat(),
                    "trainable_params": trainable_params,
                    "total_params": total_params,
                    "elapsed_time": elapsed,
                }
                with open(final_path / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)

                await self.broadcast_metrics({
                    "type": "checkpoint",
                    "step": self.status.current_step,
                    "path": str(final_path),
                    "final": True,
                })

            self.status.state = "completed"
            await self.broadcast_metrics({
                "type": "complete",
                "final_step": self.status.current_step,
                "final_train_loss": _sanitize_float(self.status.train_loss),
                "final_eval_loss": _sanitize_float(self.status.eval_loss),
                "elapsed_time": elapsed,
            })

        except Exception as e:
            logger.error(f"Fine-tuning error: {e}", exc_info=True)
            self.status.state = "error"
            await self.broadcast_metrics({
                "type": "error",
                "message": str(e),
            })
        finally:
            # Stop poller
            if self._poller_task:
                self._poller_task.cancel()
                try:
                    await self._poller_task
                except asyncio.CancelledError:
                    pass
                self._poller_task = None

            # Clean up GPU memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            logger.info("GPU memory cleaned up after fine-tuning")

    def pause_training(self):
        if self.status.state != "running":
            raise HTTPException(status_code=400, detail="Fine-tuning not running")
        self.control_flags.should_pause = True
        self.status.state = "paused"

    def resume_training(self):
        if self.status.state != "paused":
            raise HTTPException(status_code=400, detail="Fine-tuning not paused")
        self.control_flags.should_pause = False
        self.status.state = "running"

    def stop_training(self):
        if self.status.state not in ("running", "paused"):
            raise HTTPException(status_code=400, detail="Fine-tuning not active")
        self.control_flags.should_stop = True
        self.control_flags.should_pause = False
        self.status.state = "completed"

    def trigger_save_now(self):
        if self.status.state != "running":
            raise HTTPException(status_code=400, detail="Fine-tuning not running")
        self.control_flags.should_save_now = True


# Global manager
fine_tuning_manager = FineTuningManager()


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/fine-tuning", tags=["fine-tuning"])


@router.get("/status", response_model=FineTuningStatus)
async def get_status():
    """Get current fine-tuning status."""
    return fine_tuning_manager.status


@router.post("/start", response_model=FineTuningStatus)
async def start_training(config: FineTuningConfig):
    """Start a new fine-tuning run."""
    await fine_tuning_manager.start_training(config)
    return fine_tuning_manager.status


@router.post("/pause", response_model=FineTuningStatus)
async def pause_training():
    """Pause the current fine-tuning run."""
    fine_tuning_manager.pause_training()
    return fine_tuning_manager.status


@router.post("/resume", response_model=FineTuningStatus)
async def resume_training():
    """Resume a paused fine-tuning run."""
    fine_tuning_manager.resume_training()
    return fine_tuning_manager.status


@router.post("/stop", response_model=FineTuningStatus)
async def stop_training():
    """Stop the current fine-tuning run."""
    fine_tuning_manager.stop_training()
    return fine_tuning_manager.status


@router.post("/checkpoint/save-now")
async def save_checkpoint_now():
    """Trigger immediate adapter checkpoint save."""
    fine_tuning_manager.trigger_save_now()
    return {
        "message": "Checkpoint save requested. Will be saved at next training step.",
        "step": fine_tuning_manager.status.current_step,
    }


@router.get("/checkpoints", response_model=List[AdapterCheckpointInfo])
async def list_checkpoints():
    """List available adapter checkpoints."""
    if not ADAPTER_BASE_DIR.exists():
        return []

    checkpoints = []
    for adapter_dir in sorted(ADAPTER_BASE_DIR.iterdir()):
        if not adapter_dir.is_dir():
            continue
        for ckpt_dir in sorted(adapter_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if not ckpt_dir.name.startswith("checkpoint-"):
                continue

            # Try to read metadata
            metadata_path = ckpt_dir / "metadata.json"
            step = 0
            train_loss = None
            eval_loss = None
            timestamp = None

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        meta = json.load(f)
                    step = meta.get("step", 0)
                    train_loss = _sanitize_float(meta.get("train_loss"))
                    eval_loss = _sanitize_float(meta.get("eval_loss"))
                    timestamp = meta.get("timestamp")
                except (json.JSONDecodeError, KeyError):
                    pass
            else:
                # Parse step from directory name
                try:
                    step = int(ckpt_dir.name.split("-")[1])
                except (ValueError, IndexError):
                    pass

            ckpt_id = f"{adapter_dir.name}/{ckpt_dir.name}"
            checkpoints.append(AdapterCheckpointInfo(
                id=ckpt_id,
                path=str(ckpt_dir),
                step=step,
                train_loss=train_loss,
                eval_loss=eval_loss,
                timestamp=timestamp,
            ))

    return checkpoints


@router.delete("/checkpoints")
async def delete_checkpoints():
    """Delete all adapter checkpoints."""
    if not ADAPTER_BASE_DIR.exists():
        return {"deleted_count": 0}

    import shutil
    count = 0
    for adapter_dir in list(ADAPTER_BASE_DIR.iterdir()):
        if adapter_dir.is_dir():
            for ckpt_dir in list(adapter_dir.iterdir()):
                if ckpt_dir.is_dir() and ckpt_dir.name.startswith("checkpoint-"):
                    shutil.rmtree(ckpt_dir)
                    count += 1
            # Remove adapter dir if empty
            remaining = list(adapter_dir.iterdir())
            if not remaining:
                adapter_dir.rmdir()

    return {"deleted_count": count}


@router.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text using a fine-tuned adapter or the base model."""
    import torch

    # Block generation while training is active (GPU contention)
    if fine_tuning_manager.status.state in ("running", "loading", "paused"):
        raise HTTPException(
            status_code=409,
            detail="Cannot generate while fine-tuning is active. Stop training first."
        )
    try:
        from .pretraining import training_manager as pt_manager
        if pt_manager.status.state in ("running", "loading", "paused"):
            raise HTTPException(
                status_code=409,
                detail="Cannot generate while pretraining is active. Stop training first."
            )
    except ImportError:
        pass

    loop = asyncio.get_event_loop()

    def _generate():
        import psutil  # noqa: F401
        import os
        os.environ["UNSLOTH_DISABLE_TRAINER_PATCHING"] = "1"

        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template

        mgr = fine_tuning_manager

        # Check if we can reuse cached model
        if (mgr._gen_model is not None
                and mgr._gen_adapter_path == request.adapter_path):
            model = mgr._gen_model
            tokenizer = mgr._gen_tokenizer
        else:
            # Clear any previous cached model
            mgr._gen_model = None
            mgr._gen_tokenizer = None
            mgr._gen_adapter_path = None

            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Load base model
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen2.5-7B-Instruct",
                max_seq_length=1024,
                load_in_4bit=True,
                dtype=None,
            )

            if request.adapter_path:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, request.adapter_path)

            tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")
            FastLanguageModel.for_inference(model)

            mgr._gen_model = model
            mgr._gen_tokenizer = tokenizer
            mgr._gen_adapter_path = request.adapter_path

        messages = [{"role": "user", "content": request.prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            do_sample=request.temperature > 0,
        )

        response_text = tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True,
        )

        return response_text, outputs.shape[1] - inputs.shape[1]

    try:
        text, tokens_generated = await loop.run_in_executor(None, _generate)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return GenerateResponse(
        text=text,
        tokens_generated=tokens_generated,
        prompt=request.prompt,
        adapter_path=request.adapter_path,
    )


# =============================================================================
# WebSocket — mounted separately in main.py
# =============================================================================

async def websocket_fine_tuning(websocket: WebSocket):
    """WebSocket endpoint for real-time fine-tuning metrics."""
    await websocket.accept()
    fine_tuning_manager.websocket_clients.append(websocket)
    logger.info(f"Fine-tuning WS client connected. Total: {len(fine_tuning_manager.websocket_clients)}")

    try:
        # Send current status
        await websocket.send_json({
            "type": "status",
            "state": fine_tuning_manager.status.state,
            "current_step": fine_tuning_manager.status.current_step,
        })

        # Send recent history
        for metrics in fine_tuning_manager.metrics_history[-100:]:
            await websocket.send_json(metrics)

        while True:
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})

    except WebSocketDisconnect:
        logger.info("Fine-tuning WS client disconnected")
    finally:
        if websocket in fine_tuning_manager.websocket_clients:
            fine_tuning_manager.websocket_clients.remove(websocket)
