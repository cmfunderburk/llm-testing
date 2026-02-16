"""
Unified API Server for LLM Learning Lab

Main FastAPI application that serves all learning tracks:
- Pretraining: GPT training with real-time visualization
- Attention: Attention pattern extraction and analysis
- Probing: Activation extraction and representation analysis

Usage:
    uvicorn api.main:app --reload --port 8000

    Or directly:
    python -m api.main
"""

import asyncio
import argparse
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .config import config
from .routers import (
    pretraining_router,
    attention_router,
    probing_router,
    fine_tuning_router,
    education_router,
)
from .routers.pretraining import websocket_training
from .routers.fine_tuning import websocket_fine_tuning
from .services import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Pydantic Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    tracks: list[str]


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting LLM Learning Lab API server...")
    logger.info("Available tracks: pretraining, fine-tuning, attention, probing, education")
    yield
    logger.info("Shutting down API server...")

    # Cleanup: unload models
    if model_manager.is_model_loaded:
        model_manager.unload_model()

    # Stop any running training
    from .routers.pretraining import training_manager
    if training_manager.training_task:
        training_manager.should_stop = True
        await asyncio.sleep(0.5)

    # Stop any running fine-tuning
    from .routers.fine_tuning import fine_tuning_manager
    if fine_tuning_manager.training_task:
        fine_tuning_manager.control_flags.should_stop = True
        await asyncio.sleep(0.5)


# =============================================================================
# Application Factory
# =============================================================================

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="LLM Learning Lab API",
        description="""
Unified API for the LLM Learning Lab - a hands-on learning environment
for building intuition about how LLMs work.

## Tracks

- **Pretraining** (`/api/pretraining/*`): Train GPT models from scratch with real-time metrics
- **Fine-Tuning** (`/api/fine-tuning/*`): QLoRA fine-tuning with real-time metrics
- **Attention** (`/api/attention/*`): Extract and visualize attention patterns
- **Probing** (`/api/probing/*`): Analyze intermediate activations and representations
- **Education** (`/api/education/*`): Canonical educational artifacts (e.g., microgpt)

## WebSocket

- `/ws/training`: Real-time pretraining metrics stream
- `/ws/fine-tuning`: Real-time fine-tuning metrics stream
        """,
        version="0.2.0",
        lifespan=lifespan,
    )

    # Configure CORS for frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount routers
    app.include_router(pretraining_router)
    app.include_router(fine_tuning_router)
    app.include_router(attention_router)
    app.include_router(probing_router)
    app.include_router(education_router)

    return app


app = create_app()


# =============================================================================
# Root Endpoints
# =============================================================================

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.now().isoformat(),
        version="0.2.0",
        tracks=["pretraining", "fine-tuning", "attention", "probing", "education"],
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "LLM Learning Lab API",
        "version": "0.2.0",
        "docs": "/docs",
        "tracks": {
            "pretraining": "/api/pretraining",
            "fine_tuning": "/api/fine-tuning",
            "attention": "/api/attention",
            "probing": "/api/probing",
            "education": "/api/education",
        },
    }


# =============================================================================
# WebSocket Endpoint
# =============================================================================

@app.websocket("/ws/training")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time training metrics."""
    await websocket_training(websocket)


@app.websocket("/ws/fine-tuning")
async def websocket_fine_tuning_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fine-tuning metrics."""
    await websocket_fine_tuning(websocket)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run the server."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Start the LLM Learning Lab API server")
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')

    args = parser.parse_args()

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == '__main__':
    main()
