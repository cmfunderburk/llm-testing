"""
Backend API for LLM Pretraining Lab

FastAPI server providing:
- REST endpoints for training control and model analysis
- WebSocket streaming for real-time training metrics
- CORS support for frontend integration

Start the server:
    python -m experiments.pretraining.api.main

Or with uvicorn:
    uvicorn experiments.pretraining.api.main:app --reload --port 8000
"""

from .main import app, create_app

__all__ = ['app', 'create_app']
