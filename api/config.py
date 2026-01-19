"""
API Configuration

Central configuration for the unified API server.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class APIConfig:
    """API server configuration."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS settings
    cors_origins: List[str] = None

    # Model settings
    default_model: str = "unsloth/Qwen2.5-7B-Instruct"
    max_seq_length: int = 1024
    load_in_4bit: bool = True

    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = [
                "http://localhost:3000",
                "http://localhost:5173",
            ]


# Global config instance
config = APIConfig()
