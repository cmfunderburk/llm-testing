"""
Education Router

Endpoints for canonical educational artifacts used by the dashboard.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


class MicroGPTArtifactResponse(BaseModel):
    """Canonical microgpt educational artifact payload."""
    source_path: str
    docs_path: str
    source_sha256: str
    source_line_count: int
    docs_line_count: int
    source: str
    docs_markdown: str


router = APIRouter(prefix="/api/education", tags=["education"])


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_required_file(path: Path, label: str) -> str:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{label} not found at {path}")
    return path.read_text(encoding="utf-8")


@router.get("/microgpt", response_model=MicroGPTArtifactResponse)
async def get_microgpt_artifact():
    """
    Return canonical `misc/microgpt.py` source and its companion educational guide.

    This endpoint intentionally reads from repository files on each request so the
    dashboard reflects the current source of truth.
    """
    root = _repo_root()
    source_path = root / "misc" / "microgpt.py"
    docs_path = root / "docs" / "microgpt_line_by_line.md"

    source = _read_required_file(source_path, "microgpt source")
    docs_markdown = _read_required_file(docs_path, "microgpt documentation")
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()

    return MicroGPTArtifactResponse(
        source_path=str(source_path.relative_to(root)),
        docs_path=str(docs_path.relative_to(root)),
        source_sha256=source_hash,
        source_line_count=len(source.splitlines()),
        docs_line_count=len(docs_markdown.splitlines()),
        source=source,
        docs_markdown=docs_markdown,
    )

