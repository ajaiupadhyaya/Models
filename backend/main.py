"""
FastAPI app entrypoint for Financial Research & Trading Platform.
Run from repo root with PYTHONPATH=. so api, core, config, models are importable.
Usage: uvicorn backend.main:app --host 0.0.0.0 --port 8000
"""
from pathlib import Path
import sys

# Ensure project root is on path when running as backend.main
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from api.main import app  # noqa: E402

__all__ = ["app"]
