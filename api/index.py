"""Vercel Serverless Function entry point for Q-Traffic CVRP Optimizer.

Exposes the FastAPI ASGI `app` instance for serverless invocations.
"""

from __future__ import annotations
import sys
from pathlib import Path

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure qtraffic directory is also in sys.path if present
QTRAFFIC_DIR = ROOT_DIR / "qtraffic"
if QTRAFFIC_DIR.is_dir() and str(QTRAFFIC_DIR) not in sys.path:
    sys.path.insert(0, str(QTRAFFIC_DIR))

from app.main import app

# Vercel looks for `app` attribute on the module
__all__ = ["app"]
