"""Local development server runner for Q-Traffic prototype.

Usage:
    python dev.py
"""

from __future__ import annotations
import sys
from pathlib import Path
import uvicorn

# Ensure root directory is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Q-Traffic CVRP Optimizer (SIH26137)")
    print("Access prototype at: http://127.0.0.1:8000")
    print("API Documentation at: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
