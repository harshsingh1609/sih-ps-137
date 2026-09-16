"""Root runner proxy for run_demo.py."""

import sys
from pathlib import Path

QTRAFFIC_DIR = Path(__file__).resolve().parent / "qtraffic"
if str(QTRAFFIC_DIR) not in sys.path:
    sys.path.insert(0, str(QTRAFFIC_DIR))

import run_demo

if __name__ == "__main__":
    run_demo.main()
