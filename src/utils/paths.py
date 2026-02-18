from __future__ import annotations
from pathlib import Path

def project_root() -> Path:
    # src/utils/paths.py -> src/utils -> src -> repo root
    return Path(__file__).resolve().parents[2]
