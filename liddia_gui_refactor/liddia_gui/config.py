"""Central paths and default UI/runtime configuration.

Keep repo-specific constants here so future LIDDIA versions can be swapped
without hunting through the Gradio app.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

# The refactor package currently lives in liddia_gui_refactor/liddia_gui,
# while the LIDDIA runtime, dataset, and historical logs live at repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_PY = REPO_ROOT / "run.py"
LOG_ROOT = REPO_ROOT / "log"
PDB_DIR = REPO_ROOT / "dataset" / "pdb"
REPORT_TMP_DIR = Path(tempfile.gettempdir())

DEFAULT_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


def detect_targets() -> list[str]:
    """Return target names from dataset/pdb/*.pdb, with safe fallback."""
    if not PDB_DIR.exists():
        return ["EGFR", "BRAF", "JAK2", "DHFR"]
    targets = sorted({p.stem for p in PDB_DIR.glob("*.pdb")})
    return targets or ["EGFR", "BRAF", "JAK2", "DHFR"]
