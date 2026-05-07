"""Backend adapter boundary for LIDDIA v2/v3/etc.

The GUI should not know the exact CLI args, artifact names, or output layout
for every LIDDIA version. Put that knowledge behind a backend adapter.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Protocol

from .config import LOG_ROOT, RUN_PY
from .io_utils import latest_json_in_dir, safe_read_json


@dataclass(frozen=True)
class RunConfig:
    target: str
    max_iter: int
    model: str
    provider: str = "anthropic"


@dataclass(frozen=True)
class RunSnapshot:
    run_dir: Path | None
    run_json: Path | None
    data: dict[str, Any] | None


class LiddiaBackend(Protocol):
    """Interface a future backend must satisfy."""

    name: str
    log_root: Path

    def build_command(self, config: RunConfig) -> list[str]: ...
    def detect_run_dir(self, started_at: float, known_dirs: set[str]) -> Path | None: ...
    def load_snapshot(self, run_dir: Path | None, run_json: Path | None = None) -> RunSnapshot: ...


class LiddiaV2Backend:
    """Adapter for current LIDDIA v2 repo behavior.

    Assumptions isolated here:
    - Runs are launched through run.py.
    - Run artifacts are written under log/<run_id>/.
    - Each run folder contains one or more JSON files.
    """

    name = "liddia_v2"

    def __init__(self, run_py: Path = RUN_PY, log_root: Path = LOG_ROOT, python_executable: str | None = None):
        self.run_py = run_py
        self.log_root = log_root
        self.python_executable = python_executable

    @classmethod
    def resolve_python_executable(cls) -> str:
        """Choose the Python used for LIDDIA subprocesses.

        Launchers may set LIDDIA_RUN_PYTHON after activating a local conda/env.
        Otherwise, use the same interpreter that is running the GUI.
        """
        return os.environ.get("LIDDIA_RUN_PYTHON") or sys.executable

    def build_command(self, config: RunConfig) -> list[str]:
        cmd = [
            self.python_executable or self.resolve_python_executable(),
            "-u",
            str(self.run_py),
            "--target",
            config.target.strip(),
            "--max_iter",
            str(int(config.max_iter)),
            "--model",
            config.model.strip(),
        ]
        return cmd

    def detect_run_dir(self, started_at: float, known_dirs: set[str]) -> Path | None:
        if not self.log_root.exists():
            return None
        candidates: list[Path] = []
        for child in self.log_root.iterdir():
            if not child.is_dir() or child.name in known_dirs:
                continue
            try:
                if child.stat().st_mtime >= started_at - 2:
                    candidates.append(child)
            except Exception:
                continue
        return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0] if candidates else None

    def load_snapshot(self, run_dir: Path | None, run_json: Path | None = None) -> RunSnapshot:
        if run_json and run_json.exists():
            return RunSnapshot(run_json.parent, run_json, safe_read_json(run_json))
        run_json = latest_json_in_dir(run_dir)
        return RunSnapshot(run_dir, run_json, safe_read_json(run_json))


BACKENDS: dict[str, LiddiaBackend] = {
    "liddia_v2": LiddiaV2Backend(),
}
