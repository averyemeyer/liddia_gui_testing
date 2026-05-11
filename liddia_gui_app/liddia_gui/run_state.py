"""Persistent run state and lock-file recovery.

This is what lets a run continue after the browser tab closes. Gradio state is
convenient, but the source of truth must be disk files.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOG_ROOT


@dataclass
class ActiveRun:
    pid: int | None
    started_at: float
    known_dirs: list[str]
    backend_name: str = "liddia_v2"
    active_run_dir: str | None = None
    target: str | None = None
    model: str | None = None
    max_iter: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    recovered_notified: bool = False
    recovered_notice_until: float = 0.0


def lock_path(log_root: Path = LOG_ROOT) -> Path:
    return log_root / ".run.lock"


def last_run_path(log_root: Path = LOG_ROOT) -> Path:
    return log_root / ".last_run.json"


def read_lock(log_root: Path = LOG_ROOT) -> ActiveRun | None:
    path = lock_path(log_root)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return ActiveRun(**data)
    except Exception:
        return None


def write_lock(run: ActiveRun, log_root: Path = LOG_ROOT) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    lock_path(log_root).write_text(json.dumps(asdict(run), indent=2))


def clear_lock(log_root: Path = LOG_ROOT) -> None:
    lock_path(log_root).unlink(missing_ok=True)


def write_last_run(run_dir: Path | None, run_json: Path | None, log_root: Path = LOG_ROOT) -> None:
    if not run_dir and not run_json:
        return
    log_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(run_dir) if run_dir else None,
        "run_json": str(run_json) if run_json else None,
        "updated_at": datetime.now().isoformat(),
    }
    last_run_path(log_root).write_text(json.dumps(payload, indent=2))


def read_last_run(log_root: Path = LOG_ROOT) -> tuple[Path | None, Path | None]:
    path = last_run_path(log_root)
    if not path.exists():
        return latest_terminal_run(log_root)
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None, None
    run_dir = Path(data["run_dir"]) if data.get("run_dir") else None
    run_json = Path(data["run_json"]) if data.get("run_json") else None
    if run_json and not run_json.exists():
        run_json = None
    if run_dir and not run_dir.exists():
        run_dir = run_json.parent if run_json else None
    return run_dir, run_json


def latest_terminal_run(log_root: Path = LOG_ROOT) -> tuple[Path | None, Path | None]:
    """Find the most recent completed/failed run_state when no pointer exists."""
    if not log_root.exists():
        return None, None
    candidates: list[tuple[float, Path, Path | None]] = []
    for state_path in log_root.glob("*/run_state.json"):
        try:
            data = json.loads(state_path.read_text())
        except Exception:
            continue
        if data.get("status") not in {"completed", "failed", "cancelled"}:
            continue
        run_dir = Path(data["run_dir"]) if data.get("run_dir") else state_path.parent
        run_json = Path(data["run_json_path"]) if data.get("run_json_path") else None
        if run_json and not run_json.exists():
            run_json = None
        if run_json is None:
            json_candidates = [p for p in run_dir.glob("*.json") if p.name != "run_state.json"]
            run_json = sorted(json_candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0] if json_candidates else None
        if run_json:
            candidates.append((state_path.stat().st_mtime, run_dir, run_json))
    if not candidates:
        return None, None
    _, run_dir, run_json = sorted(candidates, key=lambda item: item[0], reverse=True)[0]
    return run_dir, run_json


def pid_running(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name != "nt":
        try:
            result = subprocess.run(
                ["ps", "-p", str(pid), "-o", "stat="],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if result.returncode != 0:
                return False
            if result.stdout.strip().startswith("Z"):
                return False
        except Exception:
            pass
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def run_state_path(run_dir: Path | None) -> Path | None:
    return run_dir / "run_state.json" if run_dir else None


def write_run_state(
    run_dir: Path | None,
    *,
    status: str,
    pid: int | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    target: str | None = None,
    model: str | None = None,
    max_iter: int | None = None,
    run_json_path: Path | None = None,
    stdout_log: str | None = None,
    stderr_log: str | None = None,
    backend_name: str = "liddia_v2",
) -> None:
    """Write a durable per-run status file for reload/recovery/debugging."""
    path = run_state_path(run_dir)
    if not path:
        return
    payload: dict[str, Any] = {
        "status": status,
        "backend_name": backend_name,
        "updated_at": datetime.now().isoformat(),
        "pid": pid,
        "started_at": started_at,
        "finished_at": finished_at,
        "target": target,
        "model": model,
        "max_iter": max_iter,
        "run_dir": str(run_dir),
        "run_json_path": str(run_json_path) if run_json_path else None,
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def run_has_activity_since(log_root: Path, started_at: float, known_dirs: set[str]) -> bool:
    """Detect whether a supposedly active run actually produced new artifacts."""
    if not log_root.exists():
        return False
    for child in log_root.iterdir():
        if child.is_dir() and child.name not in known_dirs:
            try:
                if child.stat().st_mtime >= started_at - 2:
                    return True
            except Exception:
                pass
    for json_file in log_root.glob("*/*.json"):
        try:
            if json_file.stat().st_mtime >= started_at - 2:
                return True
        except Exception:
            pass
    return False


def stale_lock(active: ActiveRun, log_root: Path = LOG_ROOT, max_quiet_seconds: int = 180) -> bool:
    """Return True when a lock probably survived a crashed/old process."""
    if not pid_running(active.pid):
        return True
    age = time.time() - float(active.started_at or 0)
    if age < max_quiet_seconds:
        return False
    return not run_has_activity_since(log_root, active.started_at, set(active.known_dirs or []))
