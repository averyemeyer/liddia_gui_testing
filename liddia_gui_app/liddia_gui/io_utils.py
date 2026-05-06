"""Small, defensive file I/O helpers used by the GUI."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .artifacts import discover_liddia_artifacts

_JSON_CACHE: dict[str, dict[str, Any]] = {}


def safe_read_json(path: Path | None, retries: int = 4) -> dict[str, Any] | None:
    """Read JSON that may be actively rewritten by run.py.

    A run can update its JSON while the GUI is polling, so JSONDecodeError is
    retried briefly instead of causing UI flicker.
    """
    if not path or not path.exists():
        return None
    cache_key = str(path.resolve())
    for _ in range(retries):
        try:
            text = path.read_text()
            if not text.strip():
                time.sleep(0.05)
                continue
            data = json.loads(text)
            if isinstance(data, dict):
                _JSON_CACHE[cache_key] = data
                return data
            return None
        except json.JSONDecodeError:
            time.sleep(0.05)
        except Exception:
            break
    return _JSON_CACHE.get(cache_key)


def latest_json_in_dir(run_dir: Path | None) -> Path | None:
    """Return the newest JSON artifact inside a run directory."""
    artifacts = discover_liddia_artifacts(run_dir)
    return artifacts.run_json if artifacts else None


def available_run_dirs(log_root: Path) -> list[str]:
    """Return run folder names that contain at least one JSON artifact."""
    if not log_root.exists():
        return []
    names = []
    for child in log_root.iterdir():
        if child.is_dir() and latest_json_in_dir(child):
            names.append(child.name)
    return sorted(names, reverse=True)


def run_dir_choices(log_root: Path) -> list[tuple[str, str]]:
    """Return dropdown choices as (display label, run folder name)."""
    choices: list[tuple[str, str]] = []
    for folder in available_run_dirs(log_root):
        run_dir = log_root / folder
        label = run_choice_label(run_dir, folder)
        choices.append((label, folder))
    return choices


def run_choice_label(run_dir: Path, fallback_name: str | None = None) -> str:
    """Build a compact, artifact-derived label for a run folder."""
    folder = fallback_name or run_dir.name
    run_json = latest_json_in_dir(run_dir)
    data = safe_read_json(run_json) if run_json else None
    state = safe_read_json(run_dir / "run_state.json")

    task = data.get("task") if isinstance(data, dict) and isinstance(data.get("task"), dict) else {}
    target = task.get("target") or (state or {}).get("target") or _target_from_folder(folder)
    model = (data or {}).get("model") or (state or {}).get("model") or "model unknown"
    status = _run_status(data, state)
    return f"{status} | {target or 'target unknown'} | {folder} | {model}"


def _target_from_folder(folder: str) -> str | None:
    if "_" not in folder:
        return None
    return folder.rsplit("_", 1)[-1] or None


def _run_status(data: dict[str, Any] | None, state: dict[str, Any] | None) -> str:
    status = str((state or {}).get("status") or "").upper()
    if status == "RUNNING":
        return "RUNNING"
    if isinstance(data, dict):
        if data.get("error_message"):
            return "ERROR"
        if data.get("cancelled"):
            return "CANCELLED"
        if data.get("success") is True:
            return "SUCCESS"
        if data.get("success") is False:
            return "FAILED"
    if status:
        return status
    return "UNKNOWN"
