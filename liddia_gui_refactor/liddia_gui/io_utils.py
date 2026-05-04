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
