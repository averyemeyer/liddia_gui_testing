"""Log and diagnostic helpers for the GUI.

The runner writes detached stdout/stderr files beside the normal run folders.
These helpers keep the UI resilient when a log is missing, still being written,
or referenced by a recovered lock file.
"""
from __future__ import annotations

from pathlib import Path

from .config import LOG_ROOT
from .run_state import read_lock


def tail_text(path: Path | str | None, max_chars: int = 20000) -> str:
    """Return the end of a text file without raising UI-breaking errors."""
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        text = file_path.read_text(errors="replace")
    except Exception as exc:
        return f"Could not read {file_path}: {exc}"
    return text[-max_chars:]


def active_log_text(log_root: Path = LOG_ROOT) -> str:
    """Return combined stdout/stderr for the active or most recent GUI run."""
    active = read_lock(log_root)
    stdout = active.stdout_log if active else None
    stderr = active.stderr_log if active else None

    if not stdout and not stderr and log_root.exists():
        stdout_logs = sorted(log_root.glob(".run_*.stdout.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        stderr_logs = sorted(log_root.glob(".run_*.stderr.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        stdout = str(stdout_logs[0]) if stdout_logs else None
        stderr = str(stderr_logs[0]) if stderr_logs else None

    out = tail_text(stdout)
    err = tail_text(stderr)
    if not out and not err:
        return "No CLI logs found yet."
    return f"--- STDOUT ---\n{out or '(empty)'}\n\n--- STDERR ---\n{err or '(empty)'}"

