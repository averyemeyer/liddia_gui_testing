"""Helpers for reusing run settings without auto-launching."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunSetupValues:
    target: str | None = None
    max_iter: int | None = None
    model: str | None = None


def setup_values_from_run_data(data: dict[str, Any] | None) -> RunSetupValues:
    if not data:
        return RunSetupValues()
    task = data.get("task") if isinstance(data.get("task"), dict) else {}
    runtime = data.get("runtime") if isinstance(data.get("runtime"), dict) else {}
    max_iter = runtime.get("max_iter") or task.get("resource")
    try:
        max_iter = int(max_iter) if max_iter is not None else None
    except Exception:
        max_iter = None
    return RunSetupValues(
        target=task.get("target"),
        max_iter=max_iter,
        model=data.get("model"),
    )
