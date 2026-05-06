"""LIDDIA artifact discovery.

GUI-owned files such as ``run_state.json`` are useful for recovery, but they
must never masquerade as LIDDIA result artifacts. Keep that policy here so
future backends can replace it in one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


GUI_JSON_NAMES = {"run_state.json"}


@dataclass(frozen=True)
class LiddiaArtifacts:
    run_dir: Path
    run_json: Path | None
    memory_files: tuple[Path, ...]
    pool_csvs: tuple[Path, ...]


def is_liddia_run_json(path: Path) -> bool:
    return path.suffix.lower() == ".json" and path.name not in GUI_JSON_NAMES and not path.name.startswith(".")


def discover_liddia_artifacts(run_dir: Path | None) -> LiddiaArtifacts | None:
    if not run_dir or not run_dir.exists() or not run_dir.is_dir():
        return None
    run_jsons = sorted([p for p in run_dir.glob("*.json") if is_liddia_run_json(p)], key=lambda p: p.stat().st_mtime, reverse=True)
    memory_files = tuple(sorted(run_dir.glob("*_memory.pkl"), key=lambda p: p.stat().st_mtime, reverse=True))
    pool_csvs = tuple(sorted([p for p in run_dir.glob("*.csv") if p.name.startswith("MOL")], key=lambda p: p.name))
    return LiddiaArtifacts(run_dir=run_dir, run_json=run_jsons[0] if run_jsons else None, memory_files=memory_files, pool_csvs=pool_csvs)

