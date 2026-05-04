"""Task metadata helpers derived from LIDDIA datasets."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .config import DRUGS_CSV


def task_context(target: str | None, max_iter: int | None = None, drug_csv: Path = DRUGS_CSV) -> dict[str, Any]:
    """Return the task block LIDDIA will use, best-effort for live UI state."""
    target = str(target or "").strip()
    task: dict[str, Any] = {}
    if target:
        task["target"] = target
    if max_iter is not None:
        task["resource"] = max_iter
    if not target or not drug_csv.exists():
        return task

    try:
        drugs = pd.read_csv(drug_csv)
        grouped = drugs[["NAME", "QED", "SAScore", "Lipinski Rules Followed", "Vina Score"]].groupby("NAME").mean().reset_index()
        grouped["Lipinski"] = grouped["Lipinski Rules Followed"]
        row = grouped[grouped["NAME"].astype(str) == target]
        if row.empty:
            return task
        values = row.iloc[0]
        task["requirements"] = [
            "At least 5 molecules",
            f"Vina Score must be lower than {values['Vina Score']:.2f}",
            "Novelty must be at least 0.80",
            "Diversity must be at least 0.80",
            f"QED must be better than {values['QED']:.2f}",
            f"SAScore must be better than {values['SAScore']:.2f}",
            f"Lipinski must be better than or at least {values['Lipinski']:.2f}",
        ]
        task["pocket"] = f"{target}.pdb"
    except Exception:
        return task
    return task


def merge_task_context(data: dict[str, Any], target: str | None, max_iter: int | None) -> dict[str, Any]:
    """Fill missing task fields in run JSON while preserving LIDDIA output."""
    merged = dict(data or {})
    existing = merged.get("task") if isinstance(merged.get("task"), dict) else {}
    fallback = task_context(target, max_iter)
    merged["task"] = {**fallback, **existing}
    return merged
