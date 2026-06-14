"""Molecule pool loading and table helpers.

The v2 run JSON contains summary statistics, while molecule-level properties
live in the per-run ``*_memory.pkl`` artifact. This module keeps pickle/RDKit
details out of the Gradio layout.
"""
from __future__ import annotations

import base64
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd


def resolve_run_dir(run_dir_str: str, run_json_str: str) -> Path | None:
    if run_dir_str:
        path = Path(run_dir_str)
        return path if path.exists() else None
    if run_json_str:
        path = Path(run_json_str)
        return path.parent if path.exists() else None
    return None


def load_memory(run_dir: Path | None) -> Any | None:
    if not run_dir:
        return None
    candidates = sorted(run_dir.glob("*_memory.pkl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        return None

    class DummyMemory:
        def __init__(self):
            self.stream = {}
            self.history = []

    previous = {name: sys.modules.get(name) for name in ["liddia", "liddia.memory", "liddia.action"]}
    liddia_mod = types.ModuleType("liddia")
    liddia_mod.__path__ = []
    memory_mod = types.ModuleType("liddia.memory")
    action_mod = types.ModuleType("liddia.action")
    memory_mod.Memory = DummyMemory
    for fn in ["sample_zinc", "graph_ga_optimizer", "run_code", "sample_pocket2mol"]:
        setattr(action_mod, fn, lambda *args, **kwargs: None)
    sys.modules["liddia"] = liddia_mod
    sys.modules["liddia.memory"] = memory_mod
    sys.modules["liddia.action"] = action_mod
    try:
        with candidates[-1].open("rb") as handle:
            return pickle.load(handle)
    except Exception:
        return None
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def pool_ids_for_run(run_dir_str: str, run_json_str: str) -> list[str]:
    mem = load_memory(resolve_run_dir(run_dir_str, run_json_str))
    if not mem:
        return []
    ids: list[str] = []
    for item in getattr(mem, "history", []) or []:
        pool_id = item.get("action_output")
        if pool_id and pool_id != "EMPTY SET":
            ids.append(str(pool_id))
    for pool_id, block in (getattr(mem, "stream", {}) or {}).items():
        if isinstance(block, dict) and isinstance(block.get("data"), pd.DataFrame):
            ids.append(str(pool_id))
    return list(dict.fromkeys(ids))


def pool_choices(run_dir_str: str, run_json_str: str) -> tuple[list[str], str | None]:
    choices = pool_ids_for_run(run_dir_str, run_json_str)
    return choices, choices[-1] if choices else None


def _pool_dataframe(run_dir_str: str, run_json_str: str, pool_id: str | None) -> pd.DataFrame:
    mem = load_memory(resolve_run_dir(run_dir_str, run_json_str))
    if not mem or not pool_id:
        return pd.DataFrame()
    block = getattr(mem, "stream", {}).get(pool_id, {})
    df = block.get("data")
    return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()


def normalize_metric_label(label: str) -> str:
    mapping = {
        "vina": "Vina Score",
        "vina score": "Vina Score",
        "qed": "QED",
        "sascore": "SAScore",
        "sa score": "SAScore",
        "lipinski": "Lipinski",
        "novelty": "Novelty",
        "diversity": "Diversity",
        "size": "Size",
    }
    return mapping.get(str(label).strip().lower(), str(label))


def _format_number(value: Any, decimals: int = 2) -> Any:
    if value in (None, "", "—"):
        return value
    try:
        return f"{float(value):.{decimals}f}"
    except Exception:
        return value


def _metric_stats_from_df(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        if str(col).lower() == "smiles":
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue
        stats[normalize_metric_label(str(col))] = {
            "min": _format_number(series.min()),
            "max": _format_number(series.max()),
            "median": _format_number(series.median()),
        }
    return stats


def pool_stats(run_dir_str: str, run_json_str: str, pool_id: str | None) -> dict[str, Any]:
    """Return pool-level stats from memory/CSV, normalized for parser UI helpers."""
    stats: dict[str, Any] = {"pool": pool_id, "size": None, "diversity": None, "metrics": {}}
    if not pool_id:
        return stats
    run_dir = resolve_run_dir(run_dir_str, run_json_str)
    mem = load_memory(run_dir)
    block = getattr(mem, "stream", {}).get(pool_id, {}) if mem else {}
    raw_metrics = block.get("metrics") if isinstance(block, dict) else {}
    if isinstance(raw_metrics, dict):
        stats["size"] = raw_metrics.get("size") or raw_metrics.get("Size")
        stats["diversity"] = _format_number(raw_metrics.get("diversity") or raw_metrics.get("Diversity"))
        for key, value in raw_metrics.items():
            label = normalize_metric_label(str(key))
            if label in {"Size", "Diversity"}:
                continue
            if isinstance(value, dict):
                stats["metrics"][label] = {
                    "min": _format_number(value.get("min")),
                    "max": _format_number(value.get("max")),
                    "median": _format_number(value.get("median")),
                }
    df = _pool_dataframe(run_dir_str, run_json_str, pool_id)
    if not df.empty:
        if stats["size"] is None:
            stats["size"] = len(df)
        if not stats["metrics"]:
            stats["metrics"] = _metric_stats_from_df(df)
    return stats


def enrich_parsed_with_memory(parsed: dict[str, Any], run_dir_str: str, run_json_str: str) -> dict[str, Any]:
    """Merge memory-derived metrics into parsed run data before rendering."""
    if not run_dir_str and not run_json_str:
        return parsed
    enriched = dict(parsed)
    steps = [dict(step) for step in (parsed.get("steps") or [])]
    for step in steps:
        pool_id = step.get("action_output")
        if not pool_id:
            continue
        memory_stats = pool_stats(run_dir_str, run_json_str, str(pool_id))
        existing = step.get("pool_stats") or {}
        merged = {**existing, **{k: v for k, v in memory_stats.items() if v not in (None, {}, "")}}
        if not memory_stats.get("metrics") and existing.get("metrics"):
            merged["metrics"] = existing["metrics"]
        step["pool_stats"] = merged
    enriched["steps"] = steps

    pool_ids = pool_ids_for_run(run_dir_str, run_json_str)
    final_pool_id = pool_ids[-1] if pool_ids else None
    if final_pool_id:
        enriched["final_pool"] = pool_stats(run_dir_str, run_json_str, final_pool_id)
    elif steps:
        enriched["final_pool"] = steps[-1].get("pool_stats") or {}
    return enriched


def smiles_image(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "N/A"
        image = Draw.MolToImage(mol, size=(360, 240))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        src = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"<img class='mol-thumb' src='{src}' alt='molecule'/>"
    except Exception:
        return "RDKit unavailable"


def molecule_table(run_dir_str: str, run_json_str: str, pool_id: str | None, max_rows: int = 100) -> pd.DataFrame:
    df = _pool_dataframe(run_dir_str, run_json_str, pool_id)
    if df.empty:
        return pd.DataFrame(columns=["Index", "Molecule"])
    out = df.copy()
    if "SMILES" in out.columns:
        out.insert(0, "Molecule", out["SMILES"].map(smiles_image))
        out = out.drop(columns=["SMILES"])
    out.insert(0, "Index", range(len(out)))
    for col in out.columns:
        if str(col).lower() == "index":
            continue
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: f"{float(value):.2f}" if pd.notna(value) else value)
    return out.head(max_rows)


def selected_pool_badge(pool_id: str | None) -> str:
    if not pool_id:
        return "<div class='empty-panel'>No molecule pool found for this run.</div>"
    return f"<span class='pool-badge'>Viewing pool {pool_id}</span>"


def download_all_pools_csv(run_dir_str: str, run_json_str: str):
    run_dir = resolve_run_dir(run_dir_str, run_json_str)
    ids = pool_ids_for_run(run_dir_str, run_json_str)
    if not run_dir or not ids:
        return None
    csv_exports: list[tuple[str, str]] = []
    for pool_id in ids:
        df = _pool_dataframe(run_dir_str, run_json_str, pool_id)
        if not df.empty:
            csv_exports.append((f"{pool_id}.csv", df.to_csv(index=False)))
    if not csv_exports:
        return None

    zip_path = run_dir / "all_pool_csvs.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, csv_text in csv_exports:
            archive.writestr(filename, csv_text)
    return str(zip_path)
