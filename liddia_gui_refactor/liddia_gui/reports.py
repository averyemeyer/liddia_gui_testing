"""Run report export helpers.

Reports are derived from LIDDIA-produced artifacts: the run JSON plus optional
memory-derived metrics. The GUI only creates the downloadable bundle.
"""
from __future__ import annotations

import json
import time
import zipfile
from pathlib import Path

import pandas as pd

from .config import REPORT_TMP_DIR
from .io_utils import latest_json_in_dir, safe_read_json
from .molecules import enrich_parsed_with_memory, resolve_run_dir
from .parsers import metric_rows, parse_run_data, requirements_rows, timeline_rows


def _load_parsed(run_dir_str: str, run_json_str: str) -> tuple[dict, Path | None, Path | None]:
    run_dir = resolve_run_dir(run_dir_str, run_json_str)
    run_json = Path(run_json_str) if run_json_str else latest_json_in_dir(run_dir)
    data = safe_read_json(run_json) if run_json else None
    parsed = enrich_parsed_with_memory(parse_run_data(data), str(run_dir or ""), str(run_json or ""))
    return parsed, run_dir, run_json


def build_report_json(run_dir_str: str, run_json_str: str) -> Path | None:
    parsed, run_dir, run_json = _load_parsed(run_dir_str, run_json_str)
    if not parsed or not run_json:
        return None
    payload = {
        "run_json_path": str(run_json),
        "run_dir": str(run_dir) if run_dir else None,
        "summary": parsed,
    }
    path = REPORT_TMP_DIR / f"liddia_report_{int(time.time())}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def build_report_csv(run_dir_str: str, run_json_str: str) -> Path | None:
    parsed, _, run_json = _load_parsed(run_dir_str, run_json_str)
    if not parsed or not run_json:
        return None
    rows = metric_rows(parsed)
    if not rows:
        return None
    path = REPORT_TMP_DIR / f"liddia_final_pool_metrics_{int(time.time())}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def build_report_txt(run_dir_str: str, run_json_str: str) -> Path | None:
    parsed, run_dir, run_json = _load_parsed(run_dir_str, run_json_str)
    if not parsed or not run_json:
        return None
    task = parsed.get("task") or {}
    final_pool = parsed.get("final_pool") or {}
    runtime = parsed.get("runtime") or {}
    lines = [
        "LIDDIA Run Report",
        "=" * 24,
        "",
        f"Run JSON: {run_json}",
        f"Run directory: {run_dir or '—'}",
        f"Target: {task.get('target', '—')}",
        f"Model: {parsed.get('model', '—')}",
        f"Success: {parsed.get('success', '—')}",
        f"Elapsed seconds: {runtime.get('elapsed_seconds', '—')}",
        "",
        "Final Pool",
        "-" * 10,
        f"Pool: {final_pool.get('pool', '—')}",
        f"Molecules: {final_pool.get('size', '—')}",
        f"Diversity: {final_pool.get('diversity', '—')}",
        "",
        "Requirements",
        "-" * 12,
    ]
    lines.extend(f"- {row['Requirement']}" for row in requirements_rows(parsed))
    lines.extend(["", "Timeline", "-" * 8])
    for row in timeline_rows(parsed):
        lines.append(f"- Step {row['Step']}: {row['Action']} -> {row['Output']} (Goal: {row['Goal']})")
    path = REPORT_TMP_DIR / f"liddia_report_{int(time.time())}.txt"
    path.write_text("\n".join(lines))
    return path


def build_report_bundle_file(run_dir_str: str, run_json_str: str):
    paths = [
        ("run_report.txt", build_report_txt(run_dir_str, run_json_str)),
        ("run_summary.json", build_report_json(run_dir_str, run_json_str)),
        ("final_pool_metrics.csv", build_report_csv(run_dir_str, run_json_str)),
    ]
    existing = [(name, path) for name, path in paths if path and path.exists()]
    if not existing:
        return None
    bundle_path = REPORT_TMP_DIR / f"liddia_report_bundle_{int(time.time())}.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in existing:
            archive.write(path, name)
    return str(bundle_path)

