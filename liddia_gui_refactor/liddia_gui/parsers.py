"""Pure parsing utilities for run JSON and LIDDIA text outputs."""
from __future__ import annotations

import math
import re
import json
from typing import Any

ANSWER_PATTERN = re.compile(r"Answer:\s*(YES|NO)", re.IGNORECASE)
SIZE_PATTERN = re.compile(r"Size:\s*(\d+)", re.IGNORECASE)
DIVERSITY_PATTERN = re.compile(r"Diversity:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
POOL_NAME_PATTERN = re.compile(r"Molecule\s+Set\s+(MOL\d+)", re.IGNORECASE)
METRIC_PATTERN = re.compile(
    r"^\s*-?\s*(?P<label>[A-Za-z ]+):\s*Range\s*(?P<min>-?\d+(?:\.\d+)?)\s*to\s*(?P<max>-?\d+(?:\.\d+)?)(?:,\s*Median\s*(?P<median>-?\d+(?:\.\d+)?))?",
    re.IGNORECASE | re.MULTILINE,
)


def fmt_num(value: Any, decimals: int = 2) -> str:
    """Truncate display numbers without changing underlying raw data."""
    if value in (None, ""):
        return "—"
    try:
        factor = 10**decimals
        num = math.trunc(float(value) * factor) / factor
        return f"{num:.{decimals}f}"
    except Exception:
        return str(value)


def extract_iterations(run_data: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    items = [(k, v) for k, v in run_data.items() if k.isdigit() and isinstance(v, dict)]
    return sorted(items, key=lambda kv: int(kv[0]))


def parse_goal_check(text: str) -> dict[str, str | None]:
    match = ANSWER_PATTERN.search(text or "")
    return {"answer": match.group(1).upper() if match else None}


def parse_pool_stats(text: str) -> dict[str, Any]:
    stats: dict[str, Any] = {"pool": None, "size": None, "diversity": None, "metrics": {}}
    if not text:
        return stats
    if m := POOL_NAME_PATTERN.search(text):
        stats["pool"] = m.group(1).upper()
    if m := SIZE_PATTERN.search(text):
        stats["size"] = int(m.group(1))
    if m := DIVERSITY_PATTERN.search(text):
        stats["diversity"] = fmt_num(m.group(1))
    for m in METRIC_PATTERN.finditer(text):
        label = " ".join(m.group("label").split())
        stats["metrics"][label] = {"min": fmt_num(m.group("min")), "max": fmt_num(m.group("max")), "median": fmt_num(m.group("median"))}
    return stats


def parse_run_data(run_data: dict[str, Any] | None) -> dict[str, Any]:
    """Convert raw run JSON into one stable shape for UI rendering."""
    if not run_data:
        return {"model": None, "success": None, "cancelled": None, "error_message": None, "task": {}, "runtime": {}, "steps": [], "final_pool": {}, "step_count": 0}

    steps: list[dict[str, Any]] = []
    for idx, payload in extract_iterations(run_data):
        action = payload.get("action")
        action_name = action[0] if isinstance(action, list) and action else None
        action_input = action[1] if isinstance(action, list) and len(action) > 1 else None
        pool_text = "\n".join([payload.get("input_goal_prompt", ""), payload.get("goal_response", "")])
        steps.append({
            "step": int(idx),
            "action_name": action_name,
            "action_input": action_input,
            "action_output": payload.get("action_output"),
            "goal_response": payload.get("goal_response", ""),
            "goal_eval": parse_goal_check(payload.get("goal_response", "")),
            "pool_stats": parse_pool_stats(pool_text),
            "error_message": payload.get("error_message"),
            "stop_reason": payload.get("stop_reason"),
        })

    return {
        "model": run_data.get("model"),
        "success": run_data.get("success") if "success" in run_data else None,
        "cancelled": run_data.get("cancelled", False),
        "error_message": run_data.get("error_message"),
        "stop_reason": run_data.get("stop_reason"),
        "task": run_data.get("task", {}) if isinstance(run_data.get("task"), dict) else {},
        "runtime": run_data.get("runtime", {}) if isinstance(run_data.get("runtime"), dict) else {},
        "steps": steps,
        "final_pool": steps[-1]["pool_stats"] if steps else {},
        "step_count": len(steps),
    }


def metric_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten final pool metrics for a Gradio Dataframe."""
    final_pool = parsed.get("final_pool") or {}
    rows: list[dict[str, Any]] = []
    if final_pool.get("size") is not None:
        rows.append({"Metric": "Size", "Min": "—", "Median": final_pool.get("size"), "Max": "—", "Help": "Number of molecules in the selected pool."})
    if final_pool.get("diversity") is not None:
        rows.append({"Metric": "Diversity", "Min": "—", "Median": final_pool.get("diversity"), "Max": "—", "Help": "Set-level structural spread; higher is usually better."})
    for label, values in (final_pool.get("metrics") or {}).items():
        rows.append({
            "Metric": label,
            "Min": values.get("min", "—"),
            "Median": values.get("median", "—"),
            "Max": values.get("max", "—"),
            "Help": metric_help(label),
        })
    return rows


def compact_metric_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Small metric table for the narrow monitor sidebar."""
    final_pool = parsed.get("final_pool") or {}
    rows: list[dict[str, Any]] = []
    if final_pool.get("size") is not None:
        rows.append({"Metric": "Size", "Value": final_pool.get("size")})
    if final_pool.get("diversity") is not None:
        rows.append({"Metric": "Diversity", "Value": final_pool.get("diversity")})
    for label, values in (final_pool.get("metrics") or {}).items():
        rows.append({"Metric": label, "Value": values.get("median") or values.get("min") or values.get("max") or "—"})
    return rows


def timeline_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten iterations into a compact action timeline."""
    rows: list[dict[str, Any]] = []
    for step in parsed.get("steps") or []:
        goal = (step.get("goal_eval") or {}).get("answer") or "—"
        rows.append({
            "Step": step.get("step"),
            "Action": step.get("action_name") or "—",
            "Input": compact_value(step.get("action_input")),
            "Output": step.get("action_output") or "—",
            "Goal": goal,
            "Stop/Error": step.get("stop_reason") or step.get("error_message") or "—",
        })
    return rows


def requirements_rows(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    task = parsed.get("task") or {}
    return [{"Requirement": req} for req in task.get("requirements", []) or []]


def raw_json_text(run_data: dict[str, Any] | None) -> str:
    return json.dumps(run_data, indent=2, ensure_ascii=False) if run_data else ""


def compact_value(value: Any, limit: int = 120) -> str:
    text = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else str(value if value is not None else "—")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def metric_help(label: str) -> str:
    normalized = " ".join(str(label).lower().split())
    help_by_label = {
        "vina score": "Estimated binding score from docking; lower is better.",
        "novelty": "Distance from known target ligands; higher means less similar.",
        "lipinski": "Count of Lipinski rules followed; higher is usually better.",
        "qed": "Drug-likeness estimate from 0 to 1; higher is better.",
        "sascore": "Synthetic accessibility score; lower is usually easier to synthesize.",
        "diversity": "Structural spread across the pool; higher is better.",
    }
    return help_by_label.get(normalized, "Metric reported by the current LIDDIA backend.")
