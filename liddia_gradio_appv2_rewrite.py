import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import pickle
import types
import sys

REPO_ROOT = Path(__file__).resolve().parent
RUN_PY = REPO_ROOT / "run.py"
LOG_ROOT = REPO_ROOT / "log"
PDB_DIR = REPO_ROOT / "dataset" / "pdb"

def _detect_targets() -> List[str]:
    if not PDB_DIR.exists():
        return []
    targets = []
    for path in PDB_DIR.glob("*.pdb"):
        targets.append(path.stem)
    return sorted(set(targets))


DEFAULT_TARGETS = _detect_targets() or ["EGFR", "BRAF", "JAK2", "DHFR"]
DEFAULT_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]

METRIC_PATTERN = re.compile(
    r"-\s*(?P<label>[A-Za-z ]+):\s*Range\s*(?P<min>-?\d+(?:\.\d+)?)\s*to\s*(?P<max>-?\d+(?:\.\d+)?)(?:,\s*Median\s*(?P<median>-?\d+(?:\.\d+)?))?",
    re.IGNORECASE,
)
METRIC_RANGE_SENTENCE_PATTERN = re.compile(
    r"(?P<label>Vina Score|Novelty|QED|SAScore|Lipinski)\s+range\s+of\s+(?P<min>-?\d+(?:\.\d+)?)\s+to\s+(?P<max>-?\d+(?:\.\d+)?)(?:\s*\(median\s*(?P<median>-?\d+(?:\.\d+)?)\))?",
    re.IGNORECASE,
)
METRIC_LOW_HIGH_PATTERN = re.compile(
    r"(?P<label>Vina Score|Novelty|QED|SAScore|Lipinski)\s+range\s+includes\s+values\s+as\s+low\s+as\s+(?P<min>-?\d+(?:\.\d+)?)\s+and\s+\w+\s+as\s+high\s+as\s+(?P<max>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
POOL_HEADER_PATTERN = re.compile(
    r"-\s*Molecule\s+Set\s+(?P<pool>MOL\d+):\s*\n\s*Size:\s*(?P<size>\d+)",
    re.IGNORECASE,
)
POOL_NAME_PATTERN = re.compile(r"Molecule\s+Set\s+(MOL\d+)", re.IGNORECASE)
SIZE_PATTERN = re.compile(r"Size:\s*(\d+)", re.IGNORECASE)
DIVERSITY_PATTERN = re.compile(r"Diversity:\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"Answer:\s*(YES|NO)", re.IGNORECASE)
REASON_PATTERN = re.compile(r"Reason:\s*(.+?)(?:\nAnswer:|$)", re.IGNORECASE | re.DOTALL)


# ---------- basic I/O ----------
def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _latest_run_json() -> Optional[Path]:
    if not LOG_ROOT.exists():
        return None
    candidates = sorted(
        LOG_ROOT.glob("*/*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _latest_run_json_in_dir(run_dir: Optional[Path]) -> Optional[Path]:
    if not run_dir:
        return None
    candidates = sorted(
        run_dir.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _detect_new_run_dir(existing_dirs: set, started_at: float) -> Optional[Path]:
    if not LOG_ROOT.exists():
        return None
    candidates = []
    for child in LOG_ROOT.iterdir():
        if not child.is_dir():
            continue
        if child.name in existing_dirs:
            continue
        try:
            mtime = child.stat().st_mtime
        except Exception:
            continue
        if mtime >= started_at - 2:
            candidates.append(child)
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _extract_iterations(run_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    for key, value in run_data.items():
        if key.isdigit() and isinstance(value, dict):
            items.append((key, value))
    return sorted(items, key=lambda x: int(x[0]))


# ---------- parsing helpers ----------
def _coerce_number(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_goal_check(goal_text: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"answer": None, "reason": None}
    if not goal_text:
        return result

    answer_match = ANSWER_PATTERN.search(goal_text)
    reason_match = REASON_PATTERN.search(goal_text)

    if answer_match:
        result["answer"] = answer_match.group(1).upper()
    if reason_match:
        result["reason"] = " ".join(reason_match.group(1).strip().split())

    return result


def _parse_pool_stats(text: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "pool": None,
        "size": None,
        "diversity": None,
        "metrics": {},
    }
    if not text:
        return stats

    pool_match = POOL_NAME_PATTERN.search(text)
    if pool_match:
        stats["pool"] = pool_match.group(1).upper()

    size_match = SIZE_PATTERN.search(text)
    if size_match:
        stats["size"] = int(size_match.group(1))

    diversity_match = DIVERSITY_PATTERN.search(text)
    if diversity_match:
        stats["diversity"] = float(diversity_match.group(1))

    for match in METRIC_PATTERN.finditer(text):
        label = " ".join(match.group("label").split())
        stats["metrics"][label] = {
            "min": _coerce_number(match.group("min")),
            "max": _coerce_number(match.group("max")),
            "median": _coerce_number(match.group("median")),
        }

    for match in METRIC_RANGE_SENTENCE_PATTERN.finditer(text):
        label = " ".join(match.group("label").split())
        stats["metrics"].setdefault(label, {})
        stats["metrics"][label].update(
            {
                "min": _coerce_number(match.group("min")),
                "max": _coerce_number(match.group("max")),
                "median": _coerce_number(match.group("median")),
            }
        )

    for match in METRIC_LOW_HIGH_PATTERN.finditer(text):
        label = " ".join(match.group("label").split())
        stats["metrics"].setdefault(label, {})
        stats["metrics"][label].update(
            {
                "min": _coerce_number(match.group("min")),
                "max": _coerce_number(match.group("max")),
            }
        )

    return stats


def parse_run_data(run_data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not run_data:
        return {
            "model": None,
            "success": None,
            "cancelled": None,
            "error_message": None,
            "task": {},
            "runtime": {},
            "steps": [],
            "final_pool": {},
            "step_count": 0,
        }

    steps: List[Dict[str, Any]] = []
    for idx, payload in _extract_iterations(run_data):
        action = payload.get("action")
        action_name = None
        action_input = None
        if isinstance(action, list) and action:
            action_name = action[0]
            if len(action) > 1:
                action_input = action[1]

        pool_text = "\n".join(
            [
                payload.get("input_goal_prompt", ""),
                payload.get("goal_response", ""),
            ]
        )
        pool_stats = _parse_pool_stats(pool_text)
        goal_eval = _parse_goal_check(payload.get("goal_response", ""))

        steps.append(
            {
                "step": int(idx),
                "action_name": action_name,
                "action_input": action_input,
                "action_output": payload.get("action_output"),
                "response": payload.get("response", ""),
                "input_prompt": payload.get("input_prompt", ""),
                "input_goal_prompt": payload.get("input_goal_prompt", ""),
                "goal_response": payload.get("goal_response", ""),
                "pool_stats": pool_stats,
                "goal_eval": goal_eval,
            }
        )

    final_pool = steps[-1]["pool_stats"] if steps else {}
    return {
        "model": run_data.get("model"),
        "success": run_data.get("success") if "success" in run_data else None,
        "cancelled": run_data.get("cancelled", False),
        "error_message": run_data.get("error_message"),
        "task": run_data.get("task", {}) if isinstance(run_data.get("task"), dict) else {},
        "runtime": run_data.get("runtime", {}) if isinstance(run_data.get("runtime"), dict) else {},
        "steps": steps,
        "final_pool": final_pool,
        "step_count": len(steps),
    }


# ---------- render helpers ----------
def _fmt_num(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "—"
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.{digits}f}"


def _metric_value(pool: Dict[str, Any], label: str, field: str) -> Optional[float]:
    return (((pool or {}).get("metrics") or {}).get(label) or {}).get(field)


def build_run_overview(parsed: Dict[str, Any], run_json_path: Optional[Path]) -> str:
    task = parsed.get("task", {})
    final_pool = parsed.get("final_pool", {})
    lines: List[str] = []

    has_real_data = bool(task.get("target") or parsed.get("model") or parsed.get("step_count"))
    if not has_real_data:
        return "Run overview will appear once a run has produced results."

    status = "IN PROGRESS"
    if parsed.get("success") is True:
        status = "SUCCESS"
    elif parsed.get("success") is False:
        status = "COMPLETED"
    if parsed.get("cancelled"):
        status = "CANCELLED"
    if parsed.get("error_message"):
        status = "FAILED"
    lines.append(f"Status: {status}")
    lines.append(f"Target: {task.get('target', 'Unknown')}")
    lines.append(f"Model: {parsed.get('model') or 'Unknown'}")
    lines.append(f"Iterations executed: {parsed.get('step_count', 0)}")
    if run_json_path:
        lines.append(f"Run JSON: {run_json_path}")
    if task.get("pocket"):
        lines.append(f"Pocket: {task.get('pocket')}")
    if task.get("resource") is not None:
        lines.append(f"Resource budget: {task.get('resource')}")

    metrics = task.get("metrics")
    if metrics:
        if isinstance(metrics, dict):
            lines.append("Task metrics: " + ", ".join(metrics.keys()))
        elif isinstance(metrics, list):
            lines.append("Task metrics: " + ", ".join(str(m) for m in metrics))

    if final_pool:
        lines.append("")
        lines.append("Final pool")
        lines.append(f"- Pool ID: {final_pool.get('pool') or 'Unknown'}")
        lines.append(f"- Molecules: {final_pool.get('size') or '—'}")
        lines.append(f"- Diversity: {_fmt_num(final_pool.get('diversity'))}")
        lines.append(
            f"- Vina range: {_fmt_num(_metric_value(final_pool, 'Vina Score', 'min'))} to {_fmt_num(_metric_value(final_pool, 'Vina Score', 'max'))}"
        )
        lines.append(
            f"- Novelty range: {_fmt_num(_metric_value(final_pool, 'Novelty', 'min'))} to {_fmt_num(_metric_value(final_pool, 'Novelty', 'max'))}"
        )
        lines.append(
            f"- QED range: {_fmt_num(_metric_value(final_pool, 'QED', 'min'))} to {_fmt_num(_metric_value(final_pool, 'QED', 'max'))}"
        )
        lines.append(
            f"- SAScore range: {_fmt_num(_metric_value(final_pool, 'SAScore', 'min'))} to {_fmt_num(_metric_value(final_pool, 'SAScore', 'max'))}"
        )

    if parsed.get("error_message"):
        lines.append("")
        lines.append("Error")
        lines.append(str(parsed["error_message"]))

    return "\n".join(lines)



def build_runtime_markdown(parsed: Dict[str, Any], run_json_path: Optional[Path]) -> str:
    runtime = parsed.get("runtime", {}) or {}
    steps = parsed.get("steps", []) or []
    last_action = steps[-1].get("action_name") if steps else None
    last_output = steps[-1].get("action_output") if steps else None

    lines = ["**Live status**"]
    lines.append(f"- Current iteration: {runtime.get('current_iter', parsed.get('step_count', 0))}")
    lines.append(f"- Max iterations: {runtime.get('max_iter', 'Unknown')}")
    elapsed_seconds = runtime.get("elapsed_seconds")
    if elapsed_seconds is None:
        start_iso = runtime.get("start_time")
        updated_iso = runtime.get("updated_at")
        if start_iso:
            try:
                start_dt = datetime.fromisoformat(start_iso)
                end_dt = datetime.fromisoformat(updated_iso) if updated_iso else datetime.now()
                elapsed_seconds = (end_dt - start_dt).total_seconds()
            except Exception:
                elapsed_seconds = None
    if elapsed_seconds is not None:
        elapsed = float(elapsed_seconds or 0.0)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        lines.append(f"- Time elapsed: {minutes}m {seconds}s")
    lines.append(f"- Last action: {last_action or '—'}")
    lines.append(f"- Last output: {last_output or '—'}")
    lines.append(f"- Updated at: {runtime.get('updated_at', '—')}")
    if run_json_path:
        lines.append(f"- Run JSON: {run_json_path}")
    return "\n".join(lines)


def build_progress_html(parsed: Dict[str, Any]) -> str:
    runtime = parsed.get("runtime", {}) or {}
    current_iter = runtime.get("current_iter")
    max_iter = runtime.get("max_iter")
    elapsed_seconds = runtime.get("elapsed_seconds")
    steps = parsed.get("steps", []) or []
    latest_step = steps[-1] if steps else None
    running = not _is_completed(parsed)
    if max_iter is None:
        max_iter = parsed.get("task", {}).get("resource") or parsed.get("step_count")
    if current_iter is None:
        current_iter = parsed.get("step_count")
    if current_iter is None or not max_iter:
        return (
            "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:#fff;'>"
            "<div style='font-weight:600;margin-bottom:6px;'>Progress: initializing (run started)</div>"
            "<div style='background:#f1f5f9;border-radius:8px;height:12px;overflow:hidden;'>"
            "<div style='height:12px;width:40%;background:#94a3b8;animation:pulse 1.2s ease-in-out infinite;'></div>"
            "</div>"
            "<style>@keyframes pulse{0%{margin-left:0;}50%{margin-left:60%;}100%{margin-left:0;}}</style>"
            "</div>"
        )
    effective_max = max_iter
    if parsed.get("success") and current_iter and max_iter and current_iter < max_iter:
        effective_max = current_iter
    try:
        percent = min(100, int(round((float(current_iter) / float(effective_max)) * 100)))
    except Exception:
        percent = 0
    if elapsed_seconds is None:
        start_iso = runtime.get("start_time")
        updated_iso = runtime.get("updated_at")
        if start_iso:
            try:
                start_dt = datetime.fromisoformat(start_iso)
                end_dt = datetime.fromisoformat(updated_iso) if updated_iso else datetime.now()
                elapsed_seconds = (end_dt - start_dt).total_seconds()
            except Exception:
                elapsed_seconds = None

    elapsed_text = ""

    stage_text = ""
    if latest_step:
        phase_label = _action_label(latest_step.get("action_name"))
        if latest_step.get("goal_response") or (latest_step.get("goal_eval") or {}).get("answer"):
            stage_text = f" &nbsp; | &nbsp; Action: {phase_label} → Evaluating"
        else:
            stage_text = f" &nbsp; | &nbsp; Action: {phase_label}"

    bar_style = "background:#3b82f6;"
    if running:
        bar_style = "background:linear-gradient(90deg,#60a5fa,#2563eb,#60a5fa);background-size:200% 100%;animation:flow 1.6s linear infinite;"

    ticker_script = ""

    return (
        "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:10px;background:#fff;'>"
        f"<div style='font-weight:600;margin-bottom:6px;'>Progress: {current_iter}/{effective_max} ({percent}%) {stage_text}</div>"
        "<div style='background:#f1f5f9;border-radius:8px;height:12px;overflow:hidden;'>"
        f"<div style='height:12px;width:{percent}%;{bar_style}'></div>"
        "</div></div>"
        "<style>@keyframes flow{0%{background-position:0% 50%;}100%{background-position:200% 50%;}}</style>"
    )


def build_elapsed_html(run_dir_str: str, run_json_str: str) -> str:
    if not run_dir_str and not run_json_str:
        return "<div style='padding:10px;border:1px solid #e5e7eb;border-radius:10px;'>Elapsed: —</div>"
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    run_json_path = None
    if run_json_str:
        p = Path(run_json_str)
        if p.exists():
            run_json_path = p
    if run_json_path is None and run_dir:
        run_json_path = _latest_run_json_in_dir(run_dir)
    if run_json_path is None:
        run_json_path = _latest_run_json()
    run_data = _safe_read_json(run_json_path) if run_json_path else None
    parsed = parse_run_data(run_data)
    runtime = parsed.get("runtime", {}) or {}
    start_iso = runtime.get("start_time")
    if not start_iso:
        return "<div style='padding:10px;border:1px solid #e5e7eb;border-radius:10px;'>Elapsed: —</div>"
    try:
        start_dt = datetime.fromisoformat(start_iso)
    except Exception:
        return "<div style='padding:10px;border:1px solid #e5e7eb;border-radius:10px;'>Elapsed: —</div>"
    end_iso = runtime.get("end_time")
    if not end_iso and _is_completed(parsed):
        end_iso = runtime.get("updated_at")
    if end_iso:
        try:
            end_dt = datetime.fromisoformat(end_iso)
        except Exception:
            end_dt = datetime.now()
    else:
        end_dt = datetime.now()
    elapsed = max(0, int((end_dt - start_dt).total_seconds()))
    minutes = elapsed // 60
    seconds = elapsed % 60
    label = "Elapsed"
    if _is_completed(parsed):
        label = "Final runtime"
    return (
        "<div style='padding:10px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;'>"
        f"<div style='font-weight:600;'>{label}: {minutes}m {seconds}s</div>"
        "</div>"
    )




def _is_completed(parsed: Dict[str, Any]) -> bool:
    if parsed.get("cancelled") or parsed.get("error_message"):
        return True
    if parsed.get("success") is not None:
        return True
    runtime = parsed.get("runtime", {}) or {}
    return bool(runtime.get("end_time"))


def _action_label(action_name: Optional[str]) -> str:
    return action_name or "—"

def build_action_timeline(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        return "<div style='padding:12px;border:1px solid #ddd;border-radius:8px;'>No stage timeline yet.</div>"

    stage_colors = {
        "action": "#2563eb",
        "evaluate": "#f59e0b",
        "goal_yes": "#22c55e",
        "goal_no": "#f97316",
        "done": "#16a34a",
    }

    items: List[str] = []
    for step in steps:
        stage_items: List[str] = []
        action_label = _action_label(step.get("action_name"))
        stage_items.append(
            "<div style='display:flex;align-items:center;gap:8px;margin:6px 0;'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{stage_colors['action']};'></div>"
            f"<div style='color:#0f172a;font-weight:600;'>{action_label}</div>"
            "</div>"
        )
        goal_eval = step.get("goal_eval", {}) or {}
        answer = goal_eval.get("answer")
        if step.get("goal_response") or step.get("input_goal_prompt") or answer:
            eval_label = "Evaluating (docking + scoring)"
            color = stage_colors["evaluate"]
            if answer:
                eval_label = f"Goal check: {answer}"
                color = stage_colors["goal_yes"] if answer == "YES" else stage_colors["goal_no"]
            stage_items.append(
                "<div style='display:flex;align-items:center;gap:8px;margin:6px 0;'>"
                f"<div style='width:8px;height:8px;border-radius:50%;background:{color};'></div>"
                f"<div style='color:#0f172a;font-weight:600;'>{eval_label}</div>"
                "</div>"
            )

        items.append(
            "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:10px;margin-bottom:10px;background:#fff;'>"
            f"<div style='font-weight:700;margin-bottom:6px;'>Iteration {step.get('step')}</div>"
            + "".join(stage_items)
            + "</div>"
        )

    if _is_completed(parsed):
        items.append(
            "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:10px;margin-bottom:10px;background:#fff;'>"
            "<div style='display:flex;align-items:center;gap:8px;'>"
            f"<div style='width:8px;height:8px;border-radius:50%;background:{stage_colors['done']};'></div>"
            "<div style='color:#0f172a;font-weight:600;'>Run completed — ready to review results</div>"
            "</div></div>"
        )

    return "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:12px;background:#fff;'>" + "".join(items) + "</div>"


def build_stage_panel(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        status_text = parsed.get("status_text") or ""
        if "Starting" in status_text or "Run in progress" in status_text:
            return (
                "<div style='border:1px solid #e5e7eb;border-radius:8px;padding:12px;background:#fff;'>"
                "<div style='font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>"
                "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;animation:pulseDot 1.4s ease-in-out infinite;'></span>"
                "Starting run…</div>"
                "<div style='color:#64748b;'>Waiting for first status update.</div>"
                "<style>@keyframes pulseDot{0%{transform:scale(0.8);opacity:0.5;}50%{transform:scale(1.2);opacity:1;}100%{transform:scale(0.8);opacity:0.5;}}</style>"
                "</div>"
            )
        return "<div style='padding:12px;border:1px solid #e5e7eb;border-radius:8px;'>No run yet. Click <b>Run LIDDiA</b> to start.</div>"

    completed = _is_completed(parsed)
    latest_step = steps[-1]

    if not completed:
        action_label = _action_label(latest_step.get("action_name"))
        if latest_step.get("goal_response") or (latest_step.get("goal_eval") or {}).get("answer"):
            action_label = f"{action_label} → Evaluating"
        return (
            "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:12px;background:#fff;'>"
            "<div style='font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>"
            "<span style='display:inline-block;width:10px;height:10px;border-radius:50%;background:#22c55e;animation:pulseDot 1.4s ease-in-out infinite;'></span>"
            "Current stage"
            "<span style='display:inline-block;width:12px;height:12px;border:2px solid #94a3b8;border-top-color:#2563eb;border-radius:50%;animation:spin 0.9s linear infinite;'></span>"
            "</div>"
            f"<div style='margin-bottom:6px;'>Action: {action_label}</div>"
            f"<div>Iteration: {latest_step.get('step') if latest_step.get('step') is not None else '—'}</div>"
            "<style>@keyframes pulseDot{0%{transform:scale(0.8);opacity:0.5;}50%{transform:scale(1.2);opacity:1;}100%{transform:scale(0.8);opacity:0.5;}}"
            "@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>"
            "</div>"
        )

    return build_action_timeline(parsed)


def build_live_status(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        return "<div style='padding:12px;border:1px solid #ddd;border-radius:8px;'>No live status yet.</div>"
    latest_step = steps[-1]
    action_label = _action_label(latest_step.get("action_name"))
    answer = (latest_step.get("goal_eval") or {}).get("answer")
    if latest_step.get("goal_response") or answer:
        if answer:
            action_label = f"Goal check: {answer}"
        else:
            action_label = f"{action_label} → Evaluating"
    running = not _is_completed(parsed)
    status_spinner = ""
    if running:
        status_spinner = "<span style='display:inline-block;width:10px;height:10px;border:2px solid #cbd5f5;border-top-color:#6366f1;border-radius:50%;animation:spin 0.9s linear infinite;'></span>"
    return (
        "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:12px;background:#fff;'>"
        f"<div style='font-weight:700;margin-bottom:6px;display:flex;align-items:center;gap:8px;'>Live status {status_spinner}</div>"
        f"<div style='margin-bottom:6px;' title='Action = current action for this iteration.'>Action: {action_label}</div>"
        f"<div>Iteration: {latest_step.get('step') if latest_step.get('step') is not None else '—'}</div>"
        "<style>@keyframes spin{0%{transform:rotate(0deg);}100%{transform:rotate(360deg);}}</style>"
        "</div>"
    )


def build_timeline_markdown(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        return "No iteration trace available yet."
    lines = []
    for step in steps:
        action_name = step.get("action_name") or "Unknown"
        action_output = step.get("action_output") or "Unknown"
        goal_eval = step.get("goal_eval", {}) or {}
        answer = goal_eval.get("answer") or "Unknown"
        lines.append(f"- Iteration {step.get('step')}: {action_name} -> {action_output} (Goal: {answer})")
    return "\n".join(lines)







def build_step_table(parsed: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for step in parsed.get("steps", []) or []:
        pool = step.get("pool_stats", {}) or {}
        metrics = pool.get("metrics", {}) or {}
        row = {
            "step": step.get("step"),
            "action": step.get("action_name"),
            "output_pool": step.get("action_output"),
            "goal_check": (step.get("goal_eval", {}) or {}).get("answer"),
            "pool_size": pool.get("size"),
            "diversity": pool.get("diversity"),
        }
        # include a couple metric mins/maxs if present
        vina = metrics.get("Vina Score") or metrics.get("vina")
        novelty = metrics.get("Novelty") or metrics.get("novelty")
        if isinstance(vina, dict):
            row["vina_min"] = vina.get("min")
            row["vina_max"] = vina.get("max")
        if isinstance(novelty, dict):
            row["novelty_min"] = novelty.get("min")
            row["novelty_max"] = novelty.get("max")
        rows.append(row)
    cols = ["step", "action", "output_pool", "goal_check", "pool_size", "diversity", "vina_min", "vina_max", "novelty_min", "novelty_max"]
    return pd.DataFrame(rows, columns=cols)

def build_metrics_table(parsed: Dict[str, Any]) -> pd.DataFrame:
    final_pool = parsed.get("final_pool", {}) or {}
    metrics = final_pool.get("metrics") or {}
    rows = []
    # Always include size/diversity if present
    if "size" in final_pool:
        rows.append({"metric": "Size", "min": final_pool.get("size"), "max": final_pool.get("size"), "median": None})
    if "diversity" in final_pool:
        rows.append({"metric": "Diversity", "min": final_pool.get("diversity"), "max": final_pool.get("diversity"), "median": final_pool.get("diversity")})
    for metric, stats in metrics.items():
        if isinstance(stats, dict):
            rows.append({
                "metric": metric,
                "min": stats.get("min"),
                "max": stats.get("max"),
                "median": stats.get("median"),
            })
    return pd.DataFrame(rows, columns=["metric", "min", "max", "median"])

def build_results_markdown(parsed: Dict[str, Any]) -> str:
    final_pool = parsed.get("final_pool", {}) or {}
    if not final_pool:
        return "No final pool found yet."
    lines = ["**Final Pool**"]
    pool_id = final_pool.get("pool") or "Unknown"
    size = final_pool.get("size") or "—"
    diversity = final_pool.get("diversity")
    lines.append(f"- Pool: {pool_id}")
    lines.append(f"- Molecules: {size}")
    if diversity is not None:
        lines.append(f"- Diversity: {diversity:.2f}")
    metrics = final_pool.get("metrics") or {}
    for metric, stats in metrics.items():
        if isinstance(stats, dict):
            mn = stats.get("min")
            mx = stats.get("max")
            med = stats.get("median")
            lines.append(f"- {metric}: min {mn}, max {mx}, median {med}")
    return "\n".join(lines)

def _build_summary(run_data: Dict[str, Any], run_json_path: Optional[Path]) -> str:
    parsed = parse_run_data(run_data)
    return build_run_overview(parsed, run_json_path)


# ---------- run/load functions ----------
def _render_outputs(status_text: str, run_data: Optional[Dict[str, Any]], run_json_path: Optional[Path], run_dir: Optional[Path] = None):
    parsed = parse_run_data(run_data)
    parsed = _enrich_parsed_with_memory(parsed, run_dir)
    parsed["status_text"] = status_text
    summary = build_run_overview(parsed, run_json_path)
    progress_html = build_progress_html(parsed)
    runtime_md = build_runtime_markdown(parsed, run_json_path)
    results_md = build_results_markdown(parsed)
    metrics_df = build_metrics_table(parsed)
    steps_df = build_step_table(parsed)
    stage_html = build_stage_panel(parsed)
    live_html = build_live_status(parsed)
    raw_json = json.dumps(run_data, indent=2) if run_data else ""
    runtime = parsed.get("runtime", {}) or {}
    last_event_time = runtime.get("updated_at", "—")
    status_pill = "In progress"
    status_color = "#e2e8f0"
    status_text_color = "#0f172a"
    if _is_completed(parsed):
        status_pill = "Completed"
        status_color = "#dcfce7"
        status_text_color = "#166534"
    if parsed.get("cancelled"):
        status_pill = "Cancelled"
        status_color = "#fee2e2"
        status_text_color = "#991b1b"
    if parsed.get("error_message"):
        status_pill = "Failed"
        status_color = "#fecaca"
        status_text_color = "#7f1d1d"
    run_dir_str = str(run_dir) if run_dir else ""
    run_json_str = str(run_json_path) if run_json_path else ""
    metric_trends_df = build_metric_trend_df(run_dir_str, run_json_str)
    metric_trends_fig = build_metric_plot(filter_metric_trends(metric_trends_df, "All"))
    status_badge = (
        "<div style='display:flex;align-items:center;gap:8px;'>"
        f"<span style='padding:4px 10px;border-radius:999px;background:{status_color};color:{status_text_color};font-weight:600;'>{status_pill}</span>"
        f"<span style='color:#64748b;'>Last update: {last_event_time}</span>"
        "</div>"
    )
    return status_text, summary, progress_html, runtime_md, results_md, metrics_df, steps_df, metric_trends_fig, stage_html, live_html, raw_json, run_dir_str, run_json_str, status_badge


def _run_lock_path() -> Path:
    return LOG_ROOT / ".run.lock"


def _pid_running(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _read_lock() -> Dict[str, Any]:
    lock_path = _run_lock_path()
    if not lock_path.exists():
        return {}
    try:
        return json.loads(lock_path.read_text())
    except Exception:
        return {}


def run_liddia(
    target: str,
    max_iter: int,
    model: str,
    anthropic_api_key: str,
) -> Tuple[Any, ...]:
    lock_path = _run_lock_path()
    if lock_path.exists():
        info = _read_lock()
        if not info:
            lock_path.unlink(missing_ok=True)
            info = {}
        if info and not _pid_running(info.get("pid")):
            lock_path.unlink(missing_ok=True)
        else:
            yield _render_outputs("Run already in progress. Cancel or wait.", None, None, run_dir=None)
            return
    lock_path.write_text(json.dumps({"pid": None, "started_at": time.time()}))

    # Reset UI state for a fresh run
    yield _render_outputs("Starting new run...", None, None, run_dir=None)

    if not RUN_PY.exists():
        yield _render_outputs("run.py not found.", None, None)
        lock_path.unlink(missing_ok=True)
        return

    if not anthropic_api_key or not anthropic_api_key.strip():
        yield _render_outputs("Missing Anthropic API key.", None, None)
        lock_path.unlink(missing_ok=True)
        return

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = anthropic_api_key.strip()

    cmd = [
        "python",
        "-u",
        str(RUN_PY),
        "--target",
        target.strip(),
        "--max_iter",
        str(int(max_iter)),
        "--model",
        model.strip(),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
    except Exception as e:
        yield _render_outputs(f"Failed to start run: {e}", None, None)
        lock_path.unlink(missing_ok=True)
        return
    # update lock with real pid
    lock_path.write_text(json.dumps({"pid": process.pid, "started_at": time.time()}))

    stdout_buffer: List[str] = []
    stderr_buffer: List[str] = []

    def _drain(stream, buffer):
        if stream is None:
            return
        for line in iter(stream.readline, ""):
            buffer.append(line)

    threads = [
        threading.Thread(target=_drain, args=(process.stdout, stdout_buffer), daemon=True),
        threading.Thread(target=_drain, args=(process.stderr, stderr_buffer), daemon=True),
    ]
    for t in threads:
        t.start()

    start_time = time.time()
    timeout_s = 60 * 30
    existing_dirs = {p.name for p in LOG_ROOT.iterdir()} if LOG_ROOT.exists() else set()
    active_run_dir: Optional[Path] = None

    while True:
        if process.poll() is not None:
            break
        if time.time() - start_time > timeout_s:
            process.kill()
            yield _render_outputs("Run timed out after 30 minutes.", None, None, run_dir=active_run_dir)
            lock_path.unlink(missing_ok=True)
            return

        if active_run_dir is None:
            active_run_dir = _detect_new_run_dir(existing_dirs, start_time)
        run_json_path = _latest_run_json_in_dir(active_run_dir) if active_run_dir else None
        run_data = _safe_read_json(run_json_path) if run_json_path else None
        status_text = "Run in progress..."
        if active_run_dir is None:
            status_text = "Run started. Waiting for run artifacts..."
        yield _render_outputs(status_text, run_data, run_json_path, run_dir=active_run_dir)
        time.sleep(2)

    for t in threads:
        t.join(timeout=1)

    if active_run_dir is None:
        active_run_dir = _detect_new_run_dir(existing_dirs, start_time)
    run_json_path = _latest_run_json_in_dir(active_run_dir) if active_run_dir else _latest_run_json()
    run_data = _safe_read_json(run_json_path) if run_json_path else None

    if process.returncode == 0:
        status = "Run finished successfully."
    else:
        status = f"Run failed with exit code {process.returncode}."

    yield _render_outputs(status, run_data, run_json_path, run_dir=active_run_dir)
    lock_path.unlink(missing_ok=True)


def load_latest_run() -> Tuple[Any, ...]:
    run_json_path = _latest_run_json()
    if not run_json_path:
        return _render_outputs("No runs found.", None, None)

    run_data = _safe_read_json(run_json_path)
    if not run_data:
        return _render_outputs(f"Could not read {run_json_path.name}.", None, run_json_path)

    status = f"Loaded latest run: {run_json_path.parent.name}"
    return _render_outputs(status, run_data, run_json_path)


def load_uploaded_run(run_json_file) -> Tuple[Any, ...]:
    if run_json_file is None:
        return _render_outputs("Upload a run JSON file first.", None, None)

    path = Path(run_json_file.name)
    run_data = _safe_read_json(path)
    if not run_data:
        return _render_outputs(f"Could not parse uploaded file: {path.name}", None, path)

    # If uploaded from outside log/, try to locate the matching run directory.
    run_dir = path.parent if _is_under_log_root(path.parent) else _find_run_dir_by_json(path)
    if run_dir:
        log_json = run_dir / path.name
        if log_json.exists():
            path = log_json
            run_data = _safe_read_json(path) or run_data

    return _render_outputs(f"Loaded uploaded run: {path.name}", run_data, path, run_dir=run_dir or path.parent)


def build_report(run_dir_str: str, run_json_str: str, report_type: str) -> Tuple[str, Optional[str]]:
    run_dir = Path(run_dir_str) if run_dir_str else None
    run_json_path = Path(run_json_str) if run_json_str else None
    if run_json_path and run_json_path.exists():
        run_data = _safe_read_json(run_json_path)
    else:
        run_json_path = _latest_run_json()
        run_data = _safe_read_json(run_json_path) if run_json_path else None

    parsed = parse_run_data(run_data)
    parsed = _enrich_parsed_with_memory(parsed, run_dir or (run_json_path.parent if run_json_path else None))
    report_type = (report_type or "txt").lower().strip()

    if report_type == "json":
        payload = {
            "summary": parsed,
            "run_json_path": str(run_json_path) if run_json_path else None,
        }
        tmp_path = Path("/var/folders/_7/jsnvb1yd3lxfhpllggb5cqsh0000gp/T") / f"liddia_report_{int(time.time())}.json"
        tmp_path.write_text(json.dumps(payload, indent=2))
        return "JSON report ready.", str(tmp_path)

    if report_type == "csv":
        steps_df = build_metrics_table(parsed)
        tmp_path = Path("/var/folders/_7/jsnvb1yd3lxfhpllggb5cqsh0000gp/T") / f"liddia_report_{int(time.time())}.csv"
        steps_df.to_csv(tmp_path, index=False)
        return "CSV report ready (final pool metrics).", str(tmp_path)

    lines: List[str] = []
    lines.append("LIDDIA Run Report")
    lines.append("=" * 24)
    lines.append("")
    lines.append(build_run_overview(parsed, run_json_path))
    lines.append("")
    lines.append("Live Status")
    lines.append("-" * 11)
    lines.append(build_runtime_markdown(parsed, run_json_path))
    lines.append("")
    lines.append("Iteration Timeline")
    rt = parsed.get("runtime", {}) or {}
    total = rt.get("elapsed_seconds")
    if total is not None:
        minutes = int(float(total) // 60)
        seconds = int(float(total) % 60)
        lines.append(f"Total runtime: {minutes}m {seconds}s")
        lines.append("")
    lines.append("-" * 13)
    lines.append(build_timeline_markdown(parsed))

    report_text = "\n".join(lines)
    if not report_text.strip():
        return "No run data available for report.", None

    tmp_path = Path("/var/folders/_7/jsnvb1yd3lxfhpllggb5cqsh0000gp/T") / f"liddia_report_{int(time.time())}.txt"
    tmp_path.write_text(report_text)
    return "Text report ready.", str(tmp_path)


def _load_memory_df(run_dir: Path) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    mem_candidates = list(run_dir.glob("*_memory.pkl"))
    if not mem_candidates:
        return None, None
    mem_path = sorted(mem_candidates, key=lambda p: p.stat().st_mtime)[-1]

    class DummyMemory:
        def __init__(self):
            self.stream = {}
            self.history = []

    # Stub liddia modules referenced in pickle
    liddia_mod = types.ModuleType("liddia")
    liddia_mod.__path__ = []
    submods = ["memory", "action", "environment", "evaluate", "utils", "prompt_template", "agent"]
    for name in submods:
        mod = types.ModuleType(f"liddia.{name}")
        sys.modules[f"liddia.{name}"] = mod
    sys.modules["liddia"] = liddia_mod
    sys.modules["liddia.memory"].Memory = DummyMemory
    for fn in ["sample_zinc", "graph_ga_optimizer", "run_code", "sample_pocket2mol"]:
        setattr(sys.modules["liddia.action"], fn, lambda *a, **k: None)

    with open(mem_path, "rb") as f:
        mem = pickle.load(f)

    mol_keys = [k for k, v in mem.stream.items() if isinstance(v, dict) and v.get("type") == "MOL"]
    if not mol_keys:
        return None, None
    last_mol = mol_keys[-1]
    df = mem.stream[last_mol].get("data")
    if df is None or not isinstance(df, pd.DataFrame):
        return None, None
    return df, last_mol


def _load_memory(run_dir: Path):
    mem_candidates = list(run_dir.glob("*_memory.pkl"))
    if not mem_candidates:
        return None
    mem_path = sorted(mem_candidates, key=lambda p: p.stat().st_mtime)[-1]

    class DummyMemory:
        def __init__(self):
            self.stream = {}
            self.history = []

    liddia_mod = types.ModuleType("liddia")
    liddia_mod.__path__ = []
    submods = ["memory", "action", "environment", "evaluate", "utils", "prompt_template", "agent"]
    for name in submods:
        mod = types.ModuleType(f"liddia.{name}")
        sys.modules[f"liddia.{name}"] = mod
    sys.modules["liddia"] = liddia_mod
    sys.modules["liddia.memory"].Memory = DummyMemory
    for fn in ["sample_zinc", "graph_ga_optimizer", "run_code", "sample_pocket2mol"]:
        setattr(sys.modules["liddia.action"], fn, lambda *a, **k: None)

    with open(mem_path, "rb") as f:
        mem = pickle.load(f)
    return mem


def _iteration_pool_ids(mem) -> List[str]:
    if not mem or not getattr(mem, "history", None):
        return []
    pool_ids = []
    for h in mem.history:
        pool_id = h.get("action_output")
        if pool_id and pool_id != "EMPTY SET":
            pool_ids.append(pool_id)
    return pool_ids


def build_metric_trend_df(run_dir_str: str, run_json_str: str = "") -> pd.DataFrame:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return pd.DataFrame(columns=["iteration", "metric", "value"])
    rows: List[Dict[str, Any]] = []
    mem = _load_memory(run_dir)
    if mem:
        pool_ids = _iteration_pool_ids(mem)
        for idx, pool_id in enumerate(pool_ids, start=1):
            pool_stats = _pool_stats_from_memory(mem, pool_id)
            for metric, stats in (pool_stats.get("metrics") or {}).items():
                if not isinstance(stats, dict):
                    continue
                val = stats.get("median")
                if val is None:
                    continue
                rows.append({"iteration": idx, "metric": metric, "value": val})
        if rows:
            return pd.DataFrame(rows)
    run_json_path = Path(run_json_str) if run_json_str else _latest_run_json_in_dir(run_dir)
    run_data = _safe_read_json(run_json_path) if run_json_path and run_json_path.exists() else None
    parsed = parse_run_data(run_data)
    parsed = _enrich_parsed_with_memory(parsed, run_dir)
    for step in parsed.get("steps", []) or []:
        pool = step.get("pool_stats", {}) or {}
        metrics = pool.get("metrics", {}) or {}
        step_idx = step.get("step")
        if step_idx is None:
            continue
        for metric, stats in metrics.items():
            if not isinstance(stats, dict):
                continue
            val = stats.get("median")
            if val is None:
                continue
            rows.append({"iteration": step_idx, "metric": metric, "value": val})
    return pd.DataFrame(rows)


def filter_metric_trends(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["iteration", "metric", "value"])
    if not metric or metric == "All":
        out = df.copy()
    else:
        out = df[df["metric"] == metric].copy()
    out["iteration"] = pd.to_numeric(out["iteration"], errors="coerce").fillna(0).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.sort_values("iteration")


def build_metric_plot(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    try:
        import plotly.express as px
        fig = px.line(
            df,
            x="iteration",
            y="value",
            color="metric",
            markers=True,
            hover_data={"iteration": True, "value": ":.4f", "metric": True},
        )
        fig.update_traces(line=dict(width=2), marker=dict(size=7))
        fig.update_xaxes(dtick=1, tickformat="d", title="iteration")
        fig.update_yaxes(title="value")
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
            hovermode="x unified",
        )
        return fig
    except Exception:
        return None


def update_metric_controls(run_dir_str: str, run_json_str: str):
    df = build_metric_trend_df(run_dir_str, run_json_str)
    metrics = sorted(df["metric"].unique()) if not df.empty else []
    choices = ["All"] + metrics
    value = metrics[0] if metrics else "All"
    filtered = filter_metric_trends(df, value)
    fig = build_metric_plot(filtered)
    return df, fig, gr.update(choices=choices, value=value)


def apply_metric_filter(df: pd.DataFrame, metric: str):
    return build_metric_plot(filter_metric_trends(df, metric))


def _viewer_limits(run_dir: Optional[Path]) -> Tuple[int, int]:
    if not run_dir:
        return 1, 0
    mem = _load_memory(run_dir)
    if not mem:
        return 1, 0
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return 1, 0
    max_iter = len(pool_ids)
    # default mol_index max based on last pool
    df = mem.stream.get(pool_ids[-1], {}).get("data")
    if df is None or not hasattr(df, "__len__"):
        return max_iter, 0
    return max_iter, max(0, len(df) - 1)


def build_molecule_view(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int) -> Tuple[str, str, str]:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return "No run selected.", "", ""
    mem = _load_memory(run_dir)
    if not mem:
        return "No memory.pkl found.", "", ""
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return "No molecule pools found.", "", ""
    if iteration <= 0:
        iteration = 1
    if iteration > len(pool_ids):
        iteration = len(pool_ids)
    pool_id = pool_ids[iteration - 1]
    block = mem.stream.get(pool_id, {})
    df = block.get("data")
    if df is None or "SMILES" not in df.columns:
        return "SMILES not available.", "", ""
    if mol_index < 0:
        mol_index = 0
    if mol_index >= len(df):
        mol_index = max(0, len(df) - 1)
    smiles = str(df.iloc[mol_index]["SMILES"])
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return smiles, "<div>Could not parse SMILES.</div>", ""
        svg = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(250, 200), useSVG=True)
        html = f"<div>{svg}</div>"
        return smiles, html, ""
    except Exception as e:
        return smiles, "<div>2D viewer requires RDKit.</div>", str(e)


def build_molecule_table(run_dir_str: str, run_json_str: str, iteration: int, max_rows: int = 50) -> pd.DataFrame:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return pd.DataFrame()
    mem = _load_memory(run_dir)
    if not mem:
        return pd.DataFrame()
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return pd.DataFrame()
    if iteration <= 0:
        iteration = 1
    if iteration > len(pool_ids):
        iteration = len(pool_ids)
    pool_id = pool_ids[iteration - 1]
    df = mem.stream.get(pool_id, {}).get("data")
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    # Keep SMILES + numeric properties
    cols = [c for c in df.columns]
    out = df[cols].copy()
    if max_rows and len(out) > max_rows:
        out = out.head(max_rows)
    return out


def select_molecule_from_table(run_dir_str: str, run_json_str: str, iteration: int, evt: gr.SelectData):
    # evt.index is row index in displayed table
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None:
        return gr.update(), "", "", ""
    smiles, svg, status = build_molecule_view(run_dir_str, run_json_str, int(iteration), int(idx))
    return gr.update(value=idx), smiles, svg, status


def _pool_ids_for_run(run_dir_str: str, run_json_str: str) -> List[str]:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return []
    mem = _load_memory(run_dir)
    if not mem:
        return []
    return _iteration_pool_ids(mem)


def update_pool_selector(run_dir_str: str, run_json_str: str):
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids:
        return gr.update(choices=[], value=None), gr.update(value=1)
    return gr.update(choices=pool_ids, value=pool_ids[-1]), gr.update(value=len(pool_ids))


def set_iteration_from_pool(run_dir_str: str, run_json_str: str, pool_id: str):
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids or pool_id not in pool_ids:
        return gr.update()
    return gr.update(value=pool_ids.index(pool_id) + 1)


def build_pool_badge(run_dir_str: str, run_json_str: str, iteration: int) -> str:
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids:
        return "<div style='padding:8px 10px;border:1px solid #e5e7eb;border-radius:999px;display:inline-block;'>Viewing: —</div>"
    if iteration < 1:
        iteration = 1
    if iteration > len(pool_ids):
        iteration = len(pool_ids)
    pool_id = pool_ids[iteration - 1]
    return (
        "<div style='padding:8px 12px;border:1px solid #e5e7eb;border-radius:999px;display:inline-block;background:#f8fafc;'>"
        f"<b>Viewing</b>: Iteration {iteration} · Pool {pool_id}"
        "</div>"
    )


def get_viewer_limits(run_dir_str: str, run_json_str: str):
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return gr.update(value=1), gr.update(value=0)
    mem = _load_memory(run_dir)
    if not mem:
        return gr.update(value=1), gr.update(value=0)
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return gr.update(value=1), gr.update(value=0)
    max_iter = len(pool_ids)
    df = mem.stream.get(pool_ids[-1], {}).get("data")
    max_idx = max(0, len(df) - 1) if df is not None else 0
    return gr.update(value=min(1, max_iter)), gr.update(value=0)


def update_index_limits(run_dir_str: str, run_json_str: str, iteration: int):
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return gr.update(value=0)
    mem = _load_memory(run_dir)
    if not mem:
        return gr.update(value=0)
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return gr.update(value=0)
    if iteration < 1:
        iteration = 1
    if iteration > len(pool_ids):
        iteration = len(pool_ids)
    pool_id = pool_ids[iteration - 1]
    df = mem.stream.get(pool_id, {}).get("data")
    max_idx = max(0, len(df) - 1) if df is not None else 0
    return gr.update(value=0 if max_idx >= 0 else 0)


def _update_metric_trends(run_dir_str: str, run_json_str: str) -> pd.DataFrame:
    df = build_metric_trend_df(run_dir_str, run_json_str)
    return build_metric_plot(filter_metric_trends(df, "All"))


def _update_molecule_view(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int):
    return build_molecule_view(run_dir_str, run_json_str, int(iteration), int(mol_index))


def _update_molecule_table(run_dir_str: str, run_json_str: str, iteration: int):
    return build_molecule_table(run_dir_str, run_json_str, int(iteration))


def _find_run_dir_by_json(json_path: Path) -> Optional[Path]:
    # If uploaded file isn't from the run folder, search log/ for matching JSON name.
    if LOG_ROOT.exists():
        candidates = sorted(LOG_ROOT.glob(f"*/{json_path.name}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return candidates[0].parent
    return None


def _is_under_log_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(LOG_ROOT.resolve())
        return True
    except Exception:
        return False


def _resolve_run_dir(run_dir_str: str, run_json_str: str) -> Optional[Path]:
    if run_dir_str:
        run_dir = Path(run_dir_str)
        if run_dir.exists():
            return run_dir
    if run_json_str:
        json_path = Path(run_json_str)
        if json_path.exists() and _is_under_log_root(json_path.parent):
            return json_path.parent
        found = _find_run_dir_by_json(json_path)
        if found:
            return found
    return None


def _metrics_from_df(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for col in df.columns:
        if col.lower() == "smiles":
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        rows.append(
            {
                "metric": col,
                "min": float(series.min()),
                "max": float(series.max()),
                "median": float(series.median()),
            }
        )
    return pd.DataFrame(rows)


def _normalize_metric_label(label: str) -> str:
    key = str(label).strip()
    lowered = key.lower()
    mapping = {
        "vina": "Vina Score",
        "vina score": "Vina Score",
        "qed": "QED",
        "sascore": "SAScore",
        "lipinski": "Lipinski",
        "novelty": "Novelty",
    }
    return mapping.get(lowered, key)


def _pool_stats_from_memory(mem, pool_id: str) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"pool": pool_id, "size": None, "diversity": None, "metrics": {}}
    if not mem or not pool_id:
        return stats
    block = mem.stream.get(pool_id, {}) if getattr(mem, "stream", None) else {}
    metrics = block.get("metrics") or {}
    df = block.get("data") if isinstance(block, dict) else None

    if isinstance(metrics, dict):
        stats["size"] = metrics.get("size") or metrics.get("Size")
        stats["diversity"] = metrics.get("diversity") or metrics.get("Diversity")
        for key, val in metrics.items():
            if isinstance(val, dict):
                label = _normalize_metric_label(key)
                stats["metrics"][label] = {
                    "min": val.get("min"),
                    "max": val.get("max"),
                    "median": val.get("median"),
                }

    if stats["size"] is None and isinstance(df, pd.DataFrame):
        stats["size"] = len(df)

    # If no metric dicts were present, compute from dataframe.
    if (not stats["metrics"]) and isinstance(df, pd.DataFrame):
        stats["metrics"] = _metrics_from_df(df).set_index("metric").to_dict(orient="index")

    return stats


def _enrich_parsed_with_memory(parsed: Dict[str, Any], run_dir: Optional[Path]) -> Dict[str, Any]:
    if not run_dir:
        return parsed
    mem = _load_memory(run_dir)
    if not mem:
        return parsed
    steps = parsed.get("steps", []) or []
    for step in steps:
        pool_id = step.get("action_output")
        if not pool_id:
            continue
        step["pool_stats"] = _pool_stats_from_memory(mem, pool_id)
    parsed["steps"] = steps
    parsed["final_pool"] = steps[-1]["pool_stats"] if steps else {}
    return parsed




# ---------- UI ----------
with gr.Blocks(title="LIDDIA GUI v2") as demo:
    gr.Markdown(
        "# LIDDIA GUI v2\n"
        "Local internal research interface for launching runs, monitoring progress, and reviewing results."
    )

    with gr.Tabs():
        with gr.Tab("Monitor"):
            run_dir_state = gr.State("")
            run_json_state = gr.State("")
            with gr.Row():
                with gr.Column(scale=1):
                    target = gr.Dropdown(DEFAULT_TARGETS, value="EGFR", label="Target", allow_custom_value=True)
                    max_iter = gr.Number(value=2, precision=0, label="Max iterations")
                    model = gr.Dropdown(DEFAULT_MODELS, value="claude-opus-4-6", label="Model", allow_custom_value=True)
                    anthropic_api_key = gr.Textbox(label="Anthropic API key", type="password", placeholder="sk-ant-...")
                    with gr.Row():
                        run_button = gr.Button("Run LIDDIA", variant="primary")
                        cancel_button = gr.Button("Cancel Run", variant="stop")
                        refresh_button = gr.Button("Load latest run")

                with gr.Column(scale=2):
                    status = gr.Textbox(label="Run status", interactive=False, visible=False)
                    status_badge = gr.HTML(label="Status badge", visible=False)
                    progress_html = gr.HTML(label="Run summary")
                    elapsed_html = gr.HTML(label="Elapsed time")
                    elapsed_timer = gr.Timer(1.0)
                    live_html = gr.HTML(label="Live status", visible=False)
                    stage_html = gr.HTML(label="Action activity")

                with gr.Column(scale=1):
                    metrics_df = gr.Dataframe(label="Final pool metrics", interactive=False)

        with gr.Tab("Results"):
            gr.HTML(
                """
<style>
.results-tight .block {
  margin-bottom: 6px !important;
}
.results-tight .gr-box {
  margin-bottom: 6px !important;
}
</style>
"""
            )
            with gr.Column(elem_classes=["results-tight"]):
                with gr.Row():
                    with gr.Column(scale=1):
                        overview = gr.Textbox(label="Run overview", lines=10, interactive=False)
                        runtime_md = gr.Markdown(visible=False)
                        results_md = gr.Markdown()
                        report_status = gr.Textbox(label="Report status", interactive=False)
                        report_file = gr.File(label="Download report")
                        report_txt = gr.Button("Generate TXT")
                        report_json = gr.Button("Generate JSON")
                        report_csv = gr.Button("Generate CSV")
                        gr.Markdown("### Load previous run")
                        uploaded_run = gr.File(label="Upload run JSON", file_types=[".json"])
                        load_uploaded_button = gr.Button("Load uploaded run")

                    with gr.Column(scale=2):
                        gr.Markdown("### Molecule viewer (2D)")
                        pool_select = gr.Dropdown(label="Pool", choices=[], value=None)
                        pool_badge = gr.HTML()
                        iteration_select = gr.Number(label="Iteration", value=1, precision=0)
                        mol_index = gr.Number(label="Molecule index", value=0, precision=0)
                        smiles_text = gr.Textbox(label="SMILES", interactive=False)
                        mol_svg = gr.HTML(label="2D structure")
                        mol_status = gr.Textbox(label="Viewer status", interactive=False, visible=False)
                        mol_table = gr.Dataframe(label="Molecule properties", interactive=True)

                with gr.Row():
                    with gr.Column(scale=1):
                        metric_trends_state = gr.State(pd.DataFrame())
                        metric_select = gr.Dropdown(label="Metric", choices=["All"], value="All")
                        metric_trends = gr.Plot(label="Metric trends (median)")
                    with gr.Column(scale=1):
                        steps_df = gr.Dataframe(label="Iteration rollup", interactive=False)

                with gr.Accordion("Run JSON", open=False):
                    raw_json = gr.Code(label="Run JSON", language="json")

    run_evt = run_button.click(
        fn=run_liddia,
        inputs=[target, max_iter, model, anthropic_api_key],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, steps_df, metric_trends, stage_html, live_html, raw_json, run_dir_state, run_json_state, status_badge],
    )
    run_evt.then(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    run_evt.then(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state],
        outputs=[pool_select, iteration_select],
    )
    run_evt.then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[pool_badge],
    )
    run_evt.then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_table],
        show_progress="hidden",
    )
    run_evt.then(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select, mol_index],
        outputs=[smiles_text, mol_svg, mol_status],
        show_progress="hidden",
    )
    run_evt.then(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )
    run_evt.then(
        fn=get_viewer_limits,
        inputs=[run_dir_state, run_json_state],
        outputs=[iteration_select, mol_index],
        show_progress="hidden",
    )

    def cancel_run(run_dir_str: str) -> str:
        if not run_dir_str:
            return "No active run directory found."
        try:
            Path(run_dir_str, "cancel.flag").write_text("cancelled")
        except Exception as e:
            return f"Failed to request cancel: {e}"
        return "Cancellation requested. Waiting for run to stop..."

    cancel_button.click(
        fn=cancel_run,
        inputs=[run_dir_state],
        outputs=[status],
    )

    refresh_button.click(
        fn=load_latest_run,
        inputs=[],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, steps_df, metric_trends, stage_html, live_html, raw_json, run_dir_state, run_json_state, status_badge],
    )
    refresh_button.click(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    refresh_button.click(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state],
        outputs=[pool_select, iteration_select],
    )
    refresh_button.click(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[pool_badge],
    )
    refresh_button.click(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_table],
    )
    refresh_button.click(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )
    refresh_button.click(
        fn=get_viewer_limits,
        inputs=[run_dir_state, run_json_state],
        outputs=[iteration_select, mol_index],
    )

    load_uploaded_button.click(
        fn=load_uploaded_run,
        inputs=[uploaded_run],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, steps_df, metric_trends, stage_html, live_html, raw_json, run_dir_state, run_json_state, status_badge],
    )
    load_uploaded_button.click(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    load_uploaded_button.click(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state],
        outputs=[pool_select, iteration_select],
    )
    load_uploaded_button.click(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[pool_badge],
    )
    load_uploaded_button.click(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_table],
        show_progress="hidden",
    )
    load_uploaded_button.click(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select, mol_index],
        outputs=[smiles_text, mol_svg, mol_status],
        show_progress="hidden",
    )
    load_uploaded_button.click(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )
    load_uploaded_button.click(
        fn=get_viewer_limits,
        inputs=[run_dir_state, run_json_state],
        outputs=[iteration_select, mol_index],
    )

    elapsed_timer.tick(
        fn=build_elapsed_html,
        inputs=[run_dir_state, run_json_state],
        outputs=[elapsed_html],
    )



    refresh_button.click(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )

    load_uploaded_button.click(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )

    refresh_button.click(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select, mol_index],
        outputs=[smiles_text, mol_svg, mol_status],
        show_progress="hidden",
    )

    iteration_select.change(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select, mol_index],
        outputs=[smiles_text, mol_svg, mol_status],
        show_progress="hidden",
    )
    iteration_select.change(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[pool_badge],
    )
    iteration_select.change(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_table],
        show_progress="hidden",
    )
    pool_select.change(
        fn=set_iteration_from_pool,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[iteration_select],
        show_progress="hidden",
    )
    mol_table.select(
        fn=select_molecule_from_table,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_index, smiles_text, mol_svg, mol_status],
        show_progress="hidden",
    )
    metric_select.change(
        fn=apply_metric_filter,
        inputs=[metric_trends_state, metric_select],
        outputs=[metric_trends],
    )
    iteration_select.change(
        fn=update_index_limits,
        inputs=[run_dir_state, run_json_state, iteration_select],
        outputs=[mol_index],
        show_progress="hidden",
    )

    mol_index.change(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select, mol_index],
        outputs=[smiles_text, mol_svg, mol_status],
    )

    report_txt.click(
        fn=build_report,
        inputs=[run_dir_state, run_json_state, gr.State("txt")],
        outputs=[report_status, report_file],
    )
    report_json.click(
        fn=build_report,
        inputs=[run_dir_state, run_json_state, gr.State("json")],
        outputs=[report_status, report_file],
    )
    report_csv.click(
        fn=build_report,
        inputs=[run_dir_state, run_json_state, gr.State("csv")],
        outputs=[report_status, report_file],
    )


if __name__ == "__main__":
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
