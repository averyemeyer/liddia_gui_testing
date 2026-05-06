import json
import os
import re
import io
import html
import base64
import subprocess
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import pandas as pd
import pickle
import types
import sys
import math
import tempfile

REPO_ROOT = Path(__file__).resolve().parent
RUN_PY = REPO_ROOT / "run.py"
LOG_ROOT = REPO_ROOT / "log"
PDB_DIR = REPO_ROOT / "dataset" / "pdb"
REPORT_TMP_DIR = Path(tempfile.gettempdir())
_LAST_GOOD_RUN_DATA: Dict[str, Dict[str, Any]] = {}

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
    # run.py may rewrite JSON while we're polling; retry briefly to avoid UI flicker
    for _ in range(4):
        try:
            text = path.read_text()
            if not text.strip():
                time.sleep(0.05)
                continue
            return json.loads(text)
        except json.JSONDecodeError:
            time.sleep(0.05)
            continue
        except Exception:
            return None
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


def _get_available_runs() -> List[str]:
    if not LOG_ROOT.exists():
        return []
    runs = []
    for child in LOG_ROOT.iterdir():
        if child.is_dir():
            json_files = list(child.glob("*.json"))
            if json_files:
                runs.append(child.name)
    return sorted(runs, reverse=True)  # most recent first


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
            "stop_reason": None,
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
        pool_stats = _format_pool_stats(_parse_pool_stats(pool_text))
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
                "error_message": payload.get("error_message"),
                "stop_reason": payload.get("stop_reason"),
                "pool_stats": pool_stats,
                "goal_eval": goal_eval,
            }
        )

    final_pool = _format_pool_stats(steps[-1]["pool_stats"]) if steps else {}
    return {
        "model": run_data.get("model"),
        "success": run_data.get("success") if "success" in run_data else None,
        "cancelled": run_data.get("cancelled", False),
        "error_message": run_data.get("error_message"),
        "stop_reason": run_data.get("stop_reason"),
        "task": run_data.get("task", {}) if isinstance(run_data.get("task"), dict) else {},
        "runtime": run_data.get("runtime", {}) if isinstance(run_data.get("runtime"), dict) else {},
        "steps": steps,
        "final_pool": final_pool,
        "step_count": len(steps),
    }


# ---------- render helpers ----------
def _fmt_num(value, decimals: int = 2):
    if value is None:
        return None
    try:
        num = float(value)
    except Exception:
        return value
    factor = 10 ** int(decimals)
    return math.trunc(num * factor) / factor

def _fmt_str(value, decimals: int = 2) -> str:
    try:
        if value is None:
            return "—"
        if isinstance(value, (int, float)):
            num = float(value)
            factor = 10 ** int(decimals)
            truncated = math.trunc(num * factor) / factor
            return f"{truncated:.{decimals}f}"
        s = str(value).strip()
        # strip non-numeric chars (except . - e)
        import re
        s2 = re.sub(r"[^0-9eE+\-.]", "", s)
        if s2 == "":
            return s
        num = float(s2)
        factor = 10 ** int(decimals)
        truncated = math.trunc(num * factor) / factor
        return f"{truncated:.{decimals}f}"
    except Exception:
        return str(value)


def _round_df(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        try:
            factor = 10 ** int(decimals)
            # never coerce or modify SMILES column
            if str(col).lower() == "smiles":
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].map(lambda x: (math.trunc(float(x) * factor) / factor) if pd.notna(x) else x)
            else:
                # only coerce object columns if ALL values are numeric-like
                coerced = pd.to_numeric(df[col], errors='coerce')
                if coerced.notna().all():
                    df[col] = coerced.map(lambda x: (math.trunc(float(x) * factor) / factor) if pd.notna(x) else x)
        except Exception:
            pass
    return df

def _format_df_for_display(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    for col in df.columns:
        try:
            # preserve SMILES, Index, and step columns
            if str(col).lower() in ["smiles", "index", "step"]:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].map(lambda x: _fmt_str(x, decimals) if pd.notna(x) else x)
            else:
                coerced = pd.to_numeric(df[col], errors='coerce')
                # only format columns where all values are numeric-like
                if coerced.notna().all():
                    df[col] = coerced.map(lambda x: _fmt_str(x, decimals) if pd.notna(x) else x)
        except Exception:
            pass
    return df


def _smiles_to_markdown_image(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return "N/A"
        # Render higher-res, then display at smaller size for crisp hover zoom.
        img = Draw.MolToImage(mol, size=(420, 320))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
        src = f"data:image/png;base64,{b64}"
        return (
            f"<a class='mol-thumb-link' href='{src}' target='_blank' rel='noopener noreferrer'>"
            f"<img class='mol-thumb' src='{src}' alt='molecule'/>"
            "</a>"
        )
    except Exception:
        return "N/A"



def _format_pool_stats(pool: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(pool, dict):
        return pool
    formatted = dict(pool)
    if "diversity" in formatted:
        formatted["diversity"] = _fmt_str(formatted.get("diversity"))
    if "size" in formatted:
        formatted["size"] = formatted.get("size")
    metrics = formatted.get("metrics")
    if isinstance(metrics, dict):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, dict):
                new_metrics[k] = {
                    "min": _fmt_str(v.get("min")),
                    "max": _fmt_str(v.get("max")),
                    "median": _fmt_str(v.get("median")),
                }
            else:
                new_metrics[k] = _fmt_str(v)
        formatted["metrics"] = new_metrics
    return formatted



def _metric_value(pool: Dict[str, Any], label: str, field: str) -> Optional[float]:
    return (((pool or {}).get("metrics") or {}).get(label) or {}).get(field)


def build_run_overview(parsed: Dict[str, Any], run_json_path: Optional[Path]) -> str:
    task = parsed.get("task", {}) or {}
    runtime = parsed.get("runtime", {}) or {}

    def _fmt_dt(dt_str: Any) -> str:
        if not dt_str:
            return "—"
        try:
            return datetime.fromisoformat(str(dt_str)).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(dt_str)

    def _short_path(p: Any) -> str:
        if not p:
            return "—"
        text = str(p)
        parts = text.split("/")
        if len(parts) <= 3:
            return text
        return ".../" + "/".join(parts[-2:])

    if _is_idle(parsed):
        status = "IDLE"
    else:
        status = "SUCCESS" if parsed.get("success") else ("CANCELLED" if parsed.get("cancelled") else "RUNNING")

    target = task.get("target") or "Unknown"
    model = parsed.get("model") or "Unknown"
    budget = task.get("resource")
    pocket = task.get("pocket") or "—"
    run_json_disp = _short_path(run_json_path) if run_json_path else "—"

    start_disp = _fmt_dt(runtime.get("start_time"))
    end_disp = _fmt_dt(runtime.get("end_time"))
    runtime_disp = "—"
    if runtime.get("elapsed_seconds") is not None:
        elapsed = float(runtime.get("elapsed_seconds") or 0.0)
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        runtime_disp = f"{minutes}m {seconds}s"

    final_pool = parsed.get("final_pool", {}) or {}
    pool_id = "—"
    size = "—"
    diversity = "—"
    if final_pool:
        pool_id = final_pool.get("pool") or "—"
        size = final_pool.get("size") or "—"
        div_val = final_pool.get("diversity")
        if div_val not in (None, "", "—"):
            diversity = _fmt_str(div_val)

    status_lower = str(status).lower()
    status_class = "status-running"
    if "idle" in status_lower:
        status_class = "status-idle"
    elif "success" in status_lower:
        status_class = "status-success"
    elif "fail" in status_lower or "error" in status_lower or parsed.get("error_message"):
        status_class = "status-failed"
    elif "cancel" in status_lower or "warn" in status_lower or parsed.get("warning_message"):
        status_class = "status-warning"

    safe = lambda v: html.escape(str(v if v is not None else "—"))
    if _is_idle(parsed):
        return (
            "<div class='run-overview-grid'>"
            "<div class='run-overview-section meta-card'>"
            "<div class='run-overview-title'>Run Summary</div>"
            "<div class='empty-state'>"
            "<div class='empty-state-title'>No run loaded yet</div>"
            "<div class='empty-state-copy'>Results will appear here after you start a run or load a previous run folder.</div>"
            "</div>"
            "</div>"
            "</div>"
        )
    return (
        "<div class='run-overview-grid'>"
        "<div class='run-overview-section meta-card'>"
        "<div class='run-overview-title'>Run</div>"
        "<div class='label-value-grid'>"
        f"<div class='k'>Status</div><div class='v'><span class='status-badge {status_class}'>{safe(status)}</span></div>"
        f"<div class='k'>Target</div><div class='v'>{safe(target)}</div>"
        f"<div class='k'>Model</div><div class='v'>{safe(model)}</div>"
        f"<div class='k'>Budget</div><div class='v'>{safe(budget if budget is not None else '—')}</div>"
        "</div>"
        "</div>"
        "<div class='run-overview-section meta-card'>"
        "<div class='run-overview-title'>Files</div>"
        "<div class='label-value-grid'>"
        f"<div class='k'>Pocket</div><div class='v'>{safe(pocket)}</div>"
        f"<div class='k'>Run JSON</div><div class='v mono'>{safe(run_json_disp)}</div>"
        "</div>"
        "</div>"
        "<div class='run-overview-section meta-card'>"
        "<div class='run-overview-title'>Timing</div>"
        "<div class='label-value-grid'>"
        f"<div class='k'>Start</div><div class='v'>{safe(start_disp)}</div>"
        f"<div class='k'>End</div><div class='v'>{safe(end_disp)}</div>"
        f"<div class='k'>Runtime</div><div class='v'>{safe(runtime_disp)}</div>"
        "</div>"
        "</div>"
        "<div class='run-overview-section meta-card'>"
        "<div class='run-overview-title'>Final Pool</div>"
        "<div class='label-value-grid'>"
        f"<div class='k'>Pool ID</div><div class='v'>{safe(pool_id)}</div>"
        f"<div class='k'>Molecules</div><div class='v'>{safe(size)}</div>"
        f"<div class='k'>Diversity</div><div class='v'>{safe(diversity)}</div>"
        "</div>"
        "</div>"
        "</div>"
    )


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
    if _is_idle(parsed):
        return (
            "<div class='progress-shell'>"
            "<div class='empty-state'>"
            "<div class='empty-state-title'>Run monitor is idle</div>"
            "<div class='empty-state-copy'>Set target, model, and API key in Run Setup, then click Run LIDDIA.</div>"
            "</div>"
            "<div class='progress-track'><div class='progress-fill progress-fill-idle' style='width:0%;'></div></div>"
            "</div>"
        )

    if current_iter is None or not max_iter:
        return (
            "<div class='progress-shell'>"
            "<div class='progress-head'>"
            "<span class='progress-title'>Progress: initializing</span>"
            "<span class='helper-text'>Run started, waiting for first iteration update</span>"
            "</div>"
            "<div class='progress-track'>"
            "<div class='progress-fill progress-fill-running' style='width:40%;'></div>"
            "</div>"
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

    stage_text = ""
    if latest_step:
        phase_label = _action_label(latest_step.get("action_name"))
        if latest_step.get("goal_response") or (latest_step.get("goal_eval") or {}).get("answer"):
            stage_text = f" &nbsp; | &nbsp; Action: {phase_label} → Evaluating"
        else:
            stage_text = f" &nbsp; | &nbsp; Action: {phase_label}"

    progress_fill_class = "progress-fill"
    if running:
        progress_fill_class += " progress-fill-running"

    return (
        "<div class='progress-shell'>"
        "<div class='progress-head'>"
        f"<span class='progress-title'>Progress: {current_iter}/{effective_max} ({percent}%)</span>"
        f"<span class='helper-text'>{stage_text.replace('&nbsp; | &nbsp; ', '').strip()}</span>"
        "</div>"
        "<div class='progress-track'>"
        f"<div class='{progress_fill_class}' style='width:{percent}%;'></div>"
        "</div>"
        "</div>"
    )


def build_elapsed_html(run_dir_str: str, run_json_str: str) -> str:
    if not run_dir_str and not run_json_str:
        return "<div class='meta-card compact-meta'><span class='k'>Elapsed</span><span class='v'>—</span></div>"
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
        return "<div class='meta-card compact-meta'><span class='k'>Elapsed</span><span class='v'>—</span></div>"
    try:
        start_dt = datetime.fromisoformat(start_iso)
    except Exception:
        return "<div class='meta-card compact-meta'><span class='k'>Elapsed</span><span class='v'>—</span></div>"
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
    return f"<div class='meta-card compact-meta'><span class='k'>{label}</span><span class='v'>{minutes}m {seconds}s</span></div>"




def _is_completed(parsed: Dict[str, Any]) -> bool:
    if parsed.get("cancelled") or parsed.get("error_message"):
        return True
    if parsed.get("success") is not None:
        return True
    runtime = parsed.get("runtime", {}) or {}
    return bool(runtime.get("end_time"))


def _is_idle(parsed: Dict[str, Any]) -> bool:
    status_text = str(parsed.get("status_text") or "").strip().lower()
    if status_text.startswith("no run yet"):
        return True
    runtime = parsed.get("runtime", {}) or {}
    steps = parsed.get("steps", []) or []
    task = parsed.get("task", {}) or {}
    return (
        not steps
        and not runtime.get("start_time")
        and not task
        and parsed.get("success") is None
        and not parsed.get("cancelled")
        and not parsed.get("error_message")
    )


def _action_label(action_name: Optional[str]) -> str:
    return action_name or "—"


def _get_stop_reason(parsed: Dict[str, Any]) -> Optional[str]:
    error_message = parsed.get("error_message")
    if error_message:
        first = str(error_message).strip().splitlines()[0]
        return f"Failed: {first}"

    stop_reason = parsed.get("stop_reason")
    if stop_reason:
        return str(stop_reason)

    for step in reversed(parsed.get("steps", []) or []):
        step_error = step.get("error_message")
        if step_error:
            first = str(step_error).strip().splitlines()[0]
            return f"Failed: {first}"
        step_reason = step.get("stop_reason")
        if step_reason:
            return str(step_reason)
    status_text = str(parsed.get("status_text") or "").strip()
    if status_text and ("fail" in status_text.lower() or "error" in status_text.lower()):
        return status_text
    return None

def build_action_timeline(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        return (
            "<div class='empty-state'>"
            "<div class='empty-state-title'>No stage timeline yet</div>"
            "<div class='empty-state-copy'>Run progress and evaluation checkpoints will be listed here.</div>"
            "</div>"
        )

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
            "<div class='timeline-row'>"
            f"<div class='timeline-dot' style='background:{stage_colors['action']};'></div>"
            f"<div class='timeline-value'>{action_label}</div>"
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
                "<div class='timeline-row'>"
                f"<div class='timeline-dot' style='background:{color};'></div>"
                f"<div class='timeline-value'>{eval_label}</div>"
                "</div>"
            )

        items.append(
            "<div class='meta-card timeline-card'>"
            f"<div class='section-title'>Iteration {step.get('step')}</div>"
            + "".join(stage_items)
            + "</div>"
        )

    if _is_completed(parsed):
        items.append(
            "<div class='meta-card timeline-card'>"
            "<div class='timeline-row'>"
            f"<div class='timeline-dot' style='background:{stage_colors['done']};'></div>"
            "<div class='timeline-value'>Run completed - ready to review results</div>"
            "</div></div>"
        )

    return "<div class='timeline-wrap'>" + "".join(items) + "</div>"


def build_stage_panel(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        status_text = parsed.get("status_text") or ""
        reason = _get_stop_reason(parsed)
        if reason:
            return (
                "<div class='meta-card stage-card'>"
                "<div class='section-title'>Run ended</div>"
                "<div class='k'>Stop/Fail reason</div>"
                f"<div class='v'>{reason}</div>"
                "</div>"
            )
        if "Starting" in status_text or "Run in progress" in status_text:
            return (
                "<div class='meta-card stage-card'>"
                "<div class='stage-live'>"
                "<span class='dot-live'></span>"
                "Starting run..."
                "</div>"
                "<div class='helper-text'>Waiting for first status update.</div>"
                "</div>"
            )
        return (
            "<div class='empty-state'>"
            "<div class='empty-state-title'>No active run</div>"
            "<div class='empty-state-copy'>Configure target, model, and API key, then click Run LIDDIA to begin monitoring.</div>"
            "</div>"
        )

    completed = _is_completed(parsed)
    latest_step = steps[-1]

    if not completed:
        action_label = _action_label(latest_step.get("action_name"))
        if latest_step.get("goal_response") or (latest_step.get("goal_eval") or {}).get("answer"):
            action_label = f"{action_label} → Evaluating"
        return (
            "<div class='meta-card stage-card'>"
            "<div class='stage-live'>"
            "<span class='dot-live'></span>"
            "Current stage"
            "<span class='spinner'></span>"
            "</div>"
            f"<div class='v'>Action: {action_label}</div>"
            f"<div class='k'>Iteration: {latest_step.get('step') if latest_step.get('step') is not None else '—'}</div>"
            "</div>"
        )

    timeline_html = build_action_timeline(parsed)
    reason = _get_stop_reason(parsed)
    if not reason:
        return timeline_html
    return (
        timeline_html
        + "<div class='meta-card stage-card'>"
        + "<div class='section-title'>Stop/Fail reason</div>"
        + f"<div class='v'>{reason}</div>"
        + "</div>"
    )


def build_live_status(parsed: Dict[str, Any]) -> str:
    steps = parsed.get("steps", []) or []
    if not steps:
        return (
            "<div class='empty-state'>"
            "<div class='empty-state-title'>Live status not available yet</div>"
            "<div class='empty-state-copy'>Updates will appear here once the run starts processing actions.</div>"
            "</div>"
        )
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
        status_spinner = "<span class='spinner'></span>"
    reason = _get_stop_reason(parsed)
    reason_html = ""
    if reason:
        reason_html = (
            "<div class='status-divider'>"
            "<div class='k'>Stop/Fail reason</div>"
            f"<div class='v'>{reason}</div>"
            "</div>"
        )
    return (
        "<div class='meta-card stage-card'>"
        f"<div class='stage-live'>Live status {status_spinner}</div>"
        f"<div class='v' title='Action = current action for this iteration.'>Action: {action_label}</div>"
        f"<div class='k'>Iteration: {latest_step.get('step') if latest_step.get('step') is not None else '—'}</div>"
        f"{reason_html}"
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
    all_metrics = set()
    for step in parsed.get("steps", []) or []:
        pool = step.get("pool_stats", {}) or {}
        metrics = pool.get("metrics", {}) or {}
        all_metrics.update(metrics.keys())
    
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
        # include metric mins/maxs for all available metrics
        for metric_name in sorted(all_metrics):
            metric_data = metrics.get(metric_name)
            if isinstance(metric_data, dict):
                row[f"{metric_name.lower().replace(' ', '_')}_min"] = _fmt_str(metric_data.get("min"))
                row[f"{metric_name.lower().replace(' ', '_')}_max"] = _fmt_str(metric_data.get("max"))
        rows.append(row)
    cols = ["step", "action", "output_pool", "goal_check", "pool_size", "diversity"] + [f"{m.lower().replace(' ', '_')}_min" for m in sorted(all_metrics)] + [f"{m.lower().replace(' ', '_')}_max" for m in sorted(all_metrics)]
    return _format_df_for_display(_round_df(pd.DataFrame(rows, columns=cols)))

def build_metrics_table(parsed: Dict[str, Any]) -> pd.DataFrame:
    final_pool = parsed.get("final_pool", {}) or {}
    metrics = final_pool.get("metrics") or {}
    rows = []
    # Always include size/diversity if present
    if "size" in final_pool:
        rows.append({"metric": "Size", "min": final_pool.get("size"), "max": final_pool.get("size"), "median": None})
    if "diversity" in final_pool:
        rows.append({"metric": "Diversity", "min": _fmt_str(final_pool.get("diversity")), "max": _fmt_str(final_pool.get("diversity")), "median": _fmt_str(final_pool.get("diversity"))})
    for metric, stats in metrics.items():
        if isinstance(stats, dict):
            rows.append({
                "metric": metric,
                "min": _fmt_str(stats.get("min")),
                "max": _fmt_str(stats.get("max")),
                "median": _fmt_str(stats.get("median")),
            })
    return _format_df_for_display(_round_df(pd.DataFrame(rows, columns=["metric", "min", "max", "median"])))

def build_results_markdown(parsed: Dict[str, Any]) -> str:
    # Temporarily hidden per UI request (keep logic commented for easy restore).
    #
    # final_pool = parsed.get("final_pool", {}) or {}
    # if not final_pool:
    #     return "No final pool found yet."
    # lines = ["**Final Pool**"]
    # pool_id = final_pool.get("pool") or "Unknown"
    # size = final_pool.get("size") or "—"
    # diversity = final_pool.get("diversity")
    # lines.append(f"- Pool: {pool_id}")
    # lines.append(f"- Molecules: {size}")
    # if diversity is not None:
    #     lines.append(f"- Diversity: {_fmt_str(diversity)}")
    # metrics = final_pool.get("metrics") or {}
    # for metric, stats in metrics.items():
    #     if isinstance(stats, dict):
    #         mn = stats.get("min")
    #         mx = stats.get("max")
    #         med = stats.get("median")
    #         lines.append(f"- {metric}: min {_fmt_str(mn)}, max {_fmt_str(mx)}, median {_fmt_str(med)}")
    # return "\n".join(lines)
    return ""

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
    runtime = parsed.get("runtime", {}) or {}
    last_event_time = runtime.get("updated_at", "—")
    status_pill = "RUNNING"
    status_class = "status-running"
    if _is_idle(parsed):
        status_pill = "IDLE"
        status_class = "status-idle"
    elif _is_completed(parsed):
        status_pill = "COMPLETE"
        status_class = "status-success"
    if parsed.get("cancelled"):
        status_pill = "ERROR"
        status_class = "status-warning"
    if parsed.get("error_message"):
        status_pill = "ERROR"
        status_class = "status-failed"
    run_dir_str = str(run_dir) if run_dir else ""
    run_json_str = str(run_json_path) if run_json_path else ""
    metric_trends_df = build_metric_trend_df(run_dir_str, run_json_str)
    metric_trends_fig = build_metric_plot(filter_metric_trends(metric_trends_df, "All"))
    status_badge = (
        "<div class='status-row'>"
        f"<span class='status-badge {status_class}'>{status_pill}</span>"
        f"<span class='helper-text'>Last update: {last_event_time}</span>"
        "</div>"
    )
    if "recovered active run from disk" in (status_text or "").lower():
        status_badge = (
            "<div class='status-row'>"
            f"<span class='status-badge {status_class}'>{status_pill}</span>"
            f"<span class='helper-text'>Last update: {last_event_time}</span>"
            "<span class='status-badge status-info'>Recovered active run from disk</span>"
            "</div>"
        )
    return status_text, summary, progress_html, runtime_md, results_md, metrics_df, summary, steps_df, metric_trends_fig, stage_html, live_html, run_dir_str, run_json_str, status_badge


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


def _write_lock(info: Dict[str, Any]) -> None:
    try:
        _run_lock_path().write_text(json.dumps(info, indent=2))
    except Exception:
        pass


def _run_state_path(run_dir: Optional[Path]) -> Optional[Path]:
    if not run_dir:
        return None
    return run_dir / "run_state.json"


def _write_run_state(
    run_dir: Optional[Path],
    *,
    status: str,
    pid: Optional[int] = None,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    target: Optional[str] = None,
    model: Optional[str] = None,
    max_iter: Optional[int] = None,
    run_json_path: Optional[Path] = None,
) -> None:
    path = _run_state_path(run_dir)
    if not path:
        return
    payload: Dict[str, Any] = {
        "status": status,
        "updated_at": datetime.now().isoformat(),
        "pid": pid,
        "started_at": started_at,
        "finished_at": finished_at,
        "target": target,
        "model": model,
        "max_iter": max_iter,
        "run_dir": str(run_dir),
        "run_json_path": str(run_json_path) if run_json_path else None,
    }
    try:
        path.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass


def _send_desktop_notification(title: str, message: str) -> None:
    title = (title or "LIDDiA").replace('"', "'")
    message = (message or "").replace('"', "'")
    try:
        if sys.platform == "darwin":
            subprocess.run(
                [
                    "osascript",
                    "-e",
                    f'display notification "{message}" with title "{title}"',
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        if os.name == "nt":
            # Lightweight fallback for Windows without extra dependencies.
            ps = (
                "Add-Type -AssemblyName PresentationFramework; "
                f"[System.Windows.MessageBox]::Show('{message}', '{title}') | Out-Null"
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return
        # Linux fallback (if notify-send exists)
        subprocess.run(
            ["notify-send", title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


def _find_run_dir_after(started_at: float, exclude: Optional[set] = None) -> Optional[Path]:
    if not LOG_ROOT.exists():
        return None
    exclude = exclude or set()
    candidates: List[Path] = []
    for child in LOG_ROOT.iterdir():
        if not child.is_dir() or child.name in exclude:
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


def _has_run_activity_since(started_at: float, known_dirs: Optional[set] = None) -> bool:
    if not LOG_ROOT.exists():
        return False
    known_dirs = known_dirs or set()
    for child in LOG_ROOT.iterdir():
        if not child.is_dir():
            continue
        if child.name in known_dirs:
            continue
        try:
            if child.stat().st_mtime >= started_at - 2:
                return True
        except Exception:
            continue
    for j in LOG_ROOT.glob("*/*.json"):
        try:
            if j.stat().st_mtime >= started_at - 2:
                return True
        except Exception:
            continue
    return False


def refresh_monitor_outputs(run_dir_str: str, run_json_str: str) -> Tuple[Any, ...]:
    run_dir = Path(run_dir_str) if run_dir_str else None
    run_json_path = Path(run_json_str) if run_json_str else None
    if run_json_path and not run_json_path.exists():
        run_json_path = None
    if run_dir and not run_dir.exists():
        run_dir = None

    lock_info = _read_lock()
    pid = lock_info.get("pid")
    running = _pid_running(pid)
    started_at = float(lock_info.get("started_at") or 0) if lock_info else 0
    known_dirs = set(lock_info.get("known_dirs") or []) if lock_info else set()

    # Guard against stale lock files / PID reuse showing false "Running" on app open.
    if running and started_at:
        lock_age = time.time() - started_at
        if lock_age > 180 and not _has_run_activity_since(started_at, known_dirs):
            _run_lock_path().unlink(missing_ok=True)
            lock_info = {}
            running = False

    if lock_info and not running:
        active_run_dir_str = lock_info.get("active_run_dir")
        active_run_dir = Path(active_run_dir_str) if active_run_dir_str else None
        latest_json = None
        run_outcome = "completed"
        if active_run_dir and active_run_dir.exists():
            latest_json = _latest_run_json_in_dir(active_run_dir)
            run_json = _safe_read_json(latest_json) if latest_json else None
            if isinstance(run_json, dict) and run_json.get("success") is False:
                run_outcome = "failed"
            _write_run_state(
                active_run_dir,
                status="completed",
                pid=lock_info.get("pid"),
                started_at=float(lock_info.get("started_at") or 0),
                finished_at=time.time(),
                target=lock_info.get("target"),
                model=lock_info.get("model"),
                max_iter=lock_info.get("max_iter"),
                run_json_path=latest_json,
            )
            run_name = active_run_dir.name
            _send_desktop_notification(
                "LIDDiA Run Finished",
                f"{run_name} {run_outcome}. You can reopen the GUI to review results.",
            )
        _run_lock_path().unlink(missing_ok=True)
        lock_info = {}

    # Keep active run in focus while a detached worker is running.
    if running:
        started_at = float(lock_info.get("started_at") or time.time())
        known_dirs = set(lock_info.get("known_dirs") or [])
        active_run_dir_str = lock_info.get("active_run_dir")
        active_run_dir = Path(active_run_dir_str) if active_run_dir_str else None
        recovered_from_disk = False
        notice_until = float(lock_info.get("recovered_notice_until") or 0)
        if active_run_dir and (not run_dir_str or str(active_run_dir) != run_dir_str):
            if not lock_info.get("recovered_notified"):
                lock_info["recovered_notified"] = True
                lock_info["recovered_notice_until"] = time.time() + 45
                _write_lock(lock_info)
                notice_until = float(lock_info.get("recovered_notice_until") or 0)
            recovered_from_disk = time.time() < notice_until
        elif notice_until > 0:
            recovered_from_disk = time.time() < notice_until
        # Important: while a new run is active, do not keep rendering an older run directory.
        # If the current UI state points to a pre-existing run, ignore it until a new run dir appears.
        if run_dir and run_dir.name in known_dirs:
            run_dir = None
            run_json_path = None

        if active_run_dir and active_run_dir.exists():
            run_dir = active_run_dir
        if run_dir is None:
            run_dir = _detect_new_run_dir(known_dirs, started_at) or _find_run_dir_after(started_at, known_dirs)
        if run_dir:
            if str(run_dir) != str(active_run_dir_str or ""):
                lock_info["active_run_dir"] = str(run_dir)
                _write_lock(lock_info)
            run_json_path = _latest_run_json_in_dir(run_dir)
            _write_run_state(
                run_dir,
                status="running",
                pid=lock_info.get("pid"),
                started_at=started_at,
                target=lock_info.get("target"),
                model=lock_info.get("model"),
                max_iter=lock_info.get("max_iter"),
                run_json_path=run_json_path,
            )
        run_data = _safe_read_json(run_json_path) if run_json_path else None
        prev_data = None
        prev_json = None
        if run_json_str:
            prev_json = Path(run_json_str)
            if prev_json.exists():
                prev_data = _safe_read_json(prev_json)
        # If the current snapshot is temporarily unreadable (mid-write), keep the last known JSON.
        if run_data is None and prev_data:
            run_json_path = prev_json
            run_data = prev_data
        # Guard against transient JSON rewrites that momentarily drop step history.
        if run_data and prev_data:
            try:
                cur_steps = len(_extract_iterations(run_data))
                old_steps = len(_extract_iterations(prev_data))
            except Exception:
                cur_steps = 0
                old_steps = 0
            if old_steps > 0 and cur_steps < old_steps:
                run_json_path = prev_json
                run_data = prev_data
        # Final guard: keep an in-memory last-good snapshot during active runs.
        cache_key = str(run_dir) if run_dir else ""
        if cache_key:
            cached = _LAST_GOOD_RUN_DATA.get(cache_key)
            if run_data:
                try:
                    cur_steps = len(_extract_iterations(run_data))
                except Exception:
                    cur_steps = 0
                if cached:
                    try:
                        cached_steps = len(_extract_iterations(cached))
                    except Exception:
                        cached_steps = 0
                    if cached_steps > 0 and cur_steps < cached_steps:
                        run_data = cached
                    elif cur_steps >= cached_steps:
                        _LAST_GOOD_RUN_DATA[cache_key] = run_data
                else:
                    _LAST_GOOD_RUN_DATA[cache_key] = run_data
            elif cached:
                run_data = cached
        if run_json_path:
            status_text = "Run in progress... (recovered active run from disk)" if recovered_from_disk else "Run in progress..."
        else:
            status_text = "Run started. Waiting for run artifacts..."
        return _render_outputs(status_text, run_data, run_json_path, run_dir=run_dir)

    # No active run: preserve currently loaded run, if any.
    if run_json_path:
        run_data = _safe_read_json(run_json_path)
        if run_data:
            return _render_outputs("Loaded run.", run_data, run_json_path, run_dir=run_dir or run_json_path.parent)

    if run_dir:
        run_json_path = _latest_run_json_in_dir(run_dir)
        run_data = _safe_read_json(run_json_path) if run_json_path else None
        if run_data:
            return _render_outputs("Loaded run.", run_data, run_json_path, run_dir=run_dir)

    return _render_outputs("No run yet. Click Run LIDDiA to start.", None, None, run_dir=None)


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
    if not LOG_ROOT.exists():
        LOG_ROOT.mkdir(parents=True, exist_ok=True)

    existing_dirs = {p.name for p in LOG_ROOT.iterdir()} if LOG_ROOT.exists() else set()
    start_ts = time.time()
    _write_lock({"pid": None, "started_at": start_ts, "known_dirs": sorted(existing_dirs), "active_run_dir": None})

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

    log_stamp = int(time.time())
    stdout_log = LOG_ROOT / f".run_{log_stamp}.stdout.log"
    stderr_log = LOG_ROOT / f".run_{log_stamp}.stderr.log"
    stdout_fh = open(stdout_log, "a")
    stderr_fh = open(stderr_log, "a")
    popen_kwargs: Dict[str, Any] = {
        "cwd": str(REPO_ROOT),
        "env": env,
        "stdout": stdout_fh,
        "stderr": stderr_fh,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
        )
    else:
        popen_kwargs["start_new_session"] = True

    try:
        process = subprocess.Popen(cmd, **popen_kwargs)
    except Exception as e:
        stdout_fh.close()
        stderr_fh.close()
        yield _render_outputs(f"Failed to start run: {e}", None, None)
        lock_path.unlink(missing_ok=True)
        return
    stdout_fh.close()
    stderr_fh.close()
    started_at = time.time()
    lock_info = {
        "pid": process.pid,
        "started_at": started_at,
        "known_dirs": sorted(existing_dirs),
        "active_run_dir": None,
        "target": target.strip(),
        "model": model.strip(),
        "max_iter": int(max_iter),
        "stdout_log": str(stdout_log),
        "stderr_log": str(stderr_log),
    }
    _write_lock(lock_info)

    active_run_dir = _detect_new_run_dir(existing_dirs, started_at) or _find_run_dir_after(started_at, existing_dirs)
    if active_run_dir:
        lock_info["active_run_dir"] = str(active_run_dir)
        _write_lock(lock_info)
        _write_run_state(
            active_run_dir,
            status="running",
            pid=process.pid,
            started_at=started_at,
            target=target.strip(),
            model=model.strip(),
            max_iter=int(max_iter),
            run_json_path=_latest_run_json_in_dir(active_run_dir),
        )
    run_json_path = _latest_run_json_in_dir(active_run_dir) if active_run_dir else None
    run_data = _safe_read_json(run_json_path) if run_json_path else None
    status_text = "Run in progress..." if run_json_path else "Run started. Waiting for run artifacts..."
    yield _render_outputs(status_text, run_data, run_json_path, run_dir=active_run_dir)
    return


def load_latest_run() -> Tuple[Any, ...]:
    run_json_path = _latest_run_json()
    if not run_json_path:
        return _render_outputs("No runs found.", None, None)

    run_data = _safe_read_json(run_json_path)
    if not run_data:
        return _render_outputs(f"Could not read {run_json_path.name}.", None, run_json_path)

    status = f"Loaded latest run: {run_json_path.parent.name}"
    return _render_outputs(status, run_data, run_json_path)


def load_selected_run(run_folder: str) -> Tuple[Any, ...]:
    if not run_folder:
        return _render_outputs("No run selected.", None, None)

    run_dir = LOG_ROOT / run_folder
    if not run_dir.exists() or not run_dir.is_dir():
        return _render_outputs(f"Run folder not found: {run_folder}", None, None)

    json_files = list(run_dir.glob("*.json"))
    if not json_files:
        return _render_outputs(f"No JSON file found in {run_folder}", None, None)

    run_json_path = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    run_data = _safe_read_json(run_json_path)
    if not run_data:
        return _render_outputs(f"Could not read {run_json_path.name}.", None, run_json_path)

    status = f"Loaded run: {run_folder}"
    return _render_outputs(status, run_data, run_json_path, run_dir)


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
        tmp_path = REPORT_TMP_DIR / f"liddia_report_{int(time.time())}.json"
        tmp_path.write_text(json.dumps(payload, indent=2))
        return "JSON report ready.", str(tmp_path)

    if report_type == "csv":
        steps_df = build_metrics_table(parsed)
        tmp_path = REPORT_TMP_DIR / f"liddia_report_{int(time.time())}.csv"
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

    tmp_path = REPORT_TMP_DIR / f"liddia_report_{int(time.time())}.txt"
    tmp_path.write_text(report_text)
    return "Text report ready.", str(tmp_path)


def build_report_file(run_dir_str: str, run_json_str: str, report_type: str) -> Optional[str]:
    _, file_path = build_report(run_dir_str, run_json_str, report_type)
    return file_path


def build_report_bundle_file(run_dir_str: str, run_json_str: str) -> Optional[str]:
    import zipfile
    txt_path = build_report_file(run_dir_str, run_json_str, "txt")
    csv_path = build_report_file(run_dir_str, run_json_str, "csv")
    if not txt_path and not csv_path:
        return None
    bundle_path = REPORT_TMP_DIR / f"liddia_report_bundle_{int(time.time())}.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if txt_path and Path(txt_path).exists():
            zf.write(txt_path, "run_report.txt")
        if csv_path and Path(csv_path).exists():
            zf.write(csv_path, "final_pool_metrics.csv")
    return str(bundle_path)


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




def _metric_choices(run_dir_str: str, run_json_str: str) -> list[str]:
    df = build_metric_trend_df(run_dir_str, run_json_str)
    if df is None or df.empty or "metric" not in df.columns:
        return ["All"]
    metrics = sorted(set(df["metric"].dropna().astype(str).tolist()))
    return ["All"] + metrics
def build_metric_trend_df(run_dir_str: str, run_json_str: str = "") -> pd.DataFrame:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return pd.DataFrame(columns=["iteration", "metric", "value"])
    rows: List[Dict[str, Any]] = []
    mem = _load_memory(run_dir)
    if mem:
        pool_ids = _iteration_pool_ids(mem)
        for idx, pool_id in enumerate(pool_ids):
            pool_stats = _pool_stats_from_memory(mem, pool_id, run_dir)
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
    df = df.copy()
    df["metric"] = df["metric"].astype(str)
    metric = str(metric) if metric is not None else "All"
    if not metric or metric == "All":
        out = df.copy()
    else:
        out = df[df["metric"] == metric].copy()
    out["iteration"] = pd.to_numeric(out["iteration"], errors="coerce").fillna(0).astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce").round(2)
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


def build_molecule_view(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int) -> Tuple[str, str]:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return "No run selected.", ""
    mem = _load_memory(run_dir)
    if not mem:
        return "No memory.pkl found.", ""
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return "No molecule pools found.", ""
    # iteration is 0-based index
    if iteration < 0:
        iteration = 0
    if iteration >= len(pool_ids):
        iteration = max(0, len(pool_ids) - 1)
    pool_id = pool_ids[iteration]
    block = mem.stream.get(pool_id, {})
    df = block.get("data")
    if df is None or "SMILES" not in df.columns:
        return "SMILES not available.", ""
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
            return smiles, "<div>Could not parse SMILES.</div>"
        svg = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(250, 200), useSVG=True)
        # Make the main viewer image clickable to open full-size in a new tab.
        svg_url = "data:image/svg+xml;utf8," + urllib.parse.quote(str(svg))
        html = (
            "<div>"
            f"<a href='{svg_url}' target='_blank' rel='noopener noreferrer'>"
            f"{svg}"
            "</a>"
            "</div>"
        )
        return smiles, html
    except Exception as e:
        return smiles, f"<div>2D viewer requires RDKit.</div><div style='color:#9ca3af;font-size:12px;'>{e}</div>"


def _selected_smiles(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int) -> Optional[str]:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return None
    mem = _load_memory(run_dir)
    if not mem:
        return None
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return None
    iteration = max(0, min(int(iteration), len(pool_ids) - 1))
    pool_id = pool_ids[iteration]
    df = mem.stream.get(pool_id, {}).get("data")
    if df is None or "SMILES" not in df.columns or len(df) == 0:
        return None
    mol_index = max(0, min(int(mol_index), len(df) - 1))
    return str(df.iloc[mol_index]["SMILES"])


def _smiles_to_3d_molblock(smiles: str) -> Optional[str]:
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        embed = AllChem.EmbedMolecule(mol, params)
        if embed != 0:
            # fallback embed
            embed = AllChem.EmbedMolecule(mol, randomSeed=0xC0FFEE)
            if embed != 0:
                return None
        AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        return Chem.MolToMolBlock(mol)
    except Exception:
        return None


def _extract_pdbqt_pose(model_text: str, pose_index: int) -> Optional[str]:
    pose_index = max(1, int(pose_index or 1))
    blocks = re.findall(r"(?ms)^MODEL\s+\d+.*?^ENDMDL\s*$", model_text)
    if not blocks:
        return model_text if pose_index == 1 else None
    idx = pose_index - 1
    if idx >= len(blocks):
        return None
    return blocks[idx]


def _count_pdbqt_poses(model_text: str) -> int:
    blocks = re.findall(r"(?ms)^MODEL\s+\d+.*?^ENDMDL\s*$", model_text)
    return len(blocks)


def _extract_pdbqt_vina_score(block_text: str) -> Optional[float]:
    """Extract affinity (first numeric value) from 'REMARK VINA RESULT'."""
    if not block_text:
        return None
    for line in block_text.splitlines():
        if "REMARK VINA RESULT" not in line.upper():
            continue
        match = re.search(r"REMARK\s+VINA\s+RESULT\s*:\s*([-\d\.eE+]+)", line, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        # fallback: first parseable number on the line
        for tok in line.replace(":", " ").split():
            try:
                return float(tok)
            except Exception:
                continue
    return None


def _pdbqt_block_to_pdb(block_text: str) -> str:
    """Convert a pdbqt pose block to pdb lines for more reliable 3Dmol rendering."""
    out_lines: List[str] = []
    for line in block_text.splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        record = line[0:6].strip() or "ATOM"
        try:
            serial = int((line[6:11] or "").strip())
        except Exception:
            serial = len(out_lines) + 1
        atom_name = (line[12:16] or "").strip()[:4] or "C"
        res_name = (line[17:20] or "").strip()[:3] or "LIG"
        chain = (line[21:22] or "").strip() or "A"
        try:
            res_seq = int((line[22:26] or "").strip())
        except Exception:
            res_seq = 1
        try:
            x = float((line[30:38] or "").strip())
            y = float((line[38:46] or "").strip())
            z = float((line[46:54] or "").strip())
        except Exception:
            # fallback for non-standard spacing
            parts = line.split()
            # ATOM serial name res [chain] resSeq x y z occ bfac charge atomType
            if len(parts) < 8:
                continue
            # detect chain token by checking whether parts[4] is alpha
            has_chain = len(parts) > 9 and parts[4].isalpha() and len(parts[4]) <= 2
            try:
                if has_chain:
                    chain = parts[4][:1]
                    res_seq = int(parts[5])
                    x = float(parts[6]); y = float(parts[7]); z = float(parts[8])
                else:
                    res_seq = int(parts[4])
                    x = float(parts[5]); y = float(parts[6]); z = float(parts[7])
            except Exception:
                continue
        try:
            occ = float((line[54:60] or "").strip())
        except Exception:
            occ = 1.0
        try:
            bfac = float((line[60:66] or "").strip())
        except Exception:
            bfac = 0.0

        parts = line.split()
        atom_type = parts[-1].strip().upper() if parts else ""
        element_map = {
            "OA": "O",
            "NA": "N",
            "SA": "S",
            "HD": "H",
            "A": "C",
        }
        element = element_map.get(atom_type, atom_type[:2] if atom_type else "")
        if not element or not re.match(r"^[A-Z][A-Z]?$", element):
            element = re.sub(r"[^A-Za-z]", "", atom_name)[:2].strip().upper() or "C"
        pdb_line = (
            f"{record:<6}{serial:>5} "
            f"{atom_name:<4} "
            f"{res_name:>3} "
            f"{chain:1}{res_seq:>4}    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
            f"{occ:>6.2f}{bfac:>6.2f}          "
            f"{element:>2}"
        )
        out_lines.append(pdb_line)
    if out_lines:
        out_lines.append("END")
    return "\n".join(out_lines)


def _build_3d_html(
    model_text: str,
    model_type: str,
    style: str,
    ligand_color: str = "spectrum",
    receptor_text: Optional[str] = None,
    receptor_type: str = "pdb",
    receptor_style: str = "cartoon",
    receptor_color: str = "#3b82f6",
    receptor_opacity: float = 0.80,
) -> str:
    view_id = f"mol3d_{int(time.time() * 1000)}"
    model_js = json.dumps(model_text)
    color_scheme_values = {"spectrum", "greenCarbon", "cyanCarbon", "orangeCarbon", "magentaCarbon", "whiteCarbon"}
    surface_color_map = {
        "spectrum": "#60a5fa",
        "greenCarbon": "#22c55e",
        "cyanCarbon": "#06b6d4",
        "orangeCarbon": "#f97316",
        "magentaCarbon": "#d946ef",
        "whiteCarbon": "#e5e7eb",
    }

    def _style_js(rep: str, color_value: str, *, radius: float, linewidth: float, scale: float, opacity: float = 1.0) -> str:
        rep = (rep or "stick").strip().lower()
        color_value = (color_value or "spectrum").strip()
        is_scheme = color_value in color_scheme_values
        if rep == "cartoon":
            if color_value == "spectrum":
                return f"{{cartoon:{{color:'spectrum',opacity:{opacity:.3f}}}}}"
            if is_scheme:
                return f"{{cartoon:{{colorscheme:'{color_value}',opacity:{opacity:.3f}}}}}"
            return f"{{cartoon:{{color:'{color_value}',opacity:{opacity:.3f}}}}}"
        if rep == "line":
            if is_scheme:
                return f"{{line:{{linewidth:{linewidth},colorscheme:'{color_value}'}}}}"
            return f"{{line:{{linewidth:{linewidth},color:'{color_value}'}}}}"
        if rep == "sphere":
            if is_scheme:
                return f"{{sphere:{{scale:{scale},colorscheme:'{color_value}'}}}}"
            return f"{{sphere:{{scale:{scale},color:'{color_value}'}}}}"
        # stick default
        if is_scheme:
            return f"{{stick:{{radius:{radius},colorscheme:'{color_value}'}}}}"
        return f"{{stick:{{radius:{radius},color:'{color_value}'}}}}"

    ligand_style_name = (style or "stick").strip().lower()
    receptor_style_name = (receptor_style or "cartoon").strip().lower()
    ligand_is_surface = ligand_style_name == "surface"
    receptor_is_surface = receptor_style_name == "surface"

    ligand_style_js = _style_js(ligand_style_name if not ligand_is_surface else "stick", ligand_color, radius=0.18, linewidth=1.5, scale=0.28)
    receptor_opacity = max(0.05, min(1.0, float(receptor_opacity or 0.80)))
    receptor_style_js = _style_js(
        receptor_style_name if not receptor_is_surface else "cartoon",
        receptor_color,
        radius=0.14,
        linewidth=1.2,
        scale=0.22,
        opacity=receptor_opacity,
    )
    ligand_surface_color = surface_color_map.get((ligand_color or "").strip(), ligand_color or "#60a5fa")
    receptor_surface_color = surface_color_map.get((receptor_color or "").strip(), receptor_color or "#3b82f6")

    # Render inside an iframe because Gradio may sanitize/ignore <script> in raw HTML blocks.
    receptor_js = json.dumps(receptor_text) if receptor_text else "null"
    ligand_is_surface_js = "true" if ligand_is_surface else "false"
    receptor_is_surface_js = "true" if receptor_is_surface else "false"
    ligand_surface_color_js = json.dumps(ligand_surface_color)
    receptor_surface_color_js = json.dumps(receptor_surface_color)
    iframe_doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    html, body {{ margin:0; padding:0; background:#fff; font-family:Arial,sans-serif; }}
    #{view_id} {{ width:100%; height:520px; }}
    #err {{ color:#7f1d1d; padding:12px; font-size:14px; }}
  </style>
</head>
<body>
  <div id="{view_id}"></div>
  <div id="err"></div>
  <script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
  <script>
    (function() {{
      const err = document.getElementById("err");
      function fail(msg) {{ err.textContent = msg; }}
      try {{
        if (!window.$3Dmol) {{
          fail("3Dmol.js failed to load (CDN blocked).");
          return;
        }}
        const el = document.getElementById("{view_id}");
        const v = window.$3Dmol.createViewer(el, {{ backgroundColor: "white" }});
        const receptorText = {receptor_js};
        if (receptorText) {{
          const rec = v.addModel(receptorText, "{receptor_type}");
          if ({receptor_is_surface_js}) {{
            v.addSurface(window.$3Dmol.SurfaceType.VDW, {{opacity:{receptor_opacity:.3f}, color:{receptor_surface_color_js}}}, {{model: rec}});
          }} else {{
            rec.setStyle({{}}, {receptor_style_js});
          }}
        }}
        const lig = v.addModel({model_js}, "{model_type}");
        if ({ligand_is_surface_js}) {{
          v.addSurface(window.$3Dmol.SurfaceType.VDW, {{opacity:0.85, color:{ligand_surface_color_js}}}, {{model: lig}});
        }} else {{
          lig.setStyle({{}}, {ligand_style_js});
        }}
        v.zoomTo();
        v.render();
      }} catch (e) {{
        fail("3D viewer failed: " + (e && e.message ? e.message : String(e)));
      }}
    }})();
  </script>
</body>
</html>
"""
    data_uri = "data:text/html;charset=utf-8," + urllib.parse.quote(iframe_doc)
    return (
        "<iframe "
        f"src='{data_uri}' "
        "style='width:100%;height:540px;border:1px solid #e5e7eb;border-radius:10px;background:#fff;' "
        "sandbox='allow-scripts allow-same-origin'></iframe>"
    )


def render_3d_viewer(
    source_mode: str,
    run_dir_str: str,
    run_json_str: str,
    iteration: int,
    mol_index: int,
    structure_file,
    style: str,
    ligand_color: str,
    pose_index: int,
    receptor_file,
    receptor_style: str,
    receptor_color: str,
    receptor_opacity: float,
) -> Tuple[str, str, str]:
    def _pose_badge(text: str) -> str:
        return (
            "<div style='display:inline-flex;align-items:center;gap:6px;padding:6px 10px;border:1px solid #d1d5db;"
            "border-radius:999px;background:#f8fafc;font-weight:700;color:#0f172a;'>"
            f"{text}</div>"
        )

    source_mode = (source_mode or "").strip().lower()
    if source_mode.startswith("current"):
        smiles = _selected_smiles(run_dir_str, run_json_str, int(iteration), int(mol_index))
        if not smiles:
            return "No current molecule is available yet.", "", _pose_badge("Current molecule")
        mol_block = _smiles_to_3d_molblock(smiles)
        if not mol_block:
            return "Could not generate 3D coordinates from current SMILES.", "", _pose_badge("Current molecule")
        html = _build_3d_html(
            mol_block,
            "sdf",
            style or "stick",
            ligand_color=ligand_color or "spectrum",
            receptor_style=receptor_style or "cartoon",
            receptor_color=receptor_color or "#3b82f6",
            receptor_opacity=float(receptor_opacity or 0.80),
        )
        return (
            f"Rendered current molecule (iteration {iteration}, index {mol_index}).",
            html,
            _pose_badge("Current molecule"),
        )

    if structure_file is None:
        return "Upload a .pdb, .sdf, .mol2, or .pdbqt file first.", "", _pose_badge("No file loaded")
    path = Path(structure_file.name)
    ext = path.suffix.lower()
    model_type_map = {".pdb": "pdb", ".sdf": "sdf", ".mol2": "mol2", ".pdbqt": "pdbqt"}
    model_type = model_type_map.get(ext)
    original_model_type = model_type
    if not model_type:
        return "Unsupported file type. Use .pdb, .sdf, .mol2, or .pdbqt.", "", _pose_badge("Unsupported file")
    try:
        text = path.read_text(errors="ignore")
    except Exception as e:
        return f"Could not read uploaded file: {e}", "", _pose_badge("Read error")
    if not text.strip():
        return "Uploaded file is empty.", "", _pose_badge("Empty file")
    pose_total = None
    vina_score = None
    if original_model_type == "pdbqt":
        pose_total = _count_pdbqt_poses(text)
        pose_text = _extract_pdbqt_pose(text, int(pose_index or 1))
        if not pose_text:
            if pose_total:
                return (
                    f"Pose {pose_index} not found in pdbqt (available: 1-{pose_total}).",
                    "",
                    _pose_badge(f"Pose {int(max(1, min(int(pose_index or 1), pose_total)))}/{pose_total}"),
                )
            return f"Pose {pose_index} not found in pdbqt.", "", _pose_badge("Pose not found")
        vina_score = _extract_pdbqt_vina_score(pose_text)
        # 3Dmol parsing is more reliable on plain PDB than full pdbqt pose blocks.
        pdb_text = _pdbqt_block_to_pdb(pose_text)
        if pdb_text:
            text = pdb_text
            model_type = "pdb"
        else:
            text = pose_text

    receptor_text = None
    receptor_model_type = "pdb"
    if receptor_file is not None:
        receptor_path = Path(receptor_file.name)
        try:
            receptor_text = receptor_path.read_text(errors="ignore")
            receptor_ext = receptor_path.suffix.lower()
            receptor_model_type = {
                ".pdb": "pdb",
                ".pdbqt": "pdbqt",
                ".mol2": "mol2",
            }.get(receptor_ext, "pdb")
            if receptor_model_type == "pdbqt":
                receptor_pdb_text = _pdbqt_block_to_pdb(receptor_text)
                if receptor_pdb_text:
                    receptor_text = receptor_pdb_text
                    receptor_model_type = "pdb"
        except Exception:
            receptor_text = None

    html = _build_3d_html(
        text,
        model_type,
        style or "stick",
        ligand_color=ligand_color or "spectrum",
        receptor_text=receptor_text,
        receptor_type=receptor_model_type,
        receptor_style=receptor_style or "cartoon",
        receptor_color=receptor_color or "#3b82f6",
        receptor_opacity=float(receptor_opacity or 0.80),
    )
    if original_model_type == "pdbqt":
        if pose_total:
            pose_i = int(max(1, min(int(pose_index or 1), pose_total)))
            score_suffix = f" Vina: {vina_score:.2f}" if vina_score is not None else ""
            badge = f"Pose {pose_i}/{pose_total}" + (f" • Vina {vina_score:.2f}" if vina_score is not None else "")
            return f"Rendered uploaded {ext} file (pose {pose_i} of {pose_total}).{score_suffix}", html, _pose_badge(badge)
        score_suffix = f" Vina: {vina_score:.2f}" if vina_score is not None else ""
        badge = f"Pose {int(pose_index or 1)}" + (f" • Vina {vina_score:.2f}" if vina_score is not None else "")
        return f"Rendered uploaded {ext} file (pose {int(pose_index or 1)}).{score_suffix}", html, _pose_badge(badge)
    return f"Rendered uploaded {ext} file.", html, _pose_badge("Single structure")


def _shift_pose_index(structure_file, pose_index: int, delta: int) -> int:
    current = max(1, int(pose_index or 1))
    if structure_file is None:
        return max(1, current + delta)
    try:
        path = Path(structure_file.name)
        if path.suffix.lower() != ".pdbqt":
            return max(1, current + delta)
        text = path.read_text(errors="ignore")
        n = _count_pdbqt_poses(text)
        if n <= 0:
            return max(1, current + delta)
        return max(1, min(n, current + delta))
    except Exception:
        return max(1, current + delta)


def build_molecule_table(run_dir_str: str, run_json_str: str, iteration: int, max_rows: Optional[int] = None) -> pd.DataFrame:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return pd.DataFrame()
    mem = _load_memory(run_dir)
    if not mem:
        return pd.DataFrame()
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return pd.DataFrame()
    # iteration is 0-based index
    if iteration < 0:
        iteration = 0
    if iteration >= len(pool_ids):
        iteration = max(0, len(pool_ids) - 1)
    pool_id = pool_ids[iteration]
    df = mem.stream.get(pool_id, {}).get("data")
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    # GUI display table: show molecule thumbnail instead of raw SMILES
    cols = [c for c in df.columns]
    out = df[cols].copy()
    if "SMILES" in out.columns:
        out.insert(0, "Molecule", out["SMILES"].map(_smiles_to_markdown_image))
        out = out.drop(columns=["SMILES"])
    # Add index column
    out.insert(0, "Index", range(len(out)))
    # Ensure consistent leading columns for display/update wiring.
    lead_cols = ["Index", "Molecule"]
    remaining_cols = [c for c in out.columns if c not in lead_cols]
    out = out[lead_cols + remaining_cols]
    if max_rows is not None and max_rows > 0 and len(out) > max_rows:
        out = out.head(max_rows)
    return _format_df_for_display(out, decimals=2)


def select_molecule_from_table(run_dir_str: str, run_json_str: str, iteration: int, evt: gr.SelectData):
    # evt.index is row index in displayed table
    idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
    if idx is None:
        return 0, "", ""
    smiles, svg = build_molecule_view(run_dir_str, run_json_str, int(iteration), int(idx))
    return idx, smiles, svg


def download_current_pool_csv(run_dir_str: str, run_json_str: str, iteration: int):
    from pathlib import Path
    run_dir = Path(run_dir_str)
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids or iteration >= len(pool_ids):
        return None
    pool_id = pool_ids[iteration]
    mem = _load_memory(run_dir)
    if not mem:
        return None
    df = mem.stream.get(pool_id, {}).get("data")
    if df is None or not isinstance(df, pd.DataFrame):
        return None
    csv_path = run_dir / f"{pool_id}.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def download_all_pools_csv(run_dir_str: str, run_json_str: str):
    import zipfile
    from pathlib import Path
    run_dir = Path(run_dir_str)
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    mem = _load_memory(run_dir)
    if not mem:
        return None
    zip_path = run_dir / "all_pools.zip"
    with zipfile.ZipFile(zip_path, 'w') as zf:
        for pool_id in pool_ids:
            df = mem.stream.get(pool_id, {}).get("data")
            if df is None or not isinstance(df, pd.DataFrame):
                continue
            csv_path = run_dir / f"{pool_id}.csv"
            df.to_csv(csv_path, index=False)
            zf.write(csv_path, f"{pool_id}.csv")
    return str(zip_path)


def _pool_ids_for_run(run_dir_str: str, run_json_str: str) -> List[str]:
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return []
    mem = _load_memory(run_dir)
    if not mem:
        return []
    return _iteration_pool_ids(mem)


def update_pool_selector(run_dir_str: str, run_json_str: str, current_pool: Optional[str] = None):
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids:
        return gr.update(choices=[], value=None), gr.update(value=0)
    # keep user's selected pool when possible; otherwise default to latest pool
    selected = current_pool if current_pool in pool_ids else pool_ids[-1]
    selected_idx = pool_ids.index(selected)
    return gr.update(choices=pool_ids, value=selected), gr.update(value=selected_idx)


def reset_molecule_viewer_state():
    return (
        gr.update(choices=[], value=None),  # pool_select
        gr.update(value=0),  # iteration_select_state
        gr.update(value=0),  # mol_index_state
        "<div class='status-badge status-idle'>Viewing: —</div>",  # pool_badge
        gr.update(value=pd.DataFrame()),  # mol_table
        gr.update(value=""),  # smiles_text
        gr.update(value=""),  # mol_svg
    )


def set_iteration_from_pool(run_dir_str: str, run_json_str: str, pool_id: str):
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids or pool_id not in pool_ids:
        return 0
    return pool_ids.index(pool_id)


def build_pool_badge(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int) -> str:
    pool_ids = _pool_ids_for_run(run_dir_str, run_json_str)
    if not pool_ids:
        return "<div class='status-badge status-idle'>Viewing: —</div>"
    # iteration is 0-based index
    if iteration < 0:
        iteration = 0
    if iteration >= len(pool_ids):
        iteration = max(0, len(pool_ids) - 1)
    pool_id = pool_ids[iteration]
    return (
        "<div class='status-badge status-running'>"
        f"Viewing: Iteration {iteration} | Pool {pool_id} | Molecule Index {mol_index}"
        "</div>"
    )


def get_viewer_limits(run_dir_str: str, run_json_str: str):
    run_dir = _resolve_run_dir(run_dir_str, run_json_str)
    if not run_dir:
        return 0, 0
    mem = _load_memory(run_dir)
    if not mem:
        return 0, 0
    pool_ids = _iteration_pool_ids(mem)
    if not pool_ids:
        return 0, 0
    max_iter = len(pool_ids)
    df = mem.stream.get(pool_ids[-1], {}).get("data")
    max_idx = max(0, len(df) - 1) if df is not None else 0
    # default to iteration 0 (0-based)
    return 0, 0


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
    # iteration is 0-based index
    if iteration < 0:
        iteration = 0
    if iteration >= len(pool_ids):
        iteration = max(0, len(pool_ids) - 1)
    pool_id = pool_ids[iteration]
    df = mem.stream.get(pool_id, {}).get("data")
    max_idx = max(0, len(df) - 1) if df is not None else 0
    return gr.update(value=0 if max_idx >= 0 else 0)


def _update_metric_trends(run_dir_str: str, run_json_str: str) -> pd.DataFrame:
    df = build_metric_trend_df(run_dir_str, run_json_str)
    return build_metric_plot(filter_metric_trends(df, "All"))


def _update_molecule_view(run_dir_str: str, run_json_str: str, iteration: int, mol_index: int):
    return build_molecule_view(run_dir_str, run_json_str, int(iteration), int(mol_index))


def _update_molecule_table(run_dir_str: str, run_json_str: str, iteration: int):
    df = build_molecule_table(run_dir_str, run_json_str, int(iteration))
    if df is None or df.empty:
        return gr.update(
            value=pd.DataFrame(
                [
                    {
                        "State": "No molecules available",
                        "Details": "Results will populate after a run is loaded and a pool is selected.",
                    }
                ]
            ),
            headers=["State", "Details"],
            datatype=["str", "str"],
            interactive=False,
        )
    dtypes = []
    for col in df.columns:
        dtypes.append("markdown" if str(col).lower() == "molecule" else "str")
    return gr.update(value=df, headers=list(df.columns), datatype=dtypes)


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
                "metric": _normalize_metric_label(col),
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


def _pool_stats_from_memory(mem, pool_id: str, run_dir: Optional[Path] = None) -> Dict[str, Any]:
    stats: Dict[str, Any] = {"pool": pool_id, "size": None, "diversity": None, "metrics": {}}
    if not mem or not pool_id:
        return stats
    block = mem.stream.get(pool_id, {}) if getattr(mem, "stream", None) else {}
    metrics = block.get("metrics") or {}
    df = None  # Load from CSV only if memory metrics are missing

    if isinstance(metrics, dict):
        stats["size"] = metrics.get("size") or metrics.get("Size")
        stats["diversity"] = metrics.get("diversity") or metrics.get("Diversity")
        # Use memory metrics as primary source
        stats["metrics"] = {k: v for k, v in metrics.items() if k not in ["size", "Size", "diversity", "Diversity"]}

    # Only compute from CSV if memory doesn't have metrics
    if not stats["metrics"] and run_dir:
        csv_path = run_dir / f"{pool_id}.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                stats["metrics"] = _metrics_from_df(df).set_index("metric").to_dict(orient="index")
                if stats["size"] is None:
                    stats["size"] = len(df)
            except Exception:
                pass

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
        # pull stats from memory and merge with existing parsed stats
        raw_stats = _pool_stats_from_memory(mem, pool_id, run_dir)
        existing_stats = step.get("pool_stats", {})
        # Merge: prefer memory stats, but keep parsed metrics if memory doesn't have them
        merged_stats = {**existing_stats, **raw_stats}
        if not raw_stats.get("metrics") and existing_stats.get("metrics"):
            merged_stats["metrics"] = existing_stats["metrics"]
        step["pool_stats"] = _format_pool_stats(merged_stats)
    parsed["steps"] = steps
    parsed["final_pool"] = steps[-1]["pool_stats"] if steps else {}
    return parsed




# ---------- UI ----------
with gr.Blocks(title="LIDDIA GUI v2") as demo:
    gr.HTML(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #f6f8fb;
  --card: #ffffff;
  --surface: #ffffff;
  --surface-soft: #f8fafc;
  --border: #e2e8f0;
  --border-strong: #cbd5e1;
  --text: #0f172a;
  --muted: #64748b;
  --muted-2: #94a3b8;
  --accent: #4f46e5;
  --accent-strong: #4338ca;
  --accent-soft: #eef2ff;
  --success: #16a34a;
  --danger: #dc2626;
  --warning: #d97706;
  --radius-lg: 14px;
  --radius-md: 10px;
  --space-1: 8px;
  --space-2: 16px;
  --space-3: 24px;
  --space-4: 32px;
}

body, .gradio-container, .gr-block, .gr-form, .gr-markdown, .gr-textbox, .gr-dropdown, .gr-button, .gr-dataframe, .gr-number {
  font-family: "Manrope", "Inter", "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif !important;
}
body, .gradio-container {
  background: var(--bg) !important;
}
.gradio-container {
  padding: 28px 32px !important;
}
.app-shell {
  max-width: 1440px;
  margin: 0 auto;
}
.page-header {
  margin-bottom: 22px;
}
.page-title {
  margin: 0;
  font-size: 26px;
  line-height: 1.1;
  color: var(--text);
  font-weight: 800;
  letter-spacing: -0.03em;
}
.page-subtitle {
  margin: 10px 0 0;
  font-size: 14px;
  color: #334155;
  font-weight: 500;
}
.section-title {
  margin: 0 0 4px;
  color: var(--text);
  font-size: 17px;
  line-height: 1.25;
  font-weight: 800;
  letter-spacing: -0.02em;
}
.helper-text {
  margin: 0 0 12px;
  color: var(--muted);
  font-size: 13px;
  line-height: 1.45;
  font-weight: 500;
}

/* Top tabs */
.gradio-container .tab-nav {
  border-bottom: 1px solid var(--border) !important;
  margin-bottom: 28px !important;
}
.gradio-container .tab-nav button {
  font-size: 14px !important;
  font-weight: 650 !important;
  color: #475569 !important;
  background: transparent !important;
  border: 0 !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 14px !important;
  transition: color .16s ease, border-color .16s ease !important;
}
.gradio-container .tab-nav button:hover {
  color: var(--text) !important;
}
.gradio-container .tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom-color: var(--accent) !important;
  background: transparent !important;
}

.monitor-layout,
.results-layout {
  gap: 18px !important;
  align-items: start !important;
}
.monitor-stack {
  gap: 18px !important;
}

/* Card system: fewer boxes, cleaner contrast */
.primary-card,
.secondary-card {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  overflow: visible !important;
}
.primary-card {
  padding: 20px !important;
  box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08) !important;
}
.secondary-card {
  padding: 16px !important;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.045) !important;
}

/* Remove the extra Gradio-looking nested panels that made the UI feel boxed in. */
.primary-card .gr-markdown,
.secondary-card .gr-markdown,
.primary-card .prose,
.secondary-card .prose,
.primary-card .gr-html,
.secondary-card .gr-html {
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}
.primary-card .block,
.secondary-card .block {
  border-color: var(--border) !important;
  box-shadow: none !important;
}
.primary-card label,
.secondary-card label {
  color: var(--muted) !important;
  font-weight: 700 !important;
  font-size: 12px !important;
}

/* Metadata is supportive, not a full card stack. */
.meta-card {
  background: transparent !important;
  border: 0 !important;
  border-radius: 0 !important;
  box-shadow: none !important;
  padding: 10px 0 !important;
}
.run-overview-grid {
  display: grid;
  gap: 0;
}
.run-overview-section {
  border-bottom: 1px solid var(--border) !important;
}
.run-overview-section:last-child {
  border-bottom: 0 !important;
  padding-bottom: 0 !important;
}
.run-overview-title {
  margin: 0 0 8px;
  text-transform: uppercase;
  letter-spacing: .07em;
  font-size: 11px;
  color: var(--muted);
  font-weight: 800;
}
.label-value-grid {
  display: grid;
  grid-template-columns: 96px 1fr;
  gap: 7px 12px;
  align-items: baseline;
}
.k {
  color: var(--muted);
  font-size: 12px;
  font-weight: 700;
}
.v {
  color: var(--text);
  font-size: 13px;
  font-weight: 750;
  overflow-wrap: anywhere;
}
.mono {
  font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  font-size: 12px;
  font-weight: 650;
}

/* Status + progress */
.status-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  margin-bottom: 12px;
}
.status-badge {
  display: inline-flex;
  align-items: center;
  width: fit-content;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 11px;
  line-height: 1.2;
  letter-spacing: .035em;
  font-weight: 800;
  border: 1px solid transparent;
}
.status-success { background: #ecfdf5; color: #047857; border-color: #bbf7d0; }
.status-failed { background: #fef2f2; color: #b91c1c; border-color: #fecaca; }
.status-warning { background: #fffbeb; color: #b45309; border-color: #fde68a; }
.status-running { background: var(--accent-soft); color: var(--accent-strong); border-color: #c7d2fe; }
.status-idle { background: #f1f5f9; color: #475569; border-color: #e2e8f0; }
.status-info { background: #f5f3ff; color: #6d28d9; border-color: #ddd6fe; }

.progress-shell {
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  background: var(--surface-soft);
  padding: 14px;
  margin-bottom: 12px;
}
.progress-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 10px;
  flex-wrap: wrap;
}
.progress-title {
  color: var(--text);
  font-size: 14px;
  font-weight: 800;
}
.progress-track {
  background: #e2e8f0;
  border-radius: 999px;
  height: 10px;
  overflow: hidden;
}
.progress-fill {
  height: 100%;
  background: var(--accent);
}
.progress-fill-running {
  background: linear-gradient(90deg, #818cf8, var(--accent), #818cf8);
  background-size: 200% 100%;
  animation: flow 1.5s linear infinite;
}
.progress-fill-idle { background: #cbd5e1; }
.compact-meta {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 10px 12px !important;
  margin-bottom: 12px;
}
.stage-card {
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 14px !important;
  margin-top: 8px;
}
.stage-live {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  font-weight: 800;
  color: var(--text);
  margin-bottom: 8px;
}
.dot-live {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--success);
  animation: pulseDot 1.3s ease-in-out infinite;
}
.spinner {
  width: 12px;
  height: 12px;
  border: 2px solid #c7d2fe;
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin .9s linear infinite;
}
.status-divider {
  margin-top: 10px;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.timeline-wrap { margin-top: 8px; }
.timeline-card {
  background: #fff !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  padding: 12px !important;
  margin-bottom: 8px;
}
.timeline-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 4px 0;
}
.timeline-dot { width: 8px; height: 8px; border-radius: 50%; }
.timeline-value { color: var(--text); font-size: 13px; font-weight: 650; }

.empty-state {
  border: 1px dashed #cbd5e1;
  border-radius: var(--radius-md);
  background: #f8fafc;
  padding: 14px;
}
.empty-state-title {
  font-size: 14px;
  color: var(--text);
  font-weight: 800;
  margin-bottom: 4px;
}
.empty-state-copy {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.45;
  font-weight: 500;
}

/* Buttons */
.primary-action,
.primary-action button {
  background: var(--accent) !important;
  border-color: var(--accent) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  box-shadow: none !important;
}
.primary-action:hover,
.primary-action button:hover {
  background: var(--accent-strong) !important;
  border-color: var(--accent-strong) !important;
}
.secondary-action,
.secondary-action button {
  background: #ffffff !important;
  border: 1px solid var(--border-strong) !important;
  color: #334155 !important;
  font-weight: 750 !important;
  box-shadow: none !important;
}
.secondary-action:hover,
.secondary-action button:hover {
  background: #f8fafc !important;
  color: var(--text) !important;
}

/* Inputs */
.primary-card input,
.secondary-card input,
.primary-card textarea,
.secondary-card textarea,
.primary-card select,
.secondary-card select {
  border-color: var(--border) !important;
  border-radius: 9px !important;
}

/* Tables */
.resizable-table {
  overflow-x: auto;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  background: #fff !important;
}
.resizable-table table {
  border-collapse: separate !important;
  border-spacing: 0 !important;
}
.resizable-table table th {
  background: #f8fafc !important;
  color: #475569 !important;
  border-bottom: 1px solid var(--border) !important;
  font-weight: 800 !important;
  font-size: 12px !important;
  padding: 11px 12px !important;
}
.resizable-table table td {
  border-bottom: 1px solid #eef2f7 !important;
  padding: 12px !important;
  color: #1e293b !important;
  font-size: 13px !important;
}
.resizable-table table tr:hover td {
  background: #fafcff !important;
}

@keyframes flow { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }
@keyframes pulseDot { 0%, 100% { opacity: .45; transform: scale(0.9); } 50% { opacity: 1; transform: scale(1.05); } }
@keyframes spin { to { transform: rotate(360deg); } }
@media (max-width: 980px) {
  .gradio-container { padding: 18px !important; }
  .label-value-grid { grid-template-columns: 1fr; gap: 3px; }
}
</style>
"""
    )
    gr.HTML(
        """
<div class='app-shell page-header'>
  <h1 class='page-title'>LIDDIA GUI v2</h1>
  <p class='page-subtitle'>Internal research interface for launching runs, monitoring progress, and reviewing optimization results.</p>
</div>
"""
    )

    with gr.Tabs(elem_classes=["app-shell", "app-tabs"]):
        with gr.Tab("Monitor"):
            run_dir_state = gr.State("")
            run_json_state = gr.State("")
            iteration_select_state = gr.State(0)
            mol_index_state = gr.State(0)
            with gr.Row(elem_classes=["monitor-layout"]):
                with gr.Column(scale=1, elem_classes=["monitor-stack"]):
                    with gr.Group(elem_classes=["secondary-card"]):
                        gr.Markdown("<p class='section-title'>Run Setup</p>")
                        gr.Markdown("<p class='helper-text'>Configure target, model, and launch a run.</p>")
                        target = gr.Dropdown(DEFAULT_TARGETS, value="EGFR", label="Target", allow_custom_value=True)
                        max_iter = gr.Number(value=2, precision=0, label="Max iterations")
                        gr.Markdown("<p class='k'>Provider (future)</p>")
                        provider_mock = gr.Dropdown(
                            choices=["Anthropic (current)", "OpenAI (planned)", "Local model (planned)"],
                            value="Anthropic (current)",
                            label="LLM provider",
                            interactive=False,
                        )
                        model = gr.Dropdown(DEFAULT_MODELS, value="claude-opus-4-6", label="Model", allow_custom_value=True)
                        anthropic_api_key = gr.Textbox(label="Anthropic API key", type="password", placeholder="sk-ant-...")
                        provider_note = gr.Markdown(
                            "_Placeholder UI only: provider switching is not enabled yet. Current runs use Anthropic._"
                        )
                        with gr.Row():
                            run_button = gr.Button("Run LIDDIA", variant="primary", elem_classes=["primary-action"])
                            refresh_button = gr.Button("Load latest run", variant="secondary", elem_classes=["secondary-action"])

                with gr.Column(scale=3, elem_classes=["monitor-stack"]):
                    with gr.Group(elem_classes=["primary-card", "monitor-primary"]):
                        gr.Markdown("<p class='section-title'>Live Monitor</p>")
                        gr.Markdown("<p class='helper-text'>Track progress, elapsed time, and current stage.</p>")
                        status = gr.Textbox(label="Run status", interactive=False, visible=False)
                        status_badge = gr.HTML(label="Status badge")
                        progress_html = gr.HTML(label="Run summary")
                        elapsed_html = gr.HTML(label="Elapsed time")
                        monitor_timer = gr.Timer(2.0)
                        live_html = gr.HTML(label="Live status", visible=False)
                        stage_html = gr.HTML(label="Action activity")

                with gr.Column(scale=1, elem_classes=["monitor-stack"]):
                    with gr.Group(elem_classes=["secondary-card"]):
                        gr.Markdown("<p class='section-title'>Metrics Snapshot</p>")
                        metrics_df = gr.Dataframe(label="Final pool metrics", interactive=False, elem_classes=["resizable-table"])
                    with gr.Group(elem_classes=["secondary-card"]):
                        gr.Markdown("<p class='section-title'>Run Overview</p>")
                        monitor_overview = gr.Markdown()

        with gr.Tab("Results"):
            gr.HTML(
                """
<style>
.resizable-table { overflow-x: auto; }
.resizable-table table { table-layout: fixed; width: 100%; }
.resizable-table th, .resizable-table td { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.resizable-table th { position: relative; overflow: visible; }
/* Keep index and molecule columns compact so property columns stay visible */
.resizable-table table th:nth-child(1),
.resizable-table table td:nth-child(1) { width: 60px; }
.resizable-table table th:nth-child(2),
.resizable-table table td:nth-child(2) {
  width: 260px !important;
  min-width: 260px !important;
  max-width: 260px !important;
  overflow: visible !important;
  pointer-events: auto !important;
  position: relative;
  z-index: 1;
}
.resizable-table table td:nth-child(2) {
  font-size: 0; /* prevent long markdown/data-uri text from expanding column */
  line-height: 0;
}
.resizable-table table td:nth-child(2) img {
  width: 240px !important;
  max-width: 240px !important;
  height: auto;
  display: block;
  margin: 0 auto;
}
.resizable-table .mol-thumb {
  transition: transform 0.18s ease, box-shadow 0.18s ease;
  transform-origin: center center;
  cursor: zoom-in;
  border-radius: 6px;
}
.resizable-table .mol-thumb:hover {
  transform: scale(1.9);
  position: relative;
  z-index: 10000;
  box-shadow: 0 12px 24px rgba(15, 23, 42, 0.22);
  background: #fff;
}
.resizable-table .mol-thumb-link {
  display: block;
  pointer-events: auto !important;
  position: relative;
  z-index: 1;
}
.resizable-table .mol-thumb-link:hover {
  z-index: 10000;
}
/* Dynamic metric tooltips (attached by header name, not column index) */
.resizable-table table th[data-metric-tooltip] {
  cursor: help;
  position: relative;
  overflow: visible;
}
.resizable-table table th[data-metric-tooltip]::after {
  content: attr(data-metric-tooltip);
  position: absolute;
  left: 0;
  top: calc(100% + 6px);
  z-index: 20000;
  background: #111827;
  color: #ffffff;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
  line-height: 1.3;
  white-space: nowrap;
  opacity: 0;
  transform: translateY(-2px);
  transition: opacity 0.12s ease, transform 0.12s ease;
  pointer-events: none;
}
.resizable-table table th[data-metric-tooltip]:hover::after {
  opacity: 1;
  transform: translateY(0);
}
/* Fallback tooltip behavior for molecule properties table:
   first two columns are Index/Molecule, remaining columns are metrics. */
.mol-prop-table table th:nth-child(n+3) {
  cursor: help;
  position: relative;
  overflow: visible;
}
.mol-prop-table table th:nth-child(n+3)::after {
  content: "Definition placeholder.";
  position: absolute;
  left: 0;
  top: calc(100% + 6px);
  z-index: 20000;
  background: #111827;
  color: #ffffff;
  border-radius: 6px;
  padding: 6px 8px;
  font-size: 12px;
  line-height: 1.3;
  white-space: nowrap;
  opacity: 0;
  transform: translateY(-2px);
  transition: opacity 0.12s ease, transform 0.12s ease;
  pointer-events: none;
}
.mol-prop-table table th:nth-child(n+3):hover::after {
  opacity: 1;
  transform: translateY(0);
}
/* Keep report download output compact when empty */
#report-file-output,
#report-file-output .wrap,
#report-file-output .file-preview,
#report-file-output .empty,
#report-file-output .file-drop {
  min-height: 44px !important;
  height: auto !important;
}
.results-actions {
  gap: 8px !important;
}
.results-actions button,
.results-actions .gr-button {
  font-weight: 700 !important;
  border: 1px solid #cbd5e1 !important;
  color: #334155 !important;
  background: #ffffff !important;
}
</style>
"""
            )
            with gr.Column():
                with gr.Row(elem_classes=["results-layout"]):
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["secondary-card"]):
                            gr.Markdown("<p class='section-title'>Run Summary</p>")
                            overview = gr.Markdown()
                            runtime_md = gr.Markdown(visible=False)
                            results_md = gr.Markdown(visible=False)
                            report_file = gr.DownloadButton("Download reports", value=None, variant="secondary", size="sm", elem_classes=["secondary-action"])

                        with gr.Group(elem_classes=["secondary-card"]):
                            gr.Markdown("<p class='section-title'>Load Previous Run</p>")
                            run_selector = gr.Dropdown(choices=_get_available_runs(), label="Select run folder", value=None)
                            load_selected_button = gr.Button("Load selected run", variant="secondary", elem_classes=["secondary-action"])

                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["primary-card"]):
                            gr.Markdown("<p class='section-title'>Molecule Viewer (2D)</p>")
                            gr.Markdown("<p class='helper-text'>Molecule tables and property results appear after a run is loaded or completed.</p>")
                            pool_select = gr.Dropdown(label="Pool", choices=[], value=None)
                            pool_badge = gr.HTML()
                            with gr.Row(elem_classes=["results-actions"]):
                                download_current = gr.DownloadButton("Download current pool", variant="secondary", size="sm", elem_classes=["secondary-action"])
                                download_all = gr.DownloadButton("Download all molecule property sets", variant="secondary", size="sm", elem_classes=["secondary-action"])
                            mol_table = gr.Dataframe(interactive=True, elem_classes=["resizable-table", "mol-prop-table"])
                            smiles_text = gr.Textbox(label="SMILES", interactive=False, visible=False)
                            mol_svg = gr.HTML(label="2D structure", visible=False)


        with gr.Tab("3D Viewer"):
            gr.HTML(
                """
<style>
#viewer3d_ligand, #viewer3d_receptor {
  min-height: 48px !important;
}
#viewer3d_ligand .file-preview, #viewer3d_receptor .file-preview,
#viewer3d_ligand .empty, #viewer3d_receptor .empty {
  min-height: 40px !important;
  height: 40px !important;
  padding-top: 2px !important;
  padding-bottom: 2px !important;
}
</style>
"""
            )
            with gr.Row():
                with gr.Column(scale=1):
                    viewer3d_style = gr.State("stick")
                    viewer3d_receptor_style = gr.State("surface")
                    viewer3d_file = gr.File(
                        label="Ligand file (.pdb, .sdf, .mol2, .pdbqt)",
                        file_types=[".pdb", ".sdf", ".mol2", ".pdbqt"],
                        elem_id="viewer3d_ligand",
                    )
                    viewer3d_receptor = gr.File(
                        label="Pocket/Receptor file (.pdb, .pdbqt, .mol2)",
                        file_types=[".pdb", ".pdbqt", ".mol2"],
                        elem_id="viewer3d_receptor",
                    )
                    viewer3d_render = gr.Button("Render 3D")
                    with gr.Accordion("Details", open=False):
                        viewer3d_source = gr.Dropdown(
                            choices=["Upload structure file", "Current selection (from Results)"],
                            value="Upload structure file",
                            label="Source",
                        )
                        viewer3d_pose = gr.Number(value=1, precision=0, label="Pose number (for pdbqt)")
                        viewer3d_color = gr.Dropdown(
                            choices=["spectrum", "greenCarbon", "cyanCarbon", "orangeCarbon", "magentaCarbon", "whiteCarbon"],
                            value="spectrum",
                            label="Ligand color",
                        )
                        viewer3d_receptor_color = gr.Dropdown(
                            choices=["#3b82f6", "#94a3b8", "greenCarbon", "cyanCarbon", "orangeCarbon", "magentaCarbon", "whiteCarbon"],
                            value="#3b82f6",
                            label="Pocket color",
                        )
                        viewer3d_receptor_opacity = gr.Slider(
                            minimum=0.05,
                            maximum=1.0,
                            step=0.05,
                            value=0.80,
                            label="Pocket opacity",
                        )
                        viewer3d_status = gr.Textbox(label="Viewer status", interactive=False)
                with gr.Column(scale=2):
                    viewer3d_badge = gr.HTML(
                        label="Pose",
                        value="<div style='display:inline-flex;align-items:center;padding:6px 10px;border:1px solid #d1d5db;border-radius:999px;background:#f8fafc;font-weight:700;color:#0f172a;'>No structure loaded</div>",
                    )
                    viewer3d_html = gr.HTML(label="3D structure")
                    with gr.Row():
                        viewer3d_prev = gr.Button("◀ Previous pose", size="sm", variant="secondary")
                        viewer3d_next = gr.Button("Next pose ▶", size="sm", variant="secondary")

        with gr.Tab("Trends"):
            with gr.Row():
                with gr.Column(scale=1):
                    metric_trends_state = gr.State(pd.DataFrame())
                    metric_select = gr.Dropdown(label="Metric", choices=["All"], value="All", allow_custom_value=True)
                    metric_trends = gr.Plot(label="Metric trends (median)")
                with gr.Column(scale=1):
                    steps_df = gr.Dataframe(label="Iteration rollup", interactive=False, elem_classes=["resizable-table"])

        with gr.Tab("Help"):
            gr.Markdown(
                "### GUI Guide (Placeholder)\n"
                "- **Monitor:** Start runs, watch action/iteration progress, and inspect live status.\n"
                "- **Results:** Browse pool outputs, molecule properties, and export reports.\n"
                "- **Trends:** Inspect metric behavior across iterations.\n"
                "\n"
                "### Workflow Notes (Placeholder)\n"
                "- Runs write artifacts to `log/<run_id>/`.\n"
                "- `*.json` stores run metadata and iteration history.\n"
                "- `*_memory.pkl` stores molecule pools and dataframes for viewer/tables.\n"
                "\n"
                "### Common Questions (Placeholder)\n"
                "- Why is a run not updating yet?\n"
                "- How do I load a previous run?\n"
                "- What does each action type mean (`GENERATE`, `OPTIMIZE`, `CODE`)?\n"
            )
            metric_info_df = pd.DataFrame(
                [
                    {"metric": "Diversity", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                    {"metric": "SAScore", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                    {"metric": "Lipinski", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                    {"metric": "Novelty", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                    {"metric": "Vina Score", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                    {"metric": "QED", "definition": "Definition placeholder.", "interpretation": "Interpretation placeholder."},
                ]
            )
            gr.Dataframe(
                value=metric_info_df,
                label="Metric Definitions (Placeholder)",
                interactive=False,
                wrap=True,
            )
            gr.Markdown(
                "### Example Run Narrative (Placeholder)\n"
                "1. `GENERATE` creates an initial molecule pool from target pocket context.\n"
                "2. `OPTIMIZE` or `CODE` refines pools against constraints.\n"
                "3. Goal evaluation checks if **all molecules in selected pool** satisfy the requirements.\n"
            )

    run_button.click(
        fn=reset_molecule_viewer_state,
        inputs=[],
        outputs=[pool_select, iteration_select_state, mol_index_state, pool_badge, mol_table, smiles_text, mol_svg],
        queue=False,
    )

    run_evt = run_button.click(
        fn=run_liddia,
        inputs=[target, max_iter, model, anthropic_api_key],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, monitor_overview, steps_df, metric_trends, stage_html, live_html, run_dir_state, run_json_state, status_badge],
    )
    run_evt.then(
        fn=_metric_choices,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_select],
    )
    run_evt.then(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    run_evt.then(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[pool_select, iteration_select_state],
    )
    run_evt.then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
    )
    run_evt.then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_table],
        show_progress="hidden",
    )
    run_evt.then(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[smiles_text, mol_svg],
        show_progress="hidden",
    )
    run_evt.then(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )


    refresh_evt = refresh_button.click(
        fn=load_latest_run,
        inputs=[],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, monitor_overview, steps_df, metric_trends, stage_html, live_html, run_dir_state, run_json_state, status_badge],
    )
    refresh_evt.then(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    refresh_evt.then(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[pool_select, iteration_select_state],
    )
    refresh_evt.then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
    )
    refresh_evt.then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_table],
    )
    refresh_evt.then(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )

    monitor_tick = monitor_timer.tick(
        fn=refresh_monitor_outputs,
        inputs=[run_dir_state, run_json_state],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, monitor_overview, steps_df, metric_trends, stage_html, live_html, run_dir_state, run_json_state, status_badge],
        queue=False,
        show_progress="hidden",
    )
    monitor_tick.then(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
        show_progress="hidden",
    )
    monitor_tick.then(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[pool_select, iteration_select_state],
        show_progress="hidden",
    ).then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
        show_progress="hidden",
    ).then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_table],
        show_progress="hidden",
    ).then(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[smiles_text, mol_svg],
        show_progress="hidden",
    )
    monitor_timer.tick(
        fn=build_elapsed_html,
        inputs=[run_dir_state, run_json_state],
        outputs=[elapsed_html],
        queue=False,
        show_progress="hidden",
    )

    load_evt = load_selected_button.click(
        fn=load_selected_run,
        inputs=[run_selector],
        outputs=[status, overview, progress_html, runtime_md, results_md, metrics_df, monitor_overview, steps_df, metric_trends, stage_html, live_html, run_dir_state, run_json_state, status_badge],
    )
    load_evt.then(
        fn=_update_metric_trends,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends],
    )
    load_evt.then(
        fn=update_pool_selector,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[pool_select, iteration_select_state],
    )
    load_evt.then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
    )
    load_evt.then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_table],
        show_progress="hidden",
    )
    load_evt.then(
        fn=_update_molecule_view,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[smiles_text, mol_svg],
        show_progress="hidden",
    )
    load_evt.then(
        fn=update_metric_controls,
        inputs=[run_dir_state, run_json_state],
        outputs=[metric_trends_state, metric_trends, metric_select],
    )

    pool_select.change(
        fn=set_iteration_from_pool,
        inputs=[run_dir_state, run_json_state, pool_select],
        outputs=[iteration_select_state],
        show_progress="hidden",
    ).then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
        show_progress="hidden",
    ).then(
        fn=_update_molecule_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_table],
        show_progress="hidden",
    )
    mol_table.select(
        fn=select_molecule_from_table,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[mol_index_state, smiles_text, mol_svg],
        show_progress="hidden",
    ).then(
        fn=build_pool_badge,
        inputs=[run_dir_state, run_json_state, iteration_select_state, mol_index_state],
        outputs=[pool_badge],
        show_progress="hidden",
    )
    download_current.click(
        fn=download_current_pool_csv,
        inputs=[run_dir_state, run_json_state, iteration_select_state],
        outputs=[download_current],
    )
    download_all.click(
        fn=download_all_pools_csv,
        inputs=[run_dir_state, run_json_state],
        outputs=[download_all],
    )
    metric_select.change(
        fn=apply_metric_filter,
        inputs=[metric_trends_state, metric_select],
        outputs=[metric_trends],
    )

    report_file.click(
        fn=build_report_bundle_file,
        inputs=[run_dir_state, run_json_state],
        outputs=[report_file],
    )
    viewer3d_render.click(
        fn=render_3d_viewer,
        inputs=[
            viewer3d_source,
            run_dir_state,
            run_json_state,
            iteration_select_state,
            mol_index_state,
            viewer3d_file,
            viewer3d_style,
            viewer3d_color,
            viewer3d_pose,
            viewer3d_receptor,
            viewer3d_receptor_style,
            viewer3d_receptor_color,
            viewer3d_receptor_opacity,
        ],
        outputs=[viewer3d_status, viewer3d_html, viewer3d_badge],
    )
    prev_evt = viewer3d_prev.click(
        fn=_shift_pose_index,
        inputs=[viewer3d_file, viewer3d_pose, gr.State(-1)],
        outputs=[viewer3d_pose],
        queue=False,
    )
    prev_evt.then(
        fn=render_3d_viewer,
        inputs=[
            viewer3d_source,
            run_dir_state,
            run_json_state,
            iteration_select_state,
            mol_index_state,
            viewer3d_file,
            viewer3d_style,
            viewer3d_color,
            viewer3d_pose,
            viewer3d_receptor,
            viewer3d_receptor_style,
            viewer3d_receptor_color,
            viewer3d_receptor_opacity,
        ],
        outputs=[viewer3d_status, viewer3d_html, viewer3d_badge],
    )
    next_evt = viewer3d_next.click(
        fn=_shift_pose_index,
        inputs=[viewer3d_file, viewer3d_pose, gr.State(1)],
        outputs=[viewer3d_pose],
        queue=False,
    )
    next_evt.then(
        fn=render_3d_viewer,
        inputs=[
            viewer3d_source,
            run_dir_state,
            run_json_state,
            iteration_select_state,
            mol_index_state,
            viewer3d_file,
            viewer3d_style,
            viewer3d_color,
            viewer3d_pose,
            viewer3d_receptor,
            viewer3d_receptor_style,
            viewer3d_receptor_color,
            viewer3d_receptor_opacity,
        ],
        outputs=[viewer3d_status, viewer3d_html, viewer3d_badge],
    )
    viewer3d_pose.change(
        fn=render_3d_viewer,
        inputs=[
            viewer3d_source,
            run_dir_state,
            run_json_state,
            iteration_select_state,
            mol_index_state,
            viewer3d_file,
            viewer3d_style,
            viewer3d_color,
            viewer3d_pose,
            viewer3d_receptor,
            viewer3d_receptor_style,
            viewer3d_receptor_color,
            viewer3d_receptor_opacity,
        ],
        outputs=[viewer3d_status, viewer3d_html, viewer3d_badge],
    )


if __name__ == "__main__":
    demo.queue()
    demo.launch(inbrowser=True, theme=gr.themes.Soft())
