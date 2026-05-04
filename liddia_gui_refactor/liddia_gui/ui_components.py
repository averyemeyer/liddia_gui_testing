"""Small HTML builders. Keep Gradio layout separate from display logic."""
from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Any


def is_completed(parsed: dict[str, Any]) -> bool:
    if parsed.get("cancelled") or parsed.get("error_message"):
        return True
    if parsed.get("success") is not None:
        return True
    return bool((parsed.get("runtime") or {}).get("end_time"))


def status_badge(parsed: dict[str, Any], recovered: bool = False) -> str:
    if not parsed.get("steps") and not (parsed.get("runtime") or {}).get("start_time"):
        label, css = "IDLE", "status-idle"
    elif parsed.get("error_message"):
        label, css = "ERROR", "status-failed"
    elif is_completed(parsed):
        label, css = "COMPLETE", "status-success"
    else:
        label, css = "RUNNING", "status-running"
    extra = "<span class='status-badge status-info'>Recovered active run from disk</span>" if recovered else ""
    updated = html.escape(str((parsed.get("runtime") or {}).get("updated_at", "—")))
    return f"<div class='status-row'><span class='status-badge {css}'>{label}</span><span class='helper-text'>Last update: {updated}</span>{extra}</div>"


def run_overview(parsed: dict[str, Any], run_json: Path | None) -> str:
    task = parsed.get("task") or {}
    runtime = parsed.get("runtime") or {}
    final_pool = parsed.get("final_pool") or {}

    def clean(v: Any) -> str:
        return html.escape(str(v if v not in (None, "") else "—"))

    def dt(v: Any) -> str:
        try:
            return datetime.fromisoformat(str(v)).strftime("%Y-%m-%d %H:%M:%S") if v else "—"
        except Exception:
            return str(v)

    status = "SUCCESS" if parsed.get("success") else "RUNNING"
    if parsed.get("cancelled"):
        status = "CANCELLED"
    if parsed.get("error_message"):
        status = "ERROR"

    rows = [
        ("Status", status),
        ("Target", task.get("target")),
        ("Model", parsed.get("model")),
        ("Budget", task.get("resource")),
        ("Run JSON", run_json.name if run_json else None),
        ("Start", dt(runtime.get("start_time"))),
        ("End", dt(runtime.get("end_time"))),
        ("Pool", final_pool.get("pool")),
        ("Molecules", final_pool.get("size")),
        ("Diversity", final_pool.get("diversity")),
    ]
    body = "".join(f"<div class='k'>{clean(k)}</div><div class='v'>{clean(v)}</div>" for k, v in rows)
    return f"<div class='label-value-grid'>{body}</div>"


def elapsed_panel(parsed: dict[str, Any]) -> str:
    runtime = parsed.get("runtime") or {}
    elapsed = runtime.get("elapsed_seconds")
    if elapsed is None:
        start = runtime.get("start_time")
        end = runtime.get("end_time") or runtime.get("updated_at")
        try:
            if start and end:
                elapsed = (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds()
        except Exception:
            elapsed = None
    if elapsed is None:
        value = "—"
    else:
        total = max(0, int(float(elapsed)))
        value = f"{total // 60}m {total % 60}s"
    return f"<div class='elapsed-panel'><span>Elapsed</span><strong>{html.escape(value)}</strong></div>"


def progress(parsed: dict[str, Any]) -> str:
    runtime = parsed.get("runtime") or {}
    current = runtime.get("current_iter", parsed.get("step_count", 0))
    max_iter = runtime.get("max_iter") or (parsed.get("task") or {}).get("resource") or current or 1
    try:
        percent = min(100, round(float(current) / float(max_iter) * 100))
    except Exception:
        percent = 0
    latest = (parsed.get("steps") or [{}])[-1]
    action = latest.get("action_name") or "waiting"
    return (
        "<div class='progress-shell'>"
        f"<div class='progress-head'><span class='progress-title'>Progress: {current}/{max_iter} ({percent}%)</span><span class='helper-text'>Action: {html.escape(str(action))}</span></div>"
        f"<div class='progress-track'><div class='progress-fill {'progress-fill-running' if not is_completed(parsed) else ''}' style='width:{percent}%;'></div></div>"
        "</div>"
    )


def action_timeline(parsed: dict[str, Any]) -> str:
    steps = parsed.get("steps") or []
    if not steps:
        return "<div class='timeline-stack'><div class='timeline-card'><span class='dot dot-muted'></span><strong>Waiting for first iteration</strong></div></div>"
    cards = []
    for step in steps:
        goal = (step.get("goal_eval") or {}).get("answer") or "—"
        goal_css = "dot-good" if goal == "YES" else "dot-warn" if goal == "NO" else "dot-muted"
        action = html.escape(str(step.get("action_name") or "—"))
        output = html.escape(str(step.get("action_output") or "—"))
        cards.append(
            "<div class='timeline-card'>"
            f"<div class='timeline-title'>Iteration {html.escape(str(step.get('step')))}</div>"
            f"<div class='timeline-row'><span class='dot dot-action'></span><strong>{action}</strong><span class='timeline-output'>{output}</span></div>"
            f"<div class='timeline-row'><span class='dot {goal_css}'></span><strong>Goal check:</strong><span>{html.escape(goal)}</span></div>"
            "</div>"
        )
    if is_completed(parsed):
        cards.append("<div class='timeline-card timeline-done'><span class='dot dot-good'></span><strong>Run completed - ready to review results</strong></div>")
    return "<div class='timeline-stack'>" + "".join(cards) + "</div>"


def error_panel(parsed: dict[str, Any]) -> str:
    error = parsed.get("error_message")
    step_errors = [s for s in parsed.get("steps") or [] if s.get("error_message")]
    if not error and not step_errors:
        return "<div class='empty-panel'>No run errors reported.</div>"
    pieces = []
    if error:
        pieces.append(f"<pre>{html.escape(str(error))}</pre>")
    for step in step_errors:
        pieces.append(f"<pre>Step {html.escape(str(step.get('step')))}: {html.escape(str(step.get('error_message')))}</pre>")
    return "<div class='error-panel'>" + "".join(pieces) + "</div>"


def help_panel() -> str:
    rows = [
        ("Vina Score", "Docking estimate; lower values indicate stronger predicted binding."),
        ("Novelty", "Similarity distance from known target ligands; higher is more novel."),
        ("Diversity", "How structurally spread out the molecule pool is; higher is broader."),
        ("QED", "Drug-likeness score from 0 to 1; higher is generally better."),
        ("SAScore", "Synthetic accessibility estimate; lower is generally easier."),
        ("Lipinski", "Rule-of-five checks passed; higher means more rules followed."),
    ]
    body = "".join(f"<div class='k'>{html.escape(k)}</div><div class='v'>{html.escape(v)}</div>" for k, v in rows)
    return f"<div class='help-panel label-value-grid'>{body}</div>"
