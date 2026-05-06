from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from nicegui import ui


REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "log"


def _latest_run_dir() -> Optional[Path]:
    if not LOG_ROOT.exists():
        return None
    runs = [p for p in LOG_ROOT.iterdir() if p.is_dir()]
    if not runs:
        return None
    return sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _load_events(run_dir: Path) -> List[Dict]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    for line in events_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def _latest_status_event(events: List[Dict]) -> Optional[Dict]:
    for ev in reversed(events):
        if ev.get("type") == "status":
            return ev
    return None


def _is_completed(events: List[Dict]) -> bool:
    for ev in reversed(events):
        if ev.get("type") == "status" and ev.get("stage") in {"completed", "cancelled"}:
            return True
    return False


def _group_by_iteration(events: List[Dict]) -> Dict[int, List[Dict]]:
    grouped: Dict[int, List[Dict]] = {}
    for ev in events:
        if ev.get("type") != "status":
            continue
        if ev.get("stage") == "initializing":
            continue
        step = ev.get("step")
        if step is None:
            continue
        try:
            step = int(step)
        except Exception:
            continue
        grouped.setdefault(step, []).append(ev)
    return grouped


@ui.page("/")
def main():
    ui.markdown("# LIDDIA Dashboard (NiceGUI prototype)")
    ui.markdown("A leaner monitoring view + run comparison foundation.")

    run_dir = _latest_run_dir()

    with ui.row().classes("w-full"):
        with ui.column().classes("w-1/4"):
            ui.label("Run controls")
            target = ui.input("Target").props("dense")
            max_iter = ui.number("Max iterations", value=2)
            model = ui.input("Model", value="claude-opus-4-6")
            api_key = ui.input("Anthropic API key", password=True)
            ui.button("Run LIDDiA", color="primary")
            ui.button("Cancel run", color="negative")

        with ui.column().classes("w-2/4"):
            status_label = ui.label("Run in progress...").classes("text-lg font-bold")
            progress = ui.linear_progress(value=0.0).classes("w-full")
            elapsed = ui.label("Elapsed: --")
            live_phase = ui.label("Phase: --")
            live_label = ui.label("Status: --")

            stage_container = ui.column().classes("w-full")

        with ui.column().classes("w-1/4"):
            ui.label("Final pool metrics (placeholder)")
            ui.table(
                columns=[
                    {"name": "metric", "label": "metric", "field": "metric"},
                    {"name": "min", "label": "min", "field": "min"},
                    {"name": "max", "label": "max", "field": "max"},
                    {"name": "median", "label": "median", "field": "median"},
                ],
                rows=[],
            ).classes("w-full")

    def refresh():
        nonlocal run_dir
        run_dir = _latest_run_dir()
        if not run_dir:
            status_label.set_text("No runs found.")
            return

        events = _load_events(run_dir)
        latest = _latest_status_event(events)
        completed = _is_completed(events)
        status_label.set_text("Run completed." if completed else "Run in progress...")

        if latest:
            runtime = latest.get("runtime", {})
            current_iter = runtime.get("current_iter") or 0
            max_iter = runtime.get("max_iter") or 1
            percent = min(1.0, float(current_iter) / float(max_iter))
            progress.value = percent
            elapsed_secs = int(runtime.get("elapsed_seconds") or 0)
            elapsed.set_text(f"Elapsed: {elapsed_secs // 60}m {elapsed_secs % 60}s")
            live_phase.set_text(f"Phase: {latest.get('stage')}")
            live_label.set_text(f"Status: {latest.get('label')}")

        stage_container.clear()
        grouped = _group_by_iteration(events)
        for step, items in grouped.items():
            with stage_container:
                ui.label(f"Iteration {step}").classes("font-bold")
                for ev in items:
                    ui.label(f"- {ev.get('label')}")

    ui.timer(1.0, refresh)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run()
