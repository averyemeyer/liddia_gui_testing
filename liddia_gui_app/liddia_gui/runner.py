"""Run launcher and recovery-oriented refresh logic."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .backend import BACKENDS, LiddiaBackend, RunConfig, RunSnapshot
from .config import LOG_ROOT, REPO_ROOT
from .logs import classify_log_text, tail_text
from .run_state import (
    ActiveRun,
    clear_lock,
    pid_running,
    read_last_run,
    read_lock,
    stale_lock,
    write_lock,
    write_last_run,
    write_run_state,
)
from .task_context import merge_task_context, task_context


def _active_placeholder_data(active: ActiveRun, *, run_json: Path | None = None) -> dict[str, Any]:
    """Minimal run data used before/while JSON is temporarily unavailable."""
    runtime: dict[str, Any] = {
        "current_iter": 0,
        "max_iter": active.max_iter,
        "updated_at": datetime.now().isoformat(),
    }
    if active.started_at:
        runtime["start_time"] = datetime.fromtimestamp(active.started_at).isoformat()
        runtime["elapsed_seconds"] = max(0.0, time.time() - float(active.started_at))
    task = task_context(active.target, active.max_iter)
    return {
        "model": active.model,
        "runtime": runtime,
        "task": task,
        "_placeholder": True,
        "_run_json_path": str(run_json) if run_json else None,
    }


def _snapshot_with_active_context(snap: RunSnapshot, active: ActiveRun) -> RunSnapshot:
    if snap.data is None:
        return RunSnapshot(snap.run_dir, snap.run_json, _active_placeholder_data(active, run_json=snap.run_json))
    return RunSnapshot(snap.run_dir, snap.run_json, merge_task_context(snap.data, active.target, active.max_iter))


def _early_failure_message(active: ActiveRun) -> str:
    stderr = tail_text(active.stderr_log, max_chars=6000)
    findings = classify_log_text(stderr)
    if findings:
        titles = ", ".join(finding["title"] for finding in findings)
        return f"Run failed before artifacts were created: {titles}."
    return "Run failed before artifacts were created. Open Errors and logs for details."


def _snapshot_has_terminal_state(snap: RunSnapshot) -> bool:
    if not snap.data:
        return False
    runtime = snap.data.get("runtime") or {}
    return snap.data.get("success") is not None and bool(runtime.get("end_time"))


def _terminal_status(snap: RunSnapshot) -> str:
    return "failed" if snap.data and snap.data.get("success") is False else "completed"


def _write_terminal_state(active: ActiveRun, snap: RunSnapshot, status: str) -> None:
    write_run_state(
        snap.run_dir,
        status=status,
        pid=active.pid,
        started_at=active.started_at,
        finished_at=time.time(),
        target=active.target,
        model=active.model,
        max_iter=active.max_iter,
        run_json_path=snap.run_json,
        stdout_log=active.stdout_log,
        stderr_log=active.stderr_log,
        backend_name=active.backend_name,
    )
    write_last_run(snap.run_dir, snap.run_json)


def notify_desktop(title: str, message: str) -> None:
    """Best-effort local notification; safe to fail silently."""
    title = title.replace('"', "'")
    message = message.replace('"', "'")
    try:
        if sys.platform == "darwin":
            subprocess.run(
                ["osascript", "-e", f'display notification "{message}" with title "{title}"'],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        elif os.name == "nt":
            ps = "Add-Type -AssemblyName PresentationFramework; " f"[System.Windows.MessageBox]::Show('{message}', '{title}') | Out-Null"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.run(["notify-send", title, message], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def launch_run(config: RunConfig, api_key: str, backend: LiddiaBackend | None = None) -> tuple[str, RunSnapshot]:
    """Launch a detached LIDDIA run and persist enough state to recover it.

    Returns a human status message and the first available snapshot, which may
    have no artifacts yet if run.py has not created the run folder.
    """
    backend = backend or BACKENDS["liddia_v2"]
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    active = read_lock(LOG_ROOT)
    if active and stale_lock(active, LOG_ROOT):
        clear_lock(LOG_ROOT)
        active = None
    if active and pid_running(active.pid):
        return "Run already in progress. Wait or add a Cancel button.", RunSnapshot(None, None, None)

    if not api_key.strip():
        return "Missing Anthropic API key.", RunSnapshot(None, None, None)

    known_dirs = {p.name for p in backend.log_root.iterdir()} if backend.log_root.exists() else set()
    started_at = time.time()
    write_lock(ActiveRun(pid=None, started_at=started_at, known_dirs=sorted(known_dirs), backend_name=backend.name), LOG_ROOT)

    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = api_key.strip()
    stdout_log = LOG_ROOT / f".run_{int(started_at)}.stdout.log"
    stderr_log = LOG_ROOT / f".run_{int(started_at)}.stderr.log"

    popen_kwargs = dict(
        cwd=str(REPO_ROOT),
        env=env,
        stdout=open(stdout_log, "a"),
        stderr=open(stderr_log, "a"),
        text=True,
        bufsize=1,
    )
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS  # type: ignore[attr-defined]
    else:
        popen_kwargs["start_new_session"] = True

    try:
        proc = subprocess.Popen(backend.build_command(config), **popen_kwargs)
    finally:
        popen_kwargs["stdout"].close()
        popen_kwargs["stderr"].close()

    active = ActiveRun(
        pid=proc.pid,
        started_at=started_at,
        known_dirs=sorted(known_dirs),
        backend_name=backend.name,
        target=config.target,
        model=config.model,
        max_iter=config.max_iter,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
    )
    run_dir = backend.detect_run_dir(started_at, known_dirs)
    if run_dir:
        active.active_run_dir = str(run_dir)
    write_lock(active, LOG_ROOT)

    time.sleep(0.4)
    if proc.poll() is not None:
        snap = RunSnapshot(None, None, _active_placeholder_data(active))
        clear_lock(LOG_ROOT)
        return _early_failure_message(active), snap

    snap = _snapshot_with_active_context(backend.load_snapshot(run_dir), active)
    write_run_state(
        run_dir,
        status="running",
        pid=proc.pid,
        started_at=started_at,
        target=config.target,
        model=config.model,
        max_iter=config.max_iter,
        run_json_path=snap.run_json,
        stdout_log=active.stdout_log,
        stderr_log=active.stderr_log,
        backend_name=backend.name,
    )
    return "Run started. Waiting for artifacts..." if not snap.run_json else "Run in progress...", snap


def recover_active_run() -> tuple[str, RunSnapshot]:
    """Recover the active run from disk, or return an empty snapshot."""
    active = read_lock(LOG_ROOT)
    if not active:
        run_dir, run_json = read_last_run(LOG_ROOT)
        if run_dir or run_json:
            backend = BACKENDS["liddia_v2"]
            snap = backend.load_snapshot(run_dir, run_json)
            if snap.data:
                status = "failed" if snap.data.get("success") is False else "finished"
                return f"Last run {status}.", snap
        return "No active run.", RunSnapshot(None, None, None)

    backend = BACKENDS.get(active.backend_name, BACKENDS["liddia_v2"])
    run_dir = Path(active.active_run_dir) if active.active_run_dir else None
    if run_dir:
        snap = _snapshot_with_active_context(backend.load_snapshot(run_dir), active)
        if _snapshot_has_terminal_state(snap):
            status = _terminal_status(snap)
            _write_terminal_state(active, snap, status)
            clear_lock(LOG_ROOT)
            if status == "failed":
                notify_desktop("LIDDIA Run Failed", f"{run_dir.name} failed.")
                return "Run failed. Open Errors and logs for details.", snap
            notify_desktop("LIDDIA Run Finished", f"{run_dir.name} completed or stopped.")
            return "Run finished.", snap

    if stale_lock(active, LOG_ROOT):
        snap = _snapshot_with_active_context(backend.load_snapshot(run_dir), active)
        failed_before_artifacts = run_dir is None and snap.run_json is None
        failed_json = _terminal_status(snap) == "failed" and bool(snap.data and snap.data.get("error_message"))
        failed = failed_before_artifacts or failed_json
        write_run_state(
            run_dir,
            status="failed" if failed else "completed",
            pid=active.pid,
            started_at=active.started_at,
            finished_at=time.time(),
            target=active.target,
            model=active.model,
            max_iter=active.max_iter,
            run_json_path=snap.run_json,
            stdout_log=active.stdout_log,
            stderr_log=active.stderr_log,
            backend_name=backend.name,
        )
        if failed_before_artifacts:
            notify_desktop("LIDDIA Run Failed", "Run stopped before creating artifacts.")
        elif failed_json:
            notify_desktop("LIDDIA Run Failed", f"{run_dir.name if run_dir else 'Run'} failed.")
        else:
            notify_desktop("LIDDIA Run Finished", f"{run_dir.name if run_dir else 'Run'} completed or stopped.")
        clear_lock(LOG_ROOT)
        if failed_before_artifacts:
            return _early_failure_message(active), snap
        if failed_json:
            return "Run failed. Open Errors and logs for details.", snap
        return "Run finished.", snap

    run_dir = Path(active.active_run_dir) if active.active_run_dir else None
    if not run_dir:
        run_dir = backend.detect_run_dir(active.started_at, set(active.known_dirs or []))
        if run_dir:
            active.active_run_dir = str(run_dir)
            write_lock(active, LOG_ROOT)
    snap = _snapshot_with_active_context(backend.load_snapshot(run_dir), active)
    write_run_state(
        run_dir,
        status="running",
        pid=active.pid,
        started_at=active.started_at,
        target=active.target,
        model=active.model,
        max_iter=active.max_iter,
        run_json_path=snap.run_json,
        stdout_log=active.stdout_log,
        stderr_log=active.stderr_log,
        backend_name=backend.name,
    )
    return "Run in progress...", snap
