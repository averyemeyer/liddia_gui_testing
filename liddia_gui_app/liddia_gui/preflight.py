"""Pre-run system checks for friendlier setup and launch failures."""
from __future__ import annotations

import html
import importlib.util
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .backend import LiddiaV2Backend
from .config import LOG_ROOT, PDB_DIR, RUN_PY


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    detail: str
    action: str = ""
    blocking: bool = False


def _module_available(module_name: str, *, python_executable: str | None = None) -> bool:
    if python_executable:
        try:
            result = subprocess.run(
                [python_executable, "-c", f"import {module_name}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=8,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False
    return importlib.util.find_spec(module_name) is not None


def _ok(name: str, detail: str) -> PreflightCheck:
    return PreflightCheck(name=name, status="ok", detail=detail)


def _warn(name: str, detail: str, action: str = "") -> PreflightCheck:
    return PreflightCheck(name=name, status="warn", detail=detail, action=action)


def _fail(name: str, detail: str, action: str = "") -> PreflightCheck:
    return PreflightCheck(name=name, status="fail", detail=detail, action=action, blocking=True)


def run_preflight(
    *,
    target: str | None = None,
    api_key: str = "",
    module_checker: Callable[..., bool] = _module_available,
) -> list[PreflightCheck]:
    """Return pre-run checks without mutating LIDDIA state."""
    checks: list[PreflightCheck] = []
    run_python = LiddiaV2Backend.resolve_python_executable()

    checks.append(_ok("GUI Python", sys.executable))
    checks.append(_ok("Run Python", run_python))

    checks.append(_ok("run.py", str(RUN_PY)) if RUN_PY.exists() else _fail("run.py", "Could not find run.py.", "Launch from a complete LIDDIA checkout."))

    if module_checker("gradio"):
        checks.append(_ok("Gradio", "Available in the GUI Python environment."))
    else:
        checks.append(_fail("Gradio", "The GUI Python cannot import Gradio.", "Install Gradio in the GUI environment."))

    if module_checker("fire", python_executable=run_python):
        checks.append(_ok("Fire", "Available in the run Python environment."))
    else:
        checks.append(_fail("Fire", "The run Python cannot import Fire.", "Activate the LIDDIA environment or install `fire`."))

    if module_checker("rdkit", python_executable=run_python):
        checks.append(_ok("RDKit", "Available in the run Python environment."))
    else:
        checks.append(_fail("RDKit", "The run Python cannot import RDKit.", "Install RDKit, usually with conda-forge."))

    if module_checker("vina", python_executable=run_python):
        checks.append(_ok("Vina", "Available in the run Python environment."))
    else:
        checks.append(_warn("Vina", "The run Python cannot import Vina.", "Docking scores may fail until Vina is installed."))

    if module_checker("MolKit", python_executable=run_python):
        checks.append(_ok("MolKit", "Available for AutoDockTools receptor preparation."))
    else:
        checks.append(_warn("MolKit", "The run Python cannot import MolKit.", "Vina receptor preparation may fail for docking runs."))

    selected_target = (target or "").strip()
    if selected_target:
        target_path = PDB_DIR / f"{selected_target}.pdb"
        if target_path.exists():
            checks.append(_ok("Target PDB", str(target_path)))
        else:
            checks.append(_fail("Target PDB", f"Missing {target_path}.", "Choose a target present in dataset/pdb or add the target PDB."))
    elif PDB_DIR.exists():
        checks.append(_warn("Target PDB", "No target selected yet."))
    else:
        checks.append(_fail("Target PDB directory", f"Missing {PDB_DIR}.", "Use a complete LIDDIA checkout with dataset/pdb."))

    if api_key.strip() or os.environ.get("ANTHROPIC_API_KEY"):
        checks.append(_ok("Anthropic API key", "Configured for this run or shell."))
    else:
        checks.append(_fail("Anthropic API key", "No key entered and ANTHROPIC_API_KEY is not set.", "Enter a key before launching a run."))

    try:
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        probe = LOG_ROOT / ".preflight_write_test"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        checks.append(_ok("Log directory", f"Writable: {LOG_ROOT}"))
    except Exception as exc:
        checks.append(_fail("Log directory", f"Cannot write to {LOG_ROOT}: {exc}", "Fix folder permissions before launching."))

    return checks


def preflight_can_start(checks: list[PreflightCheck]) -> bool:
    return not any(check.blocking for check in checks)


def preflight_html(checks: list[PreflightCheck]) -> str:
    if not checks:
        return "<div class='empty-panel'>Run a system check before launching.</div>"
    rows = []
    for check in checks:
        label = {"ok": "OK", "warn": "WARN", "fail": "FIX"}.get(check.status, check.status.upper())
        css = {"ok": "status-success", "warn": "status-info", "fail": "status-failed"}.get(check.status, "status-idle")
        action = f"<div class='preflight-action'>{html.escape(check.action)}</div>" if check.action else ""
        rows.append(
            "<div class='preflight-row'>"
            f"<span class='status-badge {css}'>{label}</span>"
            "<div>"
            f"<strong>{html.escape(check.name)}</strong>"
            f"<p>{html.escape(check.detail)}</p>"
            f"{action}"
            "</div>"
            "</div>"
        )
    return "<div class='preflight-panel'>" + "".join(rows) + "</div>"
