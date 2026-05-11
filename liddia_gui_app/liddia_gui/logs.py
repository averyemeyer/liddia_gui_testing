"""Log and diagnostic helpers for the GUI.

The runner writes detached stdout/stderr files beside the normal run folders.
These helpers keep the UI resilient when a log is missing, still being written,
or referenced by a recovered lock file.
"""
from __future__ import annotations

import html
from pathlib import Path

from .config import LOG_ROOT
from .run_state import read_lock


def tail_text(path: Path | str | None, max_chars: int = 20000) -> str:
    """Return the end of a text file without raising UI-breaking errors."""
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return ""
    try:
        text = file_path.read_text(errors="replace")
    except Exception as exc:
        return f"Could not read {file_path}: {exc}"
    return text[-max_chars:]


def active_log_text(log_root: Path = LOG_ROOT) -> str:
    """Return combined stdout/stderr for the active or most recent GUI run."""
    active = read_lock(log_root)
    stdout = active.stdout_log if active else None
    stderr = active.stderr_log if active else None

    if not stdout and not stderr and log_root.exists():
        stdout_logs = sorted(log_root.glob(".run_*.stdout.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        stderr_logs = sorted(log_root.glob(".run_*.stderr.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        stdout = str(stdout_logs[0]) if stdout_logs else None
        stderr = str(stderr_logs[0]) if stderr_logs else None

    out = tail_text(stdout)
    err = tail_text(stderr)
    if not out and not err:
        return "No CLI logs found yet."
    return f"--- STDOUT ---\n{out or '(empty)'}\n\n--- STDERR ---\n{err or '(empty)'}"


def classify_log_text(text: str) -> list[dict[str, str]]:
    """Return user-facing diagnostics for common runtime log patterns."""
    if not text or text == "No CLI logs found yet.":
        return []
    lower = text.lower()
    findings: list[dict[str, str]] = []

    def add(title: str, detail: str, action: str) -> None:
        findings.append({"title": title, "detail": detail, "action": action})

    if "no module named 'fire'" in lower or 'no module named "fire"' in lower:
        add(
            "Missing Fire dependency",
            "LIDDIA could not import the Fire CLI package.",
            "Activate the LIDDIA environment or install it with `python -m pip install fire`.",
        )
    if "rdkit unavailable" in lower or "no module named 'rdkit'" in lower or 'no module named "rdkit"' in lower:
        add(
            "Missing RDKit dependency",
            "Molecule thumbnails or chemistry parsing cannot run without RDKit.",
            "Use the LIDDIA environment or install RDKit, usually with `conda install -c conda-forge rdkit`.",
        )
    if "no module named 'molkit'" in lower or 'no module named "molkit"' in lower:
        add(
            "Missing MolKit dependency",
            "AutoDockTools receptor preparation could not import MolKit.",
            "Fix the AutoDockTools/MGLTools installation before relying on Vina docking scores.",
        )
    if "protein.pdbqt does not exist" in lower:
        add(
            "Docking receptor was not prepared",
            "Vina expected `protein.pdbqt`, but receptor preparation did not create it.",
            "This often follows the MolKit error. Treat Vina scores from this run as failed or incomplete.",
        )
    if "anthropic api key not found" in lower or "missing anthropic api key" in lower:
        add(
            "Missing Anthropic API key",
            "LIDDIA could not find an API key for the model provider.",
            "Enter the key in Run Setup or set `ANTHROPIC_API_KEY` before launching.",
        )
    if "authentication" in lower and ("anthropic" in lower or "api" in lower):
        add(
            "Model provider authentication failed",
            "The model provider rejected the configured credentials.",
            "Check the API key and make sure it is valid for the selected provider/model.",
        )
    if "rate limit" in lower or "rate_limit" in lower:
        add(
            "Model provider rate limit",
            "The model provider is throttling requests.",
            "Wait and retry, reduce run frequency, or use a provider/model with more quota.",
        )
    if "could not parse action/input" in lower:
        add(
            "Model response parsing failed",
            "The model did not return an action/input pair LIDDIA could parse.",
            "Try a different model or inspect the raw response in the run JSON/logs.",
        )
    if "get_metadata_from_response" in lower and "invalid syntax" in lower:
        add(
            "Model response format parsing failed",
            "The model returned action/input text with extra formatting that LIDDIA could not parse.",
            "Retry with a model that follows the required plain `Action:` and `Input:` format, or make the core parser more tolerant.",
        )
    if "get_goal_answer_response" in lower and "indexerror: list index out of range" in lower:
        add(
            "Goal-check response parsing failed",
            "The model did not return the expected plain `Answer:` line for the goal check.",
            "Retry with a model that follows the required evaluator format, or make the core goal-check parser more tolerant.",
        )
    if "bad file descriptor" in lower:
        add(
            "Subprocess stream warning",
            "A child process reported a standard-stream file descriptor issue.",
            "If the run completed, this may be secondary noise. If docking failed too, fix docking dependencies first.",
        )
    return findings


def failure_summary_html(data: dict | None, log_text: str = "") -> str:
    """Render the most actionable failure summary from run JSON plus logs."""
    text_parts = [log_text or ""]
    if data:
        error = data.get("error_message")
        if error:
            text_parts.append(str(error))
    findings = classify_log_text("\n".join(text_parts))
    if not findings and data and data.get("success") is False:
        findings = [
            {
                "title": "Run failed",
                "detail": "The run JSON reports success=false, but the failure did not match a known diagnostic pattern.",
                "action": "Open Errors and logs, then inspect the traceback or run JSON.",
            }
        ]
    if not findings:
        return "<div class='empty-panel'>No run failure detected.</div>"

    primary = findings[0]
    extra = ""
    if len(findings) > 1:
        extra = "<ul>" + "".join(f"<li>{html.escape(f['title'])}</li>" for f in findings[1:]) + "</ul>"
    return (
        "<div class='failure-summary'>"
        f"<div class='status-row'><span class='status-badge status-failed'>ATTENTION</span><strong>{html.escape(primary['title'])}</strong></div>"
        f"<p>{html.escape(primary['detail'])}</p>"
        f"<code>{html.escape(primary['action'])}</code>"
        f"{extra}"
        "</div>"
    )


def log_diagnostics_html(text: str) -> str:
    findings = classify_log_text(text)
    if not findings:
        return "<div class='empty-panel'>No recognized runtime issues in the current logs.</div>"
    cards = []
    for finding in findings:
        cards.append(
            "<div class='diagnostic-item'>"
            f"<strong>{html.escape(finding['title'])}</strong>"
            f"<p>{html.escape(finding['detail'])}</p>"
            f"<code>{html.escape(finding['action'])}</code>"
            "</div>"
        )
    return "<div class='diagnostic-panel'>" + "".join(cards) + "</div>"
