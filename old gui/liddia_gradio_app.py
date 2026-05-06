import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

REPO_ROOT = Path(__file__).resolve().parent
RUN_PY = REPO_ROOT / "run.py"
LOG_ROOT = REPO_ROOT / "log"

# Replace with a dynamic loader later if you want to read available targets from the repo.
DEFAULT_TARGETS = [
    "EGFR",
    "BRAF",
    "JAK2",
    "DHFR",
]

DEFAULT_MODELS = [
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


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


def _extract_iterations(run_data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    items: List[Tuple[str, Dict[str, Any]]] = []
    for key, value in run_data.items():
        if key.isdigit() and isinstance(value, dict):
            items.append((key, value))
    return sorted(items, key=lambda x: int(x[0]))


def _build_summary(run_data: Dict[str, Any], run_json_path: Optional[Path]) -> str:
    lines: List[str] = []
    model = run_data.get("model", "Unknown")
    success = run_data.get("success", False)
    task = run_data.get("task", {}) if isinstance(run_data.get("task", {}), dict) else {}

    lines.append(f"Model: {model}")
    lines.append(f"Success: {success}")
    if run_json_path is not None:
        lines.append(f"Run JSON: {run_json_path}")
    if task:
        lines.append("")
        lines.append("Task")
        lines.append(f"- Target: {task.get('target', 'Unknown')}")
        lines.append(f"- Pocket: {task.get('pocket', 'Unknown')}")
        lines.append(f"- Resource budget: {task.get('resource', 'Unknown')}")
        metrics = task.get("metrics")
        if metrics:
            lines.append(f"- Metrics: {', '.join(metrics)}")

    iterations = _extract_iterations(run_data)
    if iterations:
        lines.append("")
        lines.append("Iterations")
        for idx, payload in iterations:
            action = payload.get("action")
            action_output = payload.get("action_output")
            lines.append(f"- Iteration {idx}: action={action}, output={action_output}")

            goal_response = payload.get("goal_response")
            if isinstance(goal_response, str) and goal_response.strip():
                preview = goal_response.strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:200] + "..."
                lines.append(f"  Goal check: {preview}")

    if run_data.get("error_message"):
        lines.append("")
        lines.append("Error message")
        lines.append(str(run_data["error_message"]))

    return "\n".join(lines)


def run_liddia(
    target: str,
    max_iter: int,
    model: str,
    anthropic_api_key: str,
    skip_docking: bool,
    extra_args: str,
) -> Tuple[str, str, str, str]:
    if not RUN_PY.exists():
        return "run.py not found.", "", "", ""

    if not anthropic_api_key or not anthropic_api_key.strip():
        return "Missing Anthropic API key.", "", "", ""

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

    if skip_docking:
        env["LIDDIA_SKIP_DOCKING"] = "1"

    if extra_args and extra_args.strip():
        cmd.extend(shlex.split(extra_args.strip()))

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=60 * 30,
        )
    except subprocess.TimeoutExpired:
        return "Run timed out after 30 minutes.", "", "", ""
    except Exception as e:
        return f"Failed to start run: {e}", "", "", ""

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    combined = f"--- STDOUT ---\n{stdout}\n\n--- STDERR ---\n{stderr}"

    run_json_path = _latest_run_json()
    run_data = _safe_read_json(run_json_path) if run_json_path else None
    summary = _build_summary(run_data, run_json_path) if run_data else "No run JSON found yet."
    raw_json = json.dumps(run_data, indent=2) if run_data else ""

    if result.returncode == 0:
        status = "Run finished successfully."
    else:
        status = f"Run failed with exit code {result.returncode}."

    return status, summary, raw_json, combined


def load_latest_run() -> Tuple[str, str, str, str]:
    run_json_path = _latest_run_json()
    if not run_json_path:
        return "No runs found.", "", "", ""

    run_data = _safe_read_json(run_json_path)
    if not run_data:
        return f"Could not read {run_json_path.name}.", "", "", ""

    status = f"Loaded latest run: {run_json_path.parent.name}"
    summary = _build_summary(run_data, run_json_path)
    raw_json = json.dumps(run_data, indent=2)
    return status, summary, raw_json, ""


with gr.Blocks(title="LIDDIA GUI") as demo:
    gr.Markdown(
        "# LIDDIA GUI\n\nThin wrapper around `run.py` for quick local testing."
    )

    with gr.Row():
        with gr.Column(scale=1):
            target = gr.Dropdown(DEFAULT_TARGETS, value="EGFR", label="Target", allow_custom_value=True)
            max_iter = gr.Number(value=1, precision=0, label="Max iterations")
            model = gr.Dropdown(DEFAULT_MODELS, value="claude-opus-4-6", label="Model", allow_custom_value=True)
            anthropic_api_key = gr.Textbox(label="Anthropic API key", type="password", placeholder="sk-ant-...")
            skip_docking = gr.Checkbox(value=False, label="Skip docking for prototype runs")
            extra_args = gr.Textbox(
                label="Extra CLI args",
                placeholder="Optional. Example: --env_dir ./env --drug_dir ./drugs",
            )
            with gr.Row():
                run_button = gr.Button("Run LIDDIA", variant="primary")
                refresh_button = gr.Button("Load latest run")

        with gr.Column(scale=2):
            status = gr.Textbox(label="Status", interactive=False)
            summary = gr.Textbox(label="Run summary", lines=16, interactive=False)

    with gr.Tabs():
        with gr.Tab("Raw JSON"):
            raw_json = gr.Code(label="Latest run JSON", language="json")
        with gr.Tab("Logs"):
            logs = gr.Textbox(label="CLI stdout/stderr", lines=28, interactive=False)

    run_button.click(
        fn=run_liddia,
        inputs=[target, max_iter, model, anthropic_api_key, skip_docking, extra_args],
        outputs=[status, summary, raw_json, logs],
    )

    refresh_button.click(
        fn=load_latest_run,
        inputs=[],
        outputs=[status, summary, raw_json, logs],
    )

if __name__ == "__main__":
    demo.launch()
