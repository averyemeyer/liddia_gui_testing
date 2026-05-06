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


def build_summary_cards(run_data):
    if not run_data:
        return "No data"

    task = run_data.get("task", {})
    iterations = _extract_iterations(run_data)

    return f"""
    Status: {"SUCCESS" if run_data.get("success") else "FAILED"}
    Target: {task.get("target")}
    Model: {run_data.get("model")}
    Iterations: {len(iterations)}
    """

def build_timeline(run_data):
    steps = _extract_iterations(run_data)
    lines = []

    for idx, payload in steps:
        action = payload.get("action")
        output = payload.get("action_output")

        lines.append(f"Step {idx}: {action} → {output}")

    return "\n".join(lines)


def build_results_summary(run_data):
    steps = _extract_iterations(run_data)
    if not steps:
        return "No results"

    last = steps[-1][1]
    goal = last.get("goal_response", "")

    return goal[:500]  # placeholder until we parse properly





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


def build_task_json(
    target_name: str,
    pocket_path: str,
    drugs_csv: str,
    requirements_text: str,
    metrics_csv: str,
    resource_budget: int,
    model_name: str,
) -> Tuple[str, str]:
    if not target_name or not target_name.strip():
        return "Missing target name.", ""
    if not pocket_path or not pocket_path.strip():
        return "Missing pocket path.", ""

    drugs = [d.strip() for d in drugs_csv.split(",")] if drugs_csv else []
    drugs = [d for d in drugs if d]
    metrics = [m.strip() for m in metrics_csv.split(",")] if metrics_csv else []
    metrics = [m for m in metrics if m]
    requirements = [r.strip() for r in requirements_text.splitlines()] if requirements_text else []
    requirements = [r for r in requirements if r]

    try:
        resource_int = int(resource_budget)
    except Exception:
        resource_int = 0

    task = {
        "target": target_name.strip(),
        "pocket": pocket_path.strip(),
        "drugs": drugs,
        "requirements": requirements,
        "metrics": metrics,
        "resource": resource_int,
        "model": model_name.strip(),
    }

    warnings: List[str] = []
    if not drugs:
        warnings.append("No seed/reference drugs provided.")
    if not metrics:
        warnings.append("No metrics provided.")
    if not requirements:
        warnings.append("No requirements provided.")
    if pocket_path and not Path(pocket_path).exists():
        warnings.append("Pocket path does not exist on this machine.")

    status = "Task JSON built successfully."
    if warnings:
        status += " Warnings: " + " | ".join(warnings)

    return status, json.dumps(task, indent=2)


with gr.Blocks(title="LIDDIA GUI") as demo:
    gr.Markdown("# LIDDIA GUI\n\nThin wrapper around `run.py` for quick local testing.")

    with gr.Tabs():
        with gr.Tab("Run LIDDIA"):
            with gr.Row():
                with gr.Column(scale=1):
                    target = gr.Dropdown(DEFAULT_TARGETS, value="EGFR", label="Target", allow_custom_value=True)
                    max_iter = gr.Number(value=1, precision=0, label="Max iterations")
                    status = gr.Textbox(label="Status")
                    summary_cards = gr.Textbox(label="Run Overview")
                    timeline = gr.Textbox(label="Agent Timeline")
                    results_summary = gr.Textbox(label="Results Summary")
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
                outputs=[status, summary_cards, timeline, results_summary, raw_json, logs],
            )

            refresh_button.click(
                fn=load_latest_run,
                inputs=[],
                outputs=[status, summary, raw_json, logs],
            )

        with gr.Tab("Task Builder"):
            gr.Markdown("Build a LIDDIA task JSON from a guided form.")
            with gr.Row():
                with gr.Column(scale=1):
                    tb_target = gr.Textbox(label="Target name", value="KIT")
                    tb_pocket = gr.Textbox(label="Pocket path", placeholder="/absolute/or/relative/path/to/pocket.pdb")
                    tb_drugs = gr.Textbox(
                        label="Seed/reference drugs (comma-separated)",
                        placeholder="imatinib, sunitinib",
                    )
                    tb_requirements = gr.Textbox(
                        label="Requirements (one per line)",
                        lines=8,
                        value="High novelty\nGood docking score\nDrug-like properties",
                    )
                    tb_metrics = gr.Textbox(
                        label="Metrics (comma-separated)",
                        value="vina, novelty, lipinski, qed, sascore, diversity",
                    )
                    tb_resource = gr.Number(label="Resource budget", value=2, precision=0)
                    tb_model = gr.Dropdown(DEFAULT_MODELS, value="claude-opus-4-6", label="Default model", allow_custom_value=True)
                    tb_build = gr.Button("Build task JSON", variant="primary")

                with gr.Column(scale=2):
                    tb_status = gr.Textbox(label="Builder status", interactive=False)
                    tb_json = gr.Code(label="Generated task JSON", language="json")

            tb_build.click(
                fn=build_task_json,
                inputs=[tb_target, tb_pocket, tb_drugs, tb_requirements, tb_metrics, tb_resource, tb_model],
                outputs=[tb_status, tb_json],
            )

if __name__ == "__main__":
    #demo.launch()
    demo.launch(inbrowser=True)
