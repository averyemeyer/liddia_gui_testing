"""Lean Gradio shell using modular backend/run-state code.

This app intentionally shows the pattern for tabs/state/recovery without
re-copying every molecule/table feature from the old flat file. Move existing
Results/3D/Trends blocks over incrementally into modules.
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any

import gradio as gr

from .backend import RunConfig
from .config import DEFAULT_MODELS, detect_targets
from .dashboard import DashboardRender
from .io_utils import available_run_dirs, latest_json_in_dir, safe_read_json
from .molecules import (
    download_all_pools_csv,
    download_current_pool_csv,
    molecule_table,
    selected_pool_badge,
)
from .reports import build_report_bundle_file
from .runner import launch_run, recover_active_run
from .ui_components import help_panel


def choose_server_port(host: str = "127.0.0.1", preferred: int = 7960) -> int:
    """Pick an available local port, honoring GRADIO_SERVER_PORT when set."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)

    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free local ports found in {preferred}-{preferred + 99}.")


def render_snapshot(message: str, run_dir: Path | None, run_json: Path | None, data: dict[str, Any] | None):
    return DashboardRender.from_snapshot(message, run_dir, run_json, data).as_outputs()


def start_run(target: str, max_iter: int, model: str, api_key: str):
    msg, snap = launch_run(RunConfig(target=target, max_iter=int(max_iter), model=model), api_key)
    return render_snapshot(msg, snap.run_dir, snap.run_json, snap.data)


def refresh_active_or_loaded(run_dir_str: str, run_json_str: str):
    msg, snap = recover_active_run()
    if snap.run_dir or snap.run_json or snap.data:
        return render_snapshot(msg, snap.run_dir, snap.run_json, snap.data)

    run_json = Path(run_json_str) if run_json_str else None
    run_dir = Path(run_dir_str) if run_dir_str else None
    if run_json and run_json.exists():
        return render_snapshot("Loaded run.", run_json.parent, run_json, safe_read_json(run_json))
    if run_dir and run_dir.exists():
        run_json = latest_json_in_dir(run_dir)
        return render_snapshot("Loaded run.", run_dir, run_json, safe_read_json(run_json))
    return render_snapshot("No run yet.", None, None, None)


def load_selected_run(folder: str):
    from .config import LOG_ROOT
    run_dir = LOG_ROOT / folder if folder else None
    run_json = latest_json_in_dir(run_dir)
    return render_snapshot(f"Loaded run: {folder}" if run_json else "No run selected.", run_dir, run_json, safe_read_json(run_json))


def update_pool_view(run_dir_str: str, run_json_str: str, pool_id: str | None):
    return selected_pool_badge(pool_id), molecule_table(run_dir_str, run_json_str, pool_id)


with gr.Blocks(title="LIDDIA GUI v2") as demo:
    css_path = Path(__file__).with_name("styles.css")
    gr.HTML(f"<style>{css_path.read_text()}</style>")
    gr.HTML("""
    <div class='app-shell page-header'>
      <h1 class='page-title'>LIDDIA GUI v2</h1>
      <p class='page-subtitle'>Internal research interface for launching runs, monitoring progress, and reviewing optimization results.</p>
    </div>
    """)

    run_dir_state = gr.State("")
    run_json_state = gr.State("")

    with gr.Tabs(elem_classes=["app-shell"]):
        with gr.Tab("Monitor"):
            with gr.Row(elem_classes=["monitor-layout"]):
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Run Setup</p>")
                    gr.Markdown("<p class='helper-text'>Configure target, model, and launch a run.</p>")
                    target = gr.Dropdown(detect_targets(), value=detect_targets()[0], label="Target", allow_custom_value=True)
                    max_iter = gr.Number(value=2, precision=0, label="Max iterations")
                    provider = gr.Dropdown(["Anthropic (current)", "OpenAI (planned)", "Local model (planned)"], value="Anthropic (current)", label="LLM provider", interactive=False)
                    model = gr.Dropdown(DEFAULT_MODELS, value=DEFAULT_MODELS[0], label="Model", allow_custom_value=True)
                    api_key = gr.Textbox(label="Anthropic API key", type="password")
                    run_btn = gr.Button("Run LIDDIA", variant="primary")
                    latest_btn = gr.Button("Load latest / recover", variant="secondary")
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Live Monitor</p>")
                    gr.Markdown("<p class='helper-text'>Track progress, elapsed time, current stage, and recent actions.</p>")
                    status_text = gr.Textbox(label="Status", interactive=False, visible=False)
                    status_html = gr.HTML()
                    progress_html = gr.HTML()
                    elapsed_html = gr.HTML()
                    timeline_html = gr.HTML()
                    with gr.Accordion("Errors and logs", open=False):
                        errors_html = gr.HTML()
                        logs_text = gr.Textbox(label="CLI stdout/stderr", lines=18, interactive=False)
                    timer = gr.Timer(2.0)
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Metrics Snapshot</p>")
                    monitor_metrics_df = gr.Dataframe(label="Final pool metrics", interactive=False, wrap=True, show_label=True, column_widths=["55%", "45%"], max_height=300)
                    gr.Markdown("<p class='section-title'>Run Overview</p>")
                    monitor_overview_html = gr.HTML()

        with gr.Tab("Results"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Run Summary</p>")
                    results_overview_html = gr.HTML()
                    report_file = gr.DownloadButton("Download run report", value=None, variant="secondary")
                    gr.Markdown("<p class='section-title'>Load Previous Run</p>")
                    from .config import LOG_ROOT
                    run_selector = gr.Dropdown(choices=available_run_dirs(LOG_ROOT), label="Run folder")
                    load_btn = gr.Button("Load selected run")
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Molecule Viewer (2D)</p>")
                    gr.Markdown("<p class='helper-text'>Molecule tables and property results appear after a run is loaded or completed.</p>")
                    pool_select = gr.Dropdown(label="Pool", choices=[], value=None)
                    pool_badge = gr.HTML()
                    with gr.Row(elem_classes=["results-actions"]):
                        download_current = gr.DownloadButton("Download current pool", variant="secondary")
                        download_all = gr.DownloadButton("Download all molecule property sets", variant="secondary")
                    mol_table = gr.Dataframe(label="Molecule properties", interactive=True, wrap=False, datatype="html", elem_classes=["resizable-table", "mol-prop-table"], max_height=620, pinned_columns=2)
                    with gr.Accordion("Raw JSON", open=False):
                        raw_json = gr.Code(label="Latest run JSON", language="json")
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Final Pool Metrics</p>")
                    metrics_df = gr.Dataframe(label="Final pool metrics", interactive=False, wrap=True, show_label=False, max_height=320)
                    gr.Markdown("<p class='section-title'>Task Requirements</p>")
                    requirements_df = gr.Dataframe(label="Task requirements", interactive=False, wrap=True, show_label=False, max_height=320)

        with gr.Tab("3D Viewer"):
            with gr.Column(elem_classes=["primary-panel"]):
                gr.Markdown("<p class='section-title'>3D Viewer</p>")
                gr.Markdown("<p class='helper-text'>The modular shell is ready for the previous ligand/receptor viewer. Next pass will move the py3Dmol logic here.</p>")
                gr.File(label="Ligand or pose file", file_types=[".sdf", ".mol", ".mol2", ".pdb", ".pdbqt"])
                gr.File(label="Receptor file", file_types=[".pdb", ".pdbqt"])
                gr.HTML("<div class='empty-panel'>3D rendering module pending migration.</div>")

        with gr.Tab("Trends"):
            with gr.Row():
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Metric Trends</p>")
                    trend_plot_component = gr.Plot(label="Metric trends")
                with gr.Column(scale=2, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Iteration Rollup</p>")
                    trend_df = gr.Dataframe(label="Median metrics by iteration", interactive=False, wrap=True)

        with gr.Tab("Help"):
            with gr.Column(elem_classes=["primary-panel"]):
                gr.Markdown("<p class='section-title'>Metric Help</p>")
                gr.HTML(help_panel())

    dashboard_outputs = [
        status_text,
        status_html,
        progress_html,
        elapsed_html,
        monitor_overview_html,
        timeline_html,
        monitor_metrics_df,
        results_overview_html,
        metrics_df,
        requirements_df,
        errors_html,
        raw_json,
        logs_text,
        pool_select,
        pool_badge,
        mol_table,
        trend_df,
        trend_plot_component,
        run_dir_state,
        run_json_state,
    ]
    run_btn.click(start_run, [target, max_iter, model, api_key], dashboard_outputs)
    latest_btn.click(refresh_active_or_loaded, [run_dir_state, run_json_state], dashboard_outputs)
    timer.tick(refresh_active_or_loaded, [run_dir_state, run_json_state], dashboard_outputs, queue=False, show_progress="hidden")
    load_btn.click(load_selected_run, [run_selector], dashboard_outputs)
    pool_select.change(update_pool_view, [run_dir_state, run_json_state, pool_select], [pool_badge, mol_table])
    download_current.click(download_current_pool_csv, [run_dir_state, run_json_state, pool_select], [download_current])
    download_all.click(download_all_pools_csv, [run_dir_state, run_json_state], [download_all])
    report_file.click(build_report_bundle_file, [run_dir_state, run_json_state], [report_file])


if __name__ == "__main__":
    demo.queue()
    server_name = "127.0.0.1"
    server_port = choose_server_port(server_name)
    print(f"LIDDIA GUI listening at http://{server_name}:{server_port}")
    demo.launch(inbrowser=False, server_name=server_name, server_port=server_port, theme=gr.themes.Soft(), footer_links=[])
