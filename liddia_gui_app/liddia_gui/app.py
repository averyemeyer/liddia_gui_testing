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
import pandas as pd

from .backend import RunConfig
from .config import DEFAULT_MODELS, detect_targets
from .dashboard import RESULTS_EMPTY_HTML, DashboardRender, MonitorRender
from .io_utils import latest_json_in_dir, run_dir_choices, safe_read_json
from .molecules import (
    download_all_pools_csv,
    download_current_pool_csv,
    molecule_table,
    selected_pool_badge,
)
from .preflight import preflight_can_start, preflight_html, run_preflight
from .reports import build_report_bundle_file
from .runner import launch_run, recover_active_run
from .run_state import clear_last_run, pid_running, read_lock
from .trends import apply_metric_filter
from .ui_components import help_panel, recovery_card
from .viewer3d import render_uploaded_structure, shift_pose_index


def choose_server_port(host: str = "127.0.0.1", preferred: int = 7960) -> int:
    """Pick an available local port, honoring GRADIO_SERVER_PORT when set."""
    env_port = os.environ.get("GRADIO_SERVER_PORT")
    if env_port:
        return int(env_port)
    preferred = int(os.environ.get("LIDDIA_GUI_PREFERRED_PORT", preferred))

    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise OSError(f"No free local ports found in {preferred}-{preferred + 99}.")


def render_snapshot(message: str, run_dir: Path | None, run_json: Path | None, data: dict[str, Any] | None) -> DashboardRender:
    return DashboardRender.from_snapshot(message, run_dir, run_json, data)


def render_monitor_snapshot(
    message: str,
    run_dir: Path | None,
    run_json: Path | None,
    data: dict[str, Any] | None,
    *,
    include_logs: bool = False,
) -> MonitorRender:
    return MonitorRender.from_snapshot(message, run_dir, run_json, data, include_logs=include_logs)


def monitor_outputs(render: MonitorRender) -> tuple[Any, ...]:
    active = read_lock()
    return (
        render.status_text,
        render.status_html,
        render.progress_html,
        render.elapsed_html,
        render.monitor_overview_html,
        render.timeline_html,
        render.failure_summary_html,
        render.monitor_metrics_df,
        render.errors_html,
        render.log_diagnostics,
        render.logs_text,
        recovery_card(active, run_dir=render.run_dir_state, run_json=render.run_json_state, is_running=pid_running(active.pid) if active else False),
        render.run_dir_state,
        render.run_json_state,
    )


def review_outputs(render: DashboardRender) -> tuple[Any, ...]:
    return (
        render.results_overview_html,
        render.results_empty_html,
        render.metrics_df,
        render.requirements_df,
        render.raw_json,
        render.pool_select,
        render.pool_badge,
        render.mol_table,
        render.trend_state,
        render.trend_plot_component,
        render.trend_metric_select,
        render.trend_df,
        render.run_dir_state,
        render.run_json_state,
    )


def start_run(target: str, max_iter: int, model: str, api_key: str):
    checks = run_preflight(target=target, api_key=api_key)
    if not preflight_can_start(checks):
        return monitor_outputs(render_monitor_snapshot("System check failed. Fix blocking items before launching.", None, None, None, include_logs=True))
    msg, snap = launch_run(RunConfig(target=target, max_iter=int(max_iter), model=model), api_key)
    return monitor_outputs(render_monitor_snapshot(msg, snap.run_dir, snap.run_json, snap.data, include_logs=True))


def refresh_active_run():
    msg, snap = recover_active_run()
    if snap.run_dir or snap.run_json or snap.data:
        return monitor_outputs(render_monitor_snapshot(msg, snap.run_dir, snap.run_json, snap.data))
    return monitor_outputs(render_monitor_snapshot("No active run.", None, None, None))


def recover_active_run_with_logs():
    msg, snap = recover_active_run()
    if snap.run_dir or snap.run_json or snap.data:
        return monitor_outputs(render_monitor_snapshot(msg, snap.run_dir, snap.run_json, snap.data, include_logs=True))
    return monitor_outputs(render_monitor_snapshot("No active run.", None, None, None, include_logs=True))


def load_selected_run(folder: str):
    from .config import LOG_ROOT
    run_dir = LOG_ROOT / folder if folder else None
    run_json = latest_json_in_dir(run_dir)
    return review_outputs(render_snapshot(f"Loaded run: {folder}" if run_json else "No run selected.", run_dir, run_json, safe_read_json(run_json)))


def refresh_run_choices():
    from .config import LOG_ROOT
    return gr.update(choices=run_dir_choices(LOG_ROOT))


def review_active_run(active_run_dir_str: str, active_run_json_str: str):
    run_json = Path(active_run_json_str) if active_run_json_str else None
    run_dir = Path(active_run_dir_str) if active_run_dir_str else None
    if run_json and run_json.exists():
        return review_outputs(render_snapshot("Reviewing active run.", run_json.parent, run_json, safe_read_json(run_json)))
    if run_dir and run_dir.exists():
        run_json = latest_json_in_dir(run_dir)
        return review_outputs(render_snapshot("Reviewing active run.", run_dir, run_json, safe_read_json(run_json)))
    msg, snap = recover_active_run()
    if snap.run_dir or snap.run_json or snap.data:
        return review_outputs(render_snapshot(msg, snap.run_dir, snap.run_json, snap.data))
    return review_outputs(render_snapshot("No active run to review.", None, None, None))


def update_pool_view(run_dir_str: str, run_json_str: str, pool_id: str | None):
    return selected_pool_badge(pool_id), molecule_table(run_dir_str, run_json_str, pool_id)


def refresh_preflight(target_name: str, api_key_value: str):
    return preflight_html(run_preflight(target=target_name, api_key=api_key_value))


def clear_monitor_state():
    clear_last_run()
    return monitor_outputs(render_monitor_snapshot("Monitor cleared.", None, None, None, include_logs=True))


with gr.Blocks(title="LIDDIA GUI v2") as demo:
    css_path = Path(__file__).with_name("styles.css")
    gr.HTML(f"<style>{css_path.read_text()}</style>")
    gr.HTML("""
    <div class='app-shell page-header'>
      <h1 class='page-title'>LIDDIA GUI v2</h1>
      <p class='page-subtitle'>Internal research interface for launching runs, monitoring progress, and reviewing optimization results.</p>
    </div>
    """)

    active_run_dir_state = gr.State("")
    active_run_json_state = gr.State("")
    review_run_dir_state = gr.State("")
    review_run_json_state = gr.State("")
    initial_monitor = render_monitor_snapshot("No active run.", None, None, None)

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
                    with gr.Accordion("System Check", open=False):
                        preflight_panel = gr.HTML(preflight_html(run_preflight(target=detect_targets()[0], api_key="")))
                        preflight_btn = gr.Button("Refresh system check", variant="secondary")
                    run_btn = gr.Button("Run LIDDIA", variant="primary")
                    latest_btn = gr.Button("Recover latest run", variant="secondary")
                    clear_monitor_btn = gr.Button("Clear monitor view", variant="secondary")
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Live Monitor</p>")
                    gr.Markdown("<p class='helper-text'>Track progress, elapsed time, current stage, and recent actions.</p>")
                    status_text = gr.Textbox(label="Status", value=initial_monitor.status_text, interactive=False, visible=False)
                    status_html = gr.HTML(initial_monitor.status_html)
                    progress_html = gr.HTML(initial_monitor.progress_html)
                    elapsed_html = gr.HTML(initial_monitor.elapsed_html)
                    timeline_html = gr.HTML(initial_monitor.timeline_html)
                    failure_summary = gr.HTML(initial_monitor.failure_summary_html)
                    with gr.Accordion("Errors and logs", open=False):
                        errors_html = gr.HTML(initial_monitor.errors_html)
                        log_diagnostics = gr.HTML("")
                        logs_text = gr.Textbox(label="CLI stdout/stderr", value="", lines=18, interactive=False)
                    timer = gr.Timer(10.0)
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Metrics Snapshot</p>")
                    monitor_metrics_df = gr.Dataframe(value=initial_monitor.monitor_metrics_df, label="Final pool metrics", interactive=False, wrap=True, show_label=True, datatype="html", column_widths=["55%", "45%"], max_height=300)
                    gr.Markdown("<p class='section-title'>Run Overview</p>")
                    monitor_overview_html = gr.HTML(initial_monitor.monitor_overview_html)
                    gr.Markdown("<p class='section-title'>Run Recovery</p>")
                    recovery_html = gr.HTML(recovery_card(None))

        with gr.Tab("Results"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Run Summary</p>")
                    results_overview_html = gr.HTML()
                    gr.Markdown("<p class='section-title'>Load Previous Run</p>")
                    from .config import LOG_ROOT
                    run_selector = gr.Dropdown(choices=run_dir_choices(LOG_ROOT), label="Run folder")
                    refresh_runs_btn = gr.Button("Refresh run list", variant="secondary")
                    load_btn = gr.Button("Load selected run")
                    review_active_btn = gr.Button("Review active run", variant="secondary")
                    report_file = gr.DownloadButton("Download loaded run report", value=None, variant="secondary")
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Molecule Viewer (2D)</p>")
                    gr.Markdown("<p class='helper-text'>Molecule tables and property results appear after a run is loaded or completed.</p>")
                    results_empty_html = gr.HTML(RESULTS_EMPTY_HTML)
                    pool_select = gr.Dropdown(label="Pool", choices=[], value=None)
                    pool_badge = gr.HTML()
                    with gr.Row(elem_classes=["results-actions"]):
                        download_current = gr.DownloadButton("Pool CSV", variant="secondary")
                        download_all = gr.DownloadButton("All CSV", variant="secondary")
                    mol_table = gr.Dataframe(value=pd.DataFrame(columns=["Index", "Molecule"]), label="Molecule properties", interactive=True, wrap=False, datatype="html", elem_classes=["resizable-table", "mol-prop-table"], max_height=620, pinned_columns=2)
                    raw_json = gr.Code(label="Latest run JSON", language="json", visible=False)
                with gr.Column(scale=1, elem_classes=["secondary-panel"]):
                    gr.Markdown("<p class='section-title'>Final Pool Metrics</p>")
                    metrics_df = gr.Dataframe(value=pd.DataFrame(columns=["Metric", "Min", "Median", "Max"]), label="Final pool metrics", interactive=False, wrap=True, show_label=False, datatype="html", max_height=320)
                    gr.Markdown("<p class='section-title'>Task Requirements</p>")
                    requirements_df = gr.Dataframe(value=pd.DataFrame(columns=["Requirement"]), label="Task requirements", interactive=False, wrap=True, show_label=False, max_height=320)

        with gr.Tab("3D Viewer"):
            with gr.Row(elem_classes=["viewer3d-layout"]):
                with gr.Column(scale=1, min_width=280, elem_classes=["secondary-panel", "viewer3d-controls"]):
                    gr.Markdown("<p class='section-title'>3D Setup</p>")
                    gr.Markdown("<p class='viewer3d-upload-label'>Ligand / pose</p>")
                    ligand_file = gr.File(label="Ligand / pose", show_label=False, file_types=[".pdb", ".sdf", ".mol2", ".pdbqt"], elem_classes=["compact-upload"])
                    gr.Markdown("<p class='viewer3d-upload-label'>Receptor / surface</p>")
                    receptor_file = gr.File(label="Receptor / surface", show_label=False, file_types=[".pdb", ".pdbqt", ".mol2"], elem_classes=["compact-upload"])
                    pose_number = gr.State(1)
                    ligand_style = gr.State("stick")
                    ligand_color = gr.State("spectrum")
                    receptor_style = gr.State("surface")
                    receptor_color = gr.Radio(["blue", "orangeCarbon"], value="blue", label="Receptor surface color")
                    receptor_opacity = gr.Slider(0.05, 1.0, value=0.85, step=0.05, label="Receptor opacity")
                    gr.Markdown("<p class='helper-text'>Upload one or both files, then render. Receptor-only views are supported.</p>")
                    render_3d = gr.Button("Render structure", variant="primary")
                    viewer_status = gr.State("")
                with gr.Column(scale=3, elem_classes=["viewer3d-surface"]):
                    viewer_badge = gr.HTML("<span class='viewer3d-badge'>No structure loaded</span>")
                    viewer_html = gr.HTML("<div class='viewer3d-empty'><strong>No structure rendered</strong><span>Upload a ligand pose, receptor, or both, then click Render structure.</span></div>")
                    with gr.Row(elem_classes=["viewer3d-nav"]):
                        prev_pose = gr.Button("Prev pose", variant="secondary")
                        next_pose = gr.Button("Next pose", variant="secondary")

        with gr.Tab("Trends"):
            with gr.Row(elem_classes=["trends-layout"]):
                with gr.Column(scale=3, elem_classes=["primary-panel"]):
                    gr.Markdown("<p class='section-title'>Metric Trends</p>")
                    gr.Markdown("<p class='helper-text'>Median metric values by iteration for the loaded or active run.</p>")
                    trend_state = gr.State([])
                    trend_metric_select = gr.Dropdown(label="Metric", choices=["All"], value="All", allow_custom_value=False)
                    trend_plot_component = gr.Plot(label="Metric trends")
                with gr.Column(scale=2, elem_classes=["secondary-panel trend-rollup"]):
                    gr.Markdown("<p class='section-title'>Iteration Rollup</p>")
                    trend_df = gr.Dataframe(label="Median metrics by iteration", interactive=False, wrap=True, show_label=False, max_height=520)

        with gr.Tab("Help"):
            with gr.Column(elem_classes=["primary-panel"]):
                gr.Markdown("<p class='section-title'>Metric Help</p>")
                gr.HTML(help_panel())

    monitor_outputs_components = [
        status_text,
        status_html,
        progress_html,
        elapsed_html,
        monitor_overview_html,
        timeline_html,
        failure_summary,
        monitor_metrics_df,
        errors_html,
        log_diagnostics,
        logs_text,
        recovery_html,
        active_run_dir_state,
        active_run_json_state,
    ]
    review_outputs_components = [
        results_overview_html,
        results_empty_html,
        metrics_df,
        requirements_df,
        raw_json,
        pool_select,
        pool_badge,
        mol_table,
        trend_state,
        trend_plot_component,
        trend_metric_select,
        trend_df,
        review_run_dir_state,
        review_run_json_state,
    ]
    run_btn.click(start_run, [target, max_iter, model, api_key], monitor_outputs_components, queue=False, show_progress="hidden")
    preflight_btn.click(refresh_preflight, [target, api_key], [preflight_panel], queue=False, show_progress="hidden")
    latest_btn.click(recover_active_run_with_logs, [], monitor_outputs_components, queue=False, show_progress="hidden")
    clear_monitor_btn.click(clear_monitor_state, [], monitor_outputs_components, queue=False, show_progress="hidden")
    timer.tick(refresh_active_run, [], monitor_outputs_components, queue=False, show_progress="hidden")
    refresh_runs_btn.click(refresh_run_choices, [], [run_selector], queue=False, show_progress="hidden")
    load_btn.click(load_selected_run, [run_selector], review_outputs_components, queue=False, show_progress="hidden")
    review_active_btn.click(review_active_run, [active_run_dir_state, active_run_json_state], review_outputs_components, queue=False, show_progress="hidden")
    pool_select.change(update_pool_view, [review_run_dir_state, review_run_json_state, pool_select], [pool_badge, mol_table], queue=False, show_progress="hidden")
    download_current.click(download_current_pool_csv, [review_run_dir_state, review_run_json_state, pool_select], [download_current], queue=False)
    download_all.click(download_all_pools_csv, [review_run_dir_state, review_run_json_state], [download_all], queue=False)
    report_file.click(build_report_bundle_file, [review_run_dir_state, review_run_json_state], [report_file], queue=False)
    render_inputs = [ligand_file, receptor_file, ligand_style, ligand_color, receptor_style, receptor_color, receptor_opacity, pose_number]
    render_3d.click(render_uploaded_structure, render_inputs, [viewer_status, viewer_html, viewer_badge])
    prev_pose.click(shift_pose_index, [ligand_file, pose_number, gr.State(-1)], [pose_number]).then(render_uploaded_structure, render_inputs, [viewer_status, viewer_html, viewer_badge])
    next_pose.click(shift_pose_index, [ligand_file, pose_number, gr.State(1)], [pose_number]).then(render_uploaded_structure, render_inputs, [viewer_status, viewer_html, viewer_badge])
    trend_metric_select.change(apply_metric_filter, [trend_state, trend_metric_select], [trend_plot_component], show_progress="hidden")


if __name__ == "__main__":
    demo.queue()
    server_name = "127.0.0.1"
    server_port = choose_server_port(server_name)
    print(f"LIDDIA GUI listening at http://{server_name}:{server_port}")
    demo.launch(inbrowser=False, server_name=server_name, server_port=server_port, theme=gr.themes.Soft(), footer_links=[])
