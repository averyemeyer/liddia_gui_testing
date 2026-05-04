"""Named dashboard render contract for the Gradio app edge."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from .logs import active_log_text
from .molecules import enrich_parsed_with_memory, molecule_table, pool_choices, selected_pool_badge
from .parsers import compact_metric_display_rows, metric_display_rows, parse_run_data, raw_json_text, requirements_rows
from .trends import iteration_rollup, metric_choices, trend_plot, trend_rows
from .ui_components import action_timeline, elapsed_panel, error_panel, progress, run_overview, status_badge


@dataclass(frozen=True)
class MonitorRender:
    status_text: str
    status_html: str
    progress_html: str
    elapsed_html: str
    monitor_overview_html: str
    timeline_html: str
    monitor_metrics_df: pd.DataFrame
    errors_html: str
    logs_text: str
    run_dir_state: str
    run_json_state: str

    @classmethod
    def from_snapshot(
        cls,
        message: str,
        run_dir: Path | None,
        run_json: Path | None,
        data: dict[str, Any] | None,
        *,
        include_logs: bool = False,
    ) -> "MonitorRender":
        recovered = message.lower().startswith("recovered")
        run_dir_text = str(run_dir or "")
        run_json_text = str(run_json or "")
        parsed = parse_run_data(data)
        return cls(
            status_text=message,
            status_html=status_badge(parsed, recovered=recovered),
            progress_html=progress(parsed),
            elapsed_html=elapsed_panel(parsed),
            monitor_overview_html=run_overview(parsed, run_json),
            timeline_html=action_timeline(parsed),
            monitor_metrics_df=pd.DataFrame(compact_metric_display_rows(parsed)),
            errors_html=error_panel(parsed),
            logs_text=active_log_text() if include_logs else gr.skip(),
            run_dir_state=run_dir_text,
            run_json_state=run_json_text,
        )


@dataclass(frozen=True)
class DashboardRender:
    status_text: str
    status_html: str
    progress_html: str
    elapsed_html: str
    monitor_overview_html: str
    timeline_html: str
    monitor_metrics_df: pd.DataFrame
    results_overview_html: str
    metrics_df: pd.DataFrame
    requirements_df: pd.DataFrame
    errors_html: str
    raw_json: str
    logs_text: str
    pool_select: Any
    pool_badge: str
    mol_table: pd.DataFrame
    trend_state: pd.DataFrame
    trend_plot_component: Any
    trend_metric_select: Any
    trend_df: pd.DataFrame
    run_dir_state: str
    run_json_state: str

    @classmethod
    def from_snapshot(cls, message: str, run_dir: Path | None, run_json: Path | None, data: dict[str, Any] | None) -> "DashboardRender":
        recovered = message.lower().startswith("recovered")
        run_dir_text = str(run_dir or "")
        run_json_text = str(run_json or "")
        parsed = enrich_parsed_with_memory(parse_run_data(data), run_dir_text, run_json_text)
        overview = run_overview(parsed, run_json)
        metrics = pd.DataFrame(metric_display_rows(parsed))
        pool_ids, selected_pool = pool_choices(run_dir_text, run_json_text)
        trend_data = trend_rows(parsed)
        choices = metric_choices(trend_data)
        selected_metric = choices[1] if len(choices) > 1 else "All"
        return cls(
            status_text=message,
            status_html=status_badge(parsed, recovered=recovered),
            progress_html=progress(parsed),
            elapsed_html=elapsed_panel(parsed),
            monitor_overview_html=overview,
            timeline_html=action_timeline(parsed),
            monitor_metrics_df=pd.DataFrame(compact_metric_display_rows(parsed)),
            results_overview_html=overview,
            metrics_df=metrics,
            requirements_df=pd.DataFrame(requirements_rows(parsed)),
            errors_html=error_panel(parsed),
            raw_json=raw_json_text(data),
            logs_text=active_log_text(),
            pool_select=gr.update(choices=pool_ids, value=selected_pool),
            pool_badge=selected_pool_badge(selected_pool),
            mol_table=molecule_table(run_dir_text, run_json_text, selected_pool),
            trend_state=pd.DataFrame(trend_data),
            trend_plot_component=trend_plot(trend_data, selected_metric),
            trend_metric_select=gr.update(choices=choices, value=selected_metric),
            trend_df=pd.DataFrame(iteration_rollup(parsed)),
            run_dir_state=run_dir_text,
            run_json_state=run_json_text,
        )

    def as_outputs(self) -> tuple[Any, ...]:
        """Return values in the exact order expected by ``dashboard_outputs``."""
        return (
            self.status_text,
            self.status_html,
            self.progress_html,
            self.elapsed_html,
            self.monitor_overview_html,
            self.timeline_html,
            self.monitor_metrics_df,
            self.results_overview_html,
            self.metrics_df,
            self.requirements_df,
            self.errors_html,
            self.raw_json,
            self.logs_text,
            self.pool_select,
            self.pool_badge,
            self.mol_table,
            self.trend_state,
            self.trend_plot_component,
            self.trend_metric_select,
            self.trend_df,
            self.run_dir_state,
            self.run_json_state,
        )
