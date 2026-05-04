"""Metric trend helpers for completed or loaded runs."""
from __future__ import annotations

from typing import Any

import pandas as pd


def trend_rows(parsed: dict) -> list[dict]:
    rows: list[dict] = []
    for step in parsed.get("steps") or []:
        pool_stats = step.get("pool_stats") or {}
        if pool_stats.get("diversity") not in (None, "", "—"):
            rows.append({"Iteration": step.get("step"), "Metric": "Diversity", "Median": pool_stats.get("diversity")})
        for metric, values in ((step.get("pool_stats") or {}).get("metrics") or {}).items():
            if not isinstance(values, dict):
                continue
            value = values.get("median")
            if value in (None, "", "—"):
                continue
            rows.append({"Iteration": step.get("step"), "Metric": metric, "Median": value})
    return rows


def metric_choices(rows: list[dict] | pd.DataFrame) -> list[str]:
    df = _to_trend_frame(rows)
    if df.empty or "Metric" not in df.columns:
        return ["All"]
    return ["All"] + sorted(df["Metric"].dropna().astype(str).unique().tolist())


def filter_trend_rows(rows: list[dict] | pd.DataFrame, metric: str | None) -> pd.DataFrame:
    df = _to_trend_frame(rows)
    if df.empty:
        return df
    metric = str(metric or "All")
    if metric != "All":
        df = df[df["Metric"].astype(str) == metric].copy()
    df["Iteration"] = pd.to_numeric(df["Iteration"], errors="coerce")
    df["Median"] = pd.to_numeric(df["Median"], errors="coerce")
    return df.dropna(subset=["Iteration", "Median"]).sort_values(["Metric", "Iteration"])


def trend_plot(rows: list[dict] | pd.DataFrame, metric: str | None = "All"):
    df = filter_trend_rows(rows, metric)
    if df.empty:
        return None
    try:
        import plotly.express as px

        fig = px.line(df, x="Iteration", y="Median", color="Metric", markers=True)
        fig.update_traces(line=dict(width=2), marker=dict(size=7))
        fig.update_layout(height=360, margin=dict(l=12, r=12, t=12, b=12), hovermode="x unified")
        fig.update_xaxes(dtick=1, tickformat="d", title="Iteration")
        fig.update_yaxes(title="Median")
        return fig
    except Exception:
        return None


def iteration_rollup(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_names: list[str] = []
    for step in parsed.get("steps") or []:
        for metric in ((step.get("pool_stats") or {}).get("metrics") or {}).keys():
            if metric not in metric_names:
                metric_names.append(metric)

    for step in parsed.get("steps") or []:
        pool_stats = step.get("pool_stats") or {}
        row: dict[str, Any] = {
            "Iteration": step.get("step"),
            "Action": step.get("action_name") or "—",
            "Pool": step.get("action_output") or pool_stats.get("pool") or "—",
            "Goal": (step.get("goal_eval") or {}).get("answer") or "—",
            "Size": pool_stats.get("size") if pool_stats.get("size") is not None else "—",
            "Diversity": pool_stats.get("diversity") or "—",
        }
        for metric in metric_names:
            values = (pool_stats.get("metrics") or {}).get(metric) or {}
            row[metric] = values.get("median") or values.get("min") or values.get("max") or "—"
        rows.append(row)
    return rows


def apply_metric_filter(rows: list[dict] | pd.DataFrame, metric: str | None):
    return trend_plot(rows, metric)


def _to_trend_frame(rows: list[dict] | pd.DataFrame) -> pd.DataFrame:
    if rows is None:
        return pd.DataFrame(columns=["Iteration", "Metric", "Median"])
    if isinstance(rows, pd.DataFrame):
        return rows.copy()
    return pd.DataFrame(rows, columns=["Iteration", "Metric", "Median"])
