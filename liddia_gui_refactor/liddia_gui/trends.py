"""Metric trend helpers for completed or loaded runs."""
from __future__ import annotations

import pandas as pd


def trend_rows(parsed: dict) -> list[dict]:
    rows: list[dict] = []
    for step in parsed.get("steps") or []:
        for metric, values in ((step.get("pool_stats") or {}).get("metrics") or {}).items():
            if not isinstance(values, dict):
                continue
            value = values.get("median")
            if value in (None, "", "—"):
                continue
            rows.append({"Iteration": step.get("step"), "Metric": metric, "Median": value})
    return rows


def trend_plot(rows: list[dict]):
    if not rows:
        return None
    try:
        import plotly.express as px

        df = pd.DataFrame(rows)
        df["Median"] = pd.to_numeric(df["Median"], errors="coerce")
        fig = px.line(df, x="Iteration", y="Median", color="Metric", markers=True)
        fig.update_layout(height=340, margin=dict(l=12, r=12, t=12, b=12), hovermode="x unified")
        fig.update_xaxes(dtick=1)
        return fig
    except Exception:
        return None

