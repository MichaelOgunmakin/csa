from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from csa.funnel_analysis._helpers import _to_pandas_df, _build_table_fig


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
_TIME_DIVISORS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}


def _time_delta(
    d: pd.DataFrame,
    ts_a: str,
    ts_b: str,
    time_unit: str,
    aggregation: str,
) -> float:
    """Compute aggregated time delta between two timestamp columns for completed transitions."""
    mask = d[ts_a].notna() & d[ts_b].notna()
    subset = d[mask]
    if len(subset) == 0:
        return np.nan
    delta = pd.to_datetime(subset[ts_b]) - pd.to_datetime(subset[ts_a])
    values = delta.dt.total_seconds() / _TIME_DIVISORS[time_unit]
    values = values[values >= 0]  # exclude negative deltas
    if len(values) == 0:
        return np.nan
    return values.median() if aggregation == "median" else values.mean()


def _compute_ttc(
    d: pd.DataFrame,
    stages: List[str],
    timestamps: Dict[str, str],
    time_unit: str,
    aggregation: str,
    conversion: str,
) -> Dict[str, float]:
    """Return a dict of {column_label: aggregated_time} for one group."""
    results = {}

    if conversion in ("step", "both"):
        for i in range(len(stages) - 1):
            label = f"{stages[i]} → {stages[i + 1]}"
            ts_a = timestamps[stages[i]]
            ts_b = timestamps[stages[i + 1]]
            results[label] = _time_delta(d, ts_a, ts_b, time_unit, aggregation)

    if conversion in ("overall", "both"):
        ts_first = timestamps[stages[0]]
        for i in range(1, len(stages)):
            label = f"{stages[0]} → {stages[i]}"
            ts_b = timestamps[stages[i]]
            results[label] = _time_delta(d, ts_first, ts_b, time_unit, aggregation)

    return results



# ---------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class TimeToConvertResult:
    display_df: pd.DataFrame
    table_fig: go.Figure
    aggregation: str
    time_unit: str
    conversion: str
    has_segment: bool

    def show(self, title: Optional[str] = None) -> None:
        if title is not None:
            self.table_fig.update_layout(title=dict(text=title, x=0.01))
        self.table_fig.show()

    def summary_plot(self, title: Optional[str] = None) -> go.Figure:
        agg_label = self.aggregation.capitalize()
        y_label = f"{agg_label} Time ({self.time_unit})"

        # determine transition columns (exclude Segment if present)
        value_cols = [c for c in self.display_df.columns if c != "Segment"]

        if self.has_segment:
            fig = go.Figure()
            for _, row in self.display_df.iterrows():
                seg = row["Segment"]
                y_vals = [row[c] for c in value_cols]
                fig.add_trace(go.Bar(
                    name=str(seg),
                    x=value_cols,
                    y=y_vals,
                    text=[f"{v:.2f}" if pd.notna(v) else "N/A" for v in y_vals],
                    textposition="outside",
                ))
            fig.update_layout(barmode="group")
        else:
            y_vals = [self.display_df[c].iloc[0] for c in value_cols]
            fig = go.Figure(go.Bar(
                x=value_cols,
                y=y_vals,
                text=[f"{v:.2f}" if pd.notna(v) else "N/A" for v in y_vals],
                textposition="outside",
            ))

        fig.update_layout(
            title=dict(text=title or f"{agg_label} Time to Convert ({self.time_unit})", x=0.01),
            xaxis_title="Transition",
            yaxis_title=y_label,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig


# ---------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------
def funnel_time_to_convert(
    df,
    stages: List[str],
    unit: str,
    timestamps: Dict[str, str],
    time_unit: str = "hours",
    aggregation: str = "median",
    conversion: str = "step",
    segment: Optional[str] = None,
    spark_max_rows: Optional[int] = None,
) -> TimeToConvertResult:

    if time_unit not in _TIME_DIVISORS:
        raise ValueError(f"time_unit must be one of {list(_TIME_DIVISORS.keys())}.")
    if aggregation not in ("median", "mean"):
        raise ValueError("aggregation must be 'median' or 'mean'.")
    if conversion not in ("step", "overall", "both"):
        raise ValueError("conversion must be 'consecutive', 'overall', or 'both'.")

    missing = [s for s in stages if s not in timestamps]
    if missing:
        raise ValueError(f"Missing timestamps for stages: {missing}.")

    ts_cols = list(timestamps.values())
    cols = [unit] + stages + ts_cols
    if segment:
        cols.append(segment)
    d = _to_pandas_df(df, cols, spark_max_rows)

    agg_label = f"{aggregation.capitalize()} ({time_unit})"

    if segment:
        rows = []
        for seg_value, g in d.groupby(segment, dropna=False):
            row = {"Segment": str(seg_value)}
            row.update(_compute_ttc(g, stages, timestamps, time_unit, aggregation, conversion))
            rows.append(row)
        display_df = pd.DataFrame(rows)
        display_df.columns = (
            ["Segment"] +
            [f"{c} — {agg_label}" if c != "Segment" else c for c in display_df.columns[1:]]
        )
    else:
        row = _compute_ttc(d, stages, timestamps, time_unit, aggregation, conversion)
        display_df = pd.DataFrame(
            [[f"{v:.2f}" if pd.notna(v) else "N/A" for v in row.values()]],
            columns=[f"{k} — {agg_label}" for k in row.keys()],
            index=[agg_label],
        )

    table_fig = _build_table_fig(
        display_df,
        title=f"Time to Convert — {agg_label}",
    )

    return TimeToConvertResult(
        display_df=display_df,
        table_fig=table_fig,
        aggregation=aggregation,
        time_unit=time_unit,
        conversion=conversion,
        has_segment=segment is not None,
    )
