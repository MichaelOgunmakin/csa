from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import plotly.graph_objects as go

from csa.post_experiment_analysis._helpers import (
    _to_pandas_df,
    _run_comparisons,
    _build_table_fig,
    DISPLAY_COLS,
    ResultsSummary,
)

import pandas as pd


@dataclass(frozen=True)
class SegmentedResultsSummary:
    segments: Dict[str, ResultsSummary]

    def show(self, title: Optional[str] = None) -> None:
        for seg_value, result in self.segments.items():
            seg_title = f"{title} — {seg_value}" if title else f"Segment: {seg_value}"
            result.show(title=seg_title)

    def summary_plot(
        self,
        kind: str = "interval",
        metric: str = "abs_lift",
        prac_sig: Optional[float] = None,
    ) -> None:
        for seg, result in self.segments.items():
            fig = result.summary_plot(kind=kind, metric=metric, prac_sig=prac_sig)
            fig.update_layout(title=dict(text=f"Segment: {seg} — {fig.layout.title.text}", x=0.01))
            fig.show()


def experiment_summary_by_segment(
    df,
    treatment: str,
    kpi: str,
    segment: str,
    alpha: float = 0.05,
    spark_max_rows: Optional[int] = None,
) -> SegmentedResultsSummary:

    d = _to_pandas_df(df, [treatment, kpi, segment], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    segments: Dict[str, ResultsSummary] = {}

    for seg_value, g in d.groupby(segment, dropna=False):
        display_rows, detail_rows = _run_comparisons(g, treatment, kpi, alpha)
        if not display_rows:
            continue

        display_df = pd.DataFrame(display_rows, columns=DISPLAY_COLS)
        detail_df = pd.DataFrame(detail_rows)
        table_fig = _build_table_fig(display_df, f"Segment: {seg_value}")

        segments[str(seg_value)] = ResultsSummary(
            df=display_df, detail=detail_df, table_fig=table_fig, alpha=alpha,
        )

    return SegmentedResultsSummary(segments=segments)
