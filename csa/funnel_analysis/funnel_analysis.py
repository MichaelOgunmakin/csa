from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from csa.funnel_analysis._helpers import _to_pandas_df, _build_table_fig


def _compute_funnel(
    d: pd.DataFrame,
    stages: List[str],
    unit: str,
    count_method: str,
    conversion: str,
    display_format: str,
) -> tuple:
    """Compute funnel counts and conversion rates."""
    counts = []
    for stage in stages:
        if count_method == "distinct":
            n = d.loc[d[stage].astype(float) == 1, unit].nunique()
        else:
            n = int(d[stage].astype(float).sum())
        counts.append(n)

    baseline = counts[0] if counts[0] > 0 else 1

    conversions = []
    for i, n in enumerate(counts):
        if conversion == "overall":
            conv = n / baseline
        else:  # step
            prev = counts[i - 1] if i > 0 else counts[0]
            conv = n / prev if prev > 0 else np.nan
        conversions.append(conv)

    overall_conv = counts[-1] / baseline if baseline > 0 else np.nan

    volume_row = [f"{n:,}" for n in counts] + [f"{counts[-1]:,}"]
    conv_row = [f"{c:.2%}" if np.isfinite(c) else "N/A" for c in conversions] + [f"{overall_conv:.2%}"]

    cols = stages + ["Overall"]
    if display_format == "volume":
        display_df = pd.DataFrame([volume_row], columns=cols, index=["Volume"])
    elif display_format == "percentage":
        display_df = pd.DataFrame([conv_row], columns=cols, index=["Conversion Rate"])
    else:  # both
        display_df = pd.DataFrame([volume_row, conv_row], columns=cols, index=["Volume", "Conversion Rate"])

    detail_df = pd.DataFrame({
        "stage": stages,
        "count": counts,
        "conversion": conversions,
    })

    return display_df, detail_df, counts, overall_conv



@dataclass(frozen=True)
class FunnelResult:
    display_df: pd.DataFrame
    detail_df: pd.DataFrame
    table_fig: go.Figure
    stages: List[str]
    counts: List[int]
    overall_conversion: float
    conversion: str
    display_format: str

    def show(self, title: Optional[str] = None) -> None:
        if title is not None:
            self.table_fig.update_layout(title=dict(text=title, x=0.01))
        self.table_fig.show()

    def summary_plot(self, kind: str = "funnel", title: Optional[str] = None) -> go.Figure:
        if kind == "funnel":
            return self._funnel_plot(title)
        elif kind == "bar":
            return self._bar_plot(title)
        else:
            raise ValueError("kind must be 'funnel' or 'bar'")

    def _funnel_plot(self, title: Optional[str] = None) -> go.Figure:
        first = self.counts[0] if self.counts[0] > 0 else 1
        positions = ["inside" if c / first >= 0.10 else "outside" for c in self.counts]

        pct_mode = "percent initial" if self.conversion == "overall" else "percent previous"
        if self.display_format == "volume":
            textinfo = "value"
        elif self.display_format == "percentage":
            textinfo = pct_mode
        else:  # both
            textinfo = f"value+{pct_mode}"

        fig = go.Figure(go.Funnel(
            y=self.stages,
            x=self.counts,
            textinfo=textinfo,
            textposition=positions,
            textfont=dict(size=13),
            constraintext="none",
            connector=dict(line=dict(width=1)),
        ))

        fig.update_layout(
            title=dict(text=title or "Funnel Analysis", x=0.01),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=150 + len(self.stages) * 120,
            margin=dict(l=150),
        )
        return fig

    def _bar_plot(self, title: Optional[str] = None) -> go.Figure:
        conv_values = [c * 100 for c in self.detail_df["conversion"]]
        remainder_values = [100 - c for c in conv_values]
        overall_pct = self.overall_conversion * 100

        if self.display_format == "volume":
            bar_text = [f"{n:,}" for n in self.counts]
        elif self.display_format == "percentage":
            bar_text = [f"{c:.1f}%" for c in conv_values]
        else:  # both
            bar_text = [f"{n:,} | {c:.1f}%" for n, c in zip(self.counts, conv_values)]

        # filled bars — conversion portion
        filled = go.Bar(
            x=self.stages,
            y=conv_values,
            text=bar_text,
            textposition="inside",
            textfont=dict(color="white", size=12),
            marker_color="#1f77b4",
            name="Converted",
        )

        # hatched bars — drop-off portion
        remainder = go.Bar(
            x=self.stages,
            y=remainder_values,
            marker=dict(
                color="rgba(31, 119, 180, 0.15)",
                pattern=dict(shape="/", fgcolor="#1f77b4", size=6),
            ),
            name="Drop-off",
            hoverinfo="skip",
        )

        fig = go.Figure(data=[filled, remainder])

        fig.update_layout(
            barmode="stack",
            title=dict(text=title or f"Funnel — Overall Conversion: {overall_pct:.1f}%", x=0.01),
            xaxis_title="Stage",
            yaxis=dict(title="Conversion Rate (%)", range=[0, 110], ticksuffix="%"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="left", x=0),
        )
        return fig


def funnel_analysis(
    df,
    stages: List[str],
    unit: str,
    count_method: str = "distinct",
    conversion: str = "overall",
    display_format: str = "both",
    spark_max_rows: Optional[int] = None,
) -> FunnelResult:

    if count_method not in ("distinct", "total"):
        raise ValueError("count_method must be 'distinct' or 'total'.")
    if conversion not in ("overall", "step"):
        raise ValueError("conversion must be 'overall' or 'step'.")
    if display_format not in ("volume", "percentage", "both"):
        raise ValueError("display_format must be 'volume', 'percentage', or 'both'.")

    cols = [unit] + stages
    d = _to_pandas_df(df, cols, spark_max_rows)

    display_df, detail_df, counts, overall_conv = _compute_funnel(
        d, stages, unit, count_method, conversion, display_format,
    )

    table_fig = _build_table_fig(display_df, "Funnel Analysis", include_index=True)

    return FunnelResult(
        display_df=display_df,
        detail_df=detail_df,
        table_fig=table_fig,
        stages=stages,
        counts=counts,
        overall_conversion=overall_conv,
        conversion=conversion,
        display_format=display_format,
    )
