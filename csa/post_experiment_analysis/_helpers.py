from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from scipy.stats import ttest_ind
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from statsmodels.stats.weightstats import DescrStatsW, CompareMeans


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------
def _to_pandas_df(df, cols: List[str], spark_max_rows: Optional[int] = None) -> pd.DataFrame:
    """Convert pandas or Spark DataFrame to pandas, selecting only needed columns."""
    if isinstance(df, pd.DataFrame):
        return df[cols].copy()
    if hasattr(df, "select") and hasattr(df, "toPandas"):
        sdf = df.select(*cols)
        if spark_max_rows is not None:
            sdf = sdf.limit(int(spark_max_rows))
        return sdf.toPandas()
    raise TypeError("df must be a pandas or PySpark DataFrame")


# ---------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------
def _is_binary(vals: np.ndarray) -> bool:
    return set(np.unique(vals)).issubset({0.0, 1.0})


def _compare_binary(vals_a: np.ndarray, vals_b: np.ndarray, alpha: float) -> Dict[str, Any]:
    n_a, n_b = vals_a.size, vals_b.size
    x_a, x_b = vals_a.sum(), vals_b.sum()
    p_a, p_b = x_a / n_a, x_b / n_b

    _, pval = proportions_ztest([x_a, x_b], [n_a, n_b])
    lo, hi = confint_proportions_2indep(
        count1=x_b, nobs1=n_b, count2=x_a, nobs2=n_a,
        compare="diff", alpha=alpha,
    )
    return dict(
        kpi_a=p_a, kpi_b=p_b,
        abs_lift=p_b - p_a,
        rel_lift=(p_b - p_a) / p_a if p_a else np.nan,
        pval=pval, ci_lo=lo, ci_hi=hi,
        rel_ci_lo=lo / p_a if p_a else np.nan,
        rel_ci_hi=hi / p_a if p_a else np.nan,
    )


def _compare_continuous(vals_a: np.ndarray, vals_b: np.ndarray, alpha: float) -> Dict[str, Any]:
    m_a, m_b = vals_a.mean(), vals_b.mean()

    _, pval = ttest_ind(vals_b, vals_a, equal_var=False)
    lo, hi = CompareMeans(
        DescrStatsW(vals_b), DescrStatsW(vals_a)
    ).tconfint_diff(alpha=alpha)

    return dict(
        kpi_a=m_a, kpi_b=m_b,
        abs_lift=m_b - m_a,
        rel_lift=(m_b - m_a) / m_a if m_a else np.nan,
        pval=pval, ci_lo=lo, ci_hi=hi,
        rel_ci_lo=lo / m_a if m_a else np.nan,
        rel_ci_hi=hi / m_a if m_a else np.nan,
    )


def _run_comparisons(
    d: pd.DataFrame, treatment: str, kpi: str, alpha: float,
) -> tuple:
    """Run pairwise comparisons and return (display_rows, detail_rows)."""
    display_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    levels = sorted(d[treatment].unique())

    for a, b in combinations(levels, 2):
        vals_a = d.loc[d[treatment] == a, kpi].astype(float).to_numpy()
        vals_b = d.loc[d[treatment] == b, kpi].astype(float).to_numpy()
        if vals_a.size == 0 or vals_b.size == 0:
            continue

        binary = _is_binary(np.concatenate([vals_a, vals_b]))
        compare = _compare_binary if binary else _compare_continuous
        r = compare(vals_a, vals_b, alpha)

        comparison = f"{b} vs {a}"

        detail_rows.append(dict(
            comparison=comparison,
            n_a=vals_a.size, n_b=vals_b.size,
            kpi_a=r["kpi_a"], kpi_b=r["kpi_b"],
            abs_lift=r["abs_lift"], rel_lift=r["rel_lift"],
            pval=r["pval"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
            rel_ci_lo=r["rel_ci_lo"], rel_ci_hi=r["rel_ci_hi"],
        ))

        display_rows.append({
            "Comparison": comparison,
            "Sample Sizes": f"{a}: {vals_a.size:,}<br>{b}: {vals_b.size:,}",
            "KPI": f"{a}: {r['kpi_a']:.4f}<br>{b}: {r['kpi_b']:.4f}",
            "Absolute lift": f"{r['abs_lift']:.4f}",
            "Relative lift": f"{r['rel_lift']:.2%}" if np.isfinite(r["rel_lift"]) else "",
            "p-value": f"{r['pval']:.4g}",
            "CI lift": f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]",
        })

    return display_rows, detail_rows


# ---------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------
DISPLAY_COLS = [
    "Comparison", "Sample Sizes", "KPI",
    "Absolute lift", "Relative lift", "p-value", "CI lift",
]


def _build_table_fig(display_df: pd.DataFrame, title: str) -> go.Figure:
    """Build a Plotly table figure from a display DataFrame."""
    fig = go.Figure(
        go.Table(
            header=dict(values=DISPLAY_COLS),
            cells=dict(values=[display_df[c].tolist() for c in DISPLAY_COLS], align="left"),
        )
    )
    height = 60 + 30 * (len(display_df) + 1)
    fig.update_layout(
        title=dict(text=title, x=0.01),
        margin=dict(l=10, r=10, t=40, b=0),
        height=height,
    )
    return fig


# ---------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ResultsSummary:
    df: pd.DataFrame
    detail: pd.DataFrame
    table_fig: go.Figure
    alpha: float

    def table(self) -> go.Figure:
        return self.table_fig

    def show(self, title: Optional[str] = None) -> None:
        if title is not None:
            self.table_fig.update_layout(title=dict(text=title, x=0.01))
        self.table_fig.show()

    def summary_plot(
        self,
        kind: str = "interval",
        metric: str = "abs_lift",
        prac_sig: Optional[float] = None,
        title: Optional[str] = None,
    ) -> go.Figure:
        plot_df = self.detail.sort_values(metric).copy()

        if kind == "bar":
            is_rel = metric == "rel_lift"
            if is_rel:
                plot_df["rel_lift"] = plot_df["rel_lift"] * 100
            metric_label = "Relative lift (%)" if is_rel else "Absolute lift"
            fig = px.bar(
                plot_df, x=metric, y="comparison", orientation="h",
                title=title or f"{metric_label} by comparison",
            )
            fig.add_vline(x=0)
            if prac_sig is not None:
                ps = prac_sig * 100 if is_rel else prac_sig
                fig.add_vline(x=ps, line_dash="dash", line_color="black")
                fig.add_vline(x=-ps, line_dash="dash", line_color="black")
            if is_rel:
                fig.update_xaxes(ticksuffix="%")
            return fig

        is_rel = metric == "rel_lift"
        if is_rel:
            x = plot_df["rel_lift"].to_numpy(float) * 100
            lo = plot_df["rel_ci_lo"].to_numpy(float) * 100
            hi = plot_df["rel_ci_hi"].to_numpy(float) * 100
        else:
            x = plot_df["abs_lift"].to_numpy(float)
            lo = plot_df["ci_lo"].to_numpy(float)
            hi = plot_df["ci_hi"].to_numpy(float)

        fig = go.Figure(
            go.Scatter(
                x=x, y=plot_df["comparison"], mode="markers",
                error_x=dict(type="data", symmetric=False,
                             array=hi - x, arrayminus=x - lo),
                hovertemplate=(
                    "Comparison: %{y}<br>"
                    + ("Rel lift: %{x:.2f}%<br>" if is_rel else "Abs lift: %{x:.4f}<br>")
                    + ("CI: [%{customdata[0]:.2f}%, %{customdata[1]:.2f}%]<br>" if is_rel
                       else "CI: [%{customdata[0]:.4f}, %{customdata[1]:.4f}]<br>")
                    + "p-value: %{customdata[2]:.4g}<extra></extra>"
                ),
                customdata=np.column_stack([lo, hi, plot_df["pval"]]),
            )
        )
        fig.add_vline(x=0)
        if prac_sig is not None:
            ps = prac_sig * 100 if is_rel else prac_sig
            fig.add_vline(x=ps, line_dash="dash", line_color="black")
            fig.add_vline(x=-ps, line_dash="dash", line_color="black")
        metric_label = "Absolute lift" if not is_rel else "Relative lift (%)"
        fig.update_layout(
            title=title or f"{metric_label} with {1 - self.alpha:.0%} CI",
            xaxis_title=metric_label,
            yaxis_title="Comparison",
        )
        if is_rel:
            fig.update_xaxes(ticksuffix="%")
        return fig

    def bar_plot(
        self,
        title: str = "KPI by Treatment Group",
        y_label: str = "KPI",
    ) -> go.Figure:
        detail = self.detail

        groups: Dict[str, float] = {}
        for _, row in detail.iterrows():
            b_name, a_name = row["comparison"].split(" vs ")
            if a_name not in groups:
                groups[a_name] = row["kpi_a"]
            if b_name not in groups:
                groups[b_name] = row["kpi_b"]

        group_names = list(groups.keys())
        kpi_values = list(groups.values())

        fig = go.Figure(go.Bar(
            x=group_names,
            y=kpi_values,
            text=[f"{v:.4f}" for v in kpi_values],
            textposition="outside",
        ))

        max_kpi = max(kpi_values)
        y_upper = max_kpi * 1.2

        fig.update_layout(
            title=dict(text=title, x=0.01),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(title="Treatment Group", linecolor="black", linewidth=1),
            yaxis=dict(title=y_label, range=[0, y_upper], linecolor="black", linewidth=1),
        )
        return fig
