from __future__ import annotations

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go


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


def _build_table_fig(display_df: pd.DataFrame, title: str, include_index: bool = False) -> go.Figure:
    """Build a Plotly table figure from a display DataFrame."""
    if include_index:
        header_values = [""] + list(display_df.columns)
        cell_values = [list(display_df.index)] + [display_df[c].tolist() for c in display_df.columns]
    else:
        header_values = list(display_df.columns)
        cell_values = [display_df[c].tolist() for c in display_df.columns]

    fig = go.Figure(
        go.Table(
            header=dict(values=header_values),
            cells=dict(values=cell_values, align="left"),
        )
    )
    height = 60 + 30 * (len(display_df) + 1)
    fig.update_layout(
        title=dict(text=title, x=0.01),
        margin=dict(l=10, r=10, t=40, b=0),
        height=height,
    )
    return fig
