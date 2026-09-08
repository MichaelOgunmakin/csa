from __future__ import annotations

from typing import Optional

import pandas as pd

from csa.post_experiment_analysis._helpers import (
    _to_pandas_df,
    _run_comparisons,
    _build_table_fig,
    DISPLAY_COLS,
    ResultsSummary,
)


def experiment_summary(
    df,
    treatment: str,
    kpi: str,
    alpha: float = 0.05,
    spark_max_rows: Optional[int] = None,
) -> ResultsSummary:

    d = _to_pandas_df(df, [treatment, kpi], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    display_rows, detail_rows = _run_comparisons(d, treatment, kpi, alpha)

    display_df = pd.DataFrame(display_rows, columns=DISPLAY_COLS)
    detail_df = pd.DataFrame(detail_rows)
    table_fig = _build_table_fig(display_df, "Result Summary")

    return ResultsSummary(df=display_df, detail=detail_df, table_fig=table_fig, alpha=alpha)
