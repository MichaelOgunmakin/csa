from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from csa.post_experiment_analysis._helpers import _to_pandas_df


@dataclass(frozen=True)
class SpilloverResult:
    passed: bool
    spillover_df: pd.DataFrame

    def __repr__(self) -> str:
        if self.passed:
            return "Passed! No spillover"
        return f"Failed! Spillover exists ({len(self.spillover_df)} units)"

    def show(self) -> pd.DataFrame:
        return self.spillover_df


def check_spillover(
    df,
    unit: str,
    treatment: str,
    spark_max_rows: Optional[int] = None,
) -> SpilloverResult:

    d = _to_pandas_df(df, [unit, treatment], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    groups = d.groupby(treatment)[unit].apply(set)

    unit_groups: Dict[Any, List[str]] = {}
    for grp, units in groups.items():
        for u in units:
            unit_groups.setdefault(u, []).append(grp)

    spillover_units = {u: grps for u, grps in unit_groups.items() if len(grps) > 1}

    all_treatments = sorted(groups.index)

    if not spillover_units:
        empty_df = pd.DataFrame(columns=[unit] + all_treatments)
        return SpilloverResult(passed=True, spillover_df=empty_df)

    rows = []
    for u, grps in spillover_units.items():
        row = {unit: u}
        for t in all_treatments:
            row[t] = 1 if t in grps else 0
        rows.append(row)

    spillover_df = pd.DataFrame(rows, columns=[unit] + all_treatments)

    return SpilloverResult(passed=False, spillover_df=spillover_df)
