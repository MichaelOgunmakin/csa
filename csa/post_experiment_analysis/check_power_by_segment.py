from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from csa.post_experiment_analysis._helpers import _to_pandas_df
from csa.post_experiment_analysis.check_power import check_power, PowerResult


@dataclass(frozen=True)
class SegmentedPowerResult:
    segments: Dict[str, PowerResult]

    def __repr__(self) -> str:
        total = len(self.segments)
        passed = sum(1 for r in self.segments.values() if r.passed)
        return f"SegmentedPowerResult: {passed}/{total} segments sufficiently powered"

    def show(self) -> pd.DataFrame:
        dfs = []
        for seg_value, result in self.segments.items():
            df = result.show()
            df.insert(0, "Segment", seg_value)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def check_power_by_segment(
    df,
    treatment: str,
    kpi: str,
    unit: str,
    segment: str,
    control_label: str,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "two_sided",
    spark_max_rows: Optional[int] = None,
) -> SegmentedPowerResult:

    d = _to_pandas_df(df, [unit, treatment, kpi, segment], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    segments: Dict[str, PowerResult] = {}

    for seg_value, g in d.groupby(segment, dropna=False):
        if len(g[treatment].unique()) < 2:
            continue

        result = check_power(
            df=g,
            treatment=treatment,
            kpi=kpi,
            unit=unit,
            power=power,
            alpha=alpha,
            test_type=test_type,
            control_label=control_label,
        )
        segments[str(seg_value)] = result

    return SegmentedPowerResult(segments=segments)
