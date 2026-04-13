from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from csa.post_experiment_analysis._helpers import _to_pandas_df
from csa.post_experiment_analysis.check_srm import check_srm, SRMResult


@dataclass(frozen=True)
class SegmentedSRMResult:
    segments: Dict[str, SRMResult]

    def __repr__(self) -> str:
        total = len(self.segments)
        failed = sum(1 for r in self.segments.values() if r.srm_detected)
        return f"SegmentedSRMResult: {failed}/{total} segments with SRM detected"

    def show(self) -> pd.DataFrame:
        rows = []
        for seg_value, result in self.segments.items():
            for grp in result.observed_counts:
                rows.append({
                    "Segment": seg_value,
                    "Group": grp,
                    "Observed": result.observed_counts[grp],
                    "Expected": round(result.expected_counts[grp], 1),
                    "SRM Detected": "Yes" if result.srm_detected else "No",
                    "p-value": f"{result.p_value:.4g}",
                })
        return pd.DataFrame(rows)


def check_srm_by_segment(
    df,
    treatment: str,
    unit: str,
    segment: str,
    expected_ratio: Dict[str, float],
    alpha: float = 0.05,
    spark_max_rows: Optional[int] = None,
) -> SegmentedSRMResult:

    d = _to_pandas_df(df, [unit, treatment, segment], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    segments: Dict[str, SRMResult] = {}

    for seg_value, g in d.groupby(segment, dropna=False):
        if len(g[treatment].unique()) < 2:
            continue

        result = check_srm(
            df=g,
            treatment=treatment,
            unit=unit,
            expected_ratio=expected_ratio,
            alpha=alpha,
        )
        segments[str(seg_value)] = result

    return SegmentedSRMResult(segments=segments)
