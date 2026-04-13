from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from csa.funnel_analysis.funnel_analysis import FunnelResult, funnel_analysis
from csa.funnel_analysis._helpers import _to_pandas_df


@dataclass(frozen=True)
class SegmentedFunnelResult:
    segments: Dict[str, FunnelResult]

    def __repr__(self) -> str:
        return f"SegmentedFunnelResult: {len(self.segments)} segments"

    def show(self, title: Optional[str] = None) -> None:
        for seg_value, result in self.segments.items():
            seg_title = f"{title} — {seg_value}" if title else f"Segment: {seg_value}"
            result.show(title=seg_title)

    def summary_plot(self, kind: str = "funnel", title: Optional[str] = None) -> None:
        for seg_value, result in self.segments.items():
            seg_title = f"{title} — {seg_value}" if title else f"Segment: {seg_value}"
            fig = result.summary_plot(kind=kind)
            fig.update_layout(title=dict(text=seg_title, x=0.01))
            fig.show()


def funnel_analysis_by_segment(
    df,
    stages: List[str],
    unit: str,
    segment: str,
    count_method: str = "distinct",
    conversion: str = "overall",
    display_format: str = "both",
    spark_max_rows: Optional[int] = None,
) -> SegmentedFunnelResult:

    cols = [unit, segment] + stages
    d = _to_pandas_df(df, cols, spark_max_rows)

    segments: Dict[str, FunnelResult] = {}

    for seg_value, g in d.groupby(segment, dropna=False):
        result = funnel_analysis(
            df=g,
            stages=stages,
            unit=unit,
            count_method=count_method,
            conversion=conversion,
            display_format=display_format,
        )
        segments[str(seg_value)] = result

    return SegmentedFunnelResult(segments=segments)
