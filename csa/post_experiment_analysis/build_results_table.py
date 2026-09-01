from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from csa.post_experiment_analysis.experiment_summary import experiment_summary
from csa.post_experiment_analysis.experiment_summary_by_segment import (
    experiment_summary_by_segment,
)
from csa.post_experiment_analysis.check_power import check_power
from csa.post_experiment_analysis.check_power_by_segment import check_power_by_segment


def experiment_results_table(
    df,
    treatment: str,
    segments: List[str],
    kpis: List[str],
    unit: str,
    control_label: str,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "two_sided",
    spark_max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """Return a master results DataFrame across all segment × KPI combinations.

    For each pairwise comparison within every segment value and KPI, one row is
    produced containing the key statistical and power metrics.

    Parameters
    ----------
    df:
        Pandas or PySpark DataFrame containing experiment data.
    treatment:
        Column name identifying treatment assignment.
    segments:
        List of column names to segment the analysis by (one dimension at a time).
    kpis:
        List of column names for the KPI metrics to evaluate.
    unit:
        Column name for the unique experimental unit (e.g. user_id). Used for
        power analysis.
    control_label:
        Value in the ``treatment`` column that identifies the control group.
        Power metrics are only computed for comparisons involving this group.
    alpha:
        Significance level. Default 0.05.
    power:
        Target power threshold used for the sufficiently-powered determination.
        Default 0.80.
    test_type:
        ``"two_sided"`` (default) or ``"one_sided"``.
    spark_max_rows:
        Optional row limit when converting a PySpark DataFrame to pandas.

    Returns
    -------
    pd.DataFrame
        One row per (segment_dimension, segment_value, kpi, pairwise_comparison).

        Columns
        -------
        segment               – segment column name (dimension)
        segment_value         – actual segment value (e.g. "US")
        kpi                   – KPI column name
        group_a               – alphabetically first group (baseline in the comparison)
        group_b               – alphabetically second group (variant in the comparison)
        kpi_a                 – mean KPI for group_a
        kpi_b                 – mean KPI for group_b
        treatment_effect_abs       – absolute lift: kpi_b − kpi_a
        ci_low                – lower bound of the absolute lift CI
        ci_hi                 – upper bound of the absolute lift CI
        treatment_effect_rel       – relative lift: (kpi_b − kpi_a) / kpi_a
        statistically_significant  – True if p_value < alpha
        p_value               – raw p-value
        alpha                 – significance level used
        sufficiently_powered  – True/False; None for non-control comparisons
        target_power_pct      – target power as a percentage (e.g. 80.0)
        achieved_power_pct    – achieved power as a percentage; None for
                                non-control comparisons
        volume_total          – combined count for group_a and group_b
        volume_a              – count for group_a
        volume_b              – count for group_b
    """
    def _extract_rows(result_summary, power_result, segment_label, segment_value_label):
        extracted = []
        for _, row in result_summary.detail.iterrows():
            # comparison format: "{b} vs {a}"
            # where a < b alphabetically; n_a/kpi_a belong to a, n_b/kpi_b to b
            parts = row["comparison"].split(" vs ", 1)
            group_b = parts[0]   # alphabetically second
            group_a = parts[1]   # alphabetically first

            volume_a = int(row["n_a"])
            volume_b = int(row["n_b"])

            # Power metrics — only available for comparisons vs control
            sufficiently_powered = None
            target_power_pct = power * 100
            achieved_power_pct = None

            if power_result is not None and control_label in (group_a, group_b):
                treat_label = group_b if group_a == control_label else group_a
                comparison_key = f"{treat_label} vs {control_label}"

                min_n = power_result.min_samples_per_group.get(comparison_key)
                if min_n is not None:
                    n_ctrl = power_result.actual_samples.get(control_label, 0)
                    n_treat = power_result.actual_samples.get(treat_label, 0)
                    sufficiently_powered = (
                        bool(n_ctrl >= min_n and n_treat >= min_n)
                        if np.isfinite(min_n)
                        else False
                    )

                ap = power_result.achieved_power.get(comparison_key)
                if ap is not None and np.isfinite(ap):
                    achieved_power_pct = round(ap * 100, 2)

            extracted.append({
                "segment": segment_label,
                "segment_value": segment_value_label,
                "kpi": kpi,
                "group_a": group_a,
                "group_b": group_b,
                "kpi_a": row["kpi_a"],
                "kpi_b": row["kpi_b"],
                "treatment_effect_abs": row["abs_lift"],
                "ci_low": row["ci_lo"],
                "ci_hi": row["ci_hi"],
                "treatment_effect_rel": row["rel_lift"],
                "statistically_significant": bool(row["pval"] < alpha),
                "p_value": row["pval"],
                "alpha": alpha,
                "sufficiently_powered": sufficiently_powered,
                "target_power_pct": target_power_pct,
                "achieved_power_pct": achieved_power_pct,
                "volume_total": volume_a + volume_b,
                "volume_a": volume_a,
                "volume_b": volume_b,
            })
        return extracted

    rows = []

    for kpi in kpis:
        overall_summary = experiment_summary(
            df=df,
            treatment=treatment,
            kpi=kpi,
            alpha=alpha,
            spark_max_rows=spark_max_rows,
        )
        overall_power = check_power(
            df=df,
            treatment=treatment,
            kpi=kpi,
            unit=unit,
            control_label=control_label,
            power=power,
            alpha=alpha,
            test_type=test_type,
            spark_max_rows=spark_max_rows,
        )
        rows.extend(_extract_rows(overall_summary, overall_power, "overall", "overall"))

    for segment in segments:
        for kpi in kpis:
            seg_summary = experiment_summary_by_segment(
                df=df,
                treatment=treatment,
                kpi=kpi,
                segment=segment,
                alpha=alpha,
                spark_max_rows=spark_max_rows,
            )

            seg_power = check_power_by_segment(
                df=df,
                treatment=treatment,
                kpi=kpi,
                unit=unit,
                segment=segment,
                control_label=control_label,
                power=power,
                alpha=alpha,
                test_type=test_type,
                spark_max_rows=spark_max_rows,
            )

            for seg_value, result_summary in seg_summary.segments.items():
                power_result = seg_power.segments.get(seg_value)
                rows.extend(_extract_rows(result_summary, power_result, segment, seg_value))

    return pd.DataFrame(rows)
