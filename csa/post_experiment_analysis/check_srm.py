from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from csa.post_experiment_analysis._helpers import _to_pandas_df


@dataclass(frozen=True)
class SRMResult:
    chi2_statistic: float
    p_value: float
    alpha: float
    observed_counts: Dict[str, int]
    expected_counts: Dict[str, float]
    srm_detected: bool

    def __repr__(self) -> str:
        if self.srm_detected:
            return f"Failed! Sample ratio mismatch detected (p={self.p_value:.4g})"
        return f"Passed! No sample ratio mismatch (p={self.p_value:.4g})"

    def show(self) -> pd.DataFrame:
        groups = list(self.observed_counts.keys())
        return pd.DataFrame({
            "Group": groups,
            "Observed": [self.observed_counts[g] for g in groups],
            "Expected": [round(self.expected_counts[g], 1) for g in groups],
        })


def check_srm(
    df=None,
    treatment: Optional[str] = None,
    unit: Optional[str] = None,
    expected_ratio: Optional[Dict[str, float]] = None,
    observed_counts: Optional[List[int]] = None,
    expected_ratios: Optional[List[float]] = None,
    alpha: float = 0.05,
    spark_max_rows: Optional[int] = None,
) -> SRMResult:

    if not (0 < alpha < 1):
        raise ValueError("alpha must be between 0 and 1.")

    # ----- Mode 1: DataFrame provided -----
    if df is not None:
        if treatment is None or unit is None or expected_ratio is None:
            raise ValueError("When df is provided, treatment, unit, and expected_ratio are required.")

        d = _to_pandas_df(df, [unit, treatment], spark_max_rows)
        d[treatment] = d[treatment].astype(str)

        obs = d.groupby(treatment)[unit].nunique()

        treatment_groups = set(obs.index)
        ratio_groups = set(expected_ratio.keys())
        if treatment_groups != ratio_groups:
            raise ValueError(
                f"expected_ratio keys {ratio_groups} must match treatment groups {treatment_groups}."
            )

        ratios = np.array([expected_ratio[g] for g in obs.index])
        if not np.isclose(ratios.sum(), 1):
            raise ValueError("expected_ratio values must sum to 1.")

        obs_array = obs.values.astype(float)
        total = obs_array.sum()
        exp_array = ratios * total

        chi2, p_value = chisquare(f_obs=obs_array, f_exp=exp_array)

        obs_dict = {g: int(obs[g]) for g in obs.index}
        exp_dict = {g: float(exp_array[i]) for i, g in enumerate(obs.index)}

    # ----- Mode 2: Raw counts provided -----
    else:
        if observed_counts is None or expected_ratios is None:
            raise ValueError("When df is not provided, observed_counts and expected_ratios are required.")

        obs_array = np.array(observed_counts, dtype=float)
        ratios = np.array(expected_ratios, dtype=float)

        if len(obs_array) != len(ratios):
            raise ValueError("observed_counts and expected_ratios must have the same length.")
        if not np.isclose(ratios.sum(), 1):
            raise ValueError("expected_ratios must sum to 1.")
        if np.any(obs_array < 0):
            raise ValueError("observed_counts cannot contain negative values.")
        if np.any(ratios < 0):
            raise ValueError("expected_ratios cannot contain negative values.")

        total = obs_array.sum()
        exp_array = ratios * total

        chi2, p_value = chisquare(f_obs=obs_array, f_exp=exp_array)

        labels = [f"Group {i+1}" for i in range(len(obs_array))]
        obs_dict = {g: int(obs_array[i]) for i, g in enumerate(labels)}
        exp_dict = {g: float(exp_array[i]) for i, g in enumerate(labels)}

    return SRMResult(
        chi2_statistic=chi2,
        p_value=p_value,
        alpha=alpha,
        observed_counts=obs_dict,
        expected_counts=exp_dict,
        srm_detected=p_value < alpha,
    )
