from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power, zt_ind_solve_power

from csa.post_experiment_analysis._helpers import _to_pandas_df, _is_binary


@dataclass(frozen=True)
class PowerResult:
    passed: bool
    baseline_rate: float
    observed_mde: float
    min_sample_per_group: int
    actual_samples: Dict[str, int]
    power: float
    alpha: float
    is_binary: bool
    test_type: str
    metric_std: float  # pooled std for continuous; 0.0 for binary

    def __repr__(self) -> str:
        status = "Passed! Experiment is sufficiently powered" if self.passed else "Failed! Experiment is underpowered"
        return (
            f"{status}\n"
            f"  Baseline rate/mean: {self.baseline_rate:.4f}\n"
            f"  Observed MDE: {self.observed_mde:.4f}\n"
            f"  Min sample per group: {self.min_sample_per_group:,}\n"
            f"  Actual samples: {self.actual_samples}\n"
            f"  Power: {self.power}  |  Alpha: {self.alpha}"
        )

    def _detectable_mde(self, n: int) -> tuple:
        """Back-solve power equations to get (abs_mde, rel_mde) detectable at given n."""
        alternative = "two-sided" if self.test_type == "two_sided" else "larger"
        try:
            if self.is_binary:
                h = zt_ind_solve_power(
                    nobs1=n, alpha=self.alpha, power=self.power, alternative=alternative,
                )
                p2 = np.sin(abs(h) / 2 + np.arcsin(np.sqrt(self.baseline_rate))) ** 2
                abs_mde = abs(p2 - self.baseline_rate)
            else:
                d = tt_ind_solve_power(
                    nobs1=n, alpha=self.alpha, power=self.power, alternative=alternative,
                )
                abs_mde = abs(d) * self.metric_std

            rel_mde = abs_mde / self.baseline_rate if self.baseline_rate else np.nan
            return abs_mde, rel_mde
        except Exception:
            return np.nan, np.nan

    def show(self) -> pd.DataFrame:
        req_abs = self.observed_mde
        req_rel = req_abs / self.baseline_rate if self.baseline_rate else np.nan

        rows = []
        for grp, n in self.actual_samples.items():
            det_abs, det_rel = self._detectable_mde(n)
            min_req = self.min_sample_per_group
            rows.append({
                "Group": grp,
                "Actual Sample": f"{n:,}",
                "Min Sample Required": f"{min_req:,}" if min_req != float("inf") else "∞",
                "Detectable MDE": f"[{det_abs:.4f}, {det_rel:.2%}]" if np.isfinite(det_abs) else "N/A",
                "Lift": f"[{req_abs:.4f}, {req_rel:.2%}]" if np.isfinite(req_rel) else f"[{req_abs:.4f}, N/A]",
                "Sufficient": "Yes" if n >= min_req else "No",
            })
        return pd.DataFrame(rows)


def check_power(
    df,
    treatment: str,
    kpi: str,
    unit: str,
    control_label: str,
    power: float = 0.80,
    alpha: float = 0.05,
    test_type: str = "two_sided",
    spark_max_rows: Optional[int] = None,
) -> PowerResult:

    if test_type not in ("two_sided", "one_sided"):
        raise ValueError("test_type must be 'two_sided' or 'one_sided'.")
    alternative = "two-sided" if test_type == "two_sided" else "larger"

    d = _to_pandas_df(df, [unit, treatment, kpi], spark_max_rows)
    d[treatment] = d[treatment].astype(str)

    groups = sorted(d[treatment].unique())
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 treatment groups, found {len(groups)}")

    if control_label not in groups:
        raise ValueError(f"control_label '{control_label}' not found in treatment groups {groups}.")

    ctrl_vals = d.loc[d[treatment] == control_label, kpi].astype(float)

    treat_groups = [g for g in groups if g != control_label]
    treat_label = treat_groups[0]
    treat_vals = d.loc[d[treatment] == treat_label, kpi].astype(float)

    all_vals = np.concatenate([ctrl_vals.to_numpy(), treat_vals.to_numpy()])
    binary = _is_binary(all_vals)

    if binary:
        baseline = ctrl_vals.mean()
        treat_rate = treat_vals.mean()
        mde = abs(treat_rate - baseline)
        metric_std = 0.0

        if mde == 0:
            return PowerResult(
                passed=False, baseline_rate=baseline, observed_mde=0,
                min_sample_per_group=float("inf"),
                actual_samples={g: d.loc[d[treatment] == g, unit].nunique() for g in groups},
                power=power, alpha=alpha, is_binary=True,
                test_type=test_type, metric_std=metric_std,
            )

        h = 2 * (np.arcsin(np.sqrt(treat_rate)) - np.arcsin(np.sqrt(baseline)))
        effect_size = abs(h)

        min_n = zt_ind_solve_power(
            effect_size=effect_size, alpha=alpha, power=power, alternative=alternative,
        )
    else:
        baseline = ctrl_vals.mean()
        treat_mean = treat_vals.mean()
        mde = abs(treat_mean - baseline)

        pooled_std = np.sqrt(
            (ctrl_vals.var(ddof=1) * (len(ctrl_vals) - 1) + treat_vals.var(ddof=1) * (len(treat_vals) - 1))
            / (len(ctrl_vals) + len(treat_vals) - 2)
        )
        metric_std = pooled_std

        if pooled_std == 0 or mde == 0:
            return PowerResult(
                passed=False, baseline_rate=baseline, observed_mde=mde,
                min_sample_per_group=float("inf"),
                actual_samples={g: d.loc[d[treatment] == g, unit].nunique() for g in groups},
                power=power, alpha=alpha, is_binary=False,
                test_type=test_type, metric_std=metric_std,
            )

        effect_size = mde / pooled_std

        min_n = tt_ind_solve_power(
            effect_size=effect_size, alpha=alpha, power=power, alternative=alternative,
        )

    min_n = int(np.ceil(min_n))
    actual = {g: d.loc[d[treatment] == g, unit].nunique() for g in groups}
    passed = all(n >= min_n for n in actual.values())

    return PowerResult(
        passed=passed,
        baseline_rate=baseline,
        observed_mde=mde,
        min_sample_per_group=min_n,
        actual_samples=actual,
        power=power,
        alpha=alpha,
        is_binary=binary,
        test_type=test_type,
        metric_std=metric_std,
    )
