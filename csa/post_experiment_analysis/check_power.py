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
    actual_samples: Dict[str, int]
    power: float
    alpha: float
    is_binary: bool
    test_type: str
    # Per-comparison fields keyed by "variant vs control"
    observed_mdes: Dict[str, float]
    min_samples_per_group: Dict[str, float]   # float to accommodate inf
    metric_stds: Dict[str, float]

    def __repr__(self) -> str:
        status = (
            "Passed! Experiment is sufficiently powered"
            if self.passed
            else "Failed! Experiment is underpowered"
        )
        lines = [
            status,
            f"  Baseline rate/mean: {self.baseline_rate:.4f}",
            f"  Power: {self.power}  |  Alpha: {self.alpha}",
        ]
        for comparison, mde in self.observed_mdes.items():
            min_n = self.min_samples_per_group[comparison]
            min_n_str = f"{int(min_n):,}" if np.isfinite(min_n) else "∞"
            lines.append(
                f"  [{comparison}]  Observed MDE: {mde:.4f},  Min sample per group: {min_n_str}"
            )
        return "\n".join(lines)

    def _detectable_mde(self, n: int, metric_std: float) -> tuple:
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
                abs_mde = abs(d) * metric_std

            rel_mde = abs_mde / self.baseline_rate if self.baseline_rate else np.nan
            return abs_mde, rel_mde
        except Exception:
            return np.nan, np.nan

    def show(self) -> pd.DataFrame:
        rows = []
        for comparison, mde in self.observed_mdes.items():
            treat_label, ctrl_label = comparison.split(" vs ", 1)
            min_n = self.min_samples_per_group[comparison]
            metric_std = self.metric_stds[comparison]

            n_ctrl = self.actual_samples.get(ctrl_label, 0)
            n_treat = self.actual_samples.get(treat_label, 0)

            # Use the binding constraint (smaller group) for detectable MDE
            det_abs, det_rel = self._detectable_mde(min(n_ctrl, n_treat), metric_std)

            req_rel = mde / self.baseline_rate if self.baseline_rate else np.nan
            sufficient = n_ctrl >= min_n and n_treat >= min_n

            rows.append({
                "Comparison": comparison,
                "Actual N (Control)": f"{n_ctrl:,}",
                "Actual N (Treatment)": f"{n_treat:,}",
                "Min Sample Required": f"{int(min_n):,}" if np.isfinite(min_n) else "∞",
                "Detectable MDE": (
                    f"[{det_abs:.4f}, {det_rel:.2%}]" if np.isfinite(det_abs) else "N/A"
                ),
                "Lift": (
                    f"[{mde:.4f}, {req_rel:.2%}]"
                    if np.isfinite(req_rel)
                    else f"[{mde:.4f}, N/A]"
                ),
                "Sufficient": "Yes" if sufficient else "No",
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
        raise ValueError(
            f"control_label '{control_label}' not found in treatment groups {groups}."
        )

    ctrl_vals = d.loc[d[treatment] == control_label, kpi].astype(float).to_numpy()
    treat_groups = [g for g in groups if g != control_label]

    binary = _is_binary(d[kpi].astype(float).to_numpy())
    baseline = ctrl_vals.mean()
    actual = {g: d.loc[d[treatment] == g, unit].nunique() for g in groups}

    observed_mdes: Dict[str, float] = {}
    min_samples_per_group: Dict[str, float] = {}
    metric_stds: Dict[str, float] = {}

    for treat_label in treat_groups:
        key = f"{treat_label} vs {control_label}"
        treat_vals = d.loc[d[treatment] == treat_label, kpi].astype(float).to_numpy()

        if binary:
            treat_rate = treat_vals.mean()
            mde = abs(treat_rate - baseline)
            metric_stds[key] = 0.0
            observed_mdes[key] = mde

            if mde == 0:
                min_samples_per_group[key] = float("inf")
                continue

            h = 2 * (np.arcsin(np.sqrt(treat_rate)) - np.arcsin(np.sqrt(baseline)))
            min_n = zt_ind_solve_power(
                effect_size=abs(h), alpha=alpha, power=power, alternative=alternative,
            )
        else:
            treat_mean = treat_vals.mean()
            mde = abs(treat_mean - baseline)
            pooled_std = np.sqrt(
                (
                    ctrl_vals.var(ddof=1) * (len(ctrl_vals) - 1)
                    + treat_vals.var(ddof=1) * (len(treat_vals) - 1)
                )
                / (len(ctrl_vals) + len(treat_vals) - 2)
            )
            metric_stds[key] = pooled_std
            observed_mdes[key] = mde

            if pooled_std == 0 or mde == 0:
                min_samples_per_group[key] = float("inf")
                continue

            min_n = tt_ind_solve_power(
                effect_size=mde / pooled_std, alpha=alpha, power=power, alternative=alternative,
            )

        min_samples_per_group[key] = int(np.ceil(min_n))

    passed = True
    for comparison, min_n in min_samples_per_group.items():
        treat_label, ctrl_label = comparison.split(" vs ", 1)
        if actual.get(ctrl_label, 0) < min_n or actual.get(treat_label, 0) < min_n:
            passed = False
            break

    return PowerResult(
        passed=passed,
        baseline_rate=baseline,
        actual_samples=actual,
        power=power,
        alpha=alpha,
        is_binary=binary,
        test_type=test_type,
        observed_mdes=observed_mdes,
        min_samples_per_group=min_samples_per_group,
        metric_stds=metric_stds,
    )
