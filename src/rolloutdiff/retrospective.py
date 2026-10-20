"""Exact L2 mean-shift candidate with an optional covariance confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ._dependence import confirm_ar1, quasi_binomial_errors
from ._series_validation import validate_series

_COMMON = ["run", "feature", "tier", "kind"]
_SEGMENT = [
    "previous_step",
    "n_before",
    "n_after",
    "mean_before",
    "mean_after",
    "delta",
    "rms_se_before",
    "rms_se_after",
]
_COSTS = [
    "normalization_scale",
    "sse_no_change_scaled",
    "sse_split_scaled",
    "gain_scaled",
    "penalty_scaled",
    "residual_variance_scaled",
    "se_variance_floor_scaled",
    "variance_scaled",
    "variance_source",
]
_DEPENDENCE = [
    "dependence",
    "independent_selected",
    "independent_reason",
    "covariance_status",
    "covariance_measurement",
    "covariance_gain",
    "covariance_penalty",
    "covariance_null_rho",
    "covariance_alternative_rho",
    "covariance_null_tau_variance_scaled",
    "covariance_alternative_tau_variance_scaled",
    "covariance_null_rho_bound",
    "covariance_alternative_rho_bound",
    "covariance_null_variance_bound",
    "covariance_alternative_variance_bound",
    "covariance_delta_scaled",
    "covariance_grid_fits",
]
_DIAGNOSTICS = [
    *_COMMON,
    "status",
    "reason",
    "n_observations",
    "n_candidates",
    "n_missing_se",
    "min_segment_steps",
    "penalty_multiplier",
    "min_effect",
    "candidate_step",
    *_SEGMENT,
    *_COSTS,
    *_DEPENDENCE,
]
_CHANGES = [*_COMMON, "changepoint_step", *_SEGMENT, "direction"]
_SCREENING = ["run", "n_series", "n_screened", "n_changes", "n_unscorable"]


@dataclass(frozen=True)
class RetrospectiveResult:
    changepoints: pd.DataFrame
    diagnostics: pd.DataFrame
    screening: pd.DataFrame


def _moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stable prefix means and residual sums, avoiding sum(x²)-sum(x)²/n."""
    means, sums = np.zeros(len(values) + 1), np.zeros(len(values) + 1)
    for count, value in enumerate(values, 1):
        delta = value - means[count - 1]
        means[count] = means[count - 1] + delta / count
        sums[count] = max(0.0, sums[count - 1] + delta * (value - means[count]))
    return means, sums


def _rms(values: np.ndarray) -> float:
    scale = float(values.max())
    return float(scale * np.sqrt(np.mean((values / scale) ** 2))) if scale else 0.0


def _fit(
    x: np.ndarray, se: np.ndarray, minimum: int, multiplier: float
) -> tuple[int, dict]:
    # Normalize squared quantities to keep the variance floor independent
    # of the feature's physical units.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        centered = x - x[0]
        scale = max(float(np.abs(centered).max()), float(se.max()))
        y = centered / scale if scale else np.zeros(len(x))
        errors = se / scale if scale else np.zeros(len(x))
        means, sums = _moments(y)
        reverse_means, reverse_sums = _moments(y[::-1])
        n = len(y)
        splits = np.arange(minimum, n - minimum + 1)
        costs = sums[splits] + reverse_sums[n - splits]
        best = float(costs.min())
        # Resolve numerical ties toward the earliest boundary with a tolerance
        # relative to the normalized full-run cost.
        tolerance = 8 * np.finfo(float).eps * sums[n]
        k = int(splits[np.flatnonzero(costs <= best + tolerance)[0]])
        split_cost = float(sums[k] + reverse_sums[n - k])
        null_cost = float(sums[n])
        residual_variance = split_cost / (n - 2)
        se_variance = float(np.mean(errors**2))
        variance = max(residual_variance, se_variance)
        penalty = multiplier * (np.log(n) * variance)
        source = (
            "zero"
            if variance == 0
            else (
                "across_prompt_se" if se_variance >= residual_variance else "residual"
            )
        )
        left, right = means[k], reverse_means[n - k]
        numbers = {
            "normalization_scale": scale,
            "mean_before": float(x[0] + left * scale),
            "mean_after": float(x[0] + right * scale),
            "delta": float((right - left) * scale),
            "rms_se_before": _rms(se[:k]),
            "rms_se_after": _rms(se[k:]),
            "sse_no_change_scaled": null_cost,
            "sse_split_scaled": split_cost,
            "gain_scaled": max(0.0, null_cost - split_cost),
            "penalty_scaled": float(penalty),
            "residual_variance_scaled": residual_variance,
            "se_variance_floor_scaled": se_variance,
            "variance_scaled": variance,
        }
        if right != left and numbers["delta"] == 0:
            raise FloatingPointError("effect underflow")
        if not all(np.isfinite(value) for value in numbers.values()):
            raise FloatingPointError
    return k, {**numbers, "variance_source": source}


def detect_retrospective(
    series: pd.DataFrame,
    *,
    min_segment_steps: int = 5,
    penalty_multiplier: float = 3.0,
    min_prevalence_change: float = 0.10,
    min_continuous_change: float = 0.0,
    dependence: str = "ar1",
) -> RetrospectiveResult:
    """Select at most one full-run mean shift per run/feature.

    The best two-segment L2 split (``changepoint_step`` starts the right
    segment) is selected when its absolute effect reaches the kind's minimum
    change and SSE0 - SSE1 > ``penalty_multiplier`` * log(n) * variance, where
    variance = max(SSE1/(n-2), mean(SE**2)). Missing SE or too few rows is
    unscorable.

    ``dependence="ar1"`` keeps a selection only if, with a stationary AR(1)
    covariance profiled under each mean model, twice the log-likelihood gain
    exceeds ``penalty_multiplier`` * log(n) and the GLS shift keeps its
    direction; exact piecewise-constant zero-SE splits stay selected as
    ``exact_zero_variance_unconfirmed``. AR lags count logged rows, ignoring
    step gaps. ``*_scaled`` diagnostics divide by ``normalization_scale``
    (squared for costs and variances); other effects keep input units.
    """
    if dependence not in ("ar1", "independent"):
        raise ValueError("dependence must be ar1 or independent")
    if type(min_segment_steps) is not int or min_segment_steps < 2:
        raise ValueError("min_segment_steps must be an integer of at least two")
    for name, value in [
        ("penalty_multiplier", penalty_multiplier),
        ("min_prevalence_change", min_prevalence_change),
        ("min_continuous_change", min_continuous_change),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and numeric")
    if (
        penalty_multiplier <= 0
        or not 0 <= min_prevalence_change <= 1
        or min_continuous_change < 0
    ):
        raise ValueError("invalid penalty or minimum effect")
    frame = validate_series(series)
    diagnostics, changes = [], []
    for (run, feature), group in frame.groupby(["run", "feature"], sort=False):
        row, change = _detect_group(
            group,
            run,
            feature,
            min_segment_steps,
            penalty_multiplier,
            min_prevalence_change,
            min_continuous_change,
            dependence,
        )
        diagnostics.append(row)
        if change is not None:
            changes.append(change)

    diagnostic_table = pd.DataFrame(diagnostics, columns=_DIAGNOSTICS)
    # Keep integer step identities exact when candidates are missing.
    for column in ("candidate_step", "previous_step"):
        diagnostic_table[column] = pd.Series(
            [row[column] for row in diagnostics], dtype=object
        )
    change_table = (
        pd.DataFrame(changes, columns=_CHANGES)
        .sort_values(["run", "changepoint_step", "feature"])
        .reset_index(drop=True)
    )
    if diagnostic_table.empty:
        screening = pd.DataFrame(columns=_SCREENING)
    else:
        screening = (
            diagnostic_table.assign(
                screened=diagnostic_table.status.isin(["change", "no_change"]),
                changed=diagnostic_table.status.eq("change"),
            )
            .groupby("run", sort=True)
            .agg(
                n_series=("feature", "size"),
                n_screened=("screened", "sum"),
                n_changes=("changed", "sum"),
            )
            .reset_index()
        )
        screening["n_unscorable"] = screening.n_series - screening.n_screened
    return RetrospectiveResult(change_table, diagnostic_table, screening)


def _select_candidate(row: dict, x: np.ndarray, kind: str, effect: float) -> None:
    # Allow float64 roundoff so an effect exactly at the cutoff qualifies.
    tolerance = (
        4
        * np.finfo(float).eps
        * (1 if kind == "binary" else max(abs(row["delta"]), effect))
    )
    if np.all(x == x[0]):
        row.update(status="no_change", reason="constant")
    elif abs(row["delta"]) < effect - tolerance:
        row.update(status="no_change", reason="below_effect")
    elif row["gain_scaled"] <= row["penalty_scaled"]:
        row.update(status="no_change", reason="penalty")
    else:
        row.update(status="change", reason="penalized_fit")


def _confirm_candidate(row, x, se, units, k, kind, penalty_multiplier, dependence):
    row.update(
        independent_selected=row["status"] == "change",
        independent_reason=row["reason"],
        covariance_status="not_requested"
        if dependence == "independent"
        else "not_needed",
    )
    if not row["independent_selected"] or dependence != "ar1":
        return
    if np.all(se == 0) and row["sse_split_scaled"] == 0:
        # An exact descriptive step has singular Gaussian likelihood.
        row["covariance_status"] = "exact_zero_variance_unconfirmed"
        return
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            scale = row["normalization_scale"]
            normalized = (x - x[0]) / scale
            errors = se / scale
            if np.any((se > 0) & (errors == 0)):
                raise FloatingPointError("measurement SE underflow")
            null_errors, row["covariance_measurement"] = errors, "step_se"
            if kind == "binary":
                constant = np.full(len(x), x.mean())
                modelled = quasi_binomial_errors(x, errors, units, constant)
                if modelled is not None:
                    segments = np.where(
                        np.arange(len(x)) < k, row["mean_before"], row["mean_after"]
                    )
                    null_errors, row["covariance_measurement"] = (
                        modelled,
                        "quasi_binomial_se",
                    )
                    errors = quasi_binomial_errors(x, errors, units, segments)
            row.update(
                confirm_ar1(
                    normalized,
                    null_errors,
                    k,
                    float(penalty_multiplier),
                    alternative_errors=errors,
                )
            )
    except FloatingPointError:
        raise ValueError(
            "numerical overflow or underflow fitting retrospective covariance"
        ) from None
    if row["covariance_status"] != "confirmed":
        row.update(status="no_change", reason="covariance_not_confirmed")
    elif np.sign(row["covariance_delta_scaled"]) != np.sign(row["delta"]):
        row.update(
            status="no_change",
            reason="covariance_direction_mismatch",
            covariance_status="direction_mismatch",
        )


def _detect_group(
    group,
    run,
    feature,
    min_segment_steps,
    penalty_multiplier,
    min_prevalence_change,
    min_continuous_change,
    dependence,
):
    kind, tier = group.kind.iloc[0], group.tier.iloc[0]
    x, se, steps = group["mean"].to_numpy(), group.se.to_numpy(), group.step.to_numpy()
    units = group.n_sampling_units.to_numpy(dtype=float)
    n = len(x)
    effect = float(min_prevalence_change if kind == "binary" else min_continuous_change)
    row = {name: np.nan for name in _DIAGNOSTICS}
    row.update(
        run=run,
        feature=feature,
        tier=tier,
        kind=kind,
        status="insufficient_data",
        reason="too_few_observations",
        n_observations=n,
        n_candidates=0,
        n_missing_se=int(np.isnan(se).sum()),
        min_segment_steps=min_segment_steps,
        penalty_multiplier=float(penalty_multiplier),
        min_effect=effect,
        candidate_step=None,
        previous_step=None,
        variance_source=None,
        dependence=dependence,
        independent_selected=None,
        independent_reason=None,
        covariance_status="unscorable",
        covariance_grid_fits=0,
    )
    if n < 2 * min_segment_steps:
        return row, None
    if row["n_missing_se"]:
        row.update(status="unavailable_uncertainty", reason="missing_se")
        return row, None
    try:
        k, fitted = _fit(x, se, min_segment_steps, float(penalty_multiplier))
    except FloatingPointError:
        raise ValueError(
            "numerical overflow or underflow fitting retrospective segmentation"
        ) from None
    row.update(
        fitted,
        candidate_step=int(steps[k]),
        previous_step=int(steps[k - 1]),
        n_before=k,
        n_after=n - k,
        n_candidates=n - 2 * min_segment_steps + 1,
    )
    _select_candidate(row, x, kind, effect)
    _confirm_candidate(row, x, se, units, k, kind, penalty_multiplier, dependence)
    change = None
    if row["status"] == "change":
        change = {name: row[name] for name in [*_COMMON, *_SEGMENT]}
        change.update(
            changepoint_step=int(steps[k]),
            direction="up" if row["delta"] > 0 else "down",
        )
    return row, change
