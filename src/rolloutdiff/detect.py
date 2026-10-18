"""Sequential sustained-z screening with one baseline scale policy."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ._series_validation import validate_series

_ALERT_COLUMNS = [
    "run",
    "feature",
    "tier",
    "kind",
    "onset_step",
    "alert_step",
    "direction",
    "baseline_mean",
    "value_at_alert",
    "z_at_alert",
    "delta_at_alert",
    "scale",
    "se_at_alert",
    "n_prompts_at_alert",
    "n_sampling_units_at_alert",
    "n_rollouts_at_alert",
    "numerical_floor_dominant",
]
_DIAGNOSTIC_COLUMNS = [
    "run",
    "feature",
    "tier",
    "kind",
    "status",
    "baseline_n",
    "baseline_start",
    "baseline_end",
    "baseline_mean",
    "baseline_sd",
    "baseline_se_floor",
    "numerical_floor",
    "scale",
    "scale_source",
    "numerical_floor_dominant",
    "n_scored",
    "n_unscorable",
]
_SCREENING_COLUMNS = ["run", "n_series", "n_screened", "n_alerted", "n_unscorable"]


@dataclass(frozen=True)
class DetectionResult:
    alerts: pd.DataFrame
    diagnostics: pd.DataFrame
    scores: pd.DataFrame
    screening: pd.DataFrame


def detect_sequential(
    series: pd.DataFrame,
    *,
    baseline_steps: int = 15,
    z_threshold: float = 4.0,
    sustain: int = 3,
    min_prevalence_change: float = 0.10,
    numerical_floor: float = 1e-9,
) -> DetectionResult:
    """Return the first sustained alert and all screening scores per series.

    The first ``baseline_steps`` rows fix the mean and
    scale = max(sample SD, median prompt SE, numerical_floor).
    Binary crossings require z > threshold and delta >= minimum change;
    continuous crossings require abs(z) > threshold. Four float64 epsilons
    accommodate binary effect roundoff. Crossings must share a direction for
    ``sustain`` consecutive logged observations, allowing sparse step IDs.

    Missing baseline SE makes a series unscorable; missing monitoring SE resets
    its streak. Current-step SE leaves the divisor unchanged. Guard-dominated
    scores depend on input units. Error rates are uncalibrated, and full-run
    vocabulary selection remains retrospective.
    """
    if type(baseline_steps) is not int or baseline_steps < 2:
        raise ValueError("baseline_steps must be an integer of at least two")
    if type(sustain) is not int or sustain < 1:
        raise ValueError("sustain must be a positive integer")
    for name, value in [
        ("z_threshold", z_threshold),
        ("numerical_floor", numerical_floor),
        ("min_prevalence_change", min_prevalence_change),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and numeric")
    if z_threshold <= 0 or numerical_floor <= 0 or not 0 <= min_prevalence_change <= 1:
        raise ValueError("invalid detector threshold or floor")
    scores = validate_series(series)
    scores["delta"] = np.nan
    scores["z"] = np.nan
    scores["direction"] = 0
    scores["qualifies"] = False
    scores["streak"] = 0
    scores["status"] = "baseline"
    alerts, diagnostics = [], []
    for (run, feature), group in scores.groupby(["run", "feature"], sort=False):
        group_alerts, diagnostic = _detect_group(
            scores,
            group,
            run,
            feature,
            baseline_steps,
            z_threshold,
            sustain,
            min_prevalence_change,
            numerical_floor,
        )
        alerts.extend(group_alerts)
        diagnostics.append(diagnostic)

    diagnostic_table = pd.DataFrame(diagnostics, columns=_DIAGNOSTIC_COLUMNS)
    alert_table = (
        pd.DataFrame(alerts, columns=_ALERT_COLUMNS)
        .sort_values(["run", "alert_step", "feature"])
        .reset_index(drop=True)
    )
    if diagnostic_table.empty:
        screening = pd.DataFrame(columns=_SCREENING_COLUMNS)
    else:
        screening = (
            diagnostic_table.assign(
                screened=diagnostic_table.n_scored.gt(0),
                alerted=diagnostic_table.status.eq("alerted"),
            )
            .groupby("run", sort=True)
            .agg(
                n_series=("feature", "size"),
                n_screened=("screened", "sum"),
                n_alerted=("alerted", "sum"),
            )
            .reset_index()
        )
        screening["n_unscorable"] = screening.n_series - screening.n_screened
    return DetectionResult(alert_table, diagnostic_table, scores, screening)


def _monitoring_scores(
    values, errors, mean, scale, kind, z_threshold, min_prevalence_change
):
    valid = np.isfinite(errors)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            delta = values - mean
            z = np.full(len(delta), np.nan)
            z[valid] = delta[valid] / scale
    except FloatingPointError:
        raise ValueError("numerical overflow computing screening scores") from None
    if kind == "binary":
        effect_met = delta >= min_prevalence_change - 4 * np.finfo(float).eps
        qualifies = valid & (z > z_threshold) & effect_met
    else:
        qualifies = valid & (np.abs(z) > z_threshold)
    direction = np.where(qualifies, np.sign(np.nan_to_num(z, nan=0)), 0).astype(int)
    return delta, z, qualifies, direction


def _alert_streaks(direction: np.ndarray, sustain: int):
    streak, previous_direction, start = 0, 0, 0
    streaks = np.zeros(len(direction), dtype=int)
    first_alert = None
    for offset, sign in enumerate(direction):
        if sign == 0:
            streak = 0
        elif sign == previous_direction:
            streak += 1
        else:
            streak, start = 1, offset
        previous_direction = sign
        streaks[offset] = streak
        if streak >= sustain and first_alert is None:
            first_alert = start, offset
    return streaks, first_alert


def _detect_group(
    scores,
    group,
    run,
    feature,
    baseline_steps,
    z_threshold,
    sustain,
    min_prevalence_change,
    numerical_floor,
):
    indices = group.index.to_numpy()
    steps = group.step.to_numpy()
    x, errors = group["mean"].to_numpy(), group.se.to_numpy()
    kind, tier = group.kind.iloc[0], group.tier.iloc[0]
    n_base = min(baseline_steps, len(group))
    common = {"run": run, "feature": feature, "tier": tier, "kind": kind}
    diagnostic = {
        **common,
        "status": "insufficient_baseline",
        "baseline_n": n_base,
        "baseline_start": int(steps[0]),
        "baseline_end": int(steps[n_base - 1]),
        "baseline_mean": np.nan,
        "baseline_sd": np.nan,
        "baseline_se_floor": np.nan,
        "numerical_floor": float(numerical_floor),
        "scale": np.nan,
        "scale_source": None,
        "numerical_floor_dominant": False,
        "n_scored": 0,
        "n_unscorable": len(group) - n_base,
    }
    if len(group) < baseline_steps:
        return [], diagnostic
    baseline = _baseline_statistics(
        x[:baseline_steps], errors[:baseline_steps], numerical_floor
    )
    diagnostic.update(baseline)
    post_indices = indices[baseline_steps:]
    if diagnostic["status"] == "insufficient_baseline_uncertainty":
        scores.loc[post_indices, "status"] = diagnostic["status"]
        return [], diagnostic
    mean = baseline["baseline_mean"]
    scale = baseline["scale"]
    guard_dominates = baseline["numerical_floor_dominant"]
    valid = np.isfinite(errors[baseline_steps:])
    diagnostic.update(
        n_scored=int(valid.sum()),
        n_unscorable=int((~valid).sum()),
        status="no_alert" if valid.any() else "no_scorable_observations",
    )
    delta, z, qualifies, direction = _monitoring_scores(
        x[baseline_steps:],
        errors[baseline_steps:],
        mean,
        scale,
        kind,
        z_threshold,
        min_prevalence_change,
    )
    scores.loc[post_indices, "delta"] = delta
    scores.loc[post_indices, "z"] = z
    scores.loc[post_indices, "qualifies"] = qualifies
    scores.loc[post_indices, "direction"] = direction
    scores.loc[post_indices, "status"] = np.where(valid, "scored", "unavailable_se")
    streaks, first_alert = _alert_streaks(direction, sustain)
    alerts = []
    if first_alert is not None:
        start, offset = first_alert
        row = group.iloc[baseline_steps + offset]
        alerts.append(
            {
                **common,
                "onset_step": int(steps[baseline_steps + start]),
                "alert_step": int(row.step),
                "direction": "up" if direction[offset] > 0 else "down",
                "baseline_mean": mean,
                "value_at_alert": float(row["mean"]),
                "z_at_alert": float(z[offset]),
                "delta_at_alert": float(delta[offset]),
                "scale": scale,
                "se_at_alert": float(row.se),
                "n_prompts_at_alert": int(row.n_prompts),
                "n_sampling_units_at_alert": int(row.n_sampling_units),
                "n_rollouts_at_alert": int(row.n_rollouts),
                "numerical_floor_dominant": guard_dominates,
            }
        )
        diagnostic["status"] = "alerted"
    scores.loc[post_indices, "streak"] = streaks
    return alerts, diagnostic


def _baseline_statistics(
    values: np.ndarray, errors: np.ndarray, numerical_floor: float
) -> dict:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
    except FloatingPointError:
        raise ValueError("numerical overflow computing baseline statistics") from None
    baseline = {"baseline_mean": mean, "baseline_sd": sd}
    if not np.isfinite(errors).all():
        return baseline | {"status": "insufficient_baseline_uncertainty"}
    try:
        with np.errstate(over="raise", invalid="raise"):
            se_floor = float(np.median(errors))
    except FloatingPointError:
        raise ValueError("numerical overflow computing baseline SE floor") from None
    scale = max(sd, se_floor, float(numerical_floor))
    guard_dominates = max(sd, se_floor) <= numerical_floor
    source = (
        "numerical_floor"
        if guard_dominates
        else ("baseline_sd" if sd >= se_floor else "baseline_se")
    )
    return baseline | {
        "baseline_se_floor": se_floor,
        "scale": scale,
        "scale_source": source,
        "numerical_floor_dominant": guard_dominates,
    }
