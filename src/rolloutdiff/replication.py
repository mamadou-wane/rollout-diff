"""Deterministic sampling-unit split evidence for the first sequential alert."""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from inspect import signature

import numpy as np
import pandas as pd

from ._feature_values import feature_values
from ._sampling import resolve_sampling_units, subset_sampling_units
from .detect import DetectionResult, detect_sequential
from .features import FeatureTables
from .series import aggregate

_HALF_COLUMNS = [
    "half",
    "status",
    "alert_step",
    "onset_step",
    "direction",
    "alert_position",
    "n_prompts",
    "n_sampling_units",
    "n_rollouts",
    "n_steps",
    "n_scored",
    "n_unscorable",
    "missing_steps",
    "reason",
    "source_detect_status",
    "baseline_mean",
    "baseline_sd",
    "baseline_se_floor",
    "scale",
    "numerical_floor_dominant",
]
_FIXED_COLUMNS = [
    "baseline_expected",
    "baseline_required",
    "baseline_used",
    "baseline_used_steps",
    "baseline_excluded_steps",
    "monitoring_expected",
    "monitoring_available",
    "unavailable_se_steps",
    "first_possible_alert_step",
    "observed_alert_step",
    "observed_onset_step",
    "observed_direction",
    "first_alert_identified",
]
_ALIAS = "_evidence_value"


@dataclass(frozen=True)
class ReplicationEvidence:
    run: str
    feature: str
    status: str
    replicates: bool | None
    full: DetectionResult
    halves: pd.DataFrame
    assignments: pd.DataFrame
    tolerance: int
    horizon_complete: bool
    cutoff_step: int | None
    coverage_policy: str = "contiguous_prefix"


def _series(
    frame: pd.DataFrame,
    *,
    feature: str,
    kind: str,
    tier: str,
    sampling_units: pd.DataFrame,
) -> pd.DataFrame:
    features = FeatureTables(
        frame[["run", "sample_id", "_value"]].rename(columns={"_value": _ALIAS}),
        pd.DataFrame(columns=["run", "sample_id", "feature"]),
        pd.DataFrame([{"feature": _ALIAS, "tier": tier, "kind": kind}]),
    )
    result = aggregate(
        frame.drop(columns="_value"), features, sampling_units=sampling_units
    ).series
    return (
        result.loc[result.feature.eq(_ALIAS)]
        .assign(feature=feature)
        .reset_index(drop=True)
    )


def split_half_replication(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    feature: str,
    baseline_steps: int = 15,
    z_threshold: float = 4.0,
    sustain: int = 3,
    min_prevalence_change: float = 0.10,
    numerical_floor: float = 1e-9,
    tolerance: int = 10,
    sampling_units: pd.DataFrame | None = None,
    coverage_policy: str = "contiguous_prefix",
) -> ReplicationEvidence:
    """Compare first full/half alerts using whole-unit CRC32 & 1 splits.

    Replication needs all three first alerts in one direction within
    ``tolerance`` full-grid positions; observed disagreement fails regardless
    of the other half. A scorable half without an alert stays
    ``pending`` until that horizon is complete, and an unavailable-SE row that
    could hide an earlier compatible alert makes the half unavailable.
    ``contiguous_prefix`` stops a half at its first gap; ``fixed_window`` needs
    max(2, ceil(``baseline_steps``/2)) finite-SE baseline rows and resumes after
    gaps.
    """
    _validate_coverage_policy(coverage_policy)
    if type(tolerance) is not int or tolerance < 0:
        raise ValueError("tolerance must be a nonnegative integer")
    parameters = {
        "baseline_steps": baseline_steps,
        "z_threshold": z_threshold,
        "sustain": sustain,
        "min_prevalence_change": min_prevalence_change,
        "numerical_floor": numerical_floor,
    }
    full, half_series, half_schedules, assignments = _prepare_halves(
        rollouts,
        features,
        run=run,
        feature=feature,
        sampling_units=sampling_units,
        parameters=parameters,
    )
    return _replication_prepared(
        full,
        half_series,
        half_schedules,
        assignments,
        run=run,
        feature=feature,
        tolerance=tolerance,
        parameters=parameters,
        coverage_policy=coverage_policy,
    )


def _prepare_halves(rollouts, features, *, run, feature, sampling_units, parameters):
    units = resolve_sampling_units(rollouts, sampling_units)
    frame, kind = feature_values(rollouts, features, run=run, feature=feature)
    units = subset_sampling_units(units, frame)
    tier = (
        "structural"
        if feature == "scalar:reward"
        else features.catalog.set_index("feature").loc[feature, "tier"]
    )
    assignments = units.copy()
    assignments["half"] = assignments.sampling_unit_id.map(
        lambda unit: zlib.crc32(unit.encode("utf-8")) & 1
    )
    full = detect_sequential(
        _series(frame, feature=feature, kind=kind, tier=tier, sampling_units=units),
        **parameters,
    )
    half_series, half_schedules = {}, {}
    for half in (0, 1):
        prompts = assignments.loc[assignments.half.eq(half), "prompt_id"]
        subset = frame.loc[frame.prompt_id.isin(prompts)]
        half_units = subset_sampling_units(units, subset)
        half_series[half] = _series(
            subset, feature=feature, kind=kind, tier=tier, sampling_units=half_units
        )
        half_schedules[half] = (
            subset.groupby(["step", "prompt_id"], sort=True)
            .size()
            .rename("n_rollouts")
            .reset_index()
            .merge(
                half_units[["prompt_id", "sampling_unit_id"]],
                on="prompt_id",
                how="left",
                validate="many_to_one",
            )
        )
    return full, half_series, half_schedules, assignments


def _validate_coverage_policy(policy: str) -> None:
    if policy not in ("contiguous_prefix", "fixed_window"):
        raise ValueError("coverage_policy must be contiguous_prefix or fixed_window")


def _detector_parameters(parameters: dict) -> dict:
    resolved = {
        name: parameter.default
        for name, parameter in signature(detect_sequential).parameters.items()
        if name != "series"
    }
    return resolved | parameters


def _fixed_baseline_scores(series: pd.DataFrame, steps: list[int], parameters: dict):
    """Score a sparse fixed baseline; callers scan the full grid for streaks."""
    baseline_steps = parameters["baseline_steps"]
    baseline_grid, monitor_grid = steps[:baseline_steps], steps[baseline_steps:]
    prepared = series.loc[series.step.isin(steps)]
    # Empty aggregates carry an object-typed SE column; no unit means no evidence.
    finite = np.isfinite(prepared.se.to_numpy(dtype=float))
    baseline = prepared.loc[prepared.step.isin(baseline_grid) & finite]
    monitoring = prepared.loc[prepared.step.isin(monitor_grid)]
    used = set(baseline.step)
    available = set(prepared.loc[finite, "step"])
    coverage = {
        "baseline_expected": baseline_steps,
        "baseline_required": max(2, (baseline_steps + 1) // 2),
        "baseline_used": len(baseline),
        "baseline_used_steps": [int(step) for step in baseline_grid if step in used],
        "baseline_excluded_steps": [
            int(step) for step in baseline_grid if step not in used
        ],
        "monitoring_expected": len(monitor_grid),
        "monitoring_available": sum(step in available for step in monitor_grid),
        "unavailable_se_steps": [int(step) for step in prepared.loc[~finite, "step"]],
        "n_scored": 0,
        "n_unscorable": len(monitor_grid),
    }
    for name in (
        "baseline_mean",
        "baseline_sd",
        "baseline_se_floor",
        "scale",
        "numerical_floor_dominant",
    ):
        coverage[name] = None
    if len(baseline) < coverage["baseline_required"]:
        return coverage, None
    result = detect_sequential(
        pd.concat([baseline, monitoring], ignore_index=True),
        **(parameters | {"baseline_steps": len(baseline)}),
    )
    diagnostic = result.diagnostics.iloc[0]
    for name in (
        "baseline_mean",
        "baseline_sd",
        "baseline_se_floor",
        "scale",
        "numerical_floor_dominant",
        "n_scored",
    ):
        coverage[name] = diagnostic[name]
    coverage["n_unscorable"] = len(monitor_grid) - coverage["n_scored"]
    return coverage, result


def _fixed_window_half(
    half: int,
    series: pd.DataFrame,
    schedule: pd.DataFrame,
    steps: list[int],
    positions: dict,
    parameters: dict,
    horizon_complete: bool,
) -> dict:
    """Scan the full grid, distinguishing observed from possible first alerts."""
    resolved = _detector_parameters(parameters)
    coverage, result = _fixed_baseline_scores(series, steps, resolved)
    monitor_grid = steps[resolved["baseline_steps"] :]
    observed = set(schedule.step)
    row = {
        "half": half,
        "status": "unavailable",
        "alert_step": None,
        "onset_step": None,
        "direction": None,
        "alert_position": None,
        "n_prompts": int(schedule.prompt_id.nunique()),
        "n_sampling_units": int(schedule.sampling_unit_id.nunique()),
        "n_rollouts": int(schedule.n_rollouts.sum()),
        "n_steps": int(schedule.step.nunique()),
        "missing_steps": [int(step) for step in steps if step not in observed],
        "reason": "insufficient_fixed_baseline",
        "source_detect_status": "insufficient_fixed_baseline",
        "first_possible_alert_step": None,
        "observed_alert_step": None,
        "observed_onset_step": None,
        "observed_direction": None,
        "first_alert_identified": False,
        **coverage,
    }
    if result is None:
        return row
    signs = {
        int(point.step): int(point.direction)
        for point in result.scores.loc[
            result.scores.step.isin(monitor_grid)
        ].itertuples()
        if np.isfinite(point.se)
    }
    sustain = resolved["sustain"]
    allowed_signs = (1,) if series.kind.iloc[0] == "binary" else (-1, 1)
    row.update(_scan_first_alerts(signs, monitor_grid, sustain, allowed_signs))

    actual, possible = row["observed_alert_step"], row["first_possible_alert_step"]
    row["source_detect_status"] = (
        "alerted"
        if actual is not None
        else ("no_alert" if row["n_scored"] else "no_scorable_observations")
    )
    if actual is not None and possible == actual:
        row.update(
            status="alerted",
            reason="first_alert_observed",
            alert_step=actual,
            onset_step=row["observed_onset_step"],
            direction=row["observed_direction"],
            alert_position=positions[actual],
            first_alert_identified=True,
        )
    elif possible is not None:
        row["reason"] = "ambiguous_first_alert"
    elif horizon_complete:
        row.update(status="no_alert", reason="no_alert_within_complete_horizon")
    else:
        row.update(status="pending", reason="incomplete_comparison_horizon")
    return row


def _replication_prepared(
    full: DetectionResult,
    half_series: dict[int, pd.DataFrame],
    half_schedules: dict[int, pd.DataFrame],
    assignments: pd.DataFrame,
    *,
    run: str,
    feature: str,
    tolerance: int,
    parameters: dict,
    coverage_policy: str = "contiguous_prefix",
) -> ReplicationEvidence:
    """Compare first alerts using preaggregated half series."""
    _validate_coverage_policy(coverage_policy)
    columns = _HALF_COLUMNS + (
        _FIXED_COLUMNS if coverage_policy == "fixed_window" else []
    )
    if full.alerts.empty:
        status = (
            "no_full_alert"
            if full.diagnostics.n_scored.sum() > 0
            else "full_unscorable"
        )
        return ReplicationEvidence(
            run,
            feature,
            status,
            None,
            full,
            pd.DataFrame(columns=columns),
            assignments,
            tolerance,
            False,
            None,
            coverage_policy,
        )

    alert = full.alerts.iloc[0]
    steps = full.scores.step.tolist()
    positions = {step: position for position, step in enumerate(steps)}
    full_position = positions[alert.alert_step]
    horizon_complete = full_position + tolerance < len(steps)
    cutoff_position = min(full_position + tolerance, len(steps) - 1)
    required_steps = steps[: cutoff_position + 1]
    cutoff_step = int(required_steps[-1])
    rows = []
    for half in (0, 1):
        subset = half_schedules[half].loc[
            half_schedules[half].step.isin(required_steps)
        ]
        if coverage_policy == "fixed_window":
            rows.append(
                _fixed_window_half(
                    half,
                    half_series[half],
                    subset,
                    required_steps,
                    positions,
                    parameters,
                    horizon_complete,
                )
            )
            continue
        rows.append(
            _contiguous_half(
                half,
                half_series[half],
                subset,
                required_steps,
                positions,
                parameters,
                horizon_complete,
            )
        )

    halves = pd.DataFrame(rows, columns=columns)
    # Keep integer step identities exact when alerts are missing.
    step_columns = ["alert_step", "onset_step"]
    if coverage_policy == "fixed_window":
        step_columns += [
            "first_possible_alert_step",
            "observed_alert_step",
            "observed_onset_step",
        ]
    for column in step_columns:
        halves[column] = pd.Series([row[column] for row in rows], dtype=object)
    alerted = halves.loc[halves.status.eq("alerted")]
    alert_positions = [full_position, *alerted.alert_position.tolist()]
    known_failure = (
        halves.status.eq("no_alert").any()
        or alerted.direction.ne(alert.direction).any()
        or max(alert_positions) - min(alert_positions) > tolerance
    )
    if known_failure:
        status, replicates = "not_replicated", False
    elif len(alerted) == 2:
        status, replicates = "replicated", True
    elif halves.status.eq("unavailable").any():
        status, replicates = "unavailable", None
    else:
        status, replicates = "pending", None
    return ReplicationEvidence(
        run,
        feature,
        status,
        replicates,
        full,
        halves,
        assignments,
        tolerance,
        horizon_complete,
        cutoff_step,
        coverage_policy,
    )


def _scan_first_alerts(
    signs: dict, monitor_grid: list[int], sustain: int, allowed_signs: tuple[int, ...]
) -> dict:
    """Track witnessed alerts and earlier alerts that missing rows could hide."""
    witnesses = {
        "first_possible_alert_step": None,
        "observed_alert_step": None,
        "observed_onset_step": None,
        "observed_direction": None,
    }
    possible_streaks = dict.fromkeys(allowed_signs, 0)
    actual_streak, previous_sign, onset = 0, 0, None
    for step in monitor_grid:
        sign = signs.get(step)  # None is unavailable, zero is observed nonqualifying.
        for candidate_sign in allowed_signs:
            possible_streaks[candidate_sign] = (
                possible_streaks[candidate_sign] + 1
                if sign is None or sign == candidate_sign
                else 0
            )
            if (
                possible_streaks[candidate_sign] >= sustain
                and witnesses["first_possible_alert_step"] is None
            ):
                witnesses["first_possible_alert_step"] = int(step)
        if sign is None or sign == 0:
            actual_streak, previous_sign = 0, 0
        elif sign == previous_sign:
            actual_streak += 1
        else:
            actual_streak, previous_sign, onset = 1, sign, step
        if actual_streak >= sustain and witnesses["observed_alert_step"] is None:
            witnesses.update(
                observed_alert_step=int(step),
                observed_onset_step=int(onset),
                observed_direction="up" if sign > 0 else "down",
            )

    return witnesses


def _contiguous_half(
    half, series, subset, required_steps, positions, parameters, horizon_complete
):
    observed = set(subset.step)
    missing = [int(step) for step in required_steps if step not in observed]
    prefix_end = positions[missing[0]] if missing else len(required_steps)
    prefix = subset.loc[subset.step.isin(required_steps[:prefix_end])]
    prepared = series.loc[series.step.isin(required_steps[:prefix_end])]
    result = detect_sequential(prepared, **parameters)
    diagnostic = result.diagnostics.iloc[0] if not result.diagnostics.empty else None
    source_status = diagnostic.status if diagnostic is not None else "no_observations"
    n_scored = int(diagnostic.n_scored) if diagnostic is not None else 0
    n_unscorable = int(diagnostic.n_unscorable) if diagnostic is not None else 0
    row = {
        "half": half,
        "status": "unavailable",
        "alert_step": None,
        "onset_step": None,
        "direction": None,
        "alert_position": None,
        "n_prompts": int(prefix.prompt_id.nunique()),
        "n_sampling_units": int(prefix.sampling_unit_id.nunique()),
        "n_rollouts": int(prefix.n_rollouts.sum()),
        "n_steps": int(prefix.step.nunique()),
        "n_scored": n_scored,
        "n_unscorable": n_unscorable,
        "missing_steps": missing,
        "reason": None,
        "source_detect_status": source_status,
    }
    for name in (
        "baseline_mean",
        "baseline_sd",
        "baseline_se_floor",
        "scale",
        "numerical_floor_dominant",
    ):
        row[name] = diagnostic[name] if diagnostic is not None else None
    if not result.alerts.empty:
        half_alert = result.alerts.iloc[0]
        # An unavailable-SE row before the observed alert could have completed
        # an earlier compatible streak, so the first alert is then unknown.
        resolved = _detector_parameters(parameters)
        monitor_grid = required_steps[resolved["baseline_steps"] : prefix_end]
        signs = {
            int(point.step): int(point.direction)
            for point in result.scores.loc[
                result.scores.step.isin(monitor_grid)
            ].itertuples()
            if np.isfinite(point.se)
        }
        allowed_signs = (1,) if series.kind.iloc[0] == "binary" else (-1, 1)
        possible = _scan_first_alerts(
            signs, monitor_grid, resolved["sustain"], allowed_signs
        )["first_possible_alert_step"]
        if possible != int(half_alert.alert_step):
            row["reason"] = "ambiguous_first_alert"
        else:
            row.update(
                status="alerted",
                alert_step=int(half_alert.alert_step),
                onset_step=int(half_alert.onset_step),
                direction=half_alert.direction,
                alert_position=positions[half_alert.alert_step],
                reason="first_alert_observed",
            )
    elif missing:
        row["reason"] = "missing_step_grid"
    elif source_status in (
        "insufficient_baseline",
        "insufficient_baseline_uncertainty",
        "no_observations",
    ):
        row["reason"] = source_status
    elif n_unscorable:
        row["reason"] = "unavailable_monitoring_uncertainty"
    elif not horizon_complete:
        row.update(status="pending", reason="incomplete_comparison_horizon")
    elif n_scored:
        row.update(status="no_alert", reason="no_alert_within_complete_horizon")
    else:
        row["reason"] = source_status
    return row
