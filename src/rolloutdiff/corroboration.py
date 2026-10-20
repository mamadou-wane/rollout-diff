"""Bounded event-window evidence for a sequential alert in sampling-unit halves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .detect import DetectionResult
from .features import FeatureTables
from .replication import _detector_parameters, _fixed_baseline_scores, _prepare_halves

_HALF_COLUMNS = [
    "half",
    "status",
    "reason",
    "n_prompts",
    "n_sampling_units",
    "n_rollouts",
    "baseline_expected",
    "baseline_required",
    "baseline_used",
    "baseline_used_steps",
    "baseline_excluded_steps",
    "baseline_mean",
    "baseline_sd",
    "baseline_se_floor",
    "scale",
    "numerical_floor_dominant",
    "window_expected",
    "window_observed",
    "window_available",
    "n_scored",
    "n_unscorable",
    "future_steps",
    "missing_steps",
    "unavailable_se_steps",
    "match_start_step",
    "match_end_step",
    "opposite_start_step",
    "opposite_end_step",
    "mixed_direction",
]


@dataclass(frozen=True)
class EventWindowEvidence:
    run: str
    feature: str
    status: str
    corroborates: bool | None
    reason: str
    full: DetectionResult
    halves: pd.DataFrame
    assignments: pd.DataFrame
    radius: int
    window_start_step: int | None
    window_end_step: int | None
    horizon_complete: bool


def _validate_radius(radius: int) -> None:
    if type(radius) is not int or radius < 0:
        raise ValueError("radius must be a nonnegative integer")


def event_window_corroboration(
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
    radius: int = 10,
    sampling_units: pd.DataFrame | None = None,
) -> EventWindowEvidence:
    """Find same-direction sustained witnesses near the full sequential alert.

    Halves use fixed-window replication rules within ``radius`` full-grid
    positions of the alert, after the baseline; streaks cannot span missing
    rows or SE. A matching witness in each half corroborates. A half contradicts
    only when its window is complete, fully scorable, and long enough for a
    streak, and one contradicting half decides the result; anything else is
    unavailable.
    """
    _validate_radius(radius)
    parameters = {
        "baseline_steps": baseline_steps,
        "z_threshold": z_threshold,
        "sustain": sustain,
        "min_prevalence_change": min_prevalence_change,
        "numerical_floor": numerical_floor,
    }
    full, series, schedules, assignments = _prepare_halves(
        rollouts,
        features,
        run=run,
        feature=feature,
        sampling_units=sampling_units,
        parameters=parameters,
    )
    return _corroboration_prepared(
        full,
        series,
        schedules,
        assignments,
        run=run,
        feature=feature,
        radius=radius,
        parameters=parameters,
    )


def _corroboration_prepared(
    full, half_series, half_schedules, assignments, *, run, feature, radius, parameters
) -> EventWindowEvidence:
    _validate_radius(radius)
    if full.alerts.empty:
        reason = (
            "no_full_alert" if full.diagnostics.n_scored.sum() else "full_unscorable"
        )
        return EventWindowEvidence(
            run,
            feature,
            "unavailable",
            None,
            reason,
            full,
            pd.DataFrame(columns=_HALF_COLUMNS),
            assignments,
            radius,
            None,
            None,
            False,
        )
    resolved = _detector_parameters(parameters)
    steps = full.scores.step.tolist()
    alert = full.alerts.iloc[0]
    position = steps.index(alert.alert_step)
    start = max(resolved["baseline_steps"], position - radius)
    end = position + radius
    window = steps[start : end + 1]
    expected = end - start + 1
    horizon_complete = end < len(steps)
    direction = 1 if alert.direction == "up" else -1
    rows = []
    for half in (0, 1):
        rows.append(
            _corroborate_half(
                half,
                half_series[half],
                half_schedules[half],
                steps,
                window,
                expected,
                resolved,
                direction,
                horizon_complete,
            )
        )
    halves = pd.DataFrame(rows, columns=_HALF_COLUMNS)
    for name in (
        "match_start_step",
        "match_end_step",
        "opposite_start_step",
        "opposite_end_step",
    ):
        halves[name] = pd.Series([row[name] for row in rows], dtype=object)
    if halves.status.eq("contradicted").any():
        status, value, reason = (
            "contradicted",
            False,
            "complete_half_without_matching_streak",
        )
    elif halves.status.eq("corroborated").all():
        status, value, reason = (
            "corroborated",
            True,
            "both_halves_have_matching_streaks",
        )
    else:
        status, value, reason = "unavailable", None, "insufficient_half_evidence"
    return EventWindowEvidence(
        run,
        feature,
        status,
        value,
        reason,
        full,
        halves,
        assignments,
        radius,
        int(window[0]),
        int(window[-1]),
        horizon_complete,
    )


def _window_streaks(
    signs: dict, window: list[int], sustain: int, direction: int
) -> dict:
    witnesses = {
        "match_start_step": None,
        "match_end_step": None,
        "opposite_start_step": None,
        "opposite_end_step": None,
    }
    streak, previous = 0, 0
    for index, step in enumerate(window):
        sign = signs.get(step, 0)
        streak = streak + 1 if sign and sign == previous else int(bool(sign))
        previous = sign
        if streak == sustain:
            prefix = "match" if sign == direction else "opposite"
            if witnesses[prefix + "_end_step"] is None:
                witnesses[prefix + "_start_step"] = int(window[index - streak + 1])
                witnesses[prefix + "_end_step"] = int(step)
    return witnesses


def _corroborate_half(
    half,
    series,
    schedule,
    steps,
    window,
    expected,
    resolved,
    direction,
    horizon_complete,
):
    coverage, result = _fixed_baseline_scores(
        series, steps[: resolved["baseline_steps"]] + window, resolved
    )
    schedule = schedule.loc[schedule.step.isin(window)]
    observed = set(schedule.step)
    row = {name: coverage[name] for name in _HALF_COLUMNS if name in coverage}
    row.update(
        half=half,
        status="unavailable",
        reason="insufficient_fixed_baseline",
        n_prompts=int(schedule.prompt_id.nunique()),
        n_sampling_units=int(schedule.sampling_unit_id.nunique()),
        n_rollouts=int(schedule.n_rollouts.sum()),
        window_expected=expected,
        window_observed=len(observed),
        window_available=coverage["monitoring_available"],
        n_unscorable=expected - coverage["n_scored"],
        future_steps=expected - len(window),
        missing_steps=[int(step) for step in window if step not in observed],
        unavailable_se_steps=[
            step for step in coverage["unavailable_se_steps"] if step in window
        ],
        match_start_step=None,
        match_end_step=None,
        opposite_start_step=None,
        opposite_end_step=None,
        mixed_direction=False,
    )
    if result is not None:
        # Use finite-SE point directions on the full event grid; compressed
        # detector streaks and alerts do not preserve gaps.
        signs = {
            int(point.step): int(point.direction)
            for point in result.scores.loc[result.scores.step.isin(window)].itertuples()
            if np.isfinite(point.se)
        }
        row.update(_window_streaks(signs, window, resolved["sustain"], direction))
        row["mixed_direction"] = (
            row["match_end_step"] is not None and row["opposite_end_step"] is not None
        )
        if row["match_end_step"] is not None:
            row.update(status="corroborated", reason="observed_matching_streak")
        elif expected < resolved["sustain"]:
            row["reason"] = "insufficient_window_observations"
        elif horizon_complete and row["n_scored"] == expected:
            row.update(
                status="contradicted", reason="no_matching_streak_in_complete_window"
            )
        else:
            row["reason"] = "incomplete_window_evidence"
    return row
