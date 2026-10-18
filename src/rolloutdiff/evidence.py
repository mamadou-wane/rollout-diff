"""Descriptive persistence and within-group reward evidence."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ._feature_values import feature_values
from ._series_validation import validate_series
from .features import FeatureTables
from .prompt_mix import _window


@dataclass(frozen=True)
class PersistenceEvidence:
    run: str
    feature: str
    after_step: int
    reference_mean: float
    change: float
    effect_fraction: float
    threshold: float
    direction: str
    status: str
    n_observations: int
    n_evaluable: int
    n_retained: int
    n_unavailable: int
    fraction_retained: float | None
    coverage: float | None
    steps: pd.DataFrame


@dataclass(frozen=True)
class RewardAssociation:
    run: str
    feature: str
    window: tuple[int, int]
    status: str
    mean_difference: float | None
    n_groups: int
    n_mixed_groups: int
    n_rollouts: int
    n_included_rollouts: int
    groups: pd.DataFrame


def persistence(
    series: pd.DataFrame,
    *,
    run: str,
    feature: str,
    after_step: int,
    reference_mean: float,
    change: float,
    effect_fraction: float = 0.5,
) -> PersistenceEvidence:
    """Measure signed effect retention strictly after an observed anchor.

    Retained means sign(change)*(mean-reference_mean) >= effect_fraction*abs(change).
    The returned threshold is an absolute mean cutoff. Fractions use evaluable
    logged observations, excluding unknown SE and reporting reduced coverage.
    No tail or no evaluable rows yields no fraction. Sequential callers use the
    alert step/mean/delta; retrospective callers must choose an anchor explicitly.
    This is descriptive, without a pass threshold or confidence interval.
    """
    frame = validate_series(series)
    frame = frame.loc[frame.run.eq(run) & frame.feature.eq(feature)].copy()
    return _persistence_prepared(
        frame,
        run=run,
        feature=feature,
        after_step=after_step,
        reference_mean=reference_mean,
        change=change,
        effect_fraction=effect_fraction,
    )


def _persistence_prepared(
    frame: pd.DataFrame,
    *,
    run: str,
    feature: str,
    after_step: int,
    reference_mean: float,
    change: float,
    effect_fraction: float = 0.5,
) -> PersistenceEvidence:
    """Measure persistence in one validated run/feature series."""
    if type(after_step) is not int or after_step < 0:
        raise ValueError("after_step must be a nonnegative integer")
    for name, value in [
        ("reference_mean", reference_mean),
        ("change", change),
        ("effect_fraction", effect_fraction),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{name} must be finite and numeric")
    if change == 0 or not 0 < effect_fraction <= 1:
        raise ValueError("change must be nonzero and effect_fraction in (0, 1]")
    if frame.empty:
        raise ValueError("run/feature has no series")
    if not frame.step.eq(after_step).any():
        raise ValueError("after_step must identify an observed step")
    binary = frame.kind.iloc[0] == "binary"
    with np.errstate(over="raise", invalid="raise"):
        try:
            endpoint = np.float64(reference_mean) + np.float64(change)
            threshold = np.float64(reference_mean) + effect_fraction * np.float64(
                change
            )
        except FloatingPointError:
            raise ValueError(
                "numerical overflow computing persistence threshold"
            ) from None
    if binary and (
        not 0 <= reference_mean <= 1
        or not -4 * np.finfo(float).eps <= endpoint <= 1 + 4 * np.finfo(float).eps
    ):
        raise ValueError("binary reference and full effect endpoint must be in [0, 1]")
    target = effect_fraction * abs(change)
    if target == 0:
        raise ValueError("retained effect target underflow")
    tail = frame.loc[frame.step.gt(after_step)].reset_index(drop=True)
    valid = tail.se.notna()
    retained = pd.Series(pd.NA, index=tail.index, dtype="boolean")
    sign = 1 if change > 0 else -1
    try:
        with np.errstate(over="raise", invalid="raise"):
            departure = sign * (tail.loc[valid, "mean"].to_numpy() - reference_mean)
            # Relative roundoff only: an absolute epsilon can exceed a tiny
            # positive target and falsely mark an unchanged value as retained.
            tolerance = 8 * np.finfo(float).eps * np.maximum(np.abs(departure), target)
            retained.loc[valid] = departure >= target - tolerance
    except FloatingPointError:
        raise ValueError("numerical overflow computing persistence") from None
    tail["retained"] = retained
    n, evaluable = len(tail), int(valid.sum())
    n_retained = int(retained.fillna(False).sum())
    status = (
        "no_future_observations"
        if not n
        else "unavailable_uncertainty"
        if not evaluable
        else "partial"
        if evaluable < n
        else "available"
    )
    return PersistenceEvidence(
        run,
        feature,
        after_step,
        float(reference_mean),
        float(change),
        float(effect_fraction),
        float(threshold),
        "up" if sign > 0 else "down",
        status,
        n,
        evaluable,
        n_retained,
        n - evaluable,
        n_retained / evaluable if evaluable else None,
        evaluable / n if n else None,
        tail,
    )


_GROUP_COLUMNS = [
    "run",
    "step",
    "prompt_id",
    "n_rollouts",
    "n_present",
    "n_absent",
    "reward_present",
    "reward_absent",
    "difference",
]


def reward_association(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    feature: str,
    window: tuple[int, int],
) -> RewardAssociation:
    """Average present-minus-absent reward differences over mixed prompt/step groups.

    Window bounds are inclusive. Mixed groups have equal weight regardless of
    completion count; pure groups remain in the detail table only. The caller
    chooses the window. Continuous features return ``not_binary``. This descriptive
    association has no causal interpretation or inferential SE.
    """
    window = _window(window)
    frame, kind = feature_values(rollouts, features, run=run, feature=feature)
    return _reward_association_prepared(
        frame, kind, run=run, feature=feature, window=window
    )


def _reward_association_prepared(
    frame: pd.DataFrame,
    kind: str,
    *,
    run: str,
    feature: str,
    window: tuple[int, int],
) -> RewardAssociation:
    """Compute reward association from identity-aligned, validated values."""
    window = _window(window)
    frame = frame.loc[frame.step.between(*window)].copy()
    if frame.empty:
        raise ValueError("reward association window is empty")
    keys = ["run", "step", "prompt_id"]
    n_groups = len(frame[keys].drop_duplicates())
    if kind != "binary":
        return RewardAssociation(
            run,
            feature,
            window,
            "not_binary",
            None,
            n_groups,
            0,
            len(frame),
            0,
            pd.DataFrame(columns=_GROUP_COLUMNS),
        )
    present = frame["_value"].eq(1)
    frame["_present_reward"] = frame.reward.where(present)
    frame["_absent_reward"] = frame.reward.where(~present)
    groups = (
        frame.groupby(keys, sort=True)
        .agg(
            n_rollouts=("sample_id", "size"),
            n_present=("_value", "sum"),
            reward_present=("_present_reward", "mean"),
            reward_absent=("_absent_reward", "mean"),
        )
        .reset_index()
    )
    groups["n_present"] = groups.n_present.astype("int64")
    groups["n_absent"] = groups.n_rollouts - groups.n_present
    for column, count in [
        ("reward_present", "n_present"),
        ("reward_absent", "n_absent"),
    ]:
        if not np.isfinite(groups.loc[groups[count].gt(0), column]).all():
            raise ValueError("numerical overflow computing within-group reward means")
    mixed = groups.n_present.gt(0) & groups.n_absent.gt(0)
    groups["difference"] = np.nan
    mean = None
    try:
        with np.errstate(over="raise", invalid="raise"):
            differences = (
                groups.loc[mixed, "reward_present"].to_numpy()
                - groups.loc[mixed, "reward_absent"].to_numpy()
            )
            if len(differences):
                mean = float(differences.mean())
                if not np.isfinite(mean):
                    raise FloatingPointError
                groups.loc[mixed, "difference"] = differences
    except FloatingPointError:
        raise ValueError("numerical overflow computing reward association") from None
    return RewardAssociation(
        run,
        feature,
        window,
        "available" if mixed.any() else "no_mixed_groups",
        mean,
        n_groups,
        int(mixed.sum()),
        len(frame),
        int(groups.loc[mixed, "n_rollouts"].sum()),
        groups[_GROUP_COLUMNS],
    )
