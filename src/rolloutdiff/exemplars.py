"""Rollout references for sequential alerts."""

from __future__ import annotations

from fractions import Fraction
from numbers import Integral

import pandas as pd

from ._feature_values import _FeatureValueIndex
from .features import FeatureTables
from .schema import validate_rollouts
from .series import _require_columns, _validate_features

_COLUMNS = [
    "run",
    "feature",
    "kind",
    "direction",
    "onset_step",
    "alert_step",
    "status",
    "n_window_rollouts",
    "n_candidates",
    "reward_selection_status",
    "selection_order",
    "sample_id",
    "step",
    "prompt_id",
    "feature_value",
    "reward",
    "reward_group_size",
    "reward_group_mean",
    "reward_deviation",
    "reward_deviation_status",
    "selection_reason",
]
_ALERT_COLUMNS = ["run", "feature", "kind", "direction", "onset_step", "alert_step"]
_ORDER = ["step", "run", "sample_id"]


def _exemplar_table(rows=()) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=_COLUMNS)
    # Null status rows must not coerce exact integer identities through float64.
    for name in [
        "onset_step",
        "alert_step",
        "step",
        "selection_order",
        "n_window_rollouts",
        "n_candidates",
        "reward_group_size",
    ]:
        result[name] = pd.Series([row.get(name) for row in rows], dtype=object)
    for name in ["feature_value", "reward", "reward_group_mean", "reward_deviation"]:
        result[name] = pd.Series([row.get(name) for row in rows], dtype=float)
    return result


def select_exemplars(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    """Return distinct rollout references for rising binary sequential alerts.

    From onset through alert, select up to three positives: earliest, largest
    positive within-(run, step, prompt_id) reward deviation among remaining
    candidates, then prompt diversity. Ties use step and canonical identity.
    Response text is excluded. Unsupported/unavailable alerts yield status rows
    without sample_id; no alerts yield an empty table.
    """
    rollouts = validate_rollouts(rollouts)
    _validate_features(rollouts, features)
    return _select_exemplars_prepared(_FeatureValueIndex(rollouts, features), alerts)


def _reward_deviations(frame: pd.DataFrame) -> pd.DataFrame:
    """Center against all completions of the same prompt at the same step."""
    if frame.empty:
        raise ValueError("No objects to concatenate")
    result = frame.copy()
    sizes, means = [0] * len(frame), [0.0] * len(frame)
    deviations, statuses = [None] * len(frame), [None] * len(frame)
    reward_values = frame.reward.to_numpy()
    groups = frame.groupby(["run", "step", "prompt_id"], sort=True).indices
    for positions in groups.values():
        rewards = [
            Fraction.from_float(float(reward_values[position]))
            for position in positions
        ]
        # Center before rounding: large offsets must not erase small deviations.
        exact_mean = sum(rewards, Fraction()) / len(positions)
        group_mean = float(exact_mean)
        for position, reward in zip(positions, rewards, strict=True):
            sizes[position], means[position] = len(positions), group_mean
            try:
                deviation = float(reward - exact_mean)
            except OverflowError:
                deviation = None
            if len(positions) < 2:
                statuses[position] = "insufficient_group_size"
            elif deviation is None or (deviation == 0 and reward != exact_mean):
                statuses[position] = "unavailable_numeric"
            else:
                deviations[position] = deviation
                statuses[position] = "available"
    result["reward_group_size"] = sizes
    result["reward_group_mean"] = means
    result["reward_deviation"] = pd.Series(deviations, dtype=float).to_numpy()
    result["reward_deviation_status"] = statuses
    return result.sort_values(_ORDER).reset_index(drop=True)


def _choose(frame: pd.DataFrame, common: dict) -> list[dict]:
    candidates = frame.loc[frame._value.eq(1)].reset_index(drop=True)
    common["n_candidates"] = len(candidates)
    if candidates.empty:
        return [
            {
                **common,
                "status": "no_positive_rollouts",
                "reward_selection_status": "no_candidates",
            }
        ]

    chosen = [(0, "earliest_occurrence")]
    remaining = candidates.iloc[1:]
    comparable = remaining.loc[remaining.reward_deviation_status.eq("available")]
    positive = comparable.loc[comparable.reward_deviation.gt(0)]
    if remaining.empty:
        reward_status = "no_remaining_candidates"
    elif comparable.empty:
        reward_status = "no_comparable_candidates"
    elif positive.empty:
        reward_status = "no_positive_deviation"
    else:
        reward_status = "selected"
        index = positive.sort_values(
            ["reward_deviation", *_ORDER], ascending=[False, True, True, True]
        ).index[0]
        chosen.append((index, "high_within_prompt_reward_deviation"))

    while len(chosen) < min(3, len(candidates)):
        indices = [index for index, _ in chosen]
        remaining = candidates.drop(index=indices)
        new_prompts = remaining.loc[
            ~remaining.prompt_id.isin(candidates.loc[indices, "prompt_id"])
        ]
        if new_prompts.empty:
            chosen.append((remaining.index[0], "earliest_remaining"))
        else:
            chosen.append((new_prompts.index[0], "prompt_diversity"))

    rows = []
    for order, (index, reason) in enumerate(chosen, start=1):
        candidate = candidates.loc[index]
        rows.append(
            {
                **common,
                "status": "selected",
                "selection_order": order,
                "reward_selection_status": reward_status,
                **{
                    name: candidate[name]
                    for name in [
                        "sample_id",
                        "step",
                        "prompt_id",
                        "reward",
                        "reward_group_size",
                        "reward_group_mean",
                        "reward_deviation",
                        "reward_deviation_status",
                    ]
                },
                "feature_value": candidate["_value"],
                "selection_reason": reason,
            }
        )
    return rows


def _select_exemplars_prepared(
    values: _FeatureValueIndex, alerts: pd.DataFrame
) -> pd.DataFrame:
    _require_columns(alerts, _ALERT_COLUMNS, "sequential alerts")
    if alerts.duplicated(["run", "feature"]).any():
        raise ValueError("duplicate sequential alert for run/feature")
    for name in ["run", "feature"]:
        if not alerts[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"alert {name} must be a nonempty string")
    if not alerts.direction.isin(["up", "down"]).all():
        raise ValueError("alert direction must be up or down")
    if not alerts.kind.isin(["binary", "continuous"]).all():
        raise ValueError("alert kind must be binary or continuous")
    for name in ["onset_step", "alert_step"]:
        if (
            not alerts[name]
            .map(
                lambda v: isinstance(v, Integral) and not isinstance(v, bool) and v >= 0
            )
            .all()
        ):
            raise ValueError(f"{name} must be a nonnegative integer")
    if (alerts.onset_step > alerts.alert_step).any():
        raise ValueError("onset_step must not exceed alert_step")

    rows = []
    for alert in alerts.sort_values(["run", "feature"]).itertuples(index=False):
        if alert.run not in values.runs:
            raise ValueError("alert run has no rollouts")
        if (
            alert.feature != "scalar:reward"
            and alert.feature not in values.metadata.index
        ):
            raise ValueError("alert feature is not in the retained catalog")
        frame, kind = values.values(alert.run, alert.feature)
        if alert.kind != kind:
            raise ValueError("alert kind does not match the retained feature")
        window = frame.loc[frame.step.between(alert.onset_step, alert.alert_step)]
        common = {
            "run": alert.run,
            "feature": alert.feature,
            "kind": kind,
            "direction": alert.direction,
            "onset_step": int(alert.onset_step),
            "alert_step": int(alert.alert_step),
            "n_window_rollouts": len(window),
        }
        if kind != "binary":
            rows.append({**common, "status": "unsupported_kind"})
        elif alert.direction != "up":
            rows.append({**common, "status": "unsupported_direction"})
        elif not {alert.onset_step, alert.alert_step}.issubset(set(window.step)):
            rows.append({**common, "status": "unavailable_window"})
        else:
            ordered = window.sort_values(_ORDER).reset_index(drop=True)
            rows.extend(_choose(_reward_deviations(ordered), common))
    return _exemplar_table(rows)
