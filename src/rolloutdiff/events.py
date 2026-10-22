"""Heuristic associations between sequential feature alerts."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from ._feature_values import _FeatureValueIndex
from .features import FeatureTables
from .schema import validate_rollouts
from .series import _require_columns, _validate_features

_EVENT_COLUMNS = ["event_id", "run", "onset_start", "onset_end", "n_members", "members"]
_ALERT_COLUMNS = ["run", "feature", "kind", "direction", "onset_step", "alert_step"]
_MEMBER_COLUMNS = ["event_id", *_ALERT_COLUMNS, "status"]


def _event_table(rows=()) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=_EVENT_COLUMNS, dtype=object)
    result["n_members"] = result.n_members.astype("int64")
    return result


def _member_table(rows=()) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=_MEMBER_COLUMNS, dtype=object)


@dataclass(frozen=True)
class EventGrouping:
    events: pd.DataFrame
    members: pd.DataFrame


def group_events(
    rollouts: pd.DataFrame, features: FeatureTables, alerts: pd.DataFrame
) -> EventGrouping:
    """Group rising binary alerts by deterministic first fit within each run.

    Process (onset_step, alert_step, feature) order; join the first created
    event whose every member matches. Pairs require numeric onset difference
    <= 3 and positive-rollout Jaccard >= 0.5 over observed rollouts in the
    inclusive window from the earlier onset to the later alert. Sparse
    absence is zero.

    Unsupported alerts, missing anchors, or no positives in an alert's own
    onset-to-alert window yield singletons. Event IDs are analysis-local;
    grouping does not establish a shared cause.
    """
    rollouts = validate_rollouts(rollouts)
    _validate_features(rollouts, features)
    return _group_events_prepared(_FeatureValueIndex(rollouts, features), alerts)


def _validate_alerts(values: _FeatureValueIndex, alerts: pd.DataFrame) -> None:
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
    for alert in alerts.itertuples(index=False):
        if alert.run not in values.runs:
            raise ValueError("alert run has no rollouts")
        if alert.feature == "scalar:reward":
            kind = "continuous"
        elif alert.feature in values.metadata.index:
            kind = values.metadata.at[alert.feature, "kind"]
        else:
            raise ValueError("alert feature is not in the retained catalog")
        if alert.kind != kind:
            raise ValueError("alert kind does not match the retained feature")


def _matches(left: dict, right: dict, steps: np.ndarray, masks: dict) -> bool:
    if abs(left["onset_step"] - right["onset_step"]) > 3:
        return False
    start = min(left["onset_step"], right["onset_step"])
    end = max(left["alert_step"], right["alert_step"])
    low, high = steps.searchsorted(start), steps.searchsorted(end, side="right")
    a, b = masks[left["feature"]][low:high], masks[right["feature"]][low:high]
    union = np.count_nonzero(a | b)
    return bool(union and 2 * np.count_nonzero(a & b) >= union)


def _group_run(values: _FeatureValueIndex, run: str, alerts: pd.DataFrame) -> list:
    frame = values.runs[run]
    order = frame.sort_values(["step", "sample_id"]).index.to_numpy()
    steps = frame.step.to_numpy()[order]
    observed = set(steps)
    groups, masks = [], {}
    for alert in alerts[_ALERT_COLUMNS].to_dict("records"):
        status = "eligible"
        if alert["kind"] != "binary":
            status = "unsupported_kind"
        elif alert["direction"] != "up":
            status = "unsupported_direction"
        elif not {alert["onset_step"], alert["alert_step"]}.issubset(observed):
            status = "unavailable_window"
        else:
            aligned, _ = values.values(run, alert["feature"])
            # Every mask uses the same validated (run, sample_id) order.
            present = aligned._value.to_numpy()[order] == 1
            low = steps.searchsorted(alert["onset_step"])
            high = steps.searchsorted(alert["alert_step"], side="right")
            if present[low:high].any():
                masks[alert["feature"]] = present
            else:
                status = "no_positive_rollouts"
        alert["status"] = status
        if status == "eligible":
            for members in groups:
                if members[0]["status"] == "eligible" and all(
                    _matches(alert, member, steps, masks) for member in members
                ):
                    members.append(alert)
                    break
            else:
                groups.append([alert])
        else:
            groups.append([alert])
    return groups


def _group_events_prepared(
    values: _FeatureValueIndex, alerts: pd.DataFrame
) -> EventGrouping:
    _validate_alerts(values, alerts)
    ordered = alerts.sort_values(["run", "onset_step", "alert_step", "feature"])
    events, members = [], []
    for run, run_alerts in ordered.groupby("run", sort=True):
        for group in _group_run(values, run, run_alerts):
            event_id = f"event-{len(events) + 1:06d}"
            events.append(
                {
                    "event_id": event_id,
                    "run": run,
                    "onset_start": min(row["onset_step"] for row in group),
                    "onset_end": max(row["onset_step"] for row in group),
                    "n_members": len(group),
                    "members": tuple((run, row["feature"]) for row in group),
                }
            )
            for row in group:
                members.append(
                    {
                        **row,
                        "event_id": event_id,
                        "status": (
                            "grouped" if len(group) > 1 else "singleton_no_match"
                        )
                        if row["status"] == "eligible"
                        else row["status"],
                    }
                )
    return EventGrouping(_event_table(events), _member_table(members))
