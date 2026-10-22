"""Within-run inspection order from existing feature evidence."""

from __future__ import annotations

import math
from fractions import Fraction
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .events import _EVENT_COLUMNS, _MEMBER_COLUMNS
from .series import _require_columns

_EVIDENCE_FIELDS = [
    "tier",
    "kind",
    "sequential_status",
    "onset_step",
    "alert_step",
    "scale_source",
    "numerical_floor_dominant",
    "replication_status",
    "replicates",
    "replication_coverage_policy",
    "replication_horizon_complete",
    "persistence_status",
    "persistence_fraction",
    "persistence_coverage",
    "persistence_n_observations",
    "persistence_n_evaluable",
    "persistence_n_retained",
    "persistence_n_unavailable",
    "reward_status",
    "reward_mean_difference",
    "reward_n_groups",
    "reward_n_mixed_groups",
    "reward_n_rollouts",
    "reward_n_included_rollouts",
    "promptmix_status",
    "prompt_positive_prompts",
    "prompt_total_prompts",
]
_COLUMNS = [
    *_EVENT_COLUMNS,
    "rank",
    "representative_feature",
    "direction",
    "member_status",
    "ranking_status",
    "n_prioritizable_members",
    *_EVIDENCE_FIELDS,
    "replication_availability",
    "persistence_availability",
    "reward_availability",
    "breadth_availability",
    "prompt_positive_fraction",
    "missing_evidence_members",
    "numerical_floor_members",
    "numerical_floor_unavailable_members",
]


def _ranking_table(rows=()) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=_COLUMNS, dtype=object)
    for name in ["rank", "n_members", "n_prioritizable_members"]:
        result[name] = result[name].astype("int64")
    for name in [
        "replicates",
        "replication_horizon_complete",
        "numerical_floor_dominant",
    ]:
        result[name] = pd.array(result[name], dtype="boolean")
    for name in [
        "persistence_fraction",
        "persistence_coverage",
        "reward_mean_difference",
        "prompt_positive_fraction",
    ]:
        result[name] = pd.to_numeric(result[name]).astype(float)
    return result


def _validate_inputs(events, members, evidence) -> dict:
    for frame, columns, keys, name in [
        (events, _EVENT_COLUMNS, ["event_id"], "events"),
        (members, _MEMBER_COLUMNS, ["run", "feature"], "event members"),
        (
            evidence,
            ["run", "feature", *_EVIDENCE_FIELDS],
            ["run", "feature"],
            "evidence",
        ),
    ]:
        _require_columns(frame, columns, name)
        if frame.duplicated(keys).any():
            raise ValueError(f"duplicate {name} keys")
        for column in set(keys) | ({"run", "event_id"} & set(columns)):
            if (
                not frame[column]
                .map(lambda v: isinstance(v, str) and bool(v.strip()))
                .all()
            ):
                raise ValueError(f"{name} {column} must be a nonempty string")
    if (
        not members.kind.isin(["binary", "continuous"]).all()
        or not members.direction.isin(["up", "down"]).all()
    ):
        raise ValueError("invalid member kind or direction")
    for name in ["onset_step", "alert_step"]:
        if (
            not members[name]
            .map(
                lambda v: isinstance(v, Integral) and not isinstance(v, bool) and v >= 0
            )
            .all()
        ):
            raise ValueError(f"member {name} must be a nonnegative integer")
    if (members.onset_step > members.alert_step).any():
        raise ValueError("member onset exceeds alert step")
    groups = {
        key: frame.sort_values(["onset_step", "alert_step", "feature"]).to_dict(
            "records"
        )
        for key, frame in members.groupby("event_id", sort=False)
    }
    if set(groups) != set(events.event_id):
        raise ValueError("events and memberships must cover the same event IDs")
    for event in events.to_dict("records"):
        group = groups[event["event_id"]]
        refs = tuple((row["run"], row["feature"]) for row in group)
        if (
            event["n_members"] != len(group)
            or any(row["run"] != event["run"] for row in group)
            or event["onset_start"] != min(row["onset_step"] for row in group)
            or event["onset_end"] != max(row["onset_step"] for row in group)
            or not isinstance(event["members"], tuple)
            or set(event["members"]) != set(refs)
            or len(event["members"]) != len(refs)
        ):
            raise ValueError("event metadata conflicts with its membership")
    return groups


def _finite(value) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
    )


def _count(value) -> bool:
    return _finite(value) and value >= 0 and int(value) == value


def _validate_persistence(row: dict) -> None:
    status = row["persistence_status"]
    if status not in ("available", "partial"):
        return
    counts = {}
    for name in ["retained", "evaluable", "unavailable", "observations"]:
        value = row[f"persistence_n_{name}"]
        if pd.isna(value):
            counts[name] = None
        elif not _count(value):
            raise ValueError(f"persistence_n_{name} must be a nonnegative integer")
        else:
            counts[name] = int(value)
    retained, evaluable, unavailable, observed = (
        counts[name]
        for name in ["retained", "evaluable", "unavailable", "observations"]
    )
    if evaluable == 0 or observed == 0:
        raise ValueError(
            "observed persistence requires positive evaluable and total counts"
        )
    for left, right in [
        (retained, evaluable),
        (evaluable, observed),
        (retained, observed),
    ]:
        if left is not None and right is not None and left > right:
            raise ValueError("inconsistent persistence observation counts")
    if (
        unavailable is not None
        and observed is not None
        and (
            unavailable >= observed
            or (retained is not None and retained + unavailable > observed)
        )
    ):
        raise ValueError("inconsistent persistence unavailable observation count")
    expected_unavailable = unavailable
    if evaluable is not None and observed is not None:
        expected_unavailable = observed - evaluable
        if unavailable is not None and unavailable != expected_unavailable:
            raise ValueError(
                "persistence evaluable and unavailable counts must sum to observations"
            )
    if expected_unavailable is not None and (status == "available") != (
        expected_unavailable == 0
    ):
        raise ValueError("persistence status conflicts with observation counts")
    for name, numerator, denominator in [
        ("fraction", retained, evaluable),
        ("coverage", evaluable, observed),
    ]:
        value = row[f"persistence_{name}"]
        if pd.isna(value):
            continue
        if not _finite(value):
            raise ValueError(f"persistence_{name} must be finite")
        # Relative roundoff tolerance must not equate tiny positive ratios with zero.
        if (
            numerator is not None
            and denominator is not None
            and not math.isclose(
                value,
                numerator / denominator,
                rel_tol=8 * np.finfo(float).eps,
                abs_tol=0.0,
            )
        ):
            raise ValueError(f"persistence_{name} conflicts with observation counts")


def _descending(available: bool, *values) -> tuple:
    return (0, *(-value for value in values)) if available else (1,)


def _candidate(member: dict, source: dict | None) -> tuple[tuple, dict]:
    row = {
        name: source[name] if source is not None else None for name in _EVIDENCE_FIELDS
    }
    if source is not None:
        for name in ["kind", "onset_step", "alert_step"]:
            if pd.isna(source[name]) or source[name] != member[name]:
                raise ValueError("member metadata conflicts with feature evidence")
        if source["sequential_status"] != "alerted":
            raise ValueError("event member evidence must describe a sequential alert")
        if source["tier"] not in ("structural", "prose", None) and not pd.isna(
            source["tier"]
        ):
            raise ValueError("invalid feature tier")
    row.update(
        representative_feature=member["feature"],
        direction=member["direction"],
        member_status=member["status"],
        kind=member["kind"],
        onset_step=member["onset_step"],
        alert_step=member["alert_step"],
    )

    status, replicated = row["replication_status"], row["replicates"]
    known = isinstance(replicated, (bool, np.bool_))
    if known and (
        (status == "replicated" and not replicated)
        or (status == "not_replicated" and replicated)
    ):
        raise ValueError("replication status conflicts with replicates")
    replication = known and status in ("replicated", "not_replicated")
    _validate_persistence(row)
    fraction, coverage = row["persistence_fraction"], row["persistence_coverage"]
    evaluable, observed = (
        row["persistence_n_evaluable"],
        row["persistence_n_observations"],
    )
    persistence = (
        row["persistence_status"] in ("available", "partial")
        and _finite(fraction)
        and 0 <= fraction <= 1
        and _finite(coverage)
        and 0 < coverage <= 1
        and _count(evaluable)
        and _count(observed)
        and 0 < evaluable <= observed
    )
    difference, mixed = row["reward_mean_difference"], row["reward_n_mixed_groups"]
    included, total = row["reward_n_included_rollouts"], row["reward_n_rollouts"]
    groups = row["reward_n_groups"]
    reward = (
        row["reward_status"] == "available"
        and _finite(difference)
        and _count(mixed)
        and _count(groups)
        and 0 < mixed <= groups
        and _count(included)
        and _count(total)
        and 0 < included <= total
    )
    positive, prompts = row["prompt_positive_prompts"], row["prompt_total_prompts"]
    # Shared-prompt comparison status does not describe after-window breadth.
    breadth = (
        member["kind"] == "binary"
        and _count(positive)
        and _count(prompts)
        and 0 <= positive <= prompts
        and prompts > 0
    )
    for name, available in [
        ("replication", replication),
        ("persistence", persistence),
        ("reward", reward),
        ("breadth", breadth),
    ]:
        row[f"{name}_availability"] = "available" if available else "unavailable"
    breadth_fraction = Fraction(int(positive), int(prompts)) if breadth else None
    row["prompt_positive_fraction"] = float(breadth_fraction) if breadth else None
    if member["kind"] != "binary":
        ranking_status = "contextual"
    elif member["direction"] != "up":
        ranking_status = "unsupported_direction"
    elif member["status"] not in ("grouped", "singleton_no_match"):
        ranking_status = "unavailable_member"
    elif row["tier"] not in ("structural", "prose") or not any(
        [replication, persistence, reward, breadth]
    ):
        ranking_status = "unavailable_evidence"
    else:
        ranking_status = "behavioral"
    row["ranking_status"] = ranking_status
    tie = (member["onset_step"], member["alert_step"], member["feature"])
    if ranking_status != "behavioral":
        return (1, *tie), row
    return (
        0,
        0 if row["tier"] == "structural" else 1,
        0 if replication and replicated else 1 if replication else 2,
        _descending(persistence, fraction, coverage, evaluable),
        _descending(reward, difference, mixed, included),
        _descending(breadth, breadth_fraction, positive),
        *tie,
    ), row


def rank_events(
    events: pd.DataFrame, event_members: pd.DataFrame, evidence: pd.DataFrame
) -> pd.DataFrame:
    """Order each run using the best single rising-binary member's evidence.

    Lexicographic priority: structural before prose; replicated, not replicated,
    then unavailable; persistence fraction, coverage, evaluable count; signed
    reward difference, mixed groups, included rollouts; positive-prompt fraction,
    then positive count. Numeric components descend, with observed values before
    unavailable values. Ties use onset, alert, feature, then event ID.

    Contextual/unsupported events and members lacking all support follow in
    onset/alert/feature/ID order. Ranking is a heuristic inspection order.
    Floor warnings remain visible.
    """
    groups = _validate_inputs(events, event_members, evidence)
    sources = {(row["run"], row["feature"]): row for row in evidence.to_dict("records")}
    rows = []
    for event in events.to_dict("records"):
        members = groups[event["event_id"]]
        candidates, missing, floor, unknown_floor = [], [], [], []
        for member in members:
            reference = (member["run"], member["feature"])
            source = sources.get(reference)
            candidates.append(_candidate(member, source))
            if source is None:
                missing.append(reference)
            warning = source["numerical_floor_dominant"] if source is not None else None
            if not isinstance(warning, (bool, np.bool_)):
                unknown_floor.append(reference)
            elif warning:
                floor.append(reference)
        key, representative = min(candidates, key=lambda item: item[0])
        rows.append(
            {
                **event,
                **representative,
                "_key": key,
                "n_prioritizable_members": sum(
                    row["ranking_status"] == "behavioral" for _, row in candidates
                ),
                "missing_evidence_members": tuple(missing),
                "numerical_floor_members": tuple(floor),
                "numerical_floor_unavailable_members": tuple(unknown_floor),
            }
        )
    rows.sort(key=lambda row: (row["run"], row["_key"], row["event_id"]))
    previous, rank = None, 0
    for row in rows:
        rank = rank + 1 if row["run"] == previous else 1
        previous, row["rank"] = row["run"], rank
    return _ranking_table(rows)
