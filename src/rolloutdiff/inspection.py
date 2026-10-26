"""Resolve report exemplar references for explicitly requested local inspection."""

from __future__ import annotations

import pandas as pd

from ._report_validation import validate_report
from .evidence import _reward_association_prepared
from .features import DENSE_BINARY, extract
from .schema import validate_rollouts


def _unavailable_comparison(reason: str) -> dict:
    return {
        "status": "unavailable",
        "reason": reason,
        "n_rollouts": None,
        "n_present": None,
        "n_absent": None,
        "reward_present": None,
        "reward_absent": None,
        "difference": None,
        "counterpart": None,
    }


def _comparisons(event: dict, lookup: dict) -> None:
    examples = [
        example for member in event["members"] for example in member["exemplars"]
    ]
    groups = {
        (example["run"], example["step"], example["prompt_id"]): []
        for example in examples
        if example["status"] == "selected" and example["kind"] == "binary"
    }
    for row in lookup.values():
        key = row["run"], row["step"], row["prompt_id"]
        if key in groups:
            groups[key].append(row)
    extracted = {}
    for example in examples:
        if example["status"] != "selected" or example["kind"] != "binary":
            reason = (
                example["status"]
                if example["status"] != "selected"
                else "unsupported_kind"
            )
            example["comparison"] = _unavailable_comparison(reason)
            continue
        group = sorted(
            groups[(example["run"], example["step"], example["prompt_id"])],
            key=lambda row: row["sample_id"],
        )
        expected_size = example["reward_group_size"]
        if expected_size is not None and len(group) != expected_size:
            raise ValueError("selected exemplar group size does not match rollouts")
        feature, values = example["feature"], []
        for row in group:
            key = row["run"], row["sample_id"]
            if key not in extracted:
                extracted[key] = extract(row["response"])
            features = extracted[key]
            if feature == "defines_test_named":
                # Historical reports use substring matching on parsed definitions.
                value = float(
                    any(
                        name.startswith("def:") and "test" in name[4:].lower()
                        for name in features.sparse
                    )
                )
            elif feature in features.dense:
                if feature not in DENSE_BINARY:
                    raise ValueError(
                        "reported binary feature is not binary in extractor"
                    )
                value = features.dense[feature]
            else:
                value = float(feature in features.sparse)
            if row["sample_id"] == example["sample_id"] and (
                example["feature_value"] != 1 or value != 1
            ):
                raise ValueError(
                    "selected exemplar feature does not match extracted presence"
                )
            values.append(value)
        frame = pd.DataFrame(
            group, columns=["run", "step", "prompt_id", "sample_id", "reward"]
        )
        frame["_value"] = values
        association = _reward_association_prepared(
            frame,
            "binary",
            run=example["run"],
            feature=feature,
            window=(example["step"], example["step"]),
        )
        stats = association.groups.to_dict("records")[0]
        comparison = _unavailable_comparison("no_absent_completion")
        for name in (
            "n_rollouts",
            "n_present",
            "n_absent",
            "reward_present",
            "reward_absent",
            "difference",
        ):
            comparison[name] = None if pd.isna(stats[name]) else stats[name]
        absent = next(
            (row for row, value in zip(group, values, strict=True) if value == 0), None
        )
        if absent is not None:
            comparison.update(
                status="available",
                reason=None,
                counterpart={
                    **{
                        name: absent[name]
                        for name in (
                            "run",
                            "sample_id",
                            "step",
                            "prompt_id",
                            "reward",
                            "response",
                        )
                    },
                    "feature_value": 0.0,
                },
            )
        example["comparison"] = comparison


def resolve_exemplars(
    report: dict,
    rollouts: pd.DataFrame,
    *,
    run: str,
    event_rank: int,
    with_absent: bool = False,
) -> dict:
    """Return an event copy with each exemplar's response, or None if unselected.

    Resolve exact (run, sample_id) keys and require identical step, prompt_id,
    and reward. Missing references, duplicates, or mismatches raise ValueError.
    Members and exemplars follow report ordering rules; missing lists and
    unsupported statuses remain intact. With with_absent, verify extracted
    binary presence and add comparison counts/means over complete prompt-step
    groups, choosing the absent counterpart by lexically smallest sample ID. Supply all
    group completions; saved group sizes are checked when available. Comparisons
    are descriptive. Inputs are unchanged; version 1 has no response fingerprint.
    Historical defines_test_named uses substring matching for the entire group.
    """
    if type(with_absent) is not bool:
        raise ValueError("with_absent must be a boolean")
    if not isinstance(run, str) or not run.strip():
        raise ValueError("run must be a nonempty string")
    if type(event_rank) is not int or event_rank < 1:
        raise ValueError("event_rank must be a positive integer")
    report = validate_report(report)
    selected_run = next((item for item in report["runs"] if item["run"] == run), None)
    if selected_run is None:
        raise ValueError("requested run is not present in report")
    event = next(
        (item for item in selected_run["events"] if item["rank"] == event_rank), None
    )
    if event is None:
        raise ValueError("requested event rank is not present in report run")
    if not isinstance(rollouts, pd.DataFrame):
        raise ValueError("rollouts must be a DataFrame")
    canonical = validate_rollouts(rollouts)
    # Python scalar comparisons avoid NumPy rounding a large integer report reward.
    lookup = {
        (row["run"], row["sample_id"]): row for row in canonical.to_dict("records")
    }
    event["members"].sort(
        key=lambda member: (
            member["onset_step"],
            member["alert_step"],
            member["feature"],
        )
    )
    for member in event["members"]:
        member["exemplars"].sort(
            key=lambda example: (
                example["selection_order"] is None,
                example["selection_order"] or 0,
                example["sample_id"] or "",
            )
        )
        for example in member["exemplars"]:
            example["response"] = None
            if example["status"] != "selected":
                continue
            if example["reward"] is None:
                raise ValueError("selected exemplar reward is unavailable")
            rollout = lookup.get((example["run"], example["sample_id"]))
            if rollout is None:
                raise ValueError("selected exemplar reference is missing from rollouts")
            for field in ("step", "prompt_id", "reward"):
                if example[field] != rollout[field]:
                    raise ValueError(
                        f"selected exemplar {field} does not match rollout"
                    )
            example["response"] = rollout["response"]
    if with_absent:
        _comparisons(event, lookup)
    return event
