"""Resolve report exemplar references for explicitly requested local inspection."""

from __future__ import annotations

import pandas as pd

from ._report_validation import validate_report
from .schema import validate_rollouts


def resolve_exemplars(
    report: dict,
    rollouts: pd.DataFrame,
    *,
    run: str,
    event_rank: int,
) -> dict:
    """Return an event copy with each exemplar's response, or None if unselected.

    Resolve exact (run, sample_id) keys and require identical step, prompt_id,
    and reward. Missing references, duplicates, or mismatches raise ValueError.
    Members and exemplars follow report ordering rules; missing lists and
    unsupported statuses remain intact. Inputs are unchanged. Version 1 has no
    response fingerprint, so metadata checks cannot establish text integrity.
    """
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
    return event
