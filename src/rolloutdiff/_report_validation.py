"""Read the version-1 report without recomputing its evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError


def _nonempty(value: str) -> str:
    if not value.strip():
        raise ValueError("identifier must be nonempty")
    return value


Identifier = Annotated[str, AfterValidator(_nonempty)]
Count = Annotated[int, Field(ge=0)]
Number = int | float


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)


class _Reference(_Model):
    run: Identifier
    feature: Identifier


class _Replication(_Model):
    status: str | None
    coverage_policy: str | None
    horizon_complete: bool | None
    cutoff_step: Count | None
    replicates: bool | None


class _Persistence(_Model):
    status: str | None
    fraction: Number | None
    coverage: Number | None
    n_observations: Count | None
    n_evaluable: Count | None
    n_retained: Count | None
    n_unavailable: Count | None


class _Reward(_Model):
    status: str | None
    mean_difference: Number | None
    n_groups: Count | None
    n_mixed_groups: Count | None
    n_rollouts: Count | None
    n_included_rollouts: Count | None
    window_start: Count | None
    window_end: Count | None


class _Breadth(_Model):
    positive_prompts: Count | None
    total_prompts: Count | None
    promptmix_status: str | None


class _Evidence(_Model):
    status: str
    tier: str | None
    scale_source: str | None
    numerical_floor_dominant: bool | None
    replication: _Replication
    persistence: _Persistence
    reward_association: _Reward
    prompt_breadth: _Breadth


class _Exemplar(_Reference):
    kind: str
    direction: str
    onset_step: Count
    alert_step: Count
    status: str
    n_window_rollouts: Count | None
    n_candidates: Count | None
    reward_selection_status: str | None
    selection_order: Count | None
    sample_id: Identifier | None
    step: Count | None
    prompt_id: Identifier | None
    feature_value: Number | None
    reward: Number | None
    reward_group_size: Count | None
    reward_group_mean: Number | None
    reward_deviation: Number | None
    reward_deviation_status: str | None
    selection_reason: str | None


class _Member(_Reference):
    kind: str
    direction: str
    onset_step: Count
    alert_step: Count
    status: str
    evidence: _Evidence
    exemplar_status: str
    exemplars: list[_Exemplar]


class _Ranking(_Model):
    n_prioritizable_members: Count
    replication_availability: str
    persistence_availability: str
    reward_availability: str
    breadth_availability: str
    missing_evidence_members: list[_Reference]
    numerical_floor_members: list[_Reference]
    numerical_floor_unavailable_members: list[_Reference]


class _Event(_Model):
    event_id: Identifier
    run: Identifier
    rank: Annotated[int, Field(ge=1)]
    representative_feature: Identifier
    ranking_status: str
    onset_start: Count
    onset_end: Count
    onset_step: Count
    alert_step: Count
    n_members: Annotated[int, Field(ge=1)]
    ranking: _Ranking
    members: list[_Member]


class _Counts(_Model):
    series: Count
    screened_series: Count
    sequential_alerts: Count
    unscorable_series: Count
    rollouts: Count
    observed_steps: Count
    prompts: Count
    events: Count
    behavioral_events: Count
    contextual_events: Count
    other_events: Count


class _Run(_Model):
    run: Identifier
    status: str
    counts: _Counts
    events: list[_Event]


class _Report(_Model):
    schema_version: int
    runs: list[_Run]


def _check_member(member: _Member) -> None:
    if member.onset_step > member.alert_step:
        raise ValueError("report member onset exceeds its alert step")
    orders, selected = set(), set()
    for example in member.exemplars:
        if any(
            getattr(example, field) != getattr(member, field)
            for field in [
                "run",
                "feature",
                "kind",
                "direction",
                "onset_step",
                "alert_step",
            ]
        ):
            raise ValueError("report exemplar does not match its member")
        if example.selection_order in orders:
            raise ValueError("report exemplar selection orders must be unique")
        orders.add(example.selection_order)
        if example.status == "selected":
            if (
                example.sample_id is None
                or example.prompt_id is None
                or example.step is None
                or not example.selection_order
                or not member.onset_step <= example.step <= member.alert_step
                or example.sample_id in selected
            ):
                raise ValueError("report selected exemplar references are invalid")
            selected.add(example.sample_id)


def _check_event(event: _Event, run: str, seen_members: set) -> None:
    members = {(member.run, member.feature): member for member in event.members}
    representative = members.get((run, event.representative_feature))
    if (
        event.run != run
        or len(members) != len(event.members)
        or len(members) != event.n_members
        or seen_members.intersection(members)
        or any(member.run != run for member in event.members)
        or representative is None
        or event.ranking.n_prioritizable_members > event.n_members
    ):
        raise ValueError("report event membership is inconsistent")
    seen_members.update(members)
    if (
        event.onset_start != min(member.onset_step for member in event.members)
        or event.onset_end != max(member.onset_step for member in event.members)
        or event.onset_step != representative.onset_step
        or event.alert_step != representative.alert_step
    ):
        raise ValueError("report event anchors do not match its members")
    for references in [
        event.ranking.missing_evidence_members,
        event.ranking.numerical_floor_members,
        event.ranking.numerical_floor_unavailable_members,
    ]:
        keys = {(reference.run, reference.feature) for reference in references}
        if len(keys) != len(references) or not keys.issubset(members):
            raise ValueError("report ranking references do not match its members")
    for member in event.members:
        _check_member(member)


def validate_report(value: object) -> dict:
    """Validate version-1 fields and links; omit unknown fields from the copy."""
    if not isinstance(value, dict):
        raise ValueError("report must be a JSON object")
    version = value.get("schema_version")
    if type(version) is not int or version != 1:
        raise ValueError("unsupported report version; expected schema_version 1")
    try:
        report = _Report.model_validate(value)
    except (ValidationError, RecursionError):
        raise ValueError("invalid report fields or types") from None
    seen_runs, seen_events, seen_members = set(), set(), set()
    for run in report.runs:
        if run.run in seen_runs:
            raise ValueError("report run identifiers must be unique")
        seen_runs.add(run.run)
        counts = run.counts
        behavioral = sum(event.ranking_status == "behavioral" for event in run.events)
        contextual = sum(event.ranking_status == "contextual" for event in run.events)
        if (
            counts.events != len(run.events)
            or counts.sequential_alerts != sum(event.n_members for event in run.events)
            or counts.behavioral_events != behavioral
            or counts.contextual_events != contextual
            or counts.other_events != len(run.events) - behavioral - contextual
            or sorted(event.rank for event in run.events)
            != list(range(1, len(run.events) + 1))
        ):
            raise ValueError("report counts or event ranks are inconsistent")
        for event in run.events:
            if event.event_id in seen_events:
                raise ValueError("report event identifiers must be unique")
            seen_events.add(event.event_id)
            _check_event(event, run.run, seen_members)
    return report.model_dump()


def _object(pairs: list[tuple]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate report JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite report JSON number")


def load_report(path: str | Path) -> dict:
    """Read strict UTF-8 JSON with safe errors for malformed input."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(
                stream, object_pairs_hook=_object, parse_constant=_constant
            )
    except (ValueError, RecursionError):
        raise ValueError("invalid report JSON") from None
    return validate_report(value)
