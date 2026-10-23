"""Versioned investigation reports from completed analysis tables."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from numbers import Integral, Real
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from .pipeline import AnalysisResult
from .series import _require_columns

_EVENT = [
    "event_id",
    "run",
    "rank",
    "representative_feature",
    "ranking_status",
    "onset_start",
    "onset_end",
    "onset_step",
    "alert_step",
    "n_members",
]
_RANKING = [
    "n_prioritizable_members",
    "replication_availability",
    "persistence_availability",
    "reward_availability",
    "breadth_availability",
]
_REFERENCES = [
    "missing_evidence_members",
    "numerical_floor_members",
    "numerical_floor_unavailable_members",
]
_MEMBER = ["run", "feature", "kind", "direction", "onset_step", "alert_step", "status"]
_EXEMPLAR = [
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
_EVIDENCE = {
    "replication": {
        **{
            name: f"replication_{name}"
            for name in ["status", "coverage_policy", "horizon_complete", "cutoff_step"]
        },
        "replicates": "replicates",
    },
    "persistence": {
        name: f"persistence_{name}"
        for name in [
            "status",
            "fraction",
            "coverage",
            "n_observations",
            "n_evaluable",
            "n_retained",
            "n_unavailable",
        ]
    },
    "reward_association": {
        name: f"reward_{name}"
        for name in [
            "status",
            "mean_difference",
            "n_groups",
            "n_mixed_groups",
            "n_rollouts",
            "n_included_rollouts",
            "window_start",
            "window_end",
        ]
    },
    "prompt_breadth": {
        "positive_prompts": "prompt_positive_prompts",
        "total_prompts": "prompt_total_prompts",
        "promptmix_status": "promptmix_status",
    },
}
_FEATURE_EVIDENCE = ["tier", "scale_source", "numerical_floor_dominant"]
_SCREENING = {
    "series": "n_series",
    "screened_series": "n_screened",
    "sequential_alerts": "n_alerted",
    "unscorable_series": "n_unscorable",
}
_STEPS = {
    "step",
    "onset_start",
    "onset_end",
    "onset_step",
    "alert_step",
    "replication_cutoff_step",
    "reward_window_start",
    "reward_window_end",
}
_INTEGERS = _STEPS | {
    "rank",
    "selection_order",
    "reward_group_size",
    "prompt_positive_prompts",
    "prompt_total_prompts",
}
_IDENTIFIERS = {
    "run",
    "event_id",
    "feature",
    "representative_feature",
    "sample_id",
    "prompt_id",
}
# POSIX roots are recognized at identifier component boundaries.
_EMBEDDED_PATH = re.compile(
    r"(?i:(?<!\w)file://)"
    r"|[A-Za-z]:(?:\\|/(?!/))"
    r"|\\\\[^\\/\s]+[\\/][^\\/\s]+"
    r"""|(?:^|(?<=[\s=,;|"'(\[{<]))//[^/\\\s]+[\\/][^/\\\s]+"""
    r"""|(?:^|(?<=[\s:=,;|"'(\[{<]))(?:/|~[\\/])(?=[^/\\\s])"""
)


def _identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("report identifiers must be nonempty strings")
    candidate = value.removeprefix("file:").removeprefix("strlit:")
    if (
        candidate.startswith(("/", "\\", "~/", "~\\"))
        or PureWindowsPath(candidate).is_absolute()
        or _EMBEDDED_PATH.search(value)
    ):
        raise ValueError(
            "report identifiers must not contain absolute filesystem paths"
        )
    return value


def _integer(value, *, identity=False):
    if value is None or value is pd.NA or pd.isna(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("report integer fields require exact integers")
    if (isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))) or (
        not identity
        and isinstance(value, Real)
        and math.isfinite(value)
        and abs(value) < 2**53
        and int(value) == value
    ):
        number = int(value)
    else:
        raise ValueError("report integer fields require exact integers")
    if number < 0:
        raise ValueError("report integer fields must be nonnegative")
    return number


def _value(name, value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if (
        isinstance(value, Real)
        and not isinstance(value, (bool, np.bool_))
        and not math.isfinite(value)
    ):
        return None
    if name in _IDENTIFIERS:
        return _identifier(value)
    if name in _INTEGERS or name.startswith("n_") or "_n_" in name:
        return _integer(value, identity=name in _STEPS)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"unsupported report value for {name}")


def _fields(row, names):
    mapping = dict(zip(names, names, strict=True)) if isinstance(names, list) else names
    return {
        output: _value(source, row.get(source)) for output, source in mapping.items()
    }


def _index(frame, keys, columns, name):
    _require_columns(frame, list(dict.fromkeys([*keys, *columns])), name)
    if frame.duplicated(keys).any():
        raise ValueError(f"duplicate {name} keys")
    result = {}
    for row in frame[list(dict.fromkeys([*keys, *columns]))].to_dict("records"):
        key = tuple(row[k] for k in keys)
        if not all(isinstance(v, str) and v.strip() for v in key):
            raise ValueError(f"invalid {name} keys")
        result[key] = row
    return result


def _references(values, allowed):
    references = []
    for run, feature in values:
        if (run, feature) not in allowed:
            raise ValueError("ranking reference is not an event member")
        references.append({"run": _identifier(run), "feature": _identifier(feature)})
    return sorted(references, key=lambda row: (row["run"], row["feature"]))


def _member_report(member, source, exemplars):
    row = _fields(member, _MEMBER)
    evidence = source if source is not None else {}
    row["evidence"] = {
        "status": "recorded" if source is not None else "missing",
        **_fields(evidence, _FEATURE_EVIDENCE),
        **{name: _fields(evidence, columns) for name, columns in _EVIDENCE.items()},
    }
    row["exemplar_status"] = "recorded" if exemplars else "missing"
    row["exemplars"] = sorted(
        [_fields(example, _EXEMPLAR) for example in exemplars],
        key=lambda example: (
            example["selection_order"] is None,
            example["selection_order"] or 0,
            example["sample_id"] or "",
        ),
    )
    return row


def build_report(analysis: AnalysisResult, *, run: str | None = None) -> dict:
    """Build schema version 1 without recomputing analysis or exporting raw text.

    Run/event/member order is canonical; unavailable numbers become null.
    `run` supplies identity for empty input and must match any analyzed run.
    Absolute path-shaped identifiers fail export rather than changing references.
    """
    canonical = _index(analysis.features.dense, ["run", "sample_id"], [], "rollouts")
    screening = _index(
        analysis.sequential.screening, ["run"], list(_SCREENING.values()), "screening"
    )
    events = _index(
        analysis.events,
        ["event_id"],
        ["run", "onset_start", "onset_end", "n_members", "members"],
        "events",
    )
    ranking = _index(
        analysis.event_ranking,
        ["event_id"],
        [*_EVENT, *_RANKING, *_REFERENCES],
        "event ranking",
    )
    members = _index(
        analysis.event_members,
        ["run", "feature"],
        ["event_id", *_MEMBER],
        "event members",
    )
    evidence = _index(
        analysis.evidence,
        ["run", "feature"],
        [
            *_FEATURE_EVIDENCE,
            *[column for group in _EVIDENCE.values() for column in group.values()],
        ],
        "evidence",
    )
    alerts = _index(
        analysis.sequential.alerts, ["run", "feature"], [], "sequential alerts"
    )
    if set(ranking) != set(events) or set(members) != set(alerts):
        raise ValueError("event, ranking and sequential membership coverage must agree")
    actual_runs = {key[0] for key in canonical}
    if {key[0] for key in screening} != actual_runs:
        raise ValueError("screening must cover all analyzed runs")
    runs = set(actual_runs)
    if run is not None:
        _identifier(run)
        if actual_runs and actual_runs != {run}:
            raise ValueError("requested run does not match analyzed runs")
        runs.add(run)

    grouped_members, grouped_examples = defaultdict(list), defaultdict(list)
    for member in members.values():
        event = events.get((member["event_id"],))
        if event is None or event["run"] != member["run"]:
            raise ValueError("event member has no matching event")
        grouped_members[member["event_id"]].append(member)
    _require_columns(analysis.exemplars, _EXEMPLAR, "exemplars")
    if analysis.exemplars.duplicated(["run", "feature", "selection_order"]).any():
        raise ValueError("duplicate exemplar selection order")
    for example in analysis.exemplars[_EXEMPLAR].to_dict("records"):
        key = example["run"], example["feature"]
        if key not in members:
            raise ValueError("exemplar has no matching event member")
        if (
            example["status"] == "selected"
            and (example["run"], example["sample_id"]) not in canonical
        ):
            raise ValueError(
                "exemplar reference does not resolve to a canonical rollout"
            )
        grouped_examples[key].append(example)

    grouped_events = defaultdict(list)
    for key, ordered in ranking.items():
        event = events[key]
        group = sorted(
            grouped_members[event["event_id"]],
            key=lambda m: (m["onset_step"], m["alert_step"], m["feature"]),
        )
        references = {(member["run"], member["feature"]) for member in group}
        if (
            event["run"] not in actual_runs
            or ordered["run"] != event["run"]
            or event["n_members"] != len(group)
            or ordered["n_members"] != len(group)
            or set(event["members"]) != references
            or (ordered["run"], ordered["representative_feature"]) not in references
        ):
            raise ValueError("event summary conflicts with its members or ranking")
        row = _fields(ordered, _EVENT)
        row["ranking"] = {
            **_fields(ordered, _RANKING),
            **{name: _references(ordered[name], references) for name in _REFERENCES},
        }
        row["members"] = [
            _member_report(
                member,
                evidence.get((member["run"], member["feature"])),
                grouped_examples[(member["run"], member["feature"])],
            )
            for member in group
        ]
        grouped_events[event["run"]].append(row)

    _require_columns(
        analysis.aggregation.prompt_schedule,
        ["run", "step", "prompt_id"],
        "prompt schedule",
    )
    steps, prompts = defaultdict(set), defaultdict(set)
    for row in analysis.aggregation.prompt_schedule[
        ["run", "step", "prompt_id"]
    ].to_dict("records"):
        steps[row["run"]].add(row["step"])
        prompts[row["run"]].add(row["prompt_id"])
    rollout_counts = Counter(key[0] for key in canonical)
    output = []
    for name in sorted(runs):
        ordered = sorted(
            grouped_events[name], key=lambda event: (event["rank"], event["event_id"])
        )
        if [event["rank"] for event in ordered] != list(range(1, len(ordered) + 1)):
            raise ValueError("event ranks must be contiguous within each run")
        counts = (
            _fields(screening[(name,)], _SCREENING)
            if (name,) in screening
            else dict.fromkeys(_SCREENING, 0)
        )
        if counts["sequential_alerts"] != sum(key[0] == name for key in alerts):
            raise ValueError("screening alert count does not match sequential alerts")
        behavioral = sum(e["ranking_status"] == "behavioral" for e in ordered)
        contextual = sum(e["ranking_status"] == "contextual" for e in ordered)
        counts.update(
            rollouts=rollout_counts[name],
            observed_steps=len(steps[name]),
            prompts=len(prompts[name]),
            events=len(ordered),
            behavioral_events=behavioral,
            contextual_events=contextual,
            other_events=len(ordered) - behavioral - contextual,
        )
        output.append(
            {
                "run": _identifier(name),
                "status": "analyzed" if rollout_counts[name] else "empty",
                "counts": counts,
                "events": ordered,
            }
        )
    return {"schema_version": 1, "runs": output}


def report_json(report: dict) -> str:
    """Encode a report as deterministic UTF-8-compatible strict JSON."""
    return (
        json.dumps(
            report, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2
        )
        + "\n"
    )


def _check_output_path(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output directory does not exist: {path.parent}")


def write_report(report: dict, path: str | Path) -> None:
    """Publish a complete report atomically; never replace an existing path."""
    path = Path(path)
    _check_output_path(path)
    _write_text(report_json(report), path)


def _write_text(text: str, path: str | Path) -> None:
    path = Path(path)
    _check_output_path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".rolloutdiff-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # A same-filesystem hard link publishes without an overwrite race.
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
