"""Bounded, read-only views of recorded investigation evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path

import pandas as pd

from . import investigation
from ._report_validation import _constant, _object, validate_report
from .comparison_report import validate_comparison_report

_MAX_BYTES = 16 * 1024 * 1024
_IDENTITY = ("run", "sample_id", "step", "prompt_id")
_TASK_COUNTS = ("tests_applicable", "tests_completed", "tests_passed", "tests_failed")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _read(path, root, *, optional=False):
    """Require a canonical root spelling and refuse symlinks at every component."""
    root_path = Path(root).absolute()
    _require(".." not in root_path.parts, "evidence root must use a canonical path")
    path = Path(path)
    if path.is_absolute():
        try:
            relative = path.relative_to(root_path)
        except ValueError:
            raise ValueError("evidence path must remain inside the root") from None
    else:
        relative = path
    _require(
        relative.parts and all(part not in {"..", "."} for part in relative.parts),
        "evidence path must remain inside the root",
    )
    descriptors = []
    try:
        directory = os.open(root_path.anchor, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(directory)
        parts = (*root_path.parts[1:], *relative.parts)
        for part in parts[:-1]:
            directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            descriptors.append(directory)
        descriptor = os.open(
            relative.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
            dir_fd=directory,
        )
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode) and before.st_size <= _MAX_BYTES,
            "evidence input must be a regular bounded file",
        )
        with os.fdopen(os.dup(descriptor), "rb") as stream:
            content = stream.read(_MAX_BYTES + 1)
        after = os.fstat(descriptor)
        _require(
            len(content) <= _MAX_BYTES
            and (before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            "evidence input changed during reading",
        )
        return content, relative
    except FileNotFoundError:
        if optional:
            return None
        raise ValueError("evidence file is unavailable inside the root") from None
    except OSError:
        raise ValueError(
            "evidence input must be a regular file without symlinks"
        ) from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _parse(content):
    def finite(number):
        value = float(number)
        _require(math.isfinite(value), "nonfinite JSON number")
        return value

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_object,
            parse_constant=_constant,
            parse_float=finite,
        )
    except (ValueError, UnicodeError, RecursionError):
        raise ValueError("invalid evidence JSON") from None
    _require(type(value) is dict, "evidence JSON must be an object")
    return value


def _identity(value):
    _require(type(value) is dict, "missing canonical identity")
    _require(
        all(
            type(value.get(key)) is str and 0 < len(value[key]) <= 4096
            for key in ("run", "sample_id", "prompt_id")
        )
        and investigation._integer(value.get("step")),
        "invalid canonical identity",
    )
    return {key: value[key] for key in _IDENTITY}


def _reason(value):
    _require(
        value is None
        or (type(value) is str and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", value)),
        "unsupported recorded reason",
    )
    return value


def _runtime(value):
    if value is None:
        return {}
    _require(type(value) is dict, "invalid recorded runtime")
    if (
        value.get("config_verified") is True
        and value.get("cleanup_verified") is True
        and value.get("status") != "infrastructure_error"
    ):
        investigation._validate_runtime(value)
    else:
        _require(
            value.get("status")
            in {
                "completed",
                "nonzero_exit",
                "timeout",
                "output_limit",
                "infrastructure_error",
            }
            and type(value.get("config_verified")) is bool
            and type(value.get("cleanup_verified")) is bool
            and all(
                investigation._hash(value.get(stream + "_sha256"))
                and investigation._integer(value.get(stream + "_bytes"))
                for stream in ("stdout", "stderr")
            ),
            "invalid partial runtime evidence",
        )
    return value


def _condition(name, record, checks):
    historical, task = record.get("historical"), record.get("task")
    _require(
        type(historical) is dict and type(task) is dict, "missing recorded outcomes"
    )
    _require(
        historical.get("status") in {"reported", "unavailable", "unresolved"}
        and task.get("status") in {"passed", "failed", "unavailable", "unresolved"},
        "unsupported recorded outcome status",
    )
    history_runtime = _runtime(historical.get("runtime"))
    task_runtime = _runtime(task.get("runtime"))
    investigation._validate_architecture((historical, task))
    identifiers = [
        runtime.get("metadata", {}).get("container_id")
        for runtime in (history_runtime, task_runtime)
    ]
    _require(
        not all(identifiers) or identifiers[0] != identifiers[1],
        "evaluation modes require distinct containers",
    )
    history_status = historical["status"]
    history_reason = _reason(historical.get("reason"))
    claim = historical.get("reported_result")
    _require(claim is None or type(claim) is dict, "invalid historical claim")
    if claim:
        supplied = [claim.get(key) for key in ("tests_passed", "tests_evaluated")]
        _require(
            all(
                value is None or (type(value) is int and value in (0, 1))
                for value in supplied
            )
            and (None in supplied or supplied[0] <= supplied[1]),
            "contradictory historical counts",
        )
    legacy = history_status == "reported" and (
        not claim
        or any(key not in claim for key in ("tests_passed", "tests_evaluated"))
        or claim.get("tests_evaluated") == 0
        or "authority" not in historical
    )
    if legacy:
        history_status = "unavailable"
        history_reason = "legacy_historical_counters_unavailable"
    elif history_runtime:
        investigation._validate_historical(historical)
    else:
        _require(
            history_status != "reported", "historical claim requires runtime evidence"
        )
    passed = evaluated = None
    if history_status == "reported":
        passed, evaluated = claim["tests_passed"], claim["tests_evaluated"]
        _require(passed <= evaluated == 1, "contradictory historical counts")

    determinate = task.get("determinate")
    _require(type(determinate) is bool, "missing task determinate status")
    task_reason = _reason(task.get("reason"))
    counts = [task.get(key) for key in _TASK_COUNTS]
    _require(
        all(value is None or investigation._integer(value) for value in counts),
        "invalid task counts",
    )
    if all(value is not None for value in counts):
        _require(
            investigation._integer(checks) and checks > 0, "unsupported task contract"
        )
        investigation._validate_task(task, checks)
    else:
        _require(
            not determinate and task["status"] in {"unavailable", "unresolved"},
            "determinate task requires complete counts and coverage",
        )
    discrepancy = None
    if (
        determinate
        and history_status == "reported"
        and history_runtime.get("config_verified") is True
        and history_runtime.get("cleanup_verified") is True
    ):
        discrepancy = passed == 1 and task["status"] == "failed"
    return {
        "name": name,
        "historical_status": history_status,
        "historical_passed": passed,
        "historical_evaluated": evaluated,
        "historical_reason": history_reason,
        "task_status": task["status"],
        "task_determinate": determinate,
        "task_applicable": counts[0],
        "task_completed": counts[1],
        "task_passed": counts[2],
        "task_failed": counts[3],
        "task_reason": task_reason,
        "discrepancy": discrepancy,
        "answer_stream_sha256": task_runtime.get("stdout_sha256"),
        "cleanup_verified": task_runtime.get("cleanup_verified"),
    }


def _provenance(value, identity):
    _require(type(value) is dict, "missing source provenance")
    for key in ("source_sha256", "parent_response_sha256"):
        _require(investigation._hash(value.get(key)), "invalid source provenance hash")
    task = value.get("task_source", {})
    _require(type(task) is dict, "unsupported task source")
    if task:
        _require(
            task.get("prompt_id") == identity["prompt_id"]
            and investigation._hash(task.get("task_sha256"))
            and investigation._integer(task.get("checks"))
            and task["checks"] > 0,
            "unsupported task source contract",
        )
    evaluator = value.get("evaluator", {})
    _require(
        type(evaluator) is dict
        and (not task or evaluator.get("component") == "hinted_test_call"),
        "unsupported recorded evaluator",
    )
    return task.get("checks")


def _replay(value):
    _require(
        type(value.get("replay_schema_version")) is int
        and value["replay_schema_version"] == 1,
        "unsupported replay schema version",
    )
    identity = _identity(value.get("identity"))
    source = value.get("source")
    _require(type(source) is dict, "missing replay source")
    provenance = {
        "source_sha256": source.get("sha256"),
        "parent_response_sha256": source.get("response_sha256"),
        "task_source": value.get("task_source", {}),
        "evaluator": value.get("evaluator", {}),
    }
    checks = _provenance(provenance, identity)
    _require(
        value.get("mechanism_status") == "unresolved", "unsupported mechanism claim"
    )
    return identity, provenance, _condition("recorded", value, checks)


def _case(value, path, root):
    _require(
        type(value.get("case_schema_version")) is int
        and value["case_schema_version"] == 1,
        "unsupported case schema version",
    )
    fields = {
        key: item for key, item in value.items() if key not in {"case_id", "variant_id"}
    }
    case_id = "aria-" + _digest(investigation._json(fields).encode())
    _require(
        value.get("case_id") == case_id
        and value.get("variant_id") == case_id + "-variant",
        "case identity hash mismatch",
    )
    files = {}
    for name, field in [
        ("control.json", "control_result_sha256"),
        ("variant.txt", "variant_response_sha256"),
        ("source.diff", "source_diff_sha256"),
    ]:
        content, _ = _read(path.parent / name, root)
        _require(_digest(content) == value.get(field), "case file hash mismatch")
        files[name] = content
    control = _parse(files["control.json"])
    identity, provenance, condition = _replay(control)
    for key, expected in {
        "parent_identity": control["identity"],
        **provenance,
        "implementation_sha256": control.get("implementation_sha256"),
    }.items():
        _require(
            value.get(key) == expected, "case metadata disagrees with frozen control"
        )
    edit = value.get("edit")
    _require(
        type(edit) is dict
        and set(edit) == {"old", "new"}
        and all(type(item) is str and len(item) <= 65536 for item in edit.values())
        and edit["old"]
        and edit["old"] != edit["new"],
        "unsupported recorded edit",
    )
    investigation._predictions(value.get("predictions"))
    _require(len(files["source.diff"]) <= 262144, "source diff exceeds the view limit")
    try:
        source_diff = files["source.diff"].decode("utf-8")
    except UnicodeError:
        raise ValueError("invalid source diff encoding") from None
    condition["name"] = "control"
    return (
        identity,
        provenance,
        condition,
        {**edit, "occurrence": 1},
        source_diff,
        control,
    )


def _hypothesis(value):
    _require(
        type(value) is str and value.strip() and len(value) <= 4096,
        "invalid recorded hypothesis",
    )
    return value


def _predictions(rows, record, condition):
    _require(type(rows) is list, "invalid recorded predictions")
    claim = record["historical"].get("reported_result") or {}
    actual = {
        "historical_passed": condition["historical_passed"],
        "historical_evaluated": condition["historical_evaluated"],
        "exception_types": claim.get("exception_types"),
        "task_matched": condition["task_passed"],
        "task_mismatched": condition["task_failed"],
        "task_status": condition["task_status"],
        "task_stdout_sha256": condition["answer_stream_sha256"],
    }
    runtime = record["historical"].get("runtime", {})
    historical_available = (
        condition["historical_status"] == "reported"
        and runtime.get("status") == "completed"
        and runtime.get("config_verified") is True
        and runtime.get("cleanup_verified") is True
    )
    output, seen = [], set()
    for row in rows:
        _require(type(row) is dict, "invalid recorded prediction")
        name = row.get("field")
        _require(name not in seen, "duplicate recorded prediction")
        investigation._predictions({name: row.get("expected")})
        observed = row.get("observed")
        if observed is not None:
            investigation._predictions({name: observed})
        _require(
            row.get("status") in {"matched", "violated", "unavailable"},
            "invalid prediction status",
        )
        _require(
            type(observed) is type(actual[name]) and observed == actual[name],
            "recorded prediction differs from observed outcomes",
        )
        unavailable = (
            observed is None
            or (
                name in {"historical_passed", "historical_evaluated", "exception_types"}
                and not historical_available
            )
            or (
                name in {"task_matched", "task_mismatched", "task_stdout_sha256"}
                and not condition["task_determinate"]
            )
        )
        status = (
            "unavailable"
            if unavailable
            else "matched"
            if type(row["expected"]) is type(observed) and row["expected"] == observed
            else "violated"
        )
        _require(row["status"] == status, "contradictory recorded prediction status")
        seen.add(name)
        output.append(
            {key: row.get(key) for key in ("field", "expected", "observed", "status")}
        )
    return output


def _source_link(identity, provenance, rollouts, source):
    if rollouts is None:
        return {
            "status": "not_loaded",
            "reason": "No canonical rollout source is loaded.",
        }
    if identity is None or provenance is None:
        return {
            "status": "mismatch",
            "reason": "Saved report lacks a verifiable response binding.",
        }
    rows = rollouts.loc[
        (rollouts["run"] == identity["run"])
        & (rollouts["sample_id"] == identity["sample_id"])
    ]
    if len(rows) != 1:
        return {
            "status": "mismatch",
            "reason": "Canonical identity is absent or ambiguous.",
        }
    row = rows.iloc[0]
    if (
        row["step"] != identity["step"]
        or row["prompt_id"] != identity["prompt_id"]
        or type(row["response"]) is not str
        or _digest(row["response"].encode()) != provenance["parent_response_sha256"]
    ):
        return {
            "status": "mismatch",
            "reason": "Canonical metadata or response hash differs.",
        }
    original = any(
        file.get("sha256") == provenance["source_sha256"]
        for file in (source or {}).get("files", [])
    )
    return {
        "status": "matched" if original else "response_only",
        "reason": "Identity, metadata, response and original file hash match."
        if original
        else "Identity, metadata and response match; original file hash is unverified.",
    }


def load_evidence(
    path: Path,
    *,
    root: Path,
    rollouts: pd.DataFrame | None = None,
    source: dict | None = None,
) -> dict:
    """Validate saved claims without preparing, approving or executing candidates."""
    content, relative = _read(path, root)
    value = _parse(content)
    view = {
        "kind": None,
        "path": relative.as_posix(),
        "sha256": _digest(content),
        "verification": "Validated recorded structure; runtime and source claims are recorded, not re-executed or authenticated.",
        "source_link": None,
        "identity": None,
        "summary": {},
        "conditions": [],
        "hypothesis": None,
        "predictions": [],
        "edit": None,
        "source_diff": None,
        "mechanism_status": "unresolved",
    }
    provenance = None
    try:
        if "replay_schema_version" in value:
            view["kind"] = "replay"
            view["identity"], provenance, condition = _replay(value)
            view["conditions"] = [condition]
            view["summary"] = {"historical_claims_authenticated": False}
        elif "case_schema_version" in value:
            view["kind"] = "case"
            identity, provenance, condition, edit, diff, _ = _case(
                value, relative, root
            )
            view.update(
                identity=identity, conditions=[condition], edit=edit, source_diff=diff
            )
            view["hypothesis"] = _hypothesis(value.get("hypothesis"))
            view["summary"] = {
                "case_id": value["case_id"],
                "variant_id": value["variant_id"],
            }
            view["predictions"] = [
                {
                    "field": key,
                    "expected": item,
                    "observed": None,
                    "status": "unavailable",
                }
                for key, item in sorted(value["predictions"].items())
            ]
        elif "comparison_schema_version" in value:
            _require(
                type(value["comparison_schema_version"]) is int
                and value["comparison_schema_version"] == 1,
                "unsupported intervention schema version",
            )
            view["kind"] = "intervention"
            identity = view["identity"] = _identity(value.get("parent_identity"))
            provenance = value.get("provenance")
            checks = _provenance(provenance, identity)
            _require(
                value.get("mechanism_status") == "unresolved"
                and value.get("historical_claims_authenticated") is False,
                "unsupported intervention authority",
            )
            view["conditions"] = [
                _condition(name, value[name], checks) for name in ("control", "variant")
            ]
            view["hypothesis"] = _hypothesis(value.get("hypothesis"))
            view["predictions"] = _predictions(
                value.get("predictions"), value["variant"], view["conditions"][1]
            )
            equal = value.get("answer_stream_equal")
            _require(
                equal is None or type(equal) is bool,
                "invalid recorded answer comparison",
            )
            if equal is not None:
                a, b = view["conditions"]
                _require(
                    a["task_determinate"]
                    and b["task_determinate"]
                    and equal
                    == (a["answer_stream_sha256"] == b["answer_stream_sha256"]),
                    "contradictory recorded answer comparison",
                )
            _require(
                type(value.get("case_id")) is str
                and re.fullmatch(r"aria-[a-f0-9]{64}", value["case_id"]),
                "invalid intervention identity",
            )
            _require(
                value.get("variant_id") == value["case_id"] + "-variant",
                "invalid intervention variant",
            )
            view["summary"] = {
                "case_id": value["case_id"],
                "answer_stream_equal": equal,
                "historical_claims_authenticated": False,
            }
            sibling = _read(relative.parent / "case.json", root, optional=True)
            if sibling is not None:
                manifest = _parse(sibling[0])
                _, _, _, edit, diff, control = _case(
                    manifest, relative.parent / "case.json", root
                )
                _require(
                    value["control"]
                    == {key: control[key] for key in ("historical", "task")},
                    "intervention control differs from frozen control",
                )
                _require(
                    {row["field"]: row["expected"] for row in view["predictions"]}
                    == manifest["predictions"],
                    "intervention predictions differ from frozen case",
                )
                for key in (
                    "case_id",
                    "variant_id",
                    "hypothesis",
                    "parent_identity",
                    "control_result_sha256",
                    "variant_response_sha256",
                ):
                    _require(
                        value.get(key) == manifest.get(key),
                        "intervention metadata differs from case",
                    )
                for key, item in provenance.items():
                    _require(
                        item == manifest.get(key),
                        "intervention provenance differs from case",
                    )
                view.update(edit=edit, source_diff=diff)
        elif value.get("report_type") == "comparison":
            report = validate_comparison_report(value)
            view["kind"] = "comparison"
            view["summary"] = {
                "mode": report["mode"],
                "sides": report["sides"],
                "feature_count": len(report["features"]),
                "prompt_mix": report["prompt_mix"],
            }
        else:
            report = validate_report(value)
            view["kind"] = "analysis"
            view["summary"] = {
                "runs": [
                    {key: run[key] for key in ("run", "status", "counts")}
                    for run in report["runs"]
                ]
            }
        view["source_link"] = _source_link(
            view["identity"], provenance, rollouts, source
        )
    except (KeyError, TypeError, AttributeError, RecursionError):
        raise ValueError("incomplete or unsupported saved evidence") from None
    return view
