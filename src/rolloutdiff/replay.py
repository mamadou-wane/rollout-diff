"""Explicit Aria replay with historical reports and external task comparisons."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ._aria_replay import (
    ARIA_SHA256,
    historical_extract,
    historical_script,
    load_task,
    protected_cases,
    verify_artifacts,
)
from ._replay_container import SandboxConfig, run_container

EVALUATOR_COMMIT = "73695ff5533b566f7cc99b02bfeb9168936e740d"

# Only candidate code and public inputs enter this interpreter. No expected values.
_ANSWER_WORKER = r"""
import contextlib
import json
import math
import sys

def encode(value, depth=0):
    if depth > 64:
        raise ValueError("unsupported result depth")
    kind = type(value)
    if value is None:
        return ["none", None]
    if kind is bool:
        return ["bool", value]
    if kind is int:
        return ["int", value]
    if kind is str:
        return ["str", value]
    if kind is float:
        return ["float", value if math.isfinite(value) else
                ("nan" if math.isnan(value) else "inf" if value > 0 else "-inf")]
    if kind is list:
        return ["list", [encode(item, depth + 1) for item in value]]
    raise ValueError("unsupported result type")

data = json.loads(sys.stdin.buffer.read())
scope = {}
results = []
try:
    with contextlib.redirect_stdout(sys.stderr):
        exec(data["setup"] + "\n" + data["program"], scope)
except BaseException as error:
    result = {"setup_exception": type(error).__name__}
else:
    for case in data["cases"]:
        try:
            with contextlib.redirect_stdout(sys.stderr):
                value = getattr(scope["Solution"](), case["method"])(**case["kwargs"])
            results.append({"value": encode(value)})
        except BaseException as error:
            results.append({"exception": type(error).__name__})
    result = {"results": results}
sys.stdout.write(json.dumps(result, allow_nan=False))
"""


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _decode(value, depth=0):
    if depth > 64 or type(value) is not list or len(value) != 2:
        raise ValueError("invalid candidate answer encoding")
    tag, item = value
    if type(tag) is not str:
        raise ValueError("invalid candidate answer tag")
    if tag == "none" and item is None:
        return None
    if tag in {"bool", "int", "str"}:
        expected = {"bool": bool, "int": int, "str": str}[tag]
        if type(item) is expected:
            return item
    if tag == "float":
        if type(item) is float and math.isfinite(item):
            return item
        if type(item) is str and item in {"inf", "-inf", "nan"}:
            return float(item)
    if tag == "list" and type(item) is list:
        return [_decode(child, depth + 1) for child in item]
    raise ValueError("invalid candidate answer encoding")


def _exception(value):
    return (
        value
        if isinstance(value, str)
        and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value)
        else "UnrecognizedException"
    )


def assess_answers(payload: object, cases: list[dict]) -> dict:
    """Compare primitive proposed answers on the host; worker verdicts have no authority."""
    result = {
        "status": "unresolved",
        "reason": "invalid_answer_protocol",
        "determinate": False,
        "tests_applicable": len(cases),
        "tests_completed": 0,
        "tests_passed": 0,
        "tests_failed": 0,
        "reported_exceptions": [],
        "comparison_authority": "host",
        "test_coverage_complete": False,
    }
    if not cases or type(payload) is not dict:
        return result
    if set(payload) == {"setup_exception"}:
        result["reason"] = "candidate_reported_setup_failure"
        result["reported_exceptions"] = [_exception(payload["setup_exception"])]
        return result
    if set(payload) != {"results"} or type(payload["results"]) is not list:
        return result
    exact_count = len(payload["results"]) == len(cases)
    for proposed, case in zip(payload["results"], cases, strict=False):
        if type(proposed) is not dict:
            continue
        if set(proposed) == {"exception"}:
            result["reported_exceptions"].append(_exception(proposed["exception"]))
        elif set(proposed) == {"value"}:
            try:
                observed = _decode(proposed["value"])
                passed = observed == case["expected"]
            except (ValueError, RecursionError):
                continue
            result["tests_completed"] += 1
            result["tests_passed" if passed else "tests_failed"] += 1
    complete = exact_count and result["tests_completed"] == len(cases)
    result["test_coverage_complete"] = complete
    if not complete:
        result["reason"] = (
            "incomplete_or_unsupported_answers"
            if exact_count
            else "incomplete_or_extra_answers"
        )
    else:
        result.update(
            status="failed" if result["tests_failed"] else "passed",
            determinate=True,
            reason=None,
        )
    return result


def _json_output(content):
    def invalid_constant(_):
        raise ValueError("nonfinite JSON constant")

    def unique_members(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON member")
            result[key] = value
        return result

    try:
        return json.loads(
            content, parse_constant=invalid_constant, object_pairs_hook=unique_members
        )
    except (ValueError, UnicodeError, RecursionError):
        return None


def _runtime(result):
    fields = asdict(result)
    for stream in ("stdout", "stderr"):
        content = fields.pop(stream)
        fields[stream + "_sha256"] = _sha(content)
        fields[stream + "_bytes"] = len(content)
    fields["peak_memory_bytes"] = None
    fields["memory_measurement_status"] = (
        "unavailable; daemon OOM flag and limit recorded"
    )
    return fields


_FLAG_NAMES = ("success", "compiled", "timeout", "oom")
_COUNTER_NAMES = ("tests_evaluated", "tests_passed")
_HISTORICAL_UNAVAILABLE_FLAGS = {
    "reported_compile_failure": (False, False, False, False),
    "reported_inner_timeout": (False, True, True, False),
    "reported_inner_oom": (False, True, False, True),
    "reported_runtime_error": (False, True, False, False),
}


def _counter_diagnostic(value):
    # Leave headroom to serialize the diagnostic inside a complete case record.
    nodes = 1
    pending = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, (list, dict)):
            if depth >= 64:
                return {
                    "invalid_value": "nesting_limit",
                    "value_type": type(value).__name__,
                }
            children = item.values() if isinstance(item, dict) else item
            nodes += len(item)
            if nodes > 256:
                return {
                    "invalid_value": "size_limit",
                    "value_type": type(value).__name__,
                }
            pending.extend((child, depth + 1) for child in children)
    try:
        if len(json.dumps(value, allow_nan=False)) > 4096:
            return {"invalid_value": "size_limit", "value_type": type(value).__name__}
    except ValueError:
        diagnostic = {
            "invalid_value": "non_finite_number",
            "value_type": type(value).__name__,
        }
        if type(value) is float:
            diagnostic["representation"] = repr(value)
        return diagnostic
    except RecursionError:
        return {"invalid_value": "nesting_limit", "value_type": type(value).__name__}
    return value


def _historical_allows_protected(observation):
    """Apply the protocol gate after host-side runtime checks."""
    if observation["runtime"]["status"] != "completed":
        return (
            observation["runtime"]["status"]
            in {"nonzero_exit", "timeout", "output_limit"}
            and observation["status"] == "unresolved"
        )
    if observation["status"] == "reported":
        claim = observation.get("reported_result")
        if type(claim) is not dict:
            return False
        passed, errors = claim.get("tests_passed"), claim.get("exception_types")
        return (
            type(claim.get("success")) is bool
            and claim.get("compiled") is True
            and claim.get("timeout") is False
            and claim.get("oom") is False
            and type(claim.get("tests_evaluated")) is int
            and claim["tests_evaluated"] == 1
            and type(passed) is int
            and passed in (0, 1)
            and type(errors) is list
            and len(errors) == 1 - passed
            and all(
                type(e) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", e)
                for e in errors
            )
        )
    return (
        observation["status"] == "unavailable"
        and type(observation.get("reason")) is str
        and observation["reason"] in _HISTORICAL_UNAVAILABLE_FLAGS
    )


def _historical_report(result):
    record = {
        "status": "unresolved",
        "authority": "candidate_controlled_historical_report",
        "semantic_completion_authenticated": False,
        "trace_status": "reported exception types only; authenticated stack unavailable",
        "reported_result": None,
        "runtime": _runtime(result),
    }
    if result.status != "completed":
        return {**record, "reason": result.status}
    payload = _json_output(result.stdout)
    if type(payload) is not dict:
        return {**record, "reason": "invalid_historical_result_envelope"}
    if (
        set(payload) == {"historical_unavailable", "exception_type"}
        and payload["historical_unavailable"]
        == "unsupported_result_validation_or_supervisor_error"
        and type(payload["exception_type"]) is str
        and payload["exception_type"].isidentifier()
    ):
        return {
            **record,
            "status": "unavailable",
            "reason": "unsupported_historical_result",
        }
    if (
        set(payload) != {*_FLAG_NAMES, "stdout"}
        or any(type(payload[key]) is not bool for key in _FLAG_NAMES)
        or type(payload["stdout"]) is not dict
    ):
        return {**record, "reason": "invalid_historical_result_envelope"}
    counts = payload["stdout"]
    flags = tuple(payload[k] for k in _FLAG_NAMES)
    is_raw_text = set(counts) == {"raw"} and type(counts["raw"]) is str
    reason = None
    if is_raw_text:
        if flags == _HISTORICAL_UNAVAILABLE_FLAGS["reported_runtime_error"]:
            reason = "reported_runtime_error"
        elif flags == _HISTORICAL_UNAVAILABLE_FLAGS["reported_compile_failure"]:
            # The parent uses this shape for both OSError and invalid child JSON.
            reason = "reported_child_output_or_setup_failure"
    elif not counts:
        reason = next(
            (
                name
                for name, expected in _HISTORICAL_UNAVAILABLE_FLAGS.items()
                if name != "reported_runtime_error" and flags == expected
            ),
            None,
        )
    if reason:
        if is_raw_text:
            raw = json.dumps(counts["raw"], ensure_ascii=True).encode("ascii")
            record["reported_raw"] = {"sha256": _sha(raw), "bytes": len(raw)}
        return {
            **record,
            "status": "unavailable",
            "reason": reason,
            "reported_flags": dict(zip(_FLAG_NAMES, flags, strict=True)),
            "counter_observations": {
                key: {"present": False, "value": None} for key in _COUNTER_NAMES
            },
        }
    if any(
        type(counts.get(key)) is not int or counts[key] not in (0, 1)
        for key in _COUNTER_NAMES
    ):
        return {
            **record,
            "reason": "invalid_historical_counters",
            "counter_observations": {
                key: {
                    "present": key in counts,
                    "value": _counter_diagnostic(counts.get(key)),
                }
                for key in _COUNTER_NAMES
            },
        }
    # The pinned runner makes one test call. The parent may force success=false
    # on a nonzero child exit even when the child emitted counters.
    errors = counts.get("test_errors")
    if (
        flags[1:] != (True, False, False)
        or set(counts) != {*_COUNTER_NAMES, "tests_total", "test_errors"}
        or type(counts["tests_total"]) is not int
        or counts["tests_total"] != 1
        or counts["tests_evaluated"] != 1
        or type(errors) is not list
        or len(errors) != 1 - counts["tests_passed"]
        or any(type(value) is not str or ": " not in value for value in errors)
    ):
        return {**record, "reason": "invalid_historical_measurement"}
    return {
        **record,
        "status": "reported",
        "reported_result": {
            **dict(zip(_FLAG_NAMES, flags, strict=True)),
            **{key: counts[key] for key in _COUNTER_NAMES},
            "exception_types": [_exception(value.split(":", 1)[0]) for value in errors],
        },
    }


def _selected_source(source, run, sample_id, response_sha256):
    if (
        run != "aria"
        or not isinstance(sample_id, str)
        or not re.fullmatch(r"step\d{3}_row\d{3}", sample_id)
    ):
        raise ValueError("Aria replay requires an exact canonical identity")
    if not isinstance(response_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", response_sha256
    ):
        raise ValueError("replay requires a SHA-256 response hash")
    content = Path(source).read_bytes()
    if _sha(content) != ARIA_SHA256:
        raise ValueError("Aria source checksum does not match the pinned dataset")
    rows = pd.read_parquet(
        io.BytesIO(content),
        columns=["sample_id", "step", "problem_id", "messages"],
        filters=[("sample_id", "=", sample_id)],
    )
    if len(rows) != 1:
        raise ValueError("replay identity must resolve to exactly one completion")
    row = rows.iloc[0]
    messages = json.loads(row.messages)
    if [message["role"] for message in messages] != ["system", "user", "assistant"]:
        raise ValueError("unexpected canonical Aria message layout")
    response = messages[2]["content"]
    if _sha(response.encode()) != response_sha256:
        raise ValueError("selected response checksum does not match")
    return {
        "run": run,
        "sample_id": sample_id,
        "step": int(row.step),
        "prompt_id": str(row.problem_id),
        "prompt": messages[1]["content"],
        "response": response,
    }


def _replay_response(response, task, artifacts, config):
    extraction = historical_extract(response)
    program = extraction.pop("program")
    extraction["program_sha256"] = (
        _sha(program.encode()) if program is not None else None
    )
    cases = protected_cases(task)
    result = {
        "extraction": extraction,
        "historical": {"status": "unavailable", "reason": "missing_code"},
        "task": {
            "status": "unavailable",
            "reason": "missing_code",
            "determinate": False,
        },
        "evidence_status": "unresolved",
        "mechanism_status": "unresolved",
        "successful_hack_hypothesis": "unresolved",
    }
    if program is None:
        return result
    if extraction["parse_status"] == "lookup_error":
        result["historical"]["reason"] = "historical_lookup_unavailable"
        result["task"]["reason"] = "historical_lookup_unavailable"
        return result
    script, payload = historical_script(
        artifacts, task["setup_code"], program, extraction["requested_call"]
    )
    historical = run_container(script, payload, config, execution_authorized=True)
    result["historical"] = _historical_report(historical)
    from .investigation import _validate_architecture, _validate_runtime

    try:
        _validate_runtime(result["historical"]["runtime"])
    except ValueError:
        result["task"]["reason"] = "historical_container_infrastructure_failure"
        return result
    if not _historical_allows_protected(result["historical"]):
        result["task"]["reason"] = "historical_protocol_unavailable"
        return result
    payload = json.dumps(
        {
            "setup": task["setup_code"],
            "program": program,
            "cases": [
                {"method": case["method"], "kwargs": case["kwargs"]} for case in cases
            ],
        },
        allow_nan=False,
    ).encode()
    independent = run_container(
        _ANSWER_WORKER, payload, config, execution_authorized=True
    )
    comparison = assess_answers(
        _json_output(independent.stdout) if independent.status == "completed" else None,
        cases,
    )
    if independent.status != "completed":
        comparison["reason"] = independent.status
    comparison["runtime"] = _runtime(independent)
    comparison["contract"] = (
        "all original checks; fresh Solution per call; host equality"
    )
    try:
        _validate_runtime(comparison["runtime"])
        _validate_architecture((result["historical"], comparison))
        if historical.metadata["container_id"] == independent.metadata["container_id"]:
            raise ValueError("evaluation modes require distinct containers")
    except ValueError:
        comparison.update(
            status="unresolved",
            determinate=False,
            test_coverage_complete=False,
            reason="task_container_infrastructure_failure",
        )
    result["task"] = comparison
    if comparison["determinate"]:
        result["evidence_status"] = "observed_behavior"
        if comparison["status"] == "passed":
            result["successful_hack_hypothesis"] = "unsupported_under_tested_conditions"
        reported = result["historical"].get("reported_result") or {}
        if reported.get("tests_passed") == 1 and comparison["status"] == "failed":
            result["evidence_status"] = "reproduced_discrepancy"
    return result


def replay_aria(
    source: str | Path,
    *,
    run: str,
    sample_id: str,
    response_sha256: str,
    artifacts: str | Path,
    config: SandboxConfig | None = None,
    execution_authorized: bool = False,
) -> dict:
    """Replay one hash-pinned Aria completion only with explicit authorization.

    Historical stdout is an untrusted reproduced report. Original test answers
    stay on the host; candidate primitive outputs are compared there. This checks
    the available test set. It never independently authenticates a hack mechanism.
    """
    if execution_authorized is not True:
        raise ValueError("replay requires explicit execution authorization")
    config = config if config is not None else SandboxConfig()
    row = _selected_source(source, run, sample_id, response_sha256)
    record = {
        "replay_schema_version": 1,
        "identity": {k: row[k] for k in ("run", "sample_id", "step", "prompt_id")},
        "source": {
            "dataset": "gutenbergpbc/aria-reward-hacking",
            "sha256": ARIA_SHA256,
            "response_sha256": response_sha256,
        },
        "evaluator": {
            "commit": EVALUATOR_COMMIT,
            "component": "hinted_test_call",
            "reward_reconstruction": "unavailable; full historical reward not recomputed",
            "auxiliary_test_modification_checks": "not_run",
        },
        "implementation_sha256": {
            name: _sha(Path(__file__).with_name(name).read_bytes())
            for name in ("replay.py", "_aria_replay.py", "_replay_container.py")
        },
        "protected_worker_sha256": _sha(_ANSWER_WORKER.encode()),
        "configuration": {
            k: v for k, v in asdict(config).items() if k != "docker_host"
        },
        "execution_endpoint": "explicit_local_unix_socket",
        "published_labels": "excluded; compare only after replay",
        "evidence_status": "unresolved",
        "mechanism_status": "unresolved",
        "successful_hack_hypothesis": "unresolved",
    }
    try:
        verified = verify_artifacts(artifacts)
        task = load_task(verified["fixture"], row["prompt_id"], row["prompt"])
        protected_cases(task)
    except (OSError, ValueError):
        record.update(
            historical={
                "status": "unavailable",
                "reason": "missing_or_invalid_evaluator_artifacts",
            },
            task={
                "status": "unavailable",
                "reason": "missing_or_unsupported_task_contract",
                "determinate": False,
            },
        )
        return record
    record["evaluator"]["source_sha256"] = {
        name: _sha(path.read_bytes()) for name, path in verified.items()
    }
    record["task_source"] = {
        "prompt_id": row["prompt_id"],
        "task_sha256": _sha(json.dumps(task, sort_keys=True).encode()),
        "checks": len(task["gt_answer"]),
    }
    record.update(_replay_response(row["response"], task, verified, config))
    return record


def serialize_replay(record: dict) -> str:
    """Serialize replay evidence independently of investigation-report version 1."""
    if (
        type(record.get("replay_schema_version")) is not int
        or record["replay_schema_version"] != 1
    ):
        raise ValueError("unsupported replay schema version")
    return json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
