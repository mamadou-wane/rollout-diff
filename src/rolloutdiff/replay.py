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


def _historical_report(result):
    payload = _json_output(result.stdout)
    record = {
        "status": "unresolved",
        "authority": "candidate_controlled_historical_report",
        "semantic_completion_authenticated": False,
        "trace_status": "reported exception types only; authenticated stack unavailable",
        "reported_result": None,
        "runtime": _runtime(result),
    }
    if result.status != "completed" or type(payload) is not dict:
        return record
    if "historical_unavailable" in payload:
        record.update(status="unavailable", reason="unsupported_historical_result")
        return record
    boolean_fields = ("success", "compiled", "timeout", "oom")
    if (
        set(payload) != {*boolean_fields, "stdout"}
        or any(type(payload.get(name)) is not bool for name in boolean_fields)
        or type(payload["stdout"]) is not dict
    ):
        record["reason"] = "invalid_historical_result_envelope"
        return record
    counts = payload["stdout"]
    record["reported_result"] = {
        name: payload.get(name) if type(payload.get(name)) is bool else None
        for name in boolean_fields
    }
    record["reported_result"].update(
        {
            name: counts.get(name, 0)
            if type(counts.get(name, 0)) is int and 0 <= counts.get(name, 0) <= 1
            else None
            for name in ("tests_evaluated", "tests_passed")
        }
    )
    errors = counts.get("test_errors", [])
    record["reported_result"]["exception_types"] = (
        [_exception(str(value).split(":", 1)[0]) for value in errors[:8]]
        if isinstance(errors, list)
        else []
    )
    record["status"] = "reported"
    return record


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
    if not historical.cleanup_verified or historical.status == "infrastructure_error":
        result["task"]["reason"] = "historical_container_infrastructure_failure"
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
