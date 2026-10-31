"""Frozen, single-variant Aria investigations with locally consumed approvals."""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import os
import re
import stat
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from . import _aria_replay as aria
from . import _replay_container as boundary
from . import replay
from .report import _write_text

_LIMITS = {
    "memory_bytes": 1073741824,
    "cpus": 1,
    "pids": 16,
    "wall_seconds": 6,
    "output_bytes": 1048576,
    "stdin_bytes": 2097152,
    "tmp_bytes": 16777216,
}
_PREDICTIONS = {
    "historical_passed",
    "historical_evaluated",
    "exception_types",
    "task_matched",
    "task_mismatched",
    "task_stdout_sha256",
    "task_status",
}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def _read(path):
    path = Path(path)
    _require(
        not path.is_symlink()
        and stat.S_ISREG(path.stat().st_mode)
        and path.stat().st_size <= 16 * 1024 * 1024,
        "investigation input must be a regular bounded file",
    )
    return path.read_bytes()


def _load(path):
    value = replay._json_output(_read(path))
    _require(type(value) is dict, "invalid investigation JSON object")
    return value


def _save(directory, name, value):
    _write_text(_json(value), directory / name)
    # File publication is atomic; sync the directory before spending execution authority.
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash(value):
    return type(value) is str and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _integer(value):
    return type(value) is int and value >= 0


def _implementation():
    return {
        name: _sha(Path(replay.__file__).with_name(name).read_bytes())
        for name in ("replay.py", "_aria_replay.py", "_replay_container.py")
    }


def _sandbox(config):
    endpoint = urlsplit(config.docker_host)
    _require(
        config.image == boundary.DEFAULT_IMAGE
        and endpoint.scheme == "unix"
        and not endpoint.netloc
        and endpoint.path.startswith("/")
        and not endpoint.query
        and not endpoint.fragment
        and "%" not in config.docker_host
        and all(ord(char) >= 33 for char in config.docker_host),
        "unsupported investigation sandbox",
    )
    return deepcopy(
        {
            "image": config.image,
            "limits": _LIMITS,
            "host_config": boundary._HOST_CONFIG,
            "ulimits": boundary._ULIMITS,
            "uid_gid": "65534:65534",
            "cpu_limit_seconds": 3,
            "python_version": "3.12.13",
        }
    )


def _extraction(response):
    result = aria.historical_extract(response)
    program = result.pop("program")
    result["program_sha256"] = _sha(program.encode()) if program is not None else None
    return program, result


def _validate_runtime(runtime):
    _require(
        type(runtime) is dict
        and runtime.get("config_verified") is True
        and runtime.get("cleanup_verified") is True
        and runtime.get("error") is None,
        "incomplete runtime or unverified cleanup",
    )
    status, exit_code = runtime.get("status"), runtime.get("exit_code")
    _require(
        status in {"completed", "nonzero_exit", "timeout", "output_limit"},
        "incompatible runtime status",
    )
    if status in {"completed", "nonzero_exit"}:
        _require(
            type(exit_code) is int and (exit_code == 0) == (status == "completed"),
            "inconsistent runtime exit status",
        )
    metadata = runtime.get("metadata", {})
    _require(type(metadata) is dict, "invalid runtime metadata")
    _require(
        metadata.get("image_id") == boundary.DEFAULT_IMAGE
        and metadata.get("python_version") == "3.12.13"
        and metadata.get("architecture") in {"arm64", "amd64"}
        and metadata.get("runtime") == "runc"
        and metadata.get("cgroup_version") == "2"
        and metadata.get("limits") == _LIMITS
        and _hash(metadata.get("container_id")),
        "incompatible runtime configuration",
    )
    for stream in ("stdout", "stderr"):
        _require(
            _hash(runtime.get(stream + "_sha256"))
            and _integer(runtime.get(stream + "_bytes")),
            "missing stream evidence",
        )
    _require(
        runtime["stdout_bytes"] + runtime["stderr_bytes"] <= _LIMITS["output_bytes"],
        "invalid stream sizes",
    )


def _validate_architecture(modes, expected=None):
    for mode in modes:
        architecture = mode.get("runtime", {}).get("metadata", {}).get("architecture")
        if architecture is not None:
            expected = architecture if expected is None else expected
            _require(
                architecture == expected, "runtime architecture differs from control"
            )


def _validate_outcomes(record, checks, *, partial=False, architecture=None):
    _require(_integer(checks) and checks > 0, "invalid applicable task count")
    historical, task = record["historical"], record["task"]
    for mode in (historical, task):
        if partial and "runtime" not in mode:
            _require(
                mode.get("status") == "unavailable" and type(mode.get("reason")) is str,
                "missing partial mode status",
            )
        elif partial and (
            mode["runtime"].get("cleanup_verified") is not True
            or mode["runtime"].get("config_verified") is not True
            or mode["runtime"].get("status") == "infrastructure_error"
        ):
            runtime = mode["runtime"]
            _require(
                runtime.get("status")
                in {
                    "completed",
                    "nonzero_exit",
                    "timeout",
                    "output_limit",
                    "infrastructure_error",
                }
                and type(runtime.get("cleanup_verified")) is bool
                and type(runtime.get("config_verified")) is bool
                and all(
                    _hash(runtime.get(stream + "_sha256"))
                    and _integer(runtime.get(stream + "_bytes"))
                    for stream in ("stdout", "stderr")
                ),
                "invalid partial runtime",
            )
        else:
            _validate_runtime(mode.get("runtime"))
    _validate_architecture((historical, task), architecture)
    if partial and "runtime" not in historical:
        _require(
            historical == {"status": "unavailable", "reason": "not_attempted"},
            "invalid unattempted historical mode",
        )
    else:
        _validate_historical(historical)
    if partial and "runtime" not in task:
        _require(
            task
            == {
                "status": "unavailable",
                "reason": "not_attempted",
                "determinate": False,
            },
            "invalid unattempted task mode",
        )
    else:
        _validate_task(task, checks)
    identities = [
        mode.get("runtime", {}).get("metadata", {}).get("container_id")
        for mode in (historical, task)
    ]
    _require(
        identities[0] is None
        or identities[1] is None
        or identities[0] != identities[1],
        "evaluation modes require distinct containers",
    )
    _require(
        record.get("mechanism_status") == "unresolved", "unsupported mechanism claim"
    )


def _validate_historical(historical):
    _require(
        historical.get("authority") == "candidate_controlled_historical_report"
        and historical.get("semantic_completion_authenticated") is False,
        "historical claims must remain untrusted",
    )
    if historical["status"] == "reported":
        _require(
            historical["runtime"]["status"] == "completed",
            "claim from incomplete runtime",
        )
        claim = historical.get("reported_result")
        _require(type(claim) is dict, "missing historical claim")
        for key in ("success", "compiled", "timeout", "oom"):
            _require(type(claim.get(key)) is bool, "invalid historical claim")
        for key in ("tests_evaluated", "tests_passed"):
            _require(
                type(claim.get(key)) is int and claim[key] in (0, 1),
                "invalid historical counters",
            )
        _require(
            type(claim.get("exception_types")) is list
            and all(
                type(value) is str
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value)
                for value in claim["exception_types"]
            ),
            "invalid exception evidence",
        )
    else:
        _require(
            historical["status"] in {"unavailable", "unresolved"}
            and historical.get("reported_result") is None,
            "invalid unavailable historical evidence",
        )
        if historical.get("reason") == "invalid_historical_counters":
            counters = historical.get("counter_observations")
            _require(
                type(counters) is dict
                and set(counters) == {"tests_evaluated", "tests_passed"},
                "missing rejected counter observations",
            )
            for item in counters.values():
                _require(
                    type(item) is dict
                    and set(item) == {"present", "value"}
                    and type(item["present"]) is bool
                    and (item["present"] or item["value"] is None),
                    "invalid rejected counter observation",
                )
            _require(
                not all(
                    item["present"] and _integer(item["value"]) and item["value"] <= 1
                    for item in counters.values()
                ),
                "counter rejection lacks an invalid counter",
            )


def _counter_diagnostic(value):
    try:
        json.dumps(value, allow_nan=False)
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


def _historical_observation(raw):
    observation = replay._historical_report(raw)
    if observation["status"] == "reported":
        counters = replay._json_output(raw.stdout)["stdout"]
        fields = ("tests_evaluated", "tests_passed")
        if not all(
            _integer(counters.get(name)) and counters[name] <= 1 for name in fields
        ):
            observation.update(
                status="unresolved",
                reason="invalid_historical_counters",
                rejected_report=observation["reported_result"],
                reported_result=None,
                counter_observations={
                    name: {
                        "present": name in counters,
                        "value": _counter_diagnostic(counters.get(name)),
                    }
                    for name in fields
                },
            )
    return observation


def _validate_task(task, checks):
    counts = [
        task.get(key)
        for key in (
            "tests_applicable",
            "tests_completed",
            "tests_passed",
            "tests_failed",
        )
    ]
    _require(all(_integer(value) for value in counts), "missing task counts")
    total, completed, passed, failed = counts
    _require(
        total == checks and passed + failed == completed <= total,
        "task counts disagree with the task contract",
    )
    _require(
        type(task.get("determinate")) is bool
        and type(task.get("test_coverage_complete")) is bool
        and task.get("comparison_authority") == "host",
        "invalid task authority",
    )
    if task["determinate"]:
        _require(
            completed == total
            and task["test_coverage_complete"]
            and task["status"] == ("failed" if failed else "passed")
            and task.get("reason") is None
            and task["runtime"]["status"] == "completed",
            "inconsistent determinate task evidence",
        )
        _require(
            task["runtime"].get("config_verified") is True
            and task["runtime"].get("cleanup_verified") is True
            and task["runtime"].get("error") is None,
            "determinate task requires verified runtime and cleanup",
        )
    else:
        _require(
            task["status"] == "unresolved"
            and not task["test_coverage_complete"]
            and type(task.get("reason")) is str,
            "inconsistent unresolved task evidence",
        )


def _inputs(control, source, artifacts):
    try:
        _require(
            type(control.get("replay_schema_version")) is int
            and control["replay_schema_version"] == 1,
            "unsupported replay version",
        )
        identity = control["identity"]
        _require(_integer(identity.get("step")), "invalid canonical step type")
        row = replay._selected_source(
            source,
            identity["run"],
            identity["sample_id"],
            control["source"]["response_sha256"],
        )
        _require(
            identity
            == {key: row[key] for key in ("run", "sample_id", "step", "prompt_id")},
            "canonical identity mismatch",
        )
        _require(
            control["source"]["sha256"] == aria.ARIA_SHA256, "source hash mismatch"
        )
        _require(
            control["implementation_sha256"] == _implementation(),
            "replay implementation hash mismatch",
        )
        _require(
            control["protected_worker_sha256"] == _sha(replay._ANSWER_WORKER.encode()),
            "protected worker hash mismatch",
        )
        paths = aria.verify_artifacts(artifacts)
        evaluator = control["evaluator"]
        _require(
            evaluator["commit"] == aria.UPSTREAM_REVISION
            and evaluator["component"] == "hinted_test_call"
            and evaluator["source_sha256"]
            == {key: _sha(path.read_bytes()) for key, path in paths.items()},
            "evaluator provenance mismatch",
        )
        task = aria.load_task(paths["fixture"], row["prompt_id"], row["prompt"])
        cases = aria.protected_cases(task)
        _require(
            _integer(control["task_source"].get("checks")),
            "invalid task check count type",
        )
        _require(
            control["task_source"]
            == {
                "prompt_id": row["prompt_id"],
                "task_sha256": _sha(json.dumps(task, sort_keys=True).encode()),
                "checks": len(cases),
            },
            "task contract mismatch",
        )
        program, extraction = _extraction(row["response"])
        _require(
            program is not None and control["extraction"] == extraction,
            "control extraction mismatch",
        )
        _require(
            control["configuration"] == {"image": boundary.DEFAULT_IMAGE},
            "incompatible control sandbox",
        )
        _validate_outcomes(control, len(cases))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete or incompatible replay evidence") from exc
    return row, task, cases, paths


def _predictions(values):
    _require(
        type(values) is dict and values and set(values) <= _PREDICTIONS,
        "predictions must use supported outcome fields",
    )
    for name, value in values.items():
        if name == "exception_types":
            valid = type(value) is list and all(
                type(v) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", v)
                for v in value
            )
        elif name == "task_stdout_sha256":
            valid = _hash(value)
        elif name == "task_status":
            valid = value in ("passed", "failed", "unresolved", "unavailable")
        else:
            valid = _integer(value) and (
                not name.startswith("historical_") or value <= 1
            )
        _require(valid, "invalid predicted outcome")


def _workers(response, task, paths):
    program, extraction = _extraction(response)
    _require(
        program is not None and extraction["parse_status"] != "lookup_error",
        "intervention requires an extractable historical program",
    )
    historical = aria.historical_script(
        paths, task["setup_code"], program, extraction["requested_call"]
    )
    cases = aria.protected_cases(task)
    protected = (
        replay._ANSWER_WORKER,
        json.dumps(
            {
                "setup": task["setup_code"],
                "program": program,
                "cases": [
                    {"method": c["method"], "kwargs": c["kwargs"]} for c in cases
                ],
            },
            allow_nan=False,
        ).encode(),
    )
    return [historical, protected], extraction


def prepare_aria_intervention(
    control, source, *, artifacts, edit, hypothesis, predictions, directory, config=None
):
    """Freeze a control and one unique exact text replacement without execution.

    Case files contain private derived text. Use a fresh local directory outside
    source control. An operationally recorded unresolved outcome remains unresolved.
    """
    content = _read(control)
    frozen = replay._json_output(content)
    _require(type(frozen) is dict, "invalid investigation JSON object")
    row, task, _cases, paths = _inputs(frozen, source, artifacts)
    _require(
        type(edit) is dict
        and set(edit) == {"old", "new"}
        and all(type(value) is str for value in edit.values())
        and edit["old"]
        and edit["old"] != edit["new"]
        and row["response"].count(edit["old"]) == 1,
        "edit must replace exactly one unique nonempty occurrence",
    )
    _require(
        type(hypothesis) is str and hypothesis.strip() and len(hypothesis) <= 4096,
        "hypothesis must be nonempty and at most 4096 characters",
    )
    _predictions(predictions)
    config = config or boundary.SandboxConfig()
    configuration = _sandbox(config)
    variant = row["response"].replace(edit["old"], edit["new"], 1)
    workers, extraction = _workers(variant, task, paths)
    diff = "".join(
        difflib.unified_diff(
            row["response"].splitlines(keepends=True),
            variant.splitlines(keepends=True),
            fromfile="canonical-response",
            tofile="derived-response",
        )
    )
    manifest = {
        "case_schema_version": 1,
        "parent_identity": frozen["identity"],
        "parent_response_sha256": frozen["source"]["response_sha256"],
        "source_sha256": frozen["source"]["sha256"],
        "control_result_sha256": _sha(content),
        "variant_response_sha256": _sha(variant.encode()),
        "source_diff_sha256": _sha(diff.encode()),
        "variant_program_sha256": extraction["program_sha256"],
        "extraction": extraction,
        "hypothesis": hypothesis,
        "predictions": predictions,
        "edit": edit,
        "evaluator": frozen["evaluator"],
        "task_source": frozen["task_source"],
        "implementation_sha256": _implementation(),
        "workflow_sha256": _sha(Path(__file__).read_bytes()),
        "configuration": configuration,
        "docker_host": config.docker_host,
        "worker_inputs": [
            {"script_sha256": _sha(script.encode()), "payload_sha256": _sha(payload)}
            for script, payload in workers
        ],
    }
    manifest["case_id"] = "aria-" + _sha(_json(manifest).encode())
    manifest["variant_id"] = manifest["case_id"] + "-variant"
    directory = Path(directory)
    directory.mkdir(mode=0o700, exist_ok=False)
    _write_text(content.decode("utf-8"), directory / "control.json")
    _write_text(variant, directory / "variant.txt")
    _write_text(diff, directory / "source.diff")
    _save(directory, "case.json", manifest)
    return {**manifest, "case_sha256": _sha(_read(directory / "case.json"))}


def _case(directory):
    directory = Path(directory).resolve(strict=True)
    manifest = _load(directory / "case.json")
    try:
        _require(
            type(manifest["case_schema_version"]) is int
            and manifest["case_schema_version"] == 1,
            "unsupported case version",
        )
        identity_fields = {
            key: value
            for key, value in manifest.items()
            if key not in {"case_id", "variant_id"}
        }
        case_id = "aria-" + _sha(_json(identity_fields).encode())
        _require(
            manifest["case_id"] == case_id
            and manifest["variant_id"] == case_id + "-variant",
            "case identity hash mismatch",
        )
        for name, key in [
            ("control.json", "control_result_sha256"),
            ("variant.txt", "variant_response_sha256"),
            ("source.diff", "source_diff_sha256"),
        ]:
            _require(
                _sha(_read(directory / name)) == manifest[key],
                "case file hash mismatch",
            )
        _require(
            manifest["implementation_sha256"] == _implementation()
            and manifest["workflow_sha256"] == _sha(Path(__file__).read_bytes()),
            "case implementation hash mismatch",
        )
        _require(
            manifest["configuration"]
            == _sandbox(boundary.SandboxConfig(docker_host=manifest["docker_host"])),
            "case sandbox mismatch",
        )
        _predictions(manifest["predictions"])
        control = _load(directory / "control.json")
        for name, expected in {
            "parent_identity": control["identity"],
            "parent_response_sha256": control["source"]["response_sha256"],
            "source_sha256": control["source"]["sha256"],
            "evaluator": control["evaluator"],
            "task_source": control["task_source"],
            "implementation_sha256": control["implementation_sha256"],
        }.items():
            _require(
                manifest[name] == expected,
                "case metadata disagrees with frozen control",
            )
        _validate_outcomes(control, manifest["task_source"]["checks"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete case manifest") from exc
    return directory, manifest, _sha(_read(directory / "case.json"))


def approve_aria_intervention(
    directory,
    *,
    case_sha256,
    variant_sha256,
    max_containers,
    execution_authorized=False,
):
    """Write one path-bound approval; approval alone never starts a container."""
    _require(execution_authorized is True, "explicit execution authorization required")
    directory, manifest, actual = _case(directory)
    _require(not (directory / "consumed.json").exists(), "authorization is spent")
    _require(
        case_sha256 == actual and variant_sha256 == manifest["variant_response_sha256"],
        "approval hash mismatch",
    )
    _require(
        type(max_containers) is int and max_containers == 2,
        "this workflow requires a budget of exactly two evaluation containers",
    )
    approval = {
        "approval_schema_version": 1,
        "case_sha256": actual,
        "variant_response_sha256": variant_sha256,
        "case_directory": str(directory),
        "max_containers": max_containers,
        "execution_authorized": True,
    }
    _save(directory, "approval.json", approval)
    return approval


def _comparison(manifest, control, variant, *, status, origin):
    claim = variant.get("historical", {}).get("reported_result") or {}
    task = variant.get("task", {})
    actual = {
        "historical_passed": claim.get("tests_passed"),
        "historical_evaluated": claim.get("tests_evaluated"),
        "exception_types": claim.get("exception_types"),
        "task_matched": task.get("tests_passed"),
        "task_mismatched": task.get("tests_failed"),
        "task_status": task.get("status"),
        "task_stdout_sha256": task.get("runtime", {}).get("stdout_sha256"),
    }
    history_runtime = variant.get("historical", {}).get("runtime", {})
    history_available = (
        variant.get("historical", {}).get("status") == "reported"
        and history_runtime.get("status") == "completed"
        and history_runtime.get("config_verified") is True
        and history_runtime.get("cleanup_verified") is True
    )
    checks = [
        {
            "field": name,
            "expected": expected,
            "observed": actual[name],
            "status": "unavailable"
            if actual[name] is None
            or (
                name in {"historical_passed", "historical_evaluated", "exception_types"}
                and not history_available
            )
            or (
                name in {"task_matched", "task_mismatched", "task_stdout_sha256"}
                and not task.get("test_coverage_complete", False)
            )
            else "matched"
            if type(expected) is type(actual[name]) and expected == actual[name]
            else "violated",
        }
        for name, expected in sorted(manifest["predictions"].items())
    ]
    a = control["task"]["runtime"]["stdout_sha256"]
    b = actual["task_stdout_sha256"]
    return {
        "comparison_schema_version": 1,
        "case_id": manifest["case_id"],
        "variant_id": manifest["variant_id"],
        "parent_identity": manifest["parent_identity"],
        "hypothesis": manifest["hypothesis"],
        "status": status,
        "origin": origin,
        "predictions": checks,
        "control": {name: control[name] for name in ("historical", "task")},
        "variant": {name: variant[name] for name in ("historical", "task")},
        "answer_stream_equal": a == b
        if b is not None
        and control["task"]["test_coverage_complete"]
        and task.get("test_coverage_complete")
        else None,
        "mechanism_status": "unresolved",
        "historical_claims_authenticated": False,
        "interpretation": "Predictions are compared literally with recorded observations; incomplete comparisons remain unavailable. Historical counters remain candidate-controlled. No internal call trace or reward-hacking mechanism is authenticated.",
        "control_result_sha256": manifest["control_result_sha256"],
        "variant_response_sha256": manifest["variant_response_sha256"],
        "provenance": {
            key: manifest[key]
            for key in (
                "source_sha256",
                "parent_response_sha256",
                "variant_program_sha256",
                "evaluator",
                "task_source",
                "configuration",
            )
        },
    }


def compare_aria_intervention(directory, result, *, provenance=None):
    """Compare saved evidence without execution, optionally importing a pinned protocol.

    External derived-replay records require their original hash-bound protocol.
    Imported evidence confers no new execution authorization or authenticity.
    """
    directory, manifest, case_sha = _case(directory)
    variant = _load(result)
    control = _load(directory / "control.json")
    try:
        _require(
            variant["variant_response_sha256"] == manifest["variant_response_sha256"],
            "variant hash mismatch",
        )
        if provenance is None:
            _require(
                variant["case_sha256"] == case_sha
                and variant["case_id"] == manifest["case_id"],
                "result case hash mismatch",
            )
            _require(
                type(variant.get("investigation_result_version")) is int
                and variant["investigation_result_version"] == 1,
                "unsupported investigation result",
            )
            status = variant["status"]
        else:
            protocol = _load(provenance)
            _require(
                type(variant.get("derived_replay_schema_version")) is int
                and variant["derived_replay_schema_version"] == 1,
                "unsupported derived replay evidence",
            )
            _require(
                variant["protocol_sha256"] == _sha(_read(provenance)),
                "provenance hash mismatch",
            )
            for key in (
                "control_result_sha256",
                "parent_identity",
                "parent_response_sha256",
                "source_sha256",
                "variant_response_sha256",
                "variant_program_sha256",
                "implementation_sha256",
                "evaluator",
                "task_source",
                "configuration",
                "worker_inputs",
            ):
                _require(
                    protocol[key] == manifest[key],
                    "incompatible saved intervention provenance",
                )
            _require(
                variant["parent_identity"] == manifest["parent_identity"],
                "derived parent mismatch",
            )
            status = "complete"
        _require(
            variant["extraction"] == manifest["extraction"],
            "derived extraction mismatch",
        )
        _require(status in {"complete", "stopped"}, "invalid attempt status")
        _validate_outcomes(
            variant,
            manifest["task_source"]["checks"],
            partial=status == "stopped",
            architecture=control["historical"]["runtime"]["metadata"]["architecture"],
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("incomplete or incompatible derived evidence") from exc
    return _comparison(
        manifest, control, variant, status=status, origin="saved_evidence"
    )


def execute_aria_intervention(
    directory, source, *, artifacts, execution_authorized=False
):
    """Consume one approval durably, then attempt each mode at most once.

    Failures stay spent. Local filesystem ownership is trusted: deleting records
    or bypassing this API is outside the approval mechanism's threat model.
    """
    _require(execution_authorized is True, "explicit execution authorization required")
    directory, manifest, case_sha = _case(directory)
    _require(not (directory / "consumed.json").exists(), "authorization is spent")
    approval = _load(directory / "approval.json")
    _require(
        type(approval.get("approval_schema_version")) is int
        and type(approval.get("max_containers")) is int
        and approval.get("execution_authorized") is True,
        "invalid approval field types",
    )
    _require(
        approval
        == {
            "approval_schema_version": 1,
            "case_sha256": case_sha,
            "variant_response_sha256": manifest["variant_response_sha256"],
            "case_directory": str(directory),
            "max_containers": 2,
            "execution_authorized": True,
        },
        "approval hashes, directory, or budget do not match this case",
    )
    control = _load(directory / "control.json")
    row, task, cases, paths = _inputs(control, source, artifacts)
    response = _read(directory / "variant.txt").decode("utf-8")
    _require(
        response
        == row["response"].replace(manifest["edit"]["old"], manifest["edit"]["new"], 1),
        "canonical edit mismatch",
    )
    workers, extraction = _workers(response, task, paths)
    _require(
        extraction == manifest["extraction"]
        and manifest["worker_inputs"]
        == [
            {"script_sha256": _sha(script.encode()), "payload_sha256": _sha(payload)}
            for script, payload in workers
        ],
        "worker input hash mismatch",
    )
    for name in (
        "result.json",
        "comparison.json",
        "comparison.html",
        "mode-1.json",
        "mode-2.json",
    ):
        _require(not (directory / name).exists(), "execution output already exists")
    _save(
        directory,
        "consumed.json",
        {
            "approval_sha256": _sha(_read(directory / "approval.json")),
            "case_sha256": case_sha,
            "reserved_containers": 2,
            "consumed_at": datetime.now(UTC).isoformat(),
            "status": "spent",
        },
    )
    result = {
        "investigation_result_version": 1,
        "case_id": manifest["case_id"],
        "case_sha256": case_sha,
        "variant_id": manifest["variant_id"],
        "variant_response_sha256": manifest["variant_response_sha256"],
        "extraction": extraction,
        "status": "stopped",
        "mechanism_status": "unresolved",
        "historical": {"status": "unavailable", "reason": "not_attempted"},
        "task": {
            "status": "unavailable",
            "reason": "not_attempted",
            "determinate": False,
        },
    }
    config = boundary.SandboxConfig(docker_host=manifest["docker_host"])
    interruption = None
    try:
        for index, (script, payload) in enumerate(workers):
            _require(index < approval["max_containers"], "container budget exhausted")
            _save(
                directory,
                f"mode-{index + 1}-started.json",
                manifest["worker_inputs"][index],
            )
            raw = None
            try:
                raw = boundary.run_container(
                    script, payload, config, execution_authorized=True
                )
            except BaseException as error:
                raw = getattr(error, "replay_result", None)
                if raw is None:
                    raise
                interruption = error
            if index == 0:
                observation = _historical_observation(raw)
            else:
                observation = replay.assess_answers(
                    replay._json_output(raw.stdout)
                    if raw.status == "completed"
                    else None,
                    cases,
                )
                if raw.status != "completed":
                    observation["reason"] = raw.status
                observation["runtime"] = replay._runtime(raw)
                if not raw.config_verified or not raw.cleanup_verified:
                    observation.update(
                        status="unresolved",
                        determinate=False,
                        test_coverage_complete=False,
                        reason="unverified_runtime_or_cleanup",
                    )
                observation["contract"] = (
                    "all original checks; fresh Solution per call; host equality"
                )
            result["historical" if index == 0 else "task"] = observation
            _save(directory, f"mode-{index + 1}.json", observation)
            if interruption is not None:
                raise interruption
            _validate_runtime(observation["runtime"])
            _validate_architecture(
                (observation,),
                control["historical"]["runtime"]["metadata"]["architecture"],
            )
            if index == 0 and raw.status == "completed":
                _require(
                    observation["status"] == "reported",
                    observation.get("reason", "historical protocol unresolved"),
                )
                reported = observation["reported_result"]
                _require(
                    all(
                        type(reported.get(k)) is int and reported[k] in (0, 1)
                        for k in ("tests_evaluated", "tests_passed")
                    ),
                    "historical counters unresolved",
                )
        _validate_outcomes(result, len(cases))
        result["status"] = "complete"
    except BaseException as error:  # noqa: BLE001 - preserve partial evidence and spent approval
        result["stop_reason"] = (
            str(error)
            if isinstance(error, ValueError)
            else "execution interrupted or infrastructure unavailable"
        )
        result["exception_type"] = type(error).__name__
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            interruption = error
    _save(directory, "result.json", result)
    comparison = _comparison(
        manifest,
        control,
        result,
        status=result["status"],
        origin="authorized_execution",
    )
    _save(directory, "comparison.json", comparison)
    _write_text(render_aria_comparison(comparison), directory / "comparison.html")
    if interruption is not None:
        raise interruption
    return comparison


def render_aria_comparison(comparison):
    """Render claims and host comparisons without embedding response text."""
    _require(
        type(comparison) is dict and comparison.get("comparison_schema_version") == 1,
        "unsupported Aria comparison version",
    )

    def escaped(value):
        return html.escape("Unavailable" if value is None else str(value))

    rows = []
    for label, mode, field in [
        ("Historical claimed passes", "historical", "tests_passed"),
        ("Reported exception types", "historical", "exception_types"),
        ("Protected status", "task", "status"),
        ("Protected matched", "task", "tests_passed"),
        ("Protected mismatched", "task", "tests_failed"),
        ("Protected completed", "task", "tests_completed"),
        ("Protected applicable", "task", "tests_applicable"),
        ("Protected unavailable reason", "task", "reason"),
    ]:
        values = []
        for condition in ("control", "variant"):
            item = comparison[condition][mode]
            value = (
                (item.get("reported_result") or {}).get(field)
                if mode == "historical"
                else item.get(field)
            )
            if field == "reason" and item.get("determinate") is True and value is None:
                value = "Not applicable"
            values.append(value)
        rows.append(
            "<tr><th>"
            + escaped(label)
            + "</th>"
            + "".join("<td>" + escaped(v) + "</td>" for v in values)
            + "</tr>"
        )
    for mode in ("historical", "task"):
        for field in ("status", "stdout_sha256", "cleanup_verified"):
            values = [
                comparison[condition][mode].get("runtime", {}).get(field)
                for condition in ("control", "variant")
            ]
            rows.append(
                "<tr><th>"
                + escaped(mode + " runtime " + field)
                + "</th>"
                + "".join("<td>" + escaped(value) + "</td>" for value in values)
                + "</tr>"
            )
    predictions = "".join(
        "<tr>"
        + "".join(
            "<td>" + escaped(row[key]) + "</td>"
            for key in ("field", "expected", "observed", "status")
        )
        + "</tr>"
        for row in comparison["predictions"]
    )
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Aria intervention comparison</title><style>*{box-sizing:border-box}"
        "body{font:16px/1.5 system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#182128;overflow-wrap:anywhere}"
        ".table-scroll{max-width:100%;overflow-x:auto}"
        "table{border-collapse:collapse;width:100%;min-width:600px}td,th{text-align:left;min-width:100px;padding:8px;border-bottom:1px solid #ddd;overflow-wrap:anywhere}"
        "th:first-child,td:first-child{min-width:180px}"
        "code{overflow-wrap:anywhere}</style><h1>Aria intervention comparison</h1><p>Case: "
        + escaped(comparison["case_id"])
        + "</p>"
        "<p>Canonical parent: " + escaped(comparison["parent_identity"]) + "</p>"
        "<p>Status: "
        + escaped(comparison["status"])
        + "; evidence: "
        + escaped(comparison["origin"])
        + "</p>"
        "<p>Hypothesis: "
        + escaped(comparison["hypothesis"])
        + '</p><div class="table-scroll" tabindex="0" role="region" aria-label="Measurements"><table><tr><th>Measurement</th><th>Frozen control</th><th>Derived variant</th></tr>'
        + "".join(rows)
        + '</table></div><h2>Predictions</h2><div class="table-scroll" tabindex="0" role="region" aria-label="Predictions"><table><tr><th>Field</th><th>Expected</th><th>Observed</th><th>Result</th></tr>'
        + predictions
        + "</table></div>"
        "<p>Protected answer-stream hashes equal: "
        + escaped(comparison["answer_stream_equal"])
        + "</p>"
        "<p>"
        + escaped(comparison["interpretation"])
        + "</p><p>Mechanism status: unresolved. The control was not rerun.</p></html>"
    )
