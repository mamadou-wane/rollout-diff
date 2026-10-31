import copy
import json
import os
from pathlib import Path

import pytest
from test_investigation import (
    approve,
    digest,
    historical,
    prepare,
    raw_result,
    write_json,
)
from test_investigation import fixture as fixture  # noqa: PLC0414 - pytest re-export

from rolloutdiff import investigation as mod


def test_reject_misattributed_manifest_before_approval(fixture):
    prepared = prepare(fixture)
    path = fixture["directory"] / "case.json"
    manifest = json.loads(path.read_text())
    manifest["parent_identity"]["sample_id"] = "step099_row099"
    write_json(path, manifest)
    with pytest.raises(ValueError):
        mod.approve_aria_intervention(
            fixture["directory"],
            case_sha256=digest(path.read_bytes()),
            variant_sha256=prepared["variant_response_sha256"],
            max_containers=2,
            execution_authorized=True,
        )


def test_reject_historical_claim_after_timeout(fixture):
    record = copy.deepcopy(fixture["record"])
    record["historical"]["runtime"]["status"] = "timeout"
    record["historical"]["runtime"]["exit_code"] = None
    write_json(fixture["control"], record)
    with pytest.raises(ValueError):
        prepare(fixture)


def test_stopped_record_does_not_disable_evidence_validation(fixture, tmp_path):
    prepared = prepare(fixture)
    record = {
        key: copy.deepcopy(fixture["record"][key])
        for key in ("historical", "task", "mechanism_status")
    }
    record.update(
        investigation_result_version=1,
        case_id=prepared["case_id"],
        case_sha256=prepared["case_sha256"],
        variant_response_sha256=prepared["variant_response_sha256"],
        extraction=prepared["extraction"],
        status="stopped",
    )
    record["task"]["tests_applicable"] = 1
    record["task"]["tests_completed"] = 999
    record["task"]["tests_passed"] = 998
    result = tmp_path / "stopped.json"
    write_json(result, record)
    with pytest.raises(ValueError):
        mod.compare_aria_intervention(fixture["directory"], result)


def test_fifo_is_rejected_before_blocking_read(tmp_path, monkeypatch):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    monkeypatch.setattr(
        Path, "read_bytes", lambda self: pytest.fail("attempted read on FIFO")
    )
    with pytest.raises(ValueError):
        mod._read(fifo)


@pytest.mark.parametrize(
    "field,value",
    [
        ("approval_schema_version", True),
        ("execution_authorized", 1),
        ("max_containers", 2.0),
    ],
)
def test_approval_rejects_equal_but_incorrect_json_types(
    fixture, monkeypatch, field, value
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    path = fixture["directory"] / "approval.json"
    approval = json.loads(path.read_text())
    approval[field] = value
    write_json(path, approval)
    calls = []

    def transport(*args, **kwargs):
        calls.append(True)
        return raw_result(historical(0))

    monkeypatch.setattr(mod.boundary, "run_container", transport)
    with pytest.raises(ValueError):
        mod.execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert not calls
    assert not (fixture["directory"] / "consumed.json").exists()


@pytest.mark.parametrize("flag", ["cleanup_verified", "config_verified"])
def test_partial_import_rejects_determinate_task_from_unverified_runtime(
    fixture, tmp_path, flag
):
    prepared = prepare(fixture)
    record = {
        key: copy.deepcopy(fixture["record"][key])
        for key in ("historical", "task", "mechanism_status")
    }
    record.update(
        investigation_result_version=1,
        case_id=prepared["case_id"],
        case_sha256=prepared["case_sha256"],
        variant_response_sha256=prepared["variant_response_sha256"],
        extraction=prepared["extraction"],
        status="stopped",
    )
    record["task"]["runtime"][flag] = False
    path = tmp_path / "unverified.json"
    write_json(path, record)
    with pytest.raises(ValueError):
        mod.compare_aria_intervention(fixture["directory"], path)


@pytest.mark.parametrize("metadata", [None, [], "unexpected"])
def test_malformed_runtime_metadata_raises_value_error(fixture, metadata):
    fixture["record"]["historical"]["runtime"]["metadata"] = metadata
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError):
        prepare(fixture)


@pytest.mark.parametrize(
    "path,value",
    [
        (("limits", "wall_seconds"), 60),
        (("host_config", "Memory"), 2147483648),
        (("host_config", "Tmpfs", "/tmp"), "rw,size=999999999"),
        (("ulimits", "cpu"), 300),
    ],
)
def test_r1_returned_configuration_cannot_mutate_policy_or_other_cases(
    fixture, path, value
):
    policies = (mod._LIMITS, mod.boundary._HOST_CONFIG, mod.boundary._ULIMITS)
    before = copy.deepcopy(policies)
    try:
        first = prepare(fixture)
        saved = (fixture["directory"] / "case.json").read_bytes()
        target = first["configuration"]
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value
        assert policies == before
        assert (fixture["directory"] / "case.json").read_bytes() == saved
        fixture["directory"] = fixture["directory"].with_name("second-case")
        second = prepare(fixture)
        assert second["configuration"]["limits"] == before[0]
        assert second["configuration"]["host_config"] == before[1]
        assert second["configuration"]["ulimits"] == before[2]
        assert second["configuration"]["cpu_limit_seconds"] == 3
        approve(fixture, second)
    finally:
        for policy, original in zip(policies, before, strict=True):
            policy.clear()
            policy.update(original)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tests_passed", 2),
        ("tests_evaluated", 2),
        ("tests_passed", True),
        ("tests_passed", None),
    ],
)
def test_r2_invalid_counter_stopped_record_round_trips(
    fixture, monkeypatch, field, value
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    payload = historical(0)
    payload["stdout"][field] = value
    calls = []

    def run(*args, **kwargs):
        calls.append(True)
        return raw_result(payload)

    monkeypatch.setattr(mod.boundary, "run_container", run)
    report = mod.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert report["status"] == "stopped" and len(calls) == 1
    restored = mod.compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    assert restored["variant"] == report["variant"]
    assert restored["variant"]["historical"]["reported_result"] is None
    assert restored["variant"]["historical"]["reason"] == "invalid_historical_counters"
    assert all(row["status"] == "unavailable" for row in restored["predictions"])
    assert (fixture["directory"] / "consumed.json").exists()
    with pytest.raises(ValueError, match="spent"):
        mod.execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["historical", "task"])
def test_r3_import_rejects_architecture_mismatch(fixture, tmp_path, mode):
    from test_investigation import archived_result

    prepared = prepare(fixture)
    path, protocol, record = archived_result(fixture, prepared, tmp_path)
    record[mode]["runtime"]["metadata"]["architecture"] = "amd64"
    write_json(path, record)
    with pytest.raises(ValueError, match="architecture"):
        mod.compare_aria_intervention(fixture["directory"], path, provenance=protocol)


@pytest.mark.parametrize("field", ["tests_passed", "tests_evaluated"])
@pytest.mark.parametrize("value", [-1, False, 1.0, "1", [], {}, "missing"])
def test_r2_counter_observations_preserve_invalid_or_missing_values(
    fixture, monkeypatch, field, value
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    payload = historical(0)
    if value == "missing":
        del payload["stdout"][field]
    else:
        payload["stdout"][field] = value
    calls = []

    def run(*args, **kwargs):
        calls.append(True)
        return raw_result(payload)

    monkeypatch.setattr(mod.boundary, "run_container", run)
    report = mod.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert len(calls) == 1 and report["status"] == "stopped"
    restored = mod.compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    observation = restored["variant"]["historical"]
    assert observation["status"] == "unresolved"
    assert observation["reported_result"] is None
    assert observation["reason"] == "invalid_historical_counters"
    assert observation["counter_observations"][field] == {
        "present": value != "missing",
        "value": None if value == "missing" else value,
    }
    assert observation["runtime"]["stdout_sha256"] == digest(
        json.dumps(payload).encode()
    )
    assert restored["control"] == report["control"]
    assert all(check["status"] == "unavailable" for check in restored["predictions"])


@pytest.mark.parametrize("evaluated,passed", [(0, 0), (1, 0), (1, 1), (0, 1)])
def test_r2_valid_counters_round_trip_without_reinterpretation(
    fixture, monkeypatch, evaluated, passed
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    payload = historical(passed)
    payload["stdout"]["tests_evaluated"] = evaluated
    outputs = iter(
        [
            raw_result(payload),
            raw_result({"results": [{"value": ["int", 1]}, {"value": ["int", 3]}]}, 1),
        ]
    )
    monkeypatch.setattr(mod.boundary, "run_container", lambda *a, **k: next(outputs))
    report = mod.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    restored = mod.compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    assert report["status"] == restored["status"] == "complete"
    claim = restored["variant"]["historical"]["reported_result"]
    assert claim["tests_passed"] == passed and claim["tests_evaluated"] == evaluated


def test_r2_unaffected_protected_observations_survive_partial_import(
    fixture, monkeypatch
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    payload = historical(0)
    payload["stdout"]["tests_passed"] = True
    monkeypatch.setattr(
        mod.boundary, "run_container", lambda *a, **k: raw_result(payload)
    )
    mod.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    saved = json.loads((fixture["directory"] / "result.json").read_text())
    saved["task"] = copy.deepcopy(fixture["record"]["task"])
    path = fixture["directory"] / "partial-with-protected.json"
    write_json(path, saved)
    comparison = mod.compare_aria_intervention(fixture["directory"], path)
    assert comparison["variant"]["task"] == fixture["record"]["task"]
    assert comparison["variant"]["task"]["determinate"] is True
    assert (
        next(
            row
            for row in comparison["predictions"]
            if row["field"] == "historical_passed"
        )["status"]
        == "unavailable"
    )
    assert (
        next(
            row for row in comparison["predictions"] if row["field"] == "task_matched"
        )["status"]
        == "matched"
    )


@pytest.mark.parametrize("architecture", ["arm64", "amd64"])
def test_r3_matching_architectures_remain_compatible(fixture, tmp_path, architecture):
    from test_investigation import archived_result

    for mode in ("historical", "task"):
        fixture["record"][mode]["runtime"]["metadata"]["architecture"] = architecture
    write_json(fixture["control"], fixture["record"])
    prepared = prepare(fixture)
    path, protocol, _ = archived_result(fixture, prepared, tmp_path)
    report = mod.compare_aria_intervention(
        fixture["directory"], path, provenance=protocol
    )
    assert report["status"] == "complete"
    assert all(item["status"] == "matched" for item in report["predictions"])


def test_r3_control_modes_require_matching_architectures(fixture):
    fixture["record"]["task"]["runtime"]["metadata"]["architecture"] = "amd64"
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError, match="architecture"):
        prepare(fixture)


@pytest.mark.parametrize("mode", ["historical", "task"])
@pytest.mark.parametrize("status", ["complete", "stopped"])
def test_r3_native_import_checks_architecture_even_when_stopped(
    fixture, tmp_path, mode, status
):
    prepared = prepare(fixture)
    record = {
        key: copy.deepcopy(fixture["record"][key])
        for key in ("historical", "task", "mechanism_status")
    }
    record.update(
        investigation_result_version=1,
        case_id=prepared["case_id"],
        case_sha256=prepared["case_sha256"],
        variant_response_sha256=prepared["variant_response_sha256"],
        extraction=prepared["extraction"],
        status=status,
    )
    record[mode]["runtime"]["metadata"]["architecture"] = "amd64"
    path = tmp_path / "imported.json"
    write_json(path, record)
    with pytest.raises(ValueError, match="architecture"):
        mod.compare_aria_intervention(fixture["directory"], path)


@pytest.mark.parametrize("field", ["tests_passed", "tests_evaluated"])
@pytest.mark.parametrize("token", ["1e309", "-1e309", '{"nested":1e309}'])
def test_nonfinite_counter_diagnostics_are_saved_and_reimported(
    fixture, monkeypatch, field, token
):
    from dataclasses import replace

    prepared = prepare(fixture)
    approve(fixture, prepared)
    payload = historical(0)
    payload["stdout"][field] = "OVERFLOW_COUNTER"
    raw = json.dumps(payload).replace('"OVERFLOW_COUNTER"', token).encode()
    calls = []

    def transport(*args, **kwargs):
        calls.append(True)
        return replace(raw_result(historical(0)), stdout=raw)

    monkeypatch.setattr(mod.boundary, "run_container", transport)
    report = mod.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert report["status"] == "stopped" and len(calls) == 1
    for filename in ("mode-1.json", "result.json", "comparison.json"):
        saved = json.loads((fixture["directory"] / filename).read_text())
        json.dumps(saved, allow_nan=False)
    assert (fixture["directory"] / "comparison.html").is_file()
    restored = mod.compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    assert restored["variant"] == report["variant"]
    observation = restored["variant"]["historical"]
    assert (
        observation["status"] == "unresolved" and observation["reported_result"] is None
    )
    assert observation["reason"] == "invalid_historical_counters"
    assert observation["counter_observations"][field]["present"] is True
    assert observation["runtime"]["stdout_sha256"] == digest(raw)
    assert all(item["status"] == "unavailable" for item in restored["predictions"])
    assert (fixture["directory"] / "consumed.json").exists()
    with pytest.raises(ValueError, match="spent"):
        mod.execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert len(calls) == 1


@pytest.mark.parametrize(
    "value,kind",
    [
        (float("inf"), "float"),
        (float("-inf"), "float"),
        ({"nested": [0, {"overflow": float("inf")}]}, "dict"),
        ([{"negative": float("-inf")}, float("nan")], "list"),
    ],
)
def test_invalid_diagnostic_representation_is_bounded_and_strict(value, kind):
    observed = mod._counter_diagnostic(value)
    assert observed["invalid_value"] == "non_finite_number"
    assert observed["value_type"] == kind
    if kind == "float":
        assert observed["representation"] == repr(value)
    assert len(json.dumps(observed, allow_nan=False)) < 150


@pytest.mark.parametrize(
    "value", [None, False, 1.0, 1e308, {"nested": [1, "bad", None]}, []]
)
def test_finite_invalid_counter_values_remain_unchanged(value):
    before = copy.deepcopy(value)
    assert mod._counter_diagnostic(value) == value == before
    json.dumps(mod._counter_diagnostic(value), allow_nan=False)


def test_diagnostic_recursion_failure_keeps_serializer_strict(monkeypatch):
    def recursion_failure(*args, **kwargs):
        raise RecursionError

    with monkeypatch.context() as scoped:
        scoped.setattr(mod.json, "dumps", recursion_failure)
        observed = mod._counter_diagnostic([])
    assert observed == {"invalid_value": "nesting_limit", "value_type": "list"}
    json.dumps(observed, allow_nan=False)
    with pytest.raises(ValueError):
        mod._json({"invalid": float("inf")})
