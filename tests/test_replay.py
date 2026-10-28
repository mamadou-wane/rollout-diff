import hashlib
import json

import pandas as pd
import pytest

from rolloutdiff import replay as module
from rolloutdiff._replay_container import ContainerResult, SandboxConfig
from rolloutdiff.replay import (
    assess_answers,
    replay_aria,
    serialize_replay,
)


def answer(value):
    return {"results": [{"value": value}]}


def test_execution_requires_authorization_before_reading_inputs(tmp_path):
    with pytest.raises(ValueError, match="authorization"):
        replay_aria(
            tmp_path / "missing",
            run="aria",
            sample_id="step001_row000",
            response_sha256="0" * 64,
            artifacts=tmp_path,
        )


def test_host_compares_answers_and_ignores_claimed_verdicts():
    cases = [{"expected": 4}]
    assert assess_answers(answer(["int", 4]), cases)["status"] == "passed"
    assert assess_answers(answer(["int", 5]), cases)["status"] == "failed"
    forged = {"results": [{"passed": True}], "tests_passed": 1}
    result = assess_answers(forged, cases)
    assert result["status"] == "unresolved"
    assert result["tests_passed"] == 0


@pytest.mark.parametrize(
    "raw",
    [
        b'{"results":[{"value":["int",0]}],"results":[{"value":["int",1]}]}',
        b'{"results":[{"value":["int",0],"value":["int",1]}]}',
    ],
)
def test_duplicate_json_members_remain_unresolved(raw):
    payload = module._json_output(raw)
    assert payload is None
    result = assess_answers(payload, [{"expected": 1}])
    assert result["status"] == "unresolved"
    assert result["reason"] == "invalid_answer_protocol"
    assert result["determinate"] is False
    assert result["tests_completed"] == 0


@pytest.mark.parametrize("depth", [0, 1, 8])
def test_json_rejects_duplicate_members_at_every_depth(depth):
    duplicate = b'{"value":0,"val\\u0075e":1}'
    content = b'{"nested":' * depth + duplicate + b"}" * depth
    assert module._json_output(content) is None


def test_nonduplicate_json_answers_preserve_valid_values():
    raw = b'{"results":[{"value":["list",[["int",1],["bool",true]]]}]}'
    result = assess_answers(module._json_output(raw), [{"expected": [1, True]}])
    assert result["status"] == "passed"
    assert result["determinate"] is True


@pytest.mark.parametrize(
    "payload", [{}, {"stdout": {"tests_passed": 1}}, {"success": True}]
)
def test_incomplete_historical_envelopes_remain_unresolved(payload):
    result = module._historical_report(
        ContainerResult("completed", stdout=json.dumps(payload).encode())
    )
    assert result["status"] == "unresolved"
    assert result["reported_result"] is None


def test_python_equality_preserves_bool_int_and_exact_float_behavior():
    assert (
        assess_answers(answer(["bool", True]), [{"expected": 1}])["status"] == "passed"
    )
    assert (
        assess_answers(answer(["float", 0.1 + 0.2]), [{"expected": 0.3}])["status"]
        == "failed"
    )
    assert (
        assess_answers(answer(["float", "inf"]), [{"expected": float("inf")}])["status"]
        == "passed"
    )


def test_missing_extra_or_invalid_answers_never_establish_a_pass():
    for payload in [
        {"results": []},
        {"results": [{"value": ["int", 1]}] * 2},
        {},
        answer(["int", True]),
        answer(["object", "AlwaysEqual"]),
        answer([[], 1]),
    ]:
        result = assess_answers(payload, [{"expected": 1}])
        assert result["status"] == "unresolved"
        assert not result["determinate"]


def test_reported_exceptions_are_untrusted_and_distinct_from_wrong_answers():
    result = assess_answers(
        {"results": [{"exception": "RuntimeError"}]}, [{"expected": 1}]
    )
    assert result["status"] == "unresolved"
    assert result["tests_completed"] == 0
    assert result["reported_exceptions"] == ["RuntimeError"]


@pytest.mark.parametrize(
    "second",
    [
        {"exception": "RuntimeError"},
        {"value": ["int", True]},
        {"value": ["object", "unsupported"]},
        {},
        {"value": ["int", 2], "passed": True},
    ],
)
def test_mixed_incomplete_answers_preserve_comparisons_without_a_verdict(
    monkeypatch, second
):
    payload = {"results": [{"value": ["int", 0]}, second, {"value": ["int", 3]}]}
    comparison = assess_answers(payload, [{"expected": i} for i in (1, 2, 3)])
    assert comparison["tests_failed"] == 1
    assert comparison["tests_passed"] == 1
    assert comparison["tests_completed"] == 2
    assert comparison["test_coverage_complete"] is False
    assert comparison["status"] == "unresolved"
    assert comparison["determinate"] is False
    assert comparison["reason"] == "incomplete_or_unsupported_answers"
    assert comparison["reported_exceptions"] == (
        ["RuntimeError"] if "exception" in second else []
    )

    task = {
        "setup_code": "",
        "func_name": "Solution().answer",
        "gt_answer": [f"assert Solution().answer(x={i}) == {i}" for i in (1, 2, 3)],
    }
    source = "```python\nclass Solution:\n def answer(self,x): return 0\ndef run_tests(): pass\n```"
    historical = {
        "success": True,
        "compiled": True,
        "timeout": False,
        "oom": False,
        "stdout": {"tests_evaluated": 1, "tests_passed": 1},
    }
    outputs = iter([historical, payload])
    monkeypatch.setattr(module, "historical_script", lambda *a: ("unused", b""))

    def run(*args, execution_authorized):
        assert execution_authorized is True
        return ContainerResult(
            "completed",
            stdout=json.dumps(next(outputs)).encode(),
            cleanup_verified=True,
        )

    monkeypatch.setattr(module, "run_container", run)
    result = module._replay_response(source, task, {}, SandboxConfig())
    restored = json.loads(serialize_replay({"replay_schema_version": 1, **result}))
    assert all(restored["task"][key] == value for key, value in comparison.items())
    assert restored["evidence_status"] == "unresolved"
    assert restored["successful_hack_hypothesis"] == "unresolved"
    assert restored["mechanism_status"] == "unresolved"
    assert restored["historical"]["reported_result"]["tests_passed"] == 1


@pytest.mark.parametrize("first,status", [(0, "failed"), (1, "passed")])
def test_complete_answers_preserve_determinate_outcomes(first, status):
    result = assess_answers(
        {"results": [{"value": ["int", first]}, {"value": ["int", 2]}]},
        [{"expected": 1}, {"expected": 2}],
    )
    assert result["status"] == status
    assert result["determinate"] is True
    assert result["test_coverage_complete"] is True
    assert result["tests_completed"] == 2
    assert result["tests_failed"] == (first != 1)
    assert result["reason"] is None


@pytest.mark.parametrize("values,completed,passed", [([0], 1, 0), ([0, 2, 99], 2, 1)])
def test_answer_count_mismatch_retains_known_positional_comparisons(
    values, completed, passed
):
    result = assess_answers(
        {"results": [{"value": ["int", value]} for value in values]},
        [{"expected": 1}, {"expected": 2}],
    )
    assert result["tests_completed"] == completed
    assert result["tests_failed"] == 1
    assert result["tests_passed"] == passed
    assert result["test_coverage_complete"] is False
    assert result["determinate"] is False
    assert result["status"] == "unresolved"
    assert result["reason"] == "incomplete_or_extra_answers"


def test_serialization_is_stable_and_never_silently_encodes_nonfinite_values():
    record = {
        "replay_schema_version": 1,
        "identity": {"run": "aria", "sample_id": "step001_row000"},
        "unavailable": None,
    }
    first = serialize_replay(record)
    second = serialize_replay(dict(reversed(list(record.items()))))
    assert first == second
    assert json.loads(first) == record
    assert (
        hashlib.sha256(first.encode()).digest()
        == hashlib.sha256(second.encode()).digest()
    )
    with pytest.raises(ValueError):
        serialize_replay({"replay_schema_version": 1, "bad": float("nan")})
    with pytest.raises(ValueError, match="version"):
        serialize_replay({"replay_schema_version": 2})


def source_rows(tmp_path, monkeypatch):
    path = tmp_path / "source.parquet"
    path.write_bytes(b"pinned synthetic source")
    monkeypatch.setattr(
        module, "ARIA_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    response = "private response sentinel"
    rows = pd.DataFrame(
        [
            {
                "sample_id": "step001_row000",
                "step": 1,
                "problem_id": 7,
                "messages": json.dumps(
                    [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": "private prompt sentinel"},
                        {"role": "assistant", "content": response},
                    ]
                ),
            }
        ]
    )
    monkeypatch.setattr(pd, "read_parquet", lambda *a, **k: rows)
    return path, rows, hashlib.sha256(response.encode()).hexdigest()


def test_exact_source_and_response_identity_are_required(tmp_path, monkeypatch):
    path, rows, digest = source_rows(tmp_path, monkeypatch)
    selected = module._selected_source(path, "aria", "step001_row000", digest)
    assert selected["prompt_id"] == "7"
    with pytest.raises(ValueError, match="response checksum"):
        module._selected_source(path, "aria", "step001_row000", "0" * 64)
    for invalid in ["other-run", None]:
        with pytest.raises(ValueError, match="canonical identity"):
            module._selected_source(path, invalid, "step001_row000", digest)
    for invalid_rows in [rows.iloc[:0], pd.concat([rows, rows])]:
        monkeypatch.setattr(pd, "read_parquet", lambda *a, rows=invalid_rows, **k: rows)
        with pytest.raises(ValueError, match="exactly one"):
            module._selected_source(path, "aria", "step001_row000", digest)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="source checksum"):
        module._selected_source(path, "aria", "step001_row000", digest)


def test_missing_artifacts_produces_unavailable_record_without_docker(
    tmp_path, monkeypatch
):
    path, _, digest = source_rows(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module, "run_container", lambda *a, **k: pytest.fail("container started")
    )
    result = replay_aria(
        path,
        run="aria",
        sample_id="step001_row000",
        response_sha256=digest,
        artifacts=tmp_path,
        execution_authorized=True,
    )
    assert result["task"]["status"] == "unavailable"
    assert result["mechanism_status"] == "unresolved"
    assert "private response sentinel" not in serialize_replay(result)
    assert "private prompt sentinel" not in serialize_replay(result)


def test_expected_answers_never_enter_worker_and_no_raw_text_leaves_record(monkeypatch):
    task = {
        "setup_code": "",
        "func_name": "Solution().answer",
        "gt_answer": ["assert Solution().answer(x=2) == 'SECRET_CHECKER_TOKEN'"],
    }
    source = "```python\nclass Solution:\n def answer(self,x): return 'wrong'\n def run_tests(self): pass\n```"
    monkeypatch.setattr(
        module,
        "historical_script",
        lambda *a: ("fixture historical runner", b"candidate only"),
    )
    payloads = []

    def run(script, payload, config, *, execution_authorized):
        assert execution_authorized
        payloads.append(payload)
        output = (
            {
                "success": True,
                "compiled": True,
                "timeout": False,
                "oom": False,
                "stdout": {"tests_evaluated": 1, "tests_passed": 1},
            }
            if len(payloads) == 1
            else answer(["str", "wrong"])
        )
        return ContainerResult(
            "completed",
            exit_code=0,
            stdout=json.dumps(output).encode(),
            stderr=b"private diagnostic text",
            config_verified=True,
            cleanup_verified=True,
        )

    monkeypatch.setattr(module, "run_container", run)
    result = module._replay_response(source, task, {}, SandboxConfig())
    assert all(b"SECRET_CHECKER_TOKEN" not in value for value in payloads)
    assert result["historical"]["authority"] == "candidate_controlled_historical_report"
    assert result["task"]["status"] == "failed"
    assert result["evidence_status"] == "reproduced_discrepancy"
    assert result["mechanism_status"] == "unresolved"
    serialized = serialize_replay({"replay_schema_version": 1, **result})
    assert "private diagnostic text" not in serialized
    assert source not in serialized
    assert "SECRET_CHECKER_TOKEN" not in serialized


@pytest.mark.parametrize("cancelled", [False, True])
def test_unverified_cleanup_prevents_next_evaluation_mode(monkeypatch, cancelled):
    task = {
        "setup_code": "",
        "func_name": "Solution().answer",
        "gt_answer": ["assert Solution().answer() == 1"],
    }
    source = "```python\nclass Solution:\n def answer(self): return 1\ndef run_tests(): pass\n```"
    monkeypatch.setattr(module, "historical_script", lambda *a: ("unused", b""))
    calls = []
    failure = ContainerResult(
        "infrastructure_error",
        cleanup_verified=False,
        metadata={"container_id": "c" * 64},
    )
    interruption = KeyboardInterrupt("controlled cleanup uncertainty")
    interruption.replay_result = failure

    def run(*args, **kwargs):
        calls.append(True)
        if cancelled:
            raise interruption
        return failure

    monkeypatch.setattr(module, "run_container", run)
    if cancelled:
        with pytest.raises(KeyboardInterrupt) as caught:
            module._replay_response(source, task, {}, SandboxConfig())
        assert caught.value is interruption
        assert caught.value.replay_result is failure
    else:
        result = module._replay_response(source, task, {}, SandboxConfig())
        assert result["task"]["status"] == "unavailable"
        assert result["task"]["reason"] == "historical_container_infrastructure_failure"
        assert result["evidence_status"] == "unresolved"
    assert len(calls) == 1
