import json

import pytest

from rolloutdiff import replay as module
from rolloutdiff._replay_container import ContainerResult
from rolloutdiff.cli import main


def arguments(tmp_path):
    return [
        "replay-aria",
        str(tmp_path / "aria.parquet"),
        "--run",
        "aria",
        "--sample-id",
        "step111_row233",
        "--response-sha256",
        "0" * 64,
        "--artifacts",
        str(tmp_path),
        "--docker-host",
        "unix:///var/run/docker.sock",
        "--output",
        str(tmp_path / "replay.json"),
    ]


def test_cli_requires_explicit_execution_and_never_leaves_partial_output(
    tmp_path, capsys
):
    assert main(arguments(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "authorization" in captured.err
    assert "Traceback" not in captured.err
    assert not (tmp_path / "replay.json").exists()


def test_cli_preserves_existing_output_without_invoking_replay(
    tmp_path, monkeypatch, capsys
):
    target = tmp_path / "replay.json"
    target.write_text("keep")
    monkeypatch.setattr(
        "rolloutdiff.cli.replay_aria",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("executed")),
    )
    assert main([*arguments(tmp_path), "--allow-execution"]) == 1
    assert target.read_text() == "keep"
    assert "Traceback" not in capsys.readouterr().err


def test_cli_writes_structured_evidence_only_and_preserves_incomplete_status(
    tmp_path, monkeypatch, capsys
):
    evidence = {
        "replay_schema_version": 1,
        "historical": {"status": "reported"},
        "task": {"status": "unresolved", "determinate": False},
        "mechanism_status": "unresolved",
    }

    def fake(*args, **kwargs):
        assert kwargs["execution_authorized"] is True
        return evidence

    monkeypatch.setattr("rolloutdiff.cli.replay_aria", fake)
    assert main([*arguments(tmp_path), "--allow-execution"]) == 1
    assert json.loads((tmp_path / "replay.json").read_text()) == evidence
    captured = capsys.readouterr()
    assert "historical=reported" in captured.out
    assert "task=unresolved" in captured.out


@pytest.mark.parametrize(
    "raw,status,exit_code",
    [
        (
            b'{"results":[],"results":[{"value":["int",1]}]}',
            "unresolved",
            1,
        ),
        (
            b'{"results":[{"value":["int",0],"value":["int",1]}]}',
            "unresolved",
            1,
        ),
        (b'{"results":[{"value":["int",1]}]}', "passed", 0),
    ],
)
def test_cli_duplicate_json_remains_unresolved_through_replay(
    tmp_path, monkeypatch, capsys, raw, status, exit_code
):
    source = "```python\nclass Solution:\n def answer(self): return 1\ndef run_tests(): pass\n```"
    row = {
        "run": "aria",
        "sample_id": "step111_row233",
        "step": 111,
        "prompt_id": "7",
        "prompt": "handwritten fixture",
        "response": source,
    }
    task = {
        "setup_code": "",
        "func_name": "Solution().answer",
        "gt_answer": ["assert Solution().answer() == 1"],
    }
    fixture = tmp_path / "task.json"
    fixture.write_text(json.dumps(task))
    monkeypatch.setattr(module, "_selected_source", lambda *a: row)
    monkeypatch.setattr(module, "verify_artifacts", lambda *a: {"fixture": fixture})
    monkeypatch.setattr(module, "load_task", lambda *a: task)
    monkeypatch.setattr(module, "historical_script", lambda *a: ("unused", b""))
    historical = b'{"success":true,"compiled":true,"timeout":false,"oom":false,"stdout":{"tests_evaluated":1,"tests_passed":1}}'
    outputs = iter([historical, raw])

    def run(*args, execution_authorized):
        assert execution_authorized is True
        return ContainerResult("completed", stdout=next(outputs), cleanup_verified=True)

    monkeypatch.setattr(module, "run_container", run)
    assert main([*arguments(tmp_path), "--allow-execution"]) == exit_code
    result = json.loads((tmp_path / "replay.json").read_text())
    assert result["task"]["status"] == status
    assert result["task"]["determinate"] is (status == "passed")
    assert result["mechanism_status"] == "unresolved"
    assert result["evidence_status"] == (
        "observed_behavior" if status == "passed" else "unresolved"
    )
    assert result["successful_hack_hypothesis"] == (
        "unsupported_under_tested_conditions" if status == "passed" else "unresolved"
    )
    captured = capsys.readouterr()
    assert f"task={status}" in captured.out
    assert not captured.err
