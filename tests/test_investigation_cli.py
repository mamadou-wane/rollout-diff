import json
from pathlib import Path

import pytest

from rolloutdiff import cli


def prepare_args(tmp_path):
    (tmp_path / "edit.json").write_text(
        json.dumps({"old": "print(1)", "new": "assert 1"})
    )
    (tmp_path / "predictions.json").write_text(json.dumps({"historical_passed": 0}))
    return [
        "investigate-aria",
        "prepare",
        str(tmp_path / "control.json"),
        "--source",
        str(tmp_path / "aria.parquet"),
        "--artifacts",
        str(tmp_path / "artifacts"),
        "--edit",
        str(tmp_path / "edit.json"),
        "--hypothesis",
        "An assertion changes the report.",
        "--predictions",
        str(tmp_path / "predictions.json"),
        "--docker-host",
        "unix:///var/run/docker.sock",
        "--output",
        str(tmp_path / "case"),
    ]


def forbidden(*args, **kwargs):
    pytest.fail("unexpected workflow call")


def test_prepare_reads_explicit_edit_and_predictions(tmp_path, monkeypatch, capsys):
    def prepare(control, source, **kwargs):
        assert Path(control).name == "control.json"
        assert Path(source).name == "aria.parquet"
        assert kwargs["edit"] == {"old": "print(1)", "new": "assert 1"}
        assert kwargs["predictions"] == {"historical_passed": 0}
        assert kwargs["hypothesis"] == "An assertion changes the report."
        assert kwargs["config"].docker_host == "unix:///var/run/docker.sock"
        assert Path(kwargs["directory"]).name == "case"
        return {"case_sha256": "a" * 64, "variant_response_sha256": "b" * 64}

    monkeypatch.setattr(cli, "prepare_aria_intervention", prepare, raising=False)
    assert cli.main(prepare_args(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "a" * 64 in output
    assert "b" * 64 in output
    assert "print(1)" not in output


@pytest.mark.parametrize(
    "raw", ['{"old":"x","old":"y","new":"z"}', '{"old":NaN}', "[]"]
)
def test_prepare_rejects_invalid_json_before_workflow(
    tmp_path, monkeypatch, capsys, raw
):
    args = prepare_args(tmp_path)
    (tmp_path / "edit.json").write_text(raw)
    monkeypatch.setattr(cli, "prepare_aria_intervention", forbidden, raising=False)
    assert cli.main(args) == 1
    assert "JSON object" in capsys.readouterr().err
    assert not (tmp_path / "case").exists()


def test_prepare_preserves_existing_output_directory(tmp_path, monkeypatch, capsys):
    args = prepare_args(tmp_path)
    (tmp_path / "case").mkdir()
    (tmp_path / "case" / "keep").write_text("original")
    monkeypatch.setattr(cli, "prepare_aria_intervention", forbidden, raising=False)
    assert cli.main(args) == 1
    assert (tmp_path / "case" / "keep").read_text() == "original"
    assert "Traceback" not in capsys.readouterr().err


@pytest.mark.parametrize("allowed", [False, True])
def test_approval_passes_exact_hashes_budget_and_authorization(
    tmp_path, monkeypatch, capsys, allowed
):
    def approve(directory, **kwargs):
        assert Path(directory) == tmp_path
        assert kwargs == {
            "case_sha256": "a" * 64,
            "variant_sha256": "b" * 64,
            "max_containers": 2,
            "execution_authorized": allowed,
        }
        if not allowed:
            raise ValueError("explicit execution authorization required")
        return {"status": "approved"}

    monkeypatch.setattr(cli, "approve_aria_intervention", approve, raising=False)
    args = [
        "investigate-aria",
        "approve",
        str(tmp_path),
        "--case-sha256",
        "a" * 64,
        "--variant-sha256",
        "b" * 64,
        "--max-containers",
        "2",
    ]
    if allowed:
        args.append("--allow-execution")
    assert cli.main(args) == (0 if allowed else 1)
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err


@pytest.mark.parametrize("allowed", [False, True])
def test_run_requires_explicit_invocation(tmp_path, monkeypatch, capsys, allowed):
    def execute(directory, source, **kwargs):
        assert Path(directory) == tmp_path
        assert Path(source).name == "aria.parquet"
        assert kwargs["execution_authorized"] is allowed
        if not allowed:
            raise ValueError("explicit execution authorization required")
        return {"status": "completed"}

    monkeypatch.setattr(cli, "execute_aria_intervention", execute, raising=False)
    args = [
        "investigate-aria",
        "run",
        str(tmp_path),
        "--source",
        str(tmp_path / "aria.parquet"),
        "--artifacts",
        str(tmp_path / "artifacts"),
    ]
    if allowed:
        args.append("--allow-execution")
    assert cli.main(args) == (0 if allowed else 1)
    assert "Traceback" not in capsys.readouterr().err


def compare_args(tmp_path):
    return [
        "investigate-aria",
        "compare",
        str(tmp_path / "case"),
        str(tmp_path / "result.json"),
        "--provenance",
        str(tmp_path / "protocol.json"),
        "--output",
        str(tmp_path / "comparison.json"),
        "--html",
        str(tmp_path / "comparison.html"),
    ]


def test_compare_saved_records_without_execution(tmp_path, monkeypatch, capsys):
    record = {"status": "completed", "historical_authenticated": False}

    def compare(directory, result, *, provenance):
        assert Path(directory).name == "case"
        assert Path(result).name == "result.json"
        assert Path(provenance).name == "protocol.json"
        return record

    monkeypatch.setattr(cli, "compare_aria_intervention", compare, raising=False)
    monkeypatch.setattr(
        cli,
        "render_aria_comparison",
        lambda value: (
            "<html>comparison</html>"
            if value == record
            else pytest.fail("wrong report")
        ),
        raising=False,
    )
    monkeypatch.setattr(cli, "execute_aria_intervention", forbidden, raising=False)
    assert cli.main(compare_args(tmp_path)) == 0
    assert json.loads((tmp_path / "comparison.json").read_text()) == record
    assert (tmp_path / "comparison.html").read_text() == "<html>comparison</html>"
    assert "untrusted" in capsys.readouterr().out


@pytest.mark.parametrize("existing", ["comparison.json", "comparison.html"])
def test_compare_preserves_existing_outputs_before_reading_inputs(
    tmp_path, monkeypatch, capsys, existing
):
    (tmp_path / existing).write_text("keep")
    monkeypatch.setattr(cli, "compare_aria_intervention", forbidden, raising=False)
    assert cli.main(compare_args(tmp_path)) == 1
    assert (tmp_path / existing).read_text() == "keep"
    assert "Traceback" not in capsys.readouterr().err


def test_compare_rejects_colliding_outputs(tmp_path, monkeypatch, capsys):
    args = compare_args(tmp_path)
    args[-1] = str(tmp_path / "comparison.json")
    monkeypatch.setattr(cli, "compare_aria_intervention", forbidden, raising=False)
    assert cli.main(args) == 1
    assert not (tmp_path / "comparison.json").exists()
    assert "distinct" in capsys.readouterr().err


def test_render_failure_publishes_neither_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        cli, "compare_aria_intervention", lambda *a, **k: {}, raising=False
    )

    def fail(_):
        raise ValueError("malformed comparison")

    monkeypatch.setattr(cli, "render_aria_comparison", fail, raising=False)
    assert cli.main(compare_args(tmp_path)) == 1
    assert not (tmp_path / "comparison.json").exists()
    assert not (tmp_path / "comparison.html").exists()
    assert "Traceback" not in capsys.readouterr().err


def test_workflow_errors_escape_terminal_controls(tmp_path, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("failure\x1b[31m\rhidden")

    monkeypatch.setattr(cli, "compare_aria_intervention", fail, raising=False)
    assert cli.main(compare_args(tmp_path)) == 1
    captured = capsys.readouterr()
    assert "\x1b" not in captured.err and "\r" not in captured.err
    assert "\\x1b" in captured.err


def test_public_workflow_exports():
    import rolloutdiff

    for name in (
        "prepare_aria_intervention",
        "approve_aria_intervention",
        "execute_aria_intervention",
        "compare_aria_intervention",
        "render_aria_comparison",
    ):
        assert callable(getattr(rolloutdiff, name))
        assert name in rolloutdiff.__all__
