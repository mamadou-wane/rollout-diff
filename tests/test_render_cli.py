from pathlib import Path

import pytest
from test_cli import _subprocess
from test_full_pipeline import analyze, fixture

from rolloutdiff.report import build_report, report_json


@pytest.fixture(scope="module")
def report():
    return build_report(analyze(fixture()))


def _arguments(source, output):
    return ["render", str(source), "--output", str(output)]


def _input(tmp_path, report):
    source = tmp_path / "report.json"
    source.write_text(report_json(report), encoding="utf-8")
    return source


def test_render_cli_reads_existing_report_without_analysis(
    tmp_path, report, monkeypatch, capsys
):
    from rolloutdiff import cli

    def unexpected(*args, **kwargs):
        pytest.fail("render must not load rollouts or run analysis")

    for name in ("load_trl", "load_verl", "analyze_rollouts", "build_report"):
        monkeypatch.setattr(cli, name, unexpected)
    source = _input(tmp_path, report)
    original = source.read_bytes()
    output = tmp_path / "report.html"
    assert cli.main(_arguments(source, output)) == 0
    rendered = output.read_text()
    assert "<!doctype html>" in rendered.lower()
    assert "def:solve" in rendered
    assert "scalar:reward" in rendered
    assert source.read_bytes() == original
    captured = capsys.readouterr()
    assert not captured.err
    assert "r" in captured.out
    assert str(report["runs"][0]["counts"]["events"]) in captured.out
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


def test_render_cli_subprocess_produces_deterministic_html(tmp_path, report):
    source = _input(tmp_path, report)
    outputs = [tmp_path / "first.html", tmp_path / "second.html"]
    for output in outputs:
        completed = _subprocess(_arguments(source, output))
        assert completed.returncode == 0, completed.stderr
        assert not completed.stderr
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_render_cli_accepts_empty_report(tmp_path, capsys):
    from rolloutdiff.cli import main

    source = _input(tmp_path, {"schema_version": 1, "runs": []})
    output = tmp_path / "empty.html"
    assert main(_arguments(source, output)) == 0
    assert "<!doctype html>" in output.read_text().lower()
    assert "0" in capsys.readouterr().out


def test_render_cli_escapes_terminal_control_characters_in_run_summary(
    tmp_path, capsys, report
):
    from rolloutdiff.cli import main

    name = "run\x1b[2J\nextra line"
    empty = {
        "schema_version": 1,
        "runs": [
            {
                "run": name,
                "status": "empty",
                "counts": dict.fromkeys(report["runs"][0]["counts"], 0),
                "events": [],
            }
        ],
    }
    source = _input(tmp_path, empty)
    assert main(_arguments(source, tmp_path / "output.html")) == 0
    captured = capsys.readouterr()
    assert name not in captured.out
    assert repr(name) in captured.out
    assert len(captured.out.splitlines()) == 2
    assert not captured.err


@pytest.mark.parametrize(
    "payload",
    [
        '{"secret-payload":broken}',
        '{"schema_version":2,"runs":[]}',
        '{"schema_version":1,"runs":{}}',
        '{"schema_version":1,"schema_version":1,"runs":[]}',
        '{"schema_version":NaN,"runs":[]}',
        '{"schema_version":1,"runs":[Infinity]}',
        "[]",
        "\ud800",
    ],
)
def test_render_cli_rejects_malformed_report_without_payload_or_partial_output(
    tmp_path, payload
):
    source = tmp_path / "input.json"
    source.write_bytes(payload.encode("utf-8", errors="surrogatepass"))
    output = tmp_path / "output.html"
    completed = _subprocess(_arguments(source, output))
    assert completed.returncode != 0
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "secret-payload" not in completed.stderr
    assert not completed.stdout
    assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


def test_render_cli_missing_input_has_no_partial_output(tmp_path, capsys):
    from rolloutdiff.cli import main

    output = tmp_path / "output.html"
    assert main(_arguments(tmp_path / "missing.json", output)) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


@pytest.mark.parametrize(
    "case", ["existing", "directory", "dangling", "missing_parent"]
)
def test_render_cli_rejects_unsafe_output_before_reading(tmp_path, capsys, case):
    from rolloutdiff.cli import main

    output = tmp_path / "output.html"
    if case == "existing":
        output.write_text("keep existing")
    elif case == "directory":
        output.mkdir()
    elif case == "dangling":
        output.symlink_to(tmp_path / "absent-target")
    else:
        output = tmp_path / "missing" / "output.html"
    assert main(_arguments(tmp_path / "missing-input.json", output)) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "missing-input" not in captured.err
    assert "Traceback" not in captured.err
    if case == "existing":
        assert output.read_text() == "keep existing"
    elif case == "dangling":
        assert output.is_symlink()
        assert not (tmp_path / "absent-target").exists()


@pytest.mark.parametrize("failure", ["permission", "race"])
def test_render_cli_publication_failures_preserve_destination_and_clean_temporary(
    tmp_path, capsys, monkeypatch, report, failure
):
    import rolloutdiff.report as report_module
    from rolloutdiff.cli import main

    source = _input(tmp_path, report)
    output = tmp_path / "output.html"
    link = report_module.os.link

    def fail_publication(temporary, destination):
        assert destination == output
        assert "</html>" in Path(temporary).read_text().lower()
        if failure == "permission":
            raise PermissionError("Permission denied")
        output.write_text("concurrent output")
        link(temporary, destination)

    monkeypatch.setattr(report_module.os, "link", fail_publication)
    assert main(_arguments(source, output)) == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    assert not captured.out
    if failure == "race":
        assert output.read_text() == "concurrent output"
    else:
        assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


@pytest.mark.parametrize(
    "arguments",
    [
        ["render"],
        ["render", "input.json"],
        ["render", "", "--output", "out.html"],
        ["render", "input.json", "--output", " "],
        ["render", "input.json", "--out", "out.html"],
        ["render", "input.json", "--output", "out.html", "--output", "second.html"],
        ["render", "input.json", "--output", "out.html", "--run", "r"],
        ["render", "input.json", "--output", "out.html", "--format", "trl"],
    ],
)
def test_render_cli_rejects_invalid_arguments(arguments, capsys):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


def test_render_cli_help_needs_no_input():
    completed = _subprocess(["render", "--help"])
    assert completed.returncode == 0
    assert "--output" in completed.stdout
    assert "--format" not in completed.stdout
    assert not completed.stderr


def test_atomic_writer_retains_existing_output_error_precedence(tmp_path):
    from rolloutdiff.report import write_report

    output = tmp_path / "report.json"
    output.write_text("preserved")
    with pytest.raises(FileExistsError):
        write_report({"bad": float("nan")}, output)
    assert output.read_text() == "preserved"
