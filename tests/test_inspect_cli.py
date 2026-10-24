import copy

import pandas as pd
import pytest
from test_cli import _dump, _subprocess

from rolloutdiff import analyze_rollouts, build_report, load_trl, load_verl
from rolloutdiff.report import report_json


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    root = tmp_path_factory.mktemp("inspection")
    cases = {}
    for format in ("trl", "verl"):
        source = _dump(root / format, format)
        loader = load_trl if format == "trl" else load_verl
        analysis = analyze_rollouts(loader(source, run="experiment-1"))
        assert not analysis.sequential.alerts.empty
        report = build_report(analysis)
        report_path = root / f"{format}.json"
        report_path.write_text(report_json(report))
        cases[format] = source, report_path, report
    return cases


def _arguments(source, report, format="trl", *, run="experiment-1", rank=1):
    return [
        "inspect",
        str(source),
        "--format",
        format,
        "--run",
        run,
        "--report",
        str(report),
        "--event-rank",
        str(rank),
    ]


def _selected(event):
    return [
        example
        for member in event["members"]
        for example in member["exemplars"]
        if example["status"] == "selected"
    ]


def _report(tmp_path, value):
    path = tmp_path / "report.json"
    path.write_text(report_json(value))
    return path


@pytest.mark.parametrize("format", ["trl", "verl"])
def test_inspect_cli_resolves_real_alerts_without_exporting_or_modifying_reports(
    cases, format
):
    source, report_path, report = cases[format]
    event = report["runs"][0]["events"][0]
    before = report_path.read_bytes()
    completed = _subprocess(_arguments(source, report_path, format))
    assert completed.returncode == 0, completed.stderr
    assert not completed.stderr
    assert "def solve():" in completed.stdout
    assert "private completion sentinel" in completed.stdout
    assert "private prompt sentinel" not in completed.stdout
    assert event["event_id"] in completed.stdout
    assert event["representative_feature"] in completed.stdout
    for example in _selected(event):
        assert example["sample_id"] in completed.stdout
        assert example["prompt_id"] in completed.stdout
        assert example["selection_reason"] in completed.stdout
    assert report_path.read_bytes() == before
    assert "private completion sentinel" not in report_path.read_text()


def test_inspect_cli_does_not_recompute_analysis(cases, monkeypatch, capsys):
    from rolloutdiff import cli

    def unexpected(*args, **kwargs):
        pytest.fail("inspection must not rerun analysis or build reports")

    monkeypatch.setattr(cli, "analyze_rollouts", unexpected)
    monkeypatch.setattr(cli, "build_report", unexpected)
    source, report_path, _ = cases["trl"]
    assert cli.main(_arguments(source, report_path)) == 0
    assert "private completion sentinel" in capsys.readouterr().out


def test_inspect_cli_is_deterministic_with_shuffled_report_and_rollouts(
    tmp_path, cases
):
    source, report_path, report = cases["trl"]
    shuffled_source = _dump(tmp_path / "shuffled", "trl", shuffled=True)
    shuffled = copy.deepcopy(report)
    shuffled["runs"][0]["events"].reverse()
    for event in shuffled["runs"][0]["events"]:
        event["members"].reverse()
        for member in event["members"]:
            member["exemplars"].reverse()
    first = _subprocess(_arguments(source, report_path))
    second = _subprocess(_arguments(shuffled_source, _report(tmp_path, shuffled)))
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout


def test_inspect_cli_accepts_explicit_trl_reward_column(tmp_path, cases):
    source = _dump(tmp_path / "custom", "trl", reward_column="custom")
    _, report_path, _ = cases["trl"]
    completed = _subprocess(
        [*_arguments(source, report_path), "--reward-column", "custom"]
    )
    assert completed.returncode == 0, completed.stderr
    assert "private completion sentinel" in completed.stdout


@pytest.mark.parametrize(
    "field,value",
    [("step", 17), ("prompt_id", "wrong"), ("reward", 999.0), ("sample_id", "missing")],
)
def test_inspect_cli_metadata_mismatch_emits_no_partial_responses(
    tmp_path, cases, field, value
):
    source, _, report = cases["trl"]
    candidate = copy.deepcopy(report)
    examples = _selected(candidate["runs"][0]["events"][0])
    assert len(examples) > 1
    examples[-1][field] = value if field != "step" else examples[-1]["step"] + 1
    completed = _subprocess(_arguments(source, _report(tmp_path, candidate)))
    assert completed.returncode == 1
    assert not completed.stdout
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "private completion sentinel" not in completed.stderr


def test_inspect_cli_duplicate_rollout_identity_fails_without_text(tmp_path, cases):
    source = _dump(tmp_path / "duplicate", "trl")
    file = source / "completions_16.parquet"
    frame = pd.read_parquet(file)
    pd.concat([frame, frame.iloc[:1]], ignore_index=True).to_parquet(file, index=False)
    _, report_path, _ = cases["trl"]
    completed = _subprocess(_arguments(source, report_path))
    assert completed.returncode == 1
    assert "duplicate" in completed.stderr
    assert not completed.stdout
    assert "private completion sentinel" not in completed.stderr


def test_inspect_cli_keeps_unsupported_exemplars_visible(cases):
    source, report_path, report = cases["trl"]
    event = next(
        event
        for event in report["runs"][0]["events"]
        if event["representative_feature"] == "scalar:reward"
    )
    completed = _subprocess(_arguments(source, report_path, rank=event["rank"]))
    assert completed.returncode == 0, completed.stderr
    assert "scalar:reward" in completed.stdout
    assert "unsupported_kind" in completed.stdout
    assert "Unavailable" in completed.stdout
    assert "private completion sentinel" not in completed.stdout


def test_inspect_cli_keeps_missing_exemplar_records_explicit(tmp_path, cases):
    source, _, report = cases["trl"]
    candidate = copy.deepcopy(report)
    for member in candidate["runs"][0]["events"][0]["members"]:
        member["exemplar_status"] = "missing"
        member["exemplars"] = []
    completed = _subprocess(_arguments(source, _report(tmp_path, candidate)))
    assert completed.returncode == 0, completed.stderr
    assert "missing" in completed.stdout
    assert "No exemplar records" in completed.stdout
    assert "private completion sentinel" not in completed.stdout


def test_inspect_cli_escapes_response_controls_but_preserves_readable_lines(
    tmp_path, cases
):
    source = _dump(tmp_path / "controls", "trl")
    _, report_path, report = cases["trl"]
    example = _selected(report["runs"][0]["events"][0])[0]
    file = source / f"completions_{example['step']}.parquet"
    frame = pd.read_parquet(file)
    response = "café first\nnext\x1b[2J\rforged\b\t\x9b31m\u202ehidden\u2028line\x00"
    frame.loc[frame.sample_id.eq(example["sample_id"]), "completion"] = response
    frame.to_parquet(file, index=False)
    completed = _subprocess(_arguments(source, report_path))
    assert completed.returncode == 0, completed.stderr
    assert "    café first\n    next" in completed.stdout
    assert all(
        character == "\n" or character.isprintable() for character in completed.stdout
    )
    for escape in (
        r"\x1b",
        r"\r",
        r"\x08",
        r"\t",
        r"\x9b",
        r"\u202e",
        r"\u2028",
        r"\x00",
    ):
        assert escape in completed.stdout


def test_inspect_cli_escapes_run_and_selection_metadata(tmp_path, cases):
    source, _, report = cases["trl"]
    name = "experiment-1\x1b[2J\nforged\u202e"
    candidate = copy.deepcopy(report)

    def change_run(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "run":
                    value[key] = name
                else:
                    change_run(item)
        elif isinstance(value, list):
            for item in value:
                change_run(item)

    change_run(candidate)
    for example in _selected(candidate["runs"][0]["events"][0]):
        example["selection_reason"] = "reason\x1b[31m\rforged"
    completed = _subprocess(_arguments(source, _report(tmp_path, candidate), run=name))
    assert completed.returncode == 0, completed.stderr
    assert all(
        character == "\n" or character.isprintable() for character in completed.stdout
    )
    assert r"experiment-1\x1b[2J\nforged\u202e" in completed.stdout
    assert r"reason\x1b[31m\rforged" in completed.stdout


@pytest.mark.parametrize(
    "case", ["empty", "wrong_run", "wrong_rank", "bad_json", "missing_report"]
)
def test_inspect_cli_invalid_selection_and_report_have_clear_errors(
    tmp_path, cases, case
):
    source, _, report = cases["trl"]
    candidate = copy.deepcopy(report)
    arguments = {}
    if case == "empty":
        candidate["runs"] = []
    elif case == "wrong_run":
        arguments["run"] = "unknown"
    elif case == "wrong_rank":
        arguments["rank"] = 999
    report_path = _report(tmp_path, candidate)
    if case == "bad_json":
        report_path.write_text('{"private payload":broken}')
    elif case == "missing_report":
        report_path.unlink()
    completed = _subprocess(_arguments(source, report_path, **arguments))
    assert completed.returncode == 1
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "private payload" not in completed.stderr
    assert not completed.stdout


@pytest.mark.parametrize(
    "extra",
    [
        ["--event-rank", "1"],
        ["--run", "duplicate"],
        ["--report", "another.json"],
        ["--format", "trl"],
        ["--output", "out.txt"],
        ["\x1b[2Junknown"],
    ],
)
def test_inspect_cli_rejects_ambiguous_arguments_safely(tmp_path, extra, capsys):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit) as exc:
        main([*_arguments(tmp_path / "missing", tmp_path / "report.json"), *extra])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert all(
        character == "\n" or character.isprintable() for character in captured.err
    )
    assert not captured.out


@pytest.mark.parametrize("rank", ["0", "-1", "1.5", "nan"])
def test_inspect_cli_requires_positive_integer_rank(tmp_path, rank, capsys):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit) as exc:
        main(_arguments(tmp_path / "dump", tmp_path / "report.json", rank=rank))
    assert exc.value.code == 2
    assert "positive integer" in capsys.readouterr().err


def test_inspect_cli_rejects_reward_column_for_verl(tmp_path, capsys):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit) as exc:
        main(
            [
                *_arguments(tmp_path / "dump", tmp_path / "report.json", "verl"),
                "--reward-column",
                "score",
            ]
        )
    assert exc.value.code == 2
    assert "only with --format trl" in capsys.readouterr().err


def test_inspect_cli_help_needs_no_input():
    completed = _subprocess(["inspect", "--help"])
    assert completed.returncode == 0
    assert "--event-rank" in completed.stdout
    assert "--report" in completed.stdout
    assert "--output" not in completed.stdout
    assert not completed.stderr
