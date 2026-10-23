import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

import pandas as pd
import pytest


def _dump(path, format, *, reward_column="reward", changed=True, shuffled=False):
    path.mkdir()
    prompts = [
        name
        for half in (0, 1)
        for name in [
            f"p{i}" for i in range(100) if zlib.crc32(f"p{i}".encode()) & 1 == half
        ][:3]
    ]
    for step in range(1, 21):
        rows = []
        for prompt in prompts:
            for sample in range(2):
                present = changed and step > 15 and sample == 0
                row = {
                    "step": step,
                    "sample_id": f"{step}:{prompt}:{sample}",
                    "prompt_id": prompt,
                }
                response = "def solve():\n    pass" if present else "plain words."
                response += "\n# private completion sentinel"
                if format == "trl":
                    row.update(
                        prompt="private prompt sentinel " + prompt,
                        completion=response,
                        **{reward_column: 2.0 if present else 0.0},
                    )
                else:
                    row.update(
                        input="private prompt sentinel " + prompt,
                        output=response,
                        score=2.0 if present else 0.0,
                    )
                rows.append(row)
        frame = pd.DataFrame(rows)
        if shuffled:
            frame = frame.sample(frac=1, random_state=step)
        if format == "trl":
            frame.to_parquet(path / f"completions_{step}.parquet", index=False)
        else:
            frame.to_json(path / f"{step}.jsonl", orient="records", lines=True)
    return path


def _arguments(source, output, format="trl"):
    return [
        "analyze",
        str(source),
        "--format",
        format,
        "--run",
        "experiment-1",
        "--output",
        str(output),
    ]


def _subprocess(arguments):
    source = Path(__file__).resolve().parents[1] / "src"
    return subprocess.run(
        [sys.executable, "-m", "rolloutdiff.cli", *arguments],
        env=os.environ | {"PYTHONPATH": str(source)},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize("format", ["trl", "verl"])
def test_cli_analyzes_real_dumps_and_reports_canonical_references(tmp_path, format):
    source = _dump(tmp_path / "dump", format)
    output = tmp_path / "report.json"
    completed = _subprocess(_arguments(source, output, format))
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    encoded = output.read_text()
    report = json.loads(encoded)
    assert report["schema_version"] == 1
    assert len(report["runs"]) == 1
    run = report["runs"][0]
    assert run["run"] == "experiment-1"
    assert run["counts"]["rollouts"] == 240
    assert run["counts"]["observed_steps"] == 20
    assert run["counts"]["prompts"] == 6
    assert run["counts"]["screened_series"] > 0
    assert run["counts"]["sequential_alerts"] > 0
    assert run["counts"]["behavioral_events"] > 0
    assert run["counts"]["contextual_events"] > 0
    assert [event["rank"] for event in run["events"]] == list(
        range(1, len(run["events"]) + 1)
    )
    members = [member for event in run["events"] for member in event["members"]]
    assert "def:solve" in {member["feature"] for member in members}
    assert "scalar:reward" in {member["feature"] for member in members}
    examples = [example for member in members for example in member["exemplars"]]
    selected = [example for example in examples if example["status"] == "selected"]
    assert selected
    assert all(example["run"] == "experiment-1" for example in selected)
    assert all(example["sample_id"] for example in selected)
    assert all(example["feature_value"] == 1 for example in selected)
    assert all(16 <= example["step"] <= 18 for example in selected)
    assert "private prompt sentinel" not in encoded
    assert "private completion sentinel" not in encoded
    assert str(tmp_path) not in encoded
    assert "NaN" not in encoded
    assert "experiment-1" in completed.stdout
    assert "screened" in completed.stdout
    assert "alerted" in completed.stdout
    assert "behavioral" in completed.stdout
    assert "contextual" in completed.stdout
    assert "def:solve" in completed.stdout


def test_cli_reward_column_and_input_order_do_not_change_report(tmp_path):
    from rolloutdiff.cli import main

    original = _dump(tmp_path / "original", "trl", reward_column="custom")
    shuffled = _dump(
        tmp_path / "shuffled", "trl", reward_column="custom", shuffled=True
    )
    outputs = [tmp_path / "original.json", tmp_path / "shuffled.json"]
    for source, output in zip([original, shuffled], outputs, strict=True):
        assert main([*_arguments(source, output), "--reward-column", "custom"]) == 0
    assert outputs[0].read_bytes() == outputs[1].read_bytes()


def test_cli_no_alerts_keeps_run_and_counts(tmp_path):
    from rolloutdiff.cli import main

    source = _dump(tmp_path / "dump", "verl", changed=False)
    output = tmp_path / "report.json"
    assert main(_arguments(source, output, "verl")) == 0
    run = json.loads(output.read_text())["runs"][0]
    assert run["run"] == "experiment-1"
    assert run["counts"]["rollouts"] == 240
    assert run["counts"]["sequential_alerts"] == 0
    assert run["events"] == []


def test_cli_empty_dump_keeps_requested_run(tmp_path):
    from rolloutdiff.cli import main

    source = tmp_path / "1.jsonl"
    source.write_text("")
    output = tmp_path / "empty.json"
    assert main(_arguments(source, output, "verl")) == 0
    run = json.loads(output.read_text())["runs"][0]
    assert run["run"] == "experiment-1"
    assert all(count == 0 for count in run["counts"].values())
    assert run["events"] == []


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["analyze"],
        ["analyze", "dump", "--format", "csv", "--run", "r", "--output", "out"],
        ["analyze", "dump", "--for", "trl", "--run", "r", "--output", "out"],
        ["analyze", "dump", "--format", "trl", "--run", " ", "--output", "out"],
        ["analyze", "", "--format", "trl", "--run", "r", "--output", "out"],
        ["analyze", "dump", "--format", "trl", "--run", "r", "--output", " "],
        [
            "analyze",
            "dump",
            "--format",
            "verl",
            "--run",
            "r",
            "--output",
            "out",
            "--reward-column",
            "reward",
        ],
        [
            "analyze",
            "dump",
            "--format",
            "trl",
            "--run",
            "r",
            "--output",
            "out",
            "--reward-column",
            " ",
        ],
    ],
)
def test_cli_rejects_missing_unsupported_and_ambiguous_arguments(arguments, capsys):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--format", "--run", "--output", "--reward-column"])
def test_cli_rejects_repeated_options_before_loading(flag, tmp_path, capsys):
    from rolloutdiff.cli import main

    arguments = _arguments(tmp_path / "missing", tmp_path / "out")
    if flag == "--reward-column":
        arguments += [flag, "reward"]
    arguments += [flag, "trl" if flag == "--format" else "duplicate"]
    with pytest.raises(SystemExit) as exc:
        main(arguments)
    assert exc.value.code == 2
    assert "only once" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("case", ["missing", "bad_parquet", "bad_json", "bad_columns"])
def test_cli_input_failures_have_no_traceback_or_partial_output(tmp_path, capsys, case):
    from rolloutdiff.cli import main

    source = tmp_path / "completions_1.parquet"
    format = "trl"
    if case == "bad_parquet":
        source.write_text("not parquet")
    elif case == "bad_json":
        format = "verl"
        source = tmp_path / "1.jsonl"
        source.write_text("{broken\n")
    elif case == "bad_columns":
        pd.DataFrame({"step": [1], "reward": [0]}).to_parquet(source)
    output = tmp_path / "report.json"
    assert main(_arguments(source, output, format)) != 0
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "Traceback" not in stderr
    assert not output.exists()
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "case", ["existing", "directory", "dangling", "missing_parent"]
)
def test_cli_rejects_unsafe_output_before_loading(tmp_path, capsys, case):
    from rolloutdiff.cli import main

    output = tmp_path / "report.json"
    if case == "existing":
        output.write_text("keep existing")
    elif case == "directory":
        output.mkdir()
    elif case == "dangling":
        output.symlink_to(tmp_path / "absent-target")
    elif case == "missing_parent":
        output = tmp_path / "missing" / "report.json"
    assert main(_arguments(tmp_path / "missing-input", output)) != 0
    stderr = capsys.readouterr().err
    assert "error:" in stderr
    assert "missing-input" not in stderr
    assert "Traceback" not in stderr
    if case == "existing":
        assert output.read_text() == "keep existing"
    elif case == "dangling":
        assert output.is_symlink()
        assert not (tmp_path / "absent-target").exists()


def test_cli_help_is_available_without_a_dump():
    completed = _subprocess(["analyze", "--help"])
    assert completed.returncode == 0
    assert "--format" in completed.stdout
    assert "--reward-column" in completed.stdout
    assert not completed.stderr


def test_cli_rejects_private_path_features_without_writing_or_echoing_them(
    tmp_path, capsys
):
    from rolloutdiff import featurize, load_trl
    from rolloutdiff.cli import main

    source = _dump(tmp_path / "dump", "trl")
    private_path = "/tmp/private.py"
    for step in range(16, 21):
        path = source / f"completions_{step}.parquet"
        frame = pd.read_parquet(path)
        selected = frame.sample_id.str.endswith(":0")
        frame.loc[selected, "completion"] += f"\npath = {private_path!r}"
        frame.to_parquet(path, index=False)
    features = featurize(load_trl(source, run="experiment-1"))
    assert f"strlit:{private_path}" in set(features.catalog.feature)
    output = tmp_path / "report.json"
    assert main(_arguments(source, output)) != 0
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "filesystem paths" in captured.err
    assert private_path not in captured.err + captured.out
    assert "Traceback" not in captured.err
    assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


def test_cli_output_permission_failure_cleans_complete_temporary_report(
    tmp_path, capsys, monkeypatch
):
    from rolloutdiff import report
    from rolloutdiff.cli import main

    source = _dump(tmp_path / "dump", "verl")
    output = tmp_path / "report.json"

    def deny_publication(temporary, destination):
        assert destination == output
        assert json.loads(Path(temporary).read_text())["schema_version"] == 1
        raise PermissionError("Permission denied")

    monkeypatch.setattr(report.os, "link", deny_publication)
    assert main(_arguments(source, output, "verl")) != 0
    captured = capsys.readouterr()
    assert "error:" in captured.err
    assert "Permission denied" in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""
    assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))


def test_cli_rejects_embedded_exemplar_path_without_disclosure_or_output(tmp_path):
    from rolloutdiff import load_trl
    from rolloutdiff.pipeline import analyze_rollouts

    source = _dump(tmp_path / "dump", "trl")
    private_path = "/Users/private/project/raw"
    for step in range(16, 21):
        path = source / f"completions_{step}.parquet"
        frame = pd.read_parquet(path)
        present = frame.completion.str.startswith("def")
        frame.loc[present, "sample_id"] += ":" + private_path
        frame.to_parquet(path, index=False)
    selected = analyze_rollouts(load_trl(source, run="experiment-1")).exemplars
    assert selected.status.eq("selected").any()
    assert (
        selected.loc[selected.status.eq("selected"), "sample_id"]
        .str.endswith(private_path)
        .all()
    )
    output = tmp_path / "report.json"
    completed = _subprocess(_arguments(source, output))
    assert completed.returncode != 0
    assert "filesystem paths" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert private_path not in completed.stdout + completed.stderr
    assert completed.stdout == ""
    assert not output.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*.tmp"))
