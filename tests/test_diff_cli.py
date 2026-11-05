import json

import pandas as pd
import pytest
from test_cli import _dump, _subprocess


def arguments(source, report, *, format="trl"):
    return [
        "diff",
        str(source),
        "--format",
        format,
        "--run",
        "r",
        "--a",
        "1:15",
        "--b",
        "16:20",
        "--output",
        str(report),
    ]


@pytest.mark.parametrize("format", ["trl", "verl"])
def test_diff_cli_to_render_and_citation_resolution(tmp_path, format):
    source = _dump(tmp_path / "dump", format)
    report = tmp_path / "diff.json"
    html = tmp_path / "diff.html"
    completed = _subprocess(arguments(source, report, format=format))
    assert completed.returncode == 0, completed.stderr
    saved = json.loads(report.read_text())
    assert saved["report_type"] == "comparison"
    assert saved["mode"] == "windows"
    assert saved["prompt_mix"]["shared_prompts"] == 6
    feature = next(f for f in saved["features"] if f["feature"] == "def:solve")
    assert feature["a"]["mean"] == 0
    assert feature["b"]["mean"] == 0.5
    assert feature["paired"]["estimate"] == 0.5
    assert feature["citations"]["a"]["status"] == "no_positive_rollouts"
    assert len(feature["citations"]["b"]["items"]) == 3
    for side in saved["sides"]:
        assert side["source"]["format"] == format
        assert len(side["source"]["files"]) == 20
    rendered = _subprocess(["render", str(report), "--output", str(html)])
    assert rendered.returncode == 0, rendered.stderr
    assert "def:solve" in html.read_text()
    for text in (report.read_text(), html.read_text()):
        assert "private prompt sentinel" not in text
        assert "private completion sentinel" not in text
        assert str(tmp_path) not in text
    inspected = _subprocess(
        [
            "inspect",
            str(source),
            "--format",
            format,
            "--run",
            "r",
            "--report",
            str(report),
            "--side",
            "b",
            "--feature",
            "def:solve",
        ]
    )
    assert inspected.returncode == 0, inspected.stderr
    assert "private completion sentinel" in inspected.stdout
    assert "private prompt sentinel" not in inspected.stdout
    assert feature["citations"]["b"]["items"][0]["sample_id"] in inspected.stdout
    assert "Side: b" in inspected.stdout


def test_two_dump_mixed_format_reused_identities_and_inline_html(tmp_path):
    a = _dump(tmp_path / "a", "trl", changed=False)
    b = _dump(tmp_path / "b", "verl")
    report, html = tmp_path / "two.json", tmp_path / "two.html"
    result = _subprocess(
        [
            "diff",
            str(a),
            str(b),
            "--format",
            "trl",
            "--other-format",
            "verl",
            "--run",
            "r",
            "--output",
            str(report),
            "--html",
            str(html),
        ]
    )
    assert result.returncode == 0, result.stderr
    saved = json.loads(report.read_text())
    assert saved["mode"] == "dumps"
    feature = next(f for f in saved["features"] if f["feature"] == "def:solve")
    assert feature["difference"]["estimate"] == 0.125
    assert [s["run"] for s in saved["sides"]] == ["r", "r"]
    assert html.is_file()
    inspected = _subprocess(
        [
            "inspect",
            str(a),
            "--format",
            "trl",
            "--run",
            "r",
            "--report",
            str(report),
            "--side",
            "b",
            "--feature",
            "def:solve",
        ]
    )
    assert inspected.returncode != 0
    assert "private completion sentinel" not in inspected.stdout + inspected.stderr
    assert "Traceback" not in inspected.stderr


@pytest.mark.parametrize(
    "extra",
    [
        ["--a", "1:2"],
        ["--b", "1:2"],
        ["--a", "16:20", "--b", "1:15"],
        ["--a", "1:15", "--b", "15:20"],
        ["--a", "-1:4", "--b", "5:20"],
        ["--a", "1:2:3", "--b", "5:20"],
        ["--a", "80:100", "--b", "140:160"],
        ["--a", "1:15", "--b", "16:20", "--other-run", "unused"],
    ],
)
def test_invalid_windows_or_ambiguous_other_options_do_not_publish(tmp_path, extra):
    source = _dump(tmp_path / "dump", "trl")
    report = tmp_path / "bad.json"
    result = _subprocess(
        [
            "diff",
            str(source),
            "--format",
            "trl",
            "--run",
            "r",
            "--output",
            str(report),
            *extra,
        ]
    )
    assert result.returncode != 0
    assert not report.exists()
    assert "Traceback" not in result.stderr


def test_existing_output_and_html_paths_are_preserved(tmp_path):
    source = _dump(tmp_path / "dump", "trl")
    report, html = tmp_path / "d.json", tmp_path / "d.html"
    html.write_text("original")
    result = _subprocess([*arguments(source, report), "--html", str(html)])
    assert result.returncode != 0
    assert not report.exists()
    assert html.read_text() == "original"
    report.write_text("saved")
    result = _subprocess(arguments(source, report))
    assert result.returncode != 0
    assert report.read_text() == "saved"
    assert not list(tmp_path.glob(".rolloutdiff-*"))


def test_inspect_rejects_changed_source_before_any_response_output(tmp_path):
    source = _dump(tmp_path / "dump", "trl")
    report = tmp_path / "d.json"
    result = _subprocess(arguments(source, report))
    assert result.returncode == 0, result.stderr
    path = source / "completions_16.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "completion"] += " changed"
    frame.to_parquet(path, index=False)
    result = _subprocess(
        [
            "inspect",
            str(source),
            "--format",
            "trl",
            "--run",
            "r",
            "--report",
            str(report),
            "--side",
            "b",
            "--feature",
            "def:solve",
        ]
    )
    assert result.returncode != 0
    assert not result.stdout
    assert "private completion sentinel" not in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("failure", ["permission", "race"])
def test_second_output_publication_failure_leaves_no_partial_pair(
    tmp_path, monkeypatch, capsys, failure
):
    import rolloutdiff.report as publication
    from rolloutdiff.cli import main

    source = _dump(tmp_path / "dump", "trl")
    report, html = tmp_path / "paired.json", tmp_path / "paired.html"
    link = publication.os.link

    def fail_second(temporary, destination):
        if destination == html:
            if failure == "permission":
                raise PermissionError("controlled publication failure")
            html.write_text("concurrent output")
        link(temporary, destination)

    monkeypatch.setattr(publication.os, "link", fail_second)
    assert main([*arguments(source, report), "--html", str(html)]) == 1
    assert not report.exists()
    assert not capsys.readouterr().out
    if failure == "race":
        assert html.read_text() == "concurrent output"
    else:
        assert not html.exists()
    assert not list(tmp_path.glob(".rolloutdiff-*"))
