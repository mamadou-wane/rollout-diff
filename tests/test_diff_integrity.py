import copy
import hashlib
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_cli import _dump, _subprocess
from test_full_pipeline import analyze, fixture
from test_report import tables


def _pair():
    step = 2**53 + 1
    rows = [
        {
            "run": "same-run",
            "sample_id": str(2**53 + 11 + index),
            "step": step + 2 * (index // 2),
            "prompt_id": str(2**53 + 31 + index % 2),
            "prompt": f"private prompt sentinel {index % 2} /Users/private/input",
            "response": "plain words.",
            "reward": 0.0,
        }
        for index in range(4)
    ]
    a, b = pd.DataFrame(rows), pd.DataFrame(rows)
    for side, frame in (("a", a), ("b", b)):
        for index in range(4):
            positive = side == "b" or index % 2 == 0
            response = "def solve():\n    pass" if positive else "plain words."
            frame.loc[index, "response"] = (
                f"{response}\n# private completion sentinel {side}-{index}"
            )
            frame.loc[index, "reward"] = float(positive)
    return a, b


def _feature(report, name="def:solve"):
    return next(item for item in report["features"] if item["feature"] == name)


def _round_trip(report, path):
    from rolloutdiff import comparison_json, load_comparison_report

    path.write_text(comparison_json(report), encoding="utf-8")
    return load_comparison_report(path)


def test_comparison_round_trip_html_and_resolution_keep_large_exact_identities(
    tmp_path,
):
    from rolloutdiff import (
        compare_rollouts,
        render_comparison_report,
        resolve_comparison_citations,
    )

    a, b = _pair()
    report = compare_rollouts(a, b)
    saved = copy.deepcopy(report)
    path = tmp_path / "comparison.json"
    loaded = _round_trip(report, path)
    html = render_comparison_report(loaded)
    assert loaded == report == saved
    assert [side["run"] for side in loaded["sides"]] == ["same-run", "same-run"]
    assert loaded["sides"][1]["window"] == [9007199254740993, 9007199254740995]
    assert _feature(loaded)["difference"]["estimate"] == 0.5
    resolved = resolve_comparison_citations(loaded, b, side="b", feature="def:solve")
    assert [item["sample_id"] for item in resolved["items"]] == [
        "9007199254741003",
        "9007199254741004",
        "9007199254741005",
    ]
    assert [item["step"] for item in resolved["items"]] == [
        9007199254740993,
        9007199254740993,
        9007199254740995,
    ]
    for item in resolved["items"]:
        original = b.loc[b.sample_id.eq(item["sample_id"])].iloc[0]
        assert item["response"] == original.response
        assert item["prompt_id"] == original.prompt_id
        assert (
            item["response_sha256"]
            == hashlib.sha256(original.response.encode()).hexdigest()
        )
        assert item["sample_id"] in html
    assert loaded == saved
    for exported in (path.read_text(), html):
        assert "def:solve" in exported
        assert "private prompt sentinel" not in exported
        assert "private completion sentinel" not in exported
        assert "/Users/private/input" not in exported
        assert str(tmp_path) not in exported
    with pytest.raises(ValueError, match="fingerprint"):
        resolve_comparison_citations(loaded, a, side="b", feature="def:solve")


def test_shuffled_dump_rows_and_indexes_keep_serialized_report_and_resolution():
    from rolloutdiff import (
        compare_rollouts,
        comparison_json,
        resolve_comparison_citations,
    )

    a, b = _pair()
    saved_a, saved_b = a.copy(deep=True), b.copy(deep=True)
    original = compare_rollouts(a, b)
    shuffled_a = a.sample(frac=1, random_state=17).set_axis([8, 2, 90, 3])
    shuffled_b = b.sample(frac=1, random_state=29).set_axis([5, 20, 7, 4])
    reordered = compare_rollouts(shuffled_a, shuffled_b)
    assert comparison_json(reordered) == comparison_json(original)
    assert resolve_comparison_citations(
        reordered, shuffled_b, side="b", feature="def:solve"
    ) == resolve_comparison_citations(original, b, side="b", feature="def:solve")
    assert_frame_equal(a, saved_a)
    assert_frame_equal(b, saved_b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("response_sha256", "0" * 64),
        ("reward", 91.0),
        ("prompt_id", "9007199254741024"),
        ("sample_id", "missing-sample"),
    ],
)
def test_serialized_tampered_citation_is_rejected_without_report_mutation(
    tmp_path, field, value
):
    from rolloutdiff import compare_rollouts, resolve_comparison_citations

    a, b = _pair()
    report = compare_rollouts(a, b)
    target = _feature(report)["citations"]["b"]["items"][-1]
    identity = target["run"], target["sample_id"]
    for feature in report["features"]:
        for citation in feature["citations"]["b"]["items"]:
            if (citation["run"], citation["sample_id"]) == identity:
                citation[field] = value
    loaded = _round_trip(report, tmp_path / "tampered.json")
    original = copy.deepcopy(loaded)
    with pytest.raises(ValueError):
        resolve_comparison_citations(loaded, b, side="b", feature="def:solve")
    assert loaded == original


@pytest.mark.parametrize(
    "feature,status",
    [("def:solve", "no_positive_rollouts"), ("scalar:reward", "unsupported_kind")],
)
def test_unselected_citations_keep_explicit_status_through_json_and_resolution(
    tmp_path, feature, status
):
    from rolloutdiff import compare_rollouts, resolve_comparison_citations

    a, b = _pair()
    a = a.assign(response="plain words.", reward=0.0)
    loaded = _round_trip(compare_rollouts(a, b), tmp_path / "empty-citations.json")
    resolved = resolve_comparison_citations(loaded, a, side="a", feature=feature)
    assert resolved["status"] == status
    assert resolved["items"] == []


def test_missing_comparison_feature_does_not_resolve_an_unrelated_citation(tmp_path):
    from rolloutdiff import compare_rollouts, resolve_comparison_citations

    a, b = _pair()
    loaded = _round_trip(compare_rollouts(a, b), tmp_path / "comparison.json")
    with pytest.raises(ValueError):
        resolve_comparison_citations(loaded, b, side="b", feature="def:missing")


def test_comparison_leaves_every_existing_analysis_table_exactly_unchanged():
    from rolloutdiff import compare_rollouts, comparison_json

    data = fixture()
    original = data.copy(deep=True)
    before = {
        name: frame.copy(deep=True) for name, frame in tables(analyze(data)).items()
    }
    comparison_json(compare_rollouts(data, a=(1, 3), b=(4, 6), keywords=("words",)))
    after = tables(analyze(data))
    assert before.keys() == after.keys()
    for name, expected in before.items():
        assert_frame_equal(expected, after[name], check_exact=True, obj=name)
    assert_frame_equal(data, original, check_exact=True)


@pytest.fixture(scope="module")
def cli_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("diff-integrity")
    source = _dump(root / "source", "trl")
    report = root / "comparison.json"
    completed = _subprocess(
        [
            "diff",
            str(source),
            "--format",
            "trl",
            "--run",
            "r",
            "--a",
            "1:15",
            "--b",
            "16:20",
            "--output",
            str(report),
        ]
    )
    assert completed.returncode == 0, completed.stderr
    return source, json.loads(report.read_text())


def _inspect(source, report, format="trl"):
    return _subprocess(
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


@pytest.mark.parametrize("tamper", ["response_hash", "metadata", "feature_presence"])
def test_inspect_validates_last_citation_before_emitting_any_response(
    tmp_path, cli_case, tamper
):
    from rolloutdiff import comparison_json, load_trl

    source, original = cli_case
    report = copy.deepcopy(original)
    citations = _feature(report)["citations"]["b"]["items"]
    assert len(citations) == 3
    last = citations[-1]
    if tamper == "response_hash":
        last["response_sha256"] = "0" * 64
    elif tamper == "metadata":
        last["reward"] = 987.0
    else:
        rows = load_trl(source, run="r")
        absent = rows.loc[
            rows.step.eq(last["step"])
            & rows.prompt_id.eq(last["prompt_id"])
            & rows.sample_id.str.endswith(":1")
        ].iloc[0]
        for field in ("sample_id", "prompt_id", "reward"):
            last[field] = absent[field]
        last["response_sha256"] = hashlib.sha256(absent.response.encode()).hexdigest()
    path = tmp_path / "tampered.json"
    encoded = (
        comparison_json(report)
        if tamper == "feature_presence"
        else json.dumps(report, allow_nan=False)
    )
    path.write_text(encoded)
    before = path.read_bytes()
    completed = _inspect(source, path)
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "private completion sentinel" not in completed.stderr
    assert path.read_bytes() == before


def test_inspect_rejects_changed_source_bytes_with_identical_canonical_rows(tmp_path):
    from rolloutdiff import load_verl
    from rolloutdiff.comparison import canonical_fingerprint

    source = _dump(tmp_path / "source", "verl")
    path = tmp_path / "comparison.json"
    completed = _subprocess(
        [
            "diff",
            str(source),
            "--format",
            "verl",
            "--run",
            "r",
            "--a",
            "1:15",
            "--b",
            "16:20",
            "--output",
            str(path),
        ]
    )
    assert completed.returncode == 0, completed.stderr
    before = load_verl(source, run="r")
    changed = source / "1.jsonl"
    changed.write_text(
        "".join(" " + line for line in changed.read_text().splitlines(keepends=True))
    )
    after = load_verl(source, run="r")
    assert_frame_equal(before, after, check_exact=True)
    assert canonical_fingerprint(before) == canonical_fingerprint(after)
    completed = _inspect(source, path, format="verl")
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "error:" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "private completion sentinel" not in completed.stderr
