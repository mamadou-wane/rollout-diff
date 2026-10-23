import copy
import json
from dataclasses import replace

import pytest
from test_full_pipeline import analyze, fixture

from rolloutdiff import build_report


@pytest.fixture(scope="module")
def report():
    return build_report(analyze(fixture()))


def validate(value):
    from rolloutdiff._report_validation import validate_report

    return validate_report(value)


def test_valid_report_preserves_values_and_does_not_mutate_input(report):
    before = copy.deepcopy(report)
    actual = validate(report)
    assert actual == before == report
    assert actual is not report


def test_empty_reports_missing_evidence_and_large_integer_steps_are_accepted():
    assert validate({"schema_version": 1, "runs": []})["runs"] == []
    data = fixture()
    data["step"] += 2**53 + 1
    analysis = analyze(data)
    report = build_report(
        replace(
            analysis,
            evidence=analysis.evidence.iloc[:0],
            exemplars=analysis.exemplars.iloc[:0],
        )
    )
    actual = validate(report)
    assert actual == report
    assert actual["runs"][0]["events"][0]["onset_step"] > 2**53


@pytest.mark.parametrize("version", [None, True, 1.0, "1", 0, 2])
def test_report_version_requires_exact_supported_integer(version):
    with pytest.raises(ValueError, match="version"):
        validate({"schema_version": version, "runs": []})


def test_unknown_payload_fields_are_removed_at_every_depth(report):
    candidate = copy.deepcopy(report)
    event = candidate["runs"][0]["events"][0]
    member = event["members"][0]
    for row in [
        candidate,
        candidate["runs"][0],
        event,
        member,
        member["evidence"],
        member["exemplars"][0],
    ]:
        row["response"] = "PRIVATE_PAYLOAD"
    assert validate(candidate) == report


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("run",), " "),
        (("counts", "events"), True),
        (("events", 0, "onset_step"), 4.0),
        (("events", 0, "rank"), "1"),
        (("events", 0, "members", 0, "feature"), None),
        (("events", 0, "members", 0, "evidence", "numerical_floor_dominant"), "false"),
        (
            (
                "events",
                0,
                "members",
                0,
                "evidence",
                "reward_association",
                "mean_difference",
            ),
            float("inf"),
        ),
        (("events", 0, "members", 0, "evidence", "persistence", "n_evaluable"), -1),
    ],
)
def test_invalid_nested_types_are_rejected_without_echo(report, location, value):
    candidate = copy.deepcopy(report)
    target = candidate["runs"][0]
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match="report") as exc:
        validate(candidate)
    assert "input_value" not in str(exc.value)


@pytest.mark.parametrize(
    "change",
    [
        lambda run: run["counts"].update(events=999),
        lambda run: run["counts"].update(sequential_alerts=999),
        lambda run: run["events"][0].update(n_members=999),
        lambda run: run["events"][0].update(rank=999),
        lambda run: run["events"][0].update(representative_feature="missing"),
        lambda run: run["events"][0].update(run="another"),
        lambda run: run["events"][0]["members"][0].update(run="another"),
        lambda run: run["events"][0]["members"][0]["exemplars"][0].update(
            feature="missing"
        ),
        lambda run: run["events"][0]["ranking"].update(
            numerical_floor_members=[{"run": "r", "feature": "missing"}]
        ),
    ],
)
def test_inconsistent_links_and_cardinality_are_rejected(report, change):
    candidate = copy.deepcopy(report)
    change(candidate["runs"][0])
    with pytest.raises(ValueError, match="report"):
        validate(candidate)


def test_selected_exemplars_require_unique_complete_canonical_references(report):
    for field in ["sample_id", "prompt_id", "step", "selection_order"]:
        candidate = copy.deepcopy(report)
        examples = candidate["runs"][0]["events"][0]["members"][0]["exemplars"]
        assert examples[0]["status"] == "selected"
        examples[0][field] = None
        with pytest.raises(ValueError, match="report"):
            validate(candidate)
    candidate = copy.deepcopy(report)
    examples = candidate["runs"][0]["events"][0]["members"][0]["exemplars"]
    examples.append(copy.deepcopy(examples[0]))
    with pytest.raises(ValueError, match="report"):
        validate(candidate)


def test_identifiers_are_preserved_as_text_and_statuses_are_not_reinterpreted(report):
    candidate = copy.deepcopy(report)
    member = candidate["runs"][0]["events"][0]["members"][0]
    member["exemplars"][0]["sample_id"] = (
        '</script><script>alert("/private/path")</script>'
    )
    member["evidence"]["persistence"]["status"] = "new unavailable status"
    member["evidence"]["persistence"]["fraction"] = None
    assert validate(candidate) == candidate


@pytest.mark.parametrize(
    "text",
    [
        '{"schema_version": 1, "runs": [], "secret": NaN}',
        '{"schema_version": 1, "runs": [], "secret": Infinity}',
        '{"schema_version": 1, "runs": [], "runs": []}',
        '{"schema_version": 1, "runs": ["PRIVATE_PAYLOAD"',
        "[" * 2000,
    ],
)
def test_load_rejects_ambiguous_or_malformed_json_without_input_disclosure(
    tmp_path, text
):
    from rolloutdiff._report_validation import load_report

    path = tmp_path / "report.json"
    path.write_text(text)
    with pytest.raises(ValueError, match="JSON") as exc:
        load_report(path)
    assert "PRIVATE_PAYLOAD" not in str(exc.value)


def test_load_report_reads_strict_utf8_json(tmp_path, report):
    from rolloutdiff._report_validation import load_report

    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_report(path) == report
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="JSON"):
        load_report(path)
