import copy
from dataclasses import replace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import analyze, fixture

from rolloutdiff import build_report, render_report
from rolloutdiff.report import report_json


@pytest.fixture(scope="module")
def case():
    data = fixture()
    analysis = analyze(data)
    return data, analysis, build_report(analysis)


def resolve(report, rollouts, *, run="r", event_rank=1):
    from rolloutdiff import resolve_exemplars

    return resolve_exemplars(report, rollouts, run=run, event_rank=event_rank)


def examples(event):
    return [example for member in event["members"] for example in member["exemplars"]]


def test_resolves_selected_references_without_mutating_inputs_or_reports(case):
    data, _, report = case
    before_data, before_report = data.copy(deep=True), copy.deepcopy(report)
    before_json, before_html = report_json(report), render_report(report)
    event = resolve(report, data)
    assert event["rank"] == 1
    assert event["representative_feature"] == "def:solve"
    selected = examples(event)
    assert selected
    for example in selected:
        assert example["status"] == "selected"
        assert example["response"] == "def solve():\n    pass"
        assert example["reward"] == 2.0
        assert example["feature_value"] == 1
        assert 4 <= example["step"] <= 6
        assert "prompt" not in example
        del example["response"]
    assert event == report["runs"][0]["events"][0]
    assert report == before_report
    assert_frame_equal(data, before_data)
    assert report_json(report) == before_json
    assert render_report(report) == before_html


def test_identity_includes_run_and_ignores_dataframe_index(case):
    data, _, report = case
    other = data.assign(run="another", response="wrong run")
    combined = pd.concat([other, data]).sample(frac=1, random_state=9)
    combined.index = [0] * len(combined)
    event = resolve(report, combined)
    assert {example["response"] for example in examples(event)} == {
        "def solve():\n    pass"
    }
    with pytest.raises(ValueError, match="missing"):
        resolve(report, other)


def test_shared_exemplars_remain_attached_to_each_feature(case):
    data, _, report = case
    candidate = copy.deepcopy(report)
    event = candidate["runs"][0]["events"][0]
    assert len(event["members"]) > 1
    first = event["members"][0]["exemplars"][0]
    second_member = event["members"][1]
    duplicate = copy.deepcopy(first)
    duplicate.update(feature=second_member["feature"], kind=second_member["kind"])
    second_member["exemplars"] = [duplicate]
    output = resolve(candidate, data)
    assert output["members"][0]["exemplars"][0]["sample_id"] == first["sample_id"]
    assert output["members"][1]["exemplars"][0]["sample_id"] == first["sample_id"]
    assert output["members"][1]["exemplars"][0]["response"] == "def solve():\n    pass"


@pytest.mark.parametrize("field", ["step", "prompt_id", "reward"])
def test_rejects_metadata_mismatch_without_echoing_values(case, field):
    data, _, report = case
    changed = data.copy()
    example = report["runs"][0]["events"][0]["members"][0]["exemplars"][0]
    mask = changed.sample_id.eq(example["sample_id"])
    changed.loc[mask, field] = {
        "step": example["step"] + 1,
        "prompt_id": "private-mismatch-value",
        "reward": example["reward"] + 0.01,
    }[field]
    with pytest.raises(ValueError, match=field) as exc:
        resolve(report, changed)
    assert "private-mismatch-value" not in str(exc.value)


def test_rejects_missing_and_duplicate_canonical_references(case):
    data, _, report = case
    example = report["runs"][0]["events"][0]["members"][0]["exemplars"][0]
    selected = data.sample_id.eq(example["sample_id"])
    with pytest.raises(ValueError, match="missing"):
        resolve(report, data.loc[~selected])
    with pytest.raises(ValueError, match="duplicate"):
        resolve(report, pd.concat([data, data.loc[selected]]))


def test_rejects_duplicate_report_exemplars(case):
    data, _, report = case
    candidate = copy.deepcopy(report)
    member = candidate["runs"][0]["events"][0]["members"][0]
    duplicate = copy.deepcopy(member["exemplars"][0])
    duplicate["selection_order"] = 99
    member["exemplars"].append(duplicate)
    with pytest.raises(ValueError, match="references"):
        resolve(candidate, data)


@pytest.mark.parametrize("reward", [None, 2**53 + 1])
def test_selected_reward_is_required_and_compared_without_float_coercion(case, reward):
    data, _, report = case
    candidate = copy.deepcopy(report)
    example = candidate["runs"][0]["events"][0]["members"][0]["exemplars"][0]
    example["reward"] = reward
    changed = data.copy()
    changed.loc[changed.sample_id.eq(example["sample_id"]), "reward"] = float(2**53)
    with pytest.raises(ValueError, match="reward"):
        resolve(candidate, changed)


def test_preserves_large_integer_steps(case):
    data, _, report = case
    candidate = copy.deepcopy(report)
    offset = 2**53 + 1

    def shift(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in {
                    "onset_start",
                    "onset_end",
                    "onset_step",
                    "alert_step",
                    "step",
                }:
                    if item is not None:
                        value[name] += offset
                else:
                    shift(item)
        elif isinstance(value, list):
            for item in value:
                shift(item)

    shift(candidate)
    shifted = data.assign(step=data.step + offset)
    event = resolve(candidate, shifted)
    assert event["onset_step"] == offset + 4
    assert all(type(example["step"]) is int for example in examples(event))
    assert {example["response"] for example in examples(event)} == {
        "def solve():\n    pass"
    }


def test_shuffled_report_and_rollouts_preserve_member_and_selection_order(case):
    data, _, report = case
    shuffled = copy.deepcopy(report)
    shuffled["runs"][0]["events"].reverse()
    for event in shuffled["runs"][0]["events"]:
        event["members"].reverse()
        for member in event["members"]:
            member["exemplars"].reverse()
    assert resolve(shuffled, data.sample(frac=1, random_state=4)) == resolve(
        report, data
    )


def test_unsupported_and_missing_exemplars_stay_explicit(case):
    data, analysis, report = case
    contextual = next(
        event
        for event in report["runs"][0]["events"]
        if event["representative_feature"] == "scalar:reward"
    )
    event = resolve(report, data.iloc[:0], event_rank=contextual["rank"])
    assert event["ranking_status"] == "contextual"
    assert examples(event)[0]["status"] == "unsupported_kind"
    assert examples(event)[0]["response"] is None
    missing = build_report(replace(analysis, exemplars=analysis.exemplars.iloc[:0]))
    event = resolve(missing, data.iloc[:0])
    assert all(
        member["exemplar_status"] == "missing" and member["exemplars"] == []
        for member in event["members"]
    )


def test_empty_report_has_no_selectable_event(case):
    data, _, _ = case
    with pytest.raises(ValueError, match="run"):
        resolve({"schema_version": 1, "runs": []}, data)
    empty = build_report(analyze(data.iloc[:0]), run="r")
    with pytest.raises(ValueError, match="event"):
        resolve(empty, data.iloc[:0])


@pytest.mark.parametrize(
    "run,event_rank",
    [
        ("missing", 1),
        ("", 1),
        (None, 1),
        ("r", 999),
        ("r", 0),
        ("r", -1),
        ("r", True),
        ("r", 1.0),
        ("r", "1"),
    ],
)
def test_invalid_run_and_rank_fail_clearly(case, run, event_rank):
    data, _, report = case
    with pytest.raises(ValueError, match=r"run|rank"):
        resolve(report, data, run=run, event_rank=event_rank)


def test_uses_canonical_response_and_keeps_empty_response_distinct(case):
    data, _, report = case
    candidate = copy.deepcopy(report)
    example = candidate["runs"][0]["events"][0]["members"][0]["exemplars"][0]
    example["response"] = "spoofed report payload"
    changed = data.copy()
    changed.loc[changed.sample_id.eq(example["sample_id"]), "response"] = ""
    output = resolve(candidate, changed)
    assert output["members"][0]["exemplars"][0]["response"] == ""


def test_noncanonical_input_is_rejected(case):
    _, _, report = case
    with pytest.raises(ValueError, match="DataFrame"):
        resolve(report, [])
