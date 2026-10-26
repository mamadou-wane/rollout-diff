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


def resolve(report, rollouts, *, run="r", event_rank=1, **kwargs):
    from rolloutdiff import resolve_exemplars

    return resolve_exemplars(report, rollouts, run=run, event_rank=event_rank, **kwargs)


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


def matched_case(case):
    data, _, report = case
    report = copy.deepcopy(report)
    positive = data.loc[data.sample_id.eq("table:36")].iloc[0].to_dict()
    absent = data.loc[data.sample_id.eq("table:37")].iloc[0].to_dict()
    additions = [
        dict(positive, sample_id="z-present", reward=6.0),
        dict(absent, sample_id="a-absent", reward=10.0),
        dict(absent, sample_id="00-other-run", run="other", reward=-1000.0),
        dict(absent, sample_id="00-other-step", step=3, reward=-1000.0),
        dict(
            absent,
            sample_id="00-other-prompt",
            prompt_id="other",
            prompt="other",
            reward=-1000.0,
        ),
    ]
    for member in report["runs"][0]["events"][0]["members"]:
        for example in member["exemplars"]:
            if example["sample_id"] == "table:36":
                example.update(
                    reward_group_size=4, reward_group_mean=4.5, reward_deviation=-2.5
                )
    return pd.concat([data, pd.DataFrame(additions)], ignore_index=True), report


def test_matches_exact_group_and_uses_all_group_rewards(case):
    data, report = matched_case(case)
    event = resolve(report, data, with_absent=True)
    example = event["members"][0]["exemplars"][0]
    comparison = example["comparison"]
    assert example["feature"] == "def:solve"
    assert example["sample_id"] == "table:36"
    assert example["reward"] == 2.0
    assert comparison == {
        "status": "available",
        "reason": None,
        "n_rollouts": 4,
        "n_present": 2,
        "n_absent": 2,
        "reward_present": 4.0,
        "reward_absent": 5.0,
        "difference": -1.0,
        "counterpart": {
            "run": "r",
            "step": 4,
            "prompt_id": "p4",
            "sample_id": "a-absent",
            "reward": 10.0,
            "feature_value": 0.0,
            "response": "plain words.",
        },
    }


def test_counterpart_selection_ignores_rewards_and_input_order(case):
    data, report = matched_case(case)
    original = resolve(report, data, with_absent=True)
    shuffled = data.sample(frac=1, random_state=73)
    shuffled.index = [99] * len(shuffled)
    assert resolve(report, shuffled, with_absent=True) == original
    changed = data.copy()
    changed.loc[changed.sample_id.eq("a-absent"), "reward"] = -100.0
    changed.loc[changed.sample_id.eq("table:37"), "reward"] = 100.0
    other = resolve(report, changed, with_absent=True)
    assert (
        other["members"][0]["exemplars"][0]["comparison"]["counterpart"]["sample_id"]
        == "a-absent"
    )


def test_each_feature_uses_its_own_extracted_partition(case):
    data, _, report = case
    changed = data.copy()
    changed.loc[changed.sample_id.eq("table:37"), "response"] = "def other():\n    pass"
    event = resolve(report, changed, with_absent=True)
    by_feature = {
        member["feature"]: member["exemplars"][0]["comparison"]
        for member in event["members"]
    }
    assert by_feature["def:solve"]["counterpart"]["sample_id"] == "table:37"
    assert by_feature["has_code"]["counterpart"] is None
    assert by_feature["has_code"]["status"] == "unavailable"
    assert by_feature["has_code"]["reason"] == "no_absent_completion"
    assert by_feature["has_code"]["n_present"] == 2
    assert by_feature["has_code"]["n_absent"] == 0
    assert by_feature["has_code"]["reward_present"] == 1.0
    assert by_feature["has_code"]["reward_absent"] is None
    assert by_feature["has_code"]["difference"] is None


@pytest.mark.parametrize(
    "response",
    [
        "<think>def solve():\n    pass</think>plain words.",
        '```python\ntext = "def solve(): pass"\n```',
    ],
)
def test_absence_uses_extractor_not_response_substrings(case, response):
    data, _, report = case
    changed = data.copy()
    changed.loc[changed.sample_id.eq("table:37"), "response"] = response
    event = resolve(report, changed, with_absent=True)
    assert (
        event["members"][0]["exemplars"][0]["comparison"]["counterpart"]["sample_id"]
        == "table:37"
    )


def test_changed_selected_feature_fails_only_when_requested(case):
    data, _, report = case
    changed = data.copy()
    changed.loc[changed.sample_id.eq("table:40"), "response"] = "plain words."
    assert (
        resolve(report, changed)["members"][0]["exemplars"][2]["response"]
        == "plain words."
    )
    with pytest.raises(ValueError, match="feature"):
        resolve(report, changed, with_absent=True)


@pytest.mark.parametrize("value", [None, 0.0])
def test_recorded_selected_feature_value_must_support_presence(case, value):
    data, _, report = case
    candidate = copy.deepcopy(report)
    candidate["runs"][0]["events"][0]["members"][0]["exemplars"][0]["feature_value"] = (
        value
    )
    with pytest.raises(ValueError, match="feature"):
        resolve(candidate, data, with_absent=True)


def test_continuous_predicate_cannot_be_treated_as_binary(case):
    data, _, report = case

    def rename(value):
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return "n_files_declared" if value == "def:solve" else value

    with pytest.raises(ValueError, match="binary"):
        resolve(rename(report), data, with_absent=True)


def test_missing_group_rows_fail_and_invalid_rewards_are_not_dropped(case):
    data, _, report = case
    with pytest.raises(ValueError, match="group size"):
        resolve(report, data.loc[~data.sample_id.eq("table:37")], with_absent=True)
    changed = data.copy()
    changed.loc[changed.sample_id.eq("table:37"), "reward"] = float("nan")
    with pytest.raises(ValueError, match="reward"):
        resolve(report, changed, with_absent=True)


def test_matched_inspection_preserves_default_inputs_and_reports(case, monkeypatch):
    import rolloutdiff.inspection as module

    data, _, report = case
    before = copy.deepcopy(report)
    before_data = data.copy(deep=True)
    encoded, rendered = report_json(report), render_report(report)
    original = resolve(report, data)
    matched = resolve(report, data, with_absent=True)
    for example in examples(matched):
        del example["comparison"]
    assert matched == original
    assert report == before
    assert_frame_equal(data, before_data)
    assert report_json(report) == encoded
    assert render_report(report) == rendered

    def unexpected(*args, **kwargs):
        pytest.fail("default inspection must not extract features")

    monkeypatch.setattr(module, "extract", unexpected)
    assert resolve(report, data, with_absent=False) == original
    assert resolve(report, data) == original


def test_unselected_exemplars_keep_unavailable_comparisons(case):
    data, _, report = case
    candidate = copy.deepcopy(report)
    context = next(
        e
        for e in candidate["runs"][0]["events"]
        if e["representative_feature"] == "scalar:reward"
    )
    event = resolve(
        candidate, data.iloc[:0], event_rank=context["rank"], with_absent=True
    )
    comparison = examples(event)[0]["comparison"]
    assert comparison["status"] == "unavailable"
    assert comparison["reason"] == "unsupported_kind"
    assert comparison["n_present"] is comparison["n_absent"] is None
    assert comparison["counterpart"] is None
    member = candidate["runs"][0]["events"][0]["members"][0]
    member["exemplar_status"] = "missing"
    member["exemplars"] = []
    event = resolve(candidate, data, with_absent=True)
    assert event["members"][0]["exemplar_status"] == "missing"
    assert event["members"][0]["exemplars"] == []


def test_response_code_is_never_executed(case, tmp_path):
    data, _, report = case
    changed = data.copy()
    marker = tmp_path / "must-not-exist"
    changed.loc[changed.sample_id.eq("table:37"), "response"] = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')"
    )
    event = resolve(report, changed, with_absent=True)
    assert event["members"][0]["exemplars"][0]["comparison"]["status"] == "available"
    assert not marker.exists()


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_with_absent_requires_boolean(case, value):
    data, _, report = case
    with pytest.raises(ValueError, match="with_absent"):
        resolve(report, data, with_absent=value)


def _test_name_case(case):
    data, _, report = case
    names = {
        "def:solve": "defines_test_named",
        "has_code": "defines_test_token",
    }

    def rename(value):
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rename(item) for item in value]
        return names.get(value, value) if isinstance(value, str) else value

    report = rename(report)
    data = data.copy(deep=True)
    data.loc[data.response.eq("def solve():\n    pass"), "response"] = (
        "def run_tests():\n    pass"
    )
    sibling = data.loc[data.sample_id.eq("table:37")].iloc[0].to_dict()
    data.loc[data.sample_id.eq("table:37"), "response"] = (
        "def shortestPath():\n    pass"
    )
    data = pd.concat(
        [data, pd.DataFrame([dict(sibling, sample_id="z-plain", reward=4.0)])],
        ignore_index=True,
    )
    for example in examples(report["runs"][0]["events"][0]):
        if example["sample_id"] == "table:36":
            example.update(
                reward_group_size=3, reward_group_mean=2.0, reward_deviation=0.0
            )
    return data, report


def test_historical_and_token_features_partition_every_sibling_separately(case):
    data, report = _test_name_case(case)
    before_data, before_report = data.copy(deep=True), copy.deepcopy(report)
    encoded, rendered = report_json(report), render_report(report)
    ordinary = resolve(report, data)
    matched = resolve(report, data, with_absent=True)
    comparisons = {
        member["feature"]: member["exemplars"][0]["comparison"]
        for member in matched["members"]
    }
    legacy, token = (
        comparisons["defines_test_named"],
        comparisons["defines_test_token"],
    )
    assert (legacy["n_present"], legacy["n_absent"]) == (2, 1)
    assert (legacy["reward_present"], legacy["reward_absent"]) == (1.0, 4.0)
    assert legacy["difference"] == -3.0
    assert legacy["counterpart"]["sample_id"] == "z-plain"
    assert (token["n_present"], token["n_absent"]) == (1, 2)
    assert (token["reward_present"], token["reward_absent"]) == (2.0, 2.0)
    assert token["difference"] == 0.0
    assert token["counterpart"]["sample_id"] == "table:37"
    shuffled = data.sample(frac=1, random_state=31)
    shuffled.index = [0] * len(shuffled)
    assert resolve(report, shuffled, with_absent=True) == matched
    for example in examples(matched):
        del example["comparison"]
    assert matched == ordinary
    assert_frame_equal(data, before_data)
    assert report == before_report
    assert report_json(report) == encoded
    assert render_report(report) == rendered


@pytest.mark.parametrize(
    "response, present",
    [
        ("def shortestPath():\n    pass", True),
        ("class Solution:\n    async def contest(self):\n        pass", True),
        ("def helper():\n    def latest():\n        pass", True),
        ("<think>def contest():\n    pass</think>plain words.", False),
        ('text = "def contest(): pass"', False),
        ("```python\ndef contest(\n```", False),
        ("contest()", False),
    ],
)
def test_historical_presence_uses_parsed_definitions(case, response, present):
    data, report = _test_name_case(case)
    event = report["runs"][0]["events"][0]
    token = next(m for m in event["members"] if m["feature"] == "defines_test_token")
    token.update(exemplar_status="missing", exemplars=[])
    data.loc[data.sample_id.eq("table:36"), "response"] = response
    assert resolve(report, data)["members"][0]["exemplars"][0]["response"] == response
    if present:
        matched = resolve(report, data, with_absent=True)
        assert matched["members"][0]["exemplars"][0]["comparison"]["n_present"] == 2
    else:
        with pytest.raises(ValueError, match="feature does not match"):
            resolve(report, data, with_absent=True)
