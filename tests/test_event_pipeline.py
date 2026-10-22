from dataclasses import fields, is_dataclass

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import analyze, fixture

from rolloutdiff import AnalysisResult


def _tables(result, prefix=""):
    tables = {}
    for field in fields(result):
        if not prefix and field.name in {"events", "event_members"}:
            continue
        value = getattr(result, field.name)
        name = f"{prefix}{field.name}"
        if isinstance(value, pd.DataFrame):
            tables[name] = value.copy(deep=True)
        elif is_dataclass(value):
            tables.update(_tables(value, f"{name}."))
    return tables


def test_pipeline_events_match_public_api_and_join_feature_evidence():
    from rolloutdiff import EventGrouping, group_events

    data = fixture()
    data = pd.concat([data, data.assign(run="other")], ignore_index=True).sample(
        frac=1, random_state=17
    )
    saved = data.copy(deep=True)
    result = analyze(data)
    saved_tables = _tables(result)
    expected = group_events(data, result.features, result.sequential.alerts)
    assert isinstance(expected, EventGrouping)
    assert_frame_equal(result.events, expected.events, check_exact=True)
    assert_frame_equal(result.event_members, expected.members, check_exact=True)
    assert_frame_equal(data, saved, check_exact=True)
    for name, actual in _tables(result).items():
        assert_frame_equal(actual, saved_tables[name], check_exact=True, obj=name)

    members = result.event_members
    assert not members.empty
    assert not members.duplicated(["run", "feature"]).any()
    assert set(members[["run", "feature"]].itertuples(index=False, name=None)) == set(
        result.sequential.alerts[["run", "feature"]].itertuples(index=False, name=None)
    )
    joined = members.merge(
        result.evidence,
        on=["run", "feature"],
        suffixes=("_member", "_evidence"),
        validate="one_to_one",
        indicator=True,
    )
    assert joined._merge.eq("both").all()
    assert joined.sequential_status.eq("alerted").all()
    for column in ["kind", "onset_step", "alert_step"]:
        assert joined[f"{column}_member"].equals(joined[f"{column}_evidence"])
    joined_exemplars = result.exemplars.merge(
        members[["event_id", "run", "feature"]],
        on=["run", "feature"],
        validate="many_to_one",
        indicator=True,
    )
    assert len(joined_exemplars) == len(result.exemplars)
    assert joined_exemplars._merge.eq("both").all()

    assert result.events.event_id.is_unique
    for event in result.events.itertuples(index=False):
        selected = members.loc[members.event_id.eq(event.event_id)]
        assert len(selected) == event.n_members
        assert selected.run.eq(event.run).all()
        assert event.onset_start == selected.onset_step.min()
        assert event.onset_end == selected.onset_step.max()
        assert event.onset_end - event.onset_start <= 3
        references = tuple(
            selected[["run", "feature"]].itertuples(index=False, name=None)
        )
        assert references == event.members
    unsupported = members.loc[members.kind.eq("continuous")]
    assert not unsupported.empty
    assert unsupported.status.eq("unsupported_kind").all()
    assert (
        result.events.set_index("event_id")
        .loc[unsupported.event_id, "n_members"]
        .eq(1)
        .all()
    )


def test_pipeline_events_are_stable_under_rollout_order():
    data = fixture()
    result = analyze(data)
    shuffled = analyze(data.sample(frac=1, random_state=37))
    assert_frame_equal(result.events, shuffled.events, check_exact=True)
    assert_frame_equal(result.event_members, shuffled.event_members, check_exact=True)


def test_pipeline_event_tables_have_stable_empty_schema():
    result = analyze(fixture())
    for data in [
        fixture().iloc[:0],
        fixture().assign(response="constant", reward=0.0),
        fixture(single_prompt=True),
    ]:
        empty = analyze(data)
        assert empty.sequential.alerts.empty
        assert_frame_equal(empty.events, result.events.iloc[:0], check_exact=True)
        assert_frame_equal(
            empty.event_members, result.event_members.iloc[:0], check_exact=True
        )


def test_analysis_result_constructor_defaults_event_tables_independently():
    result = analyze(fixture())
    arguments = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name not in {"events", "event_members"}
    }
    legacy = AnalysisResult(**arguments)
    another = AnalysisResult(**arguments)
    assert_frame_equal(legacy.exemplars, result.exemplars, check_exact=True)
    assert_frame_equal(legacy.events, result.events.iloc[:0], check_exact=True)
    assert_frame_equal(
        legacy.event_members, result.event_members.iloc[:0], check_exact=True
    )
    assert legacy.events is not another.events
    assert legacy.event_members is not another.event_members


@pytest.mark.parametrize(
    "case", ["rising", "falling", "single_prompt", "empty", "runs"]
)
def test_grouping_preserves_every_existing_analysis_table(monkeypatch, case):
    from rolloutdiff import EventGrouping, pipeline

    data = fixture(falling=case == "falling", single_prompt=case == "single_prompt")
    if case == "empty":
        data = data.iloc[:0]
    elif case == "runs":
        data = pd.concat(
            [data, fixture(one_half=True).assign(run="other")], ignore_index=True
        ).sample(frac=1, random_state=11)
    saved = data.copy(deep=True)
    enabled = analyze(data)
    before = _tables(enabled)
    assert len(before) == 21

    def disabled(values, alerts):
        return EventGrouping(
            enabled.events.iloc[:0].copy(), enabled.event_members.iloc[:0].copy()
        )

    monkeypatch.setattr(pipeline, "_group_events_prepared", disabled)
    without_grouping = analyze(data)
    assert without_grouping.events.empty
    assert without_grouping.event_members.empty
    for actual in [_tables(enabled), _tables(without_grouping)]:
        assert actual.keys() == before.keys()
        for name, expected in before.items():
            assert_frame_equal(actual[name], expected, check_exact=True, obj=name)
    assert_frame_equal(data, saved, check_exact=True)
