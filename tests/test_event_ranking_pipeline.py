from dataclasses import fields, is_dataclass

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import analyze, fixture

from rolloutdiff import AnalysisResult


def _tables(result, prefix=""):
    tables = {}
    for field in fields(result):
        if not prefix and field.name == "event_ranking":
            continue
        value = getattr(result, field.name)
        name = f"{prefix}{field.name}"
        if isinstance(value, pd.DataFrame):
            tables[name] = value.copy(deep=True)
        elif is_dataclass(value):
            tables.update(_tables(value, f"{name}."))
    return tables


def test_pipeline_ranking_matches_public_api_and_preserves_inputs():
    from rolloutdiff import rank_events

    data = fixture()
    data = pd.concat([data, data.assign(run="other")], ignore_index=True).sample(
        frac=1, random_state=31
    )
    saved_data = data.copy(deep=True)
    result = analyze(data)
    before = _tables(result)
    actual = rank_events(result.events, result.event_members, result.evidence)
    assert not actual.empty
    assert_frame_equal(result.event_ranking, actual, check_exact=True)
    assert_frame_equal(data, saved_data, check_exact=True)
    after = _tables(result)
    for name, expected in before.items():
        assert_frame_equal(after[name], expected, check_exact=True, obj=name)


def test_pipeline_ranking_links_to_events_members_evidence_and_exemplars():
    result = analyze(fixture())
    ranking = result.event_ranking
    assert ranking.event_id.is_unique
    assert set(ranking.event_id) == set(result.events.event_id)
    event_columns = [
        "event_id",
        "run",
        "onset_start",
        "onset_end",
        "n_members",
        "members",
    ]
    assert_frame_equal(
        ranking[event_columns].sort_values("event_id").reset_index(drop=True),
        result.events.sort_values("event_id").reset_index(drop=True),
        check_exact=True,
    )
    representatives = ranking[["event_id", "run", "representative_feature"]].rename(
        columns={"representative_feature": "feature"}
    )
    joined = representatives.merge(
        result.event_members,
        on=["event_id", "run", "feature"],
        validate="one_to_one",
        indicator=True,
    )
    assert joined._merge.eq("both").all()
    joined = representatives.merge(
        result.evidence,
        on=["run", "feature"],
        validate="one_to_one",
        indicator=True,
    )
    assert joined._merge.eq("both").all()
    assert joined.sequential_status.eq("alerted").all()
    exemplars = result.exemplars.merge(
        result.event_members[["event_id", "run", "feature"]],
        on=["run", "feature"],
        validate="many_to_one",
    ).merge(ranking[["event_id", "rank"]], on="event_id", validate="many_to_one")
    assert len(exemplars) == len(result.exemplars)
    assert exemplars["rank"].notna().all()

    contextual = ranking.loc[ranking.representative_feature.eq("scalar:reward")]
    assert len(contextual) == 1
    assert contextual.ranking_status.eq("contextual").all()
    assert contextual.kind.eq("continuous").all()
    assert contextual.tier.eq("structural").all()
    assert (
        ranking.loc[ranking.kind.eq("binary"), "ranking_status"].eq("behavioral").all()
    )


def test_pipeline_ranking_preserves_representative_evidence_and_floor_warning():
    result = analyze(fixture())
    evidence = result.evidence.set_index(["run", "feature"])
    ranking = result.event_ranking
    columns = [
        "tier",
        "kind",
        "onset_step",
        "alert_step",
        "scale_source",
        "numerical_floor_dominant",
        "replication_status",
        "replicates",
        "persistence_status",
        "persistence_fraction",
        "persistence_coverage",
        "persistence_n_observations",
        "persistence_n_evaluable",
        "persistence_n_retained",
        "persistence_n_unavailable",
        "reward_status",
        "reward_mean_difference",
        "reward_n_groups",
        "reward_n_mixed_groups",
        "reward_n_rollouts",
        "reward_n_included_rollouts",
        "promptmix_status",
        "prompt_positive_prompts",
        "prompt_total_prompts",
    ]
    for row in ranking.to_dict("records"):
        source = evidence.loc[(row["run"], row["representative_feature"])]
        for column in columns:
            if pd.isna(source[column]):
                assert pd.isna(row[column]), column
            else:
                assert row[column] == source[column], column
    assert ranking.numerical_floor_dominant.any()


def test_pipeline_ranking_is_stable_under_rollout_and_table_order():
    from rolloutdiff import rank_events

    data = fixture()
    data = pd.concat([data, data.assign(run="other")], ignore_index=True)
    result = analyze(data)
    shuffled = analyze(data.sample(frac=1, random_state=53))
    assert_frame_equal(result.event_ranking, shuffled.event_ranking, check_exact=True)
    reordered = rank_events(
        result.events.sample(frac=1, random_state=2),
        result.event_members.sample(frac=1, random_state=3),
        result.evidence.sample(frac=1, random_state=5),
    )
    assert_frame_equal(result.event_ranking, reordered, check_exact=True)
    for _, ranked_run in result.event_ranking.groupby("run", sort=False):
        assert ranked_run["rank"].tolist() == list(range(1, len(ranked_run) + 1))


def test_pipeline_ranking_has_stable_empty_schema():
    expected = analyze(fixture()).event_ranking.iloc[:0]
    for data in [
        fixture().iloc[:0],
        fixture().assign(response="constant", reward=0.0),
        fixture(single_prompt=True),
    ]:
        result = analyze(data)
        assert result.events.empty
        assert_frame_equal(result.event_ranking, expected, check_exact=True)


def test_analysis_result_constructor_defaults_ranking_independently():
    result = analyze(fixture())
    arguments = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name != "event_ranking"
    }
    legacy = AnalysisResult(**arguments)
    another = AnalysisResult(**arguments)
    assert_frame_equal(legacy.event_ranking, result.event_ranking.iloc[:0])
    assert legacy.event_ranking is not another.event_ranking
    legacy_tables = _tables(legacy)
    for name, expected in _tables(result).items():
        assert_frame_equal(legacy_tables[name], expected, check_exact=True, obj=name)


@pytest.mark.parametrize(
    "case",
    ["rising", "falling", "single_prompt", "empty", "runs", "pending", "partial"],
)
def test_ranking_preserves_every_existing_analysis_table(monkeypatch, case):
    from rolloutdiff import pipeline

    data = fixture(
        falling=case == "falling",
        single_prompt=case == "single_prompt",
        last=6 if case == "pending" else 14,
        one_half=case == "pending",
        switched_prompts=case == "pending",
    )
    if case == "empty":
        data = data.iloc[:0]
    elif case == "runs":
        data = pd.concat(
            [data, fixture(one_half=True).assign(run="other")], ignore_index=True
        ).sample(frac=1, random_state=11)
    elif case == "partial":
        data = data.loc[~data.step.eq(8) | data.prompt_id.eq(data.prompt_id.iloc[0])]
    saved = data.copy(deep=True)
    enabled = analyze(data)
    before = _tables(enabled)
    assert len(before) == 23
    calls = []

    def disabled(events, members, evidence):
        assert_frame_equal(events, enabled.events, check_exact=True)
        assert_frame_equal(members, enabled.event_members, check_exact=True)
        assert_frame_equal(evidence, enabled.evidence, check_exact=True)
        calls.append(True)
        return enabled.event_ranking.iloc[:0].copy()

    monkeypatch.setattr(pipeline, "rank_events", disabled)
    without_ranking = analyze(data)
    assert calls == [True]
    assert without_ranking.event_ranking.empty
    for actual in [_tables(enabled), _tables(without_ranking)]:
        assert actual.keys() == before.keys()
        for name, expected in before.items():
            assert_frame_equal(actual[name], expected, check_exact=True, obj=name)
    assert_frame_equal(data, saved, check_exact=True)
