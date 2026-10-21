from dataclasses import fields

import pandas as pd
from pandas.testing import assert_frame_equal
from test_full_pipeline import analyze, fixture

from rolloutdiff import AnalysisResult, select_exemplars


def test_pipeline_exemplars_match_public_api_and_resolve_canonical_features():
    data = fixture()
    data = pd.concat([data, data.assign(run="other")], ignore_index=True).sample(
        frac=1, random_state=13
    )
    result = analyze(data)
    assert_frame_equal(
        result.exemplars,
        select_exemplars(data, result.features, result.sequential.alerts),
    )
    selected = result.exemplars.query('status == "selected"')
    assert not selected.empty
    assert selected.groupby(["run", "feature"]).size().le(3).all()
    assert not selected.duplicated(["run", "feature", "sample_id"]).any()
    rollouts = data.set_index(["run", "sample_id"])
    dense = result.features.dense.set_index(["run", "sample_id"])
    sparse = set(result.features.sparse.itertuples(index=False, name=None))
    for row in selected.itertuples(index=False):
        source = rollouts.loc[(row.run, row.sample_id)]
        assert source.step == row.step
        assert source.prompt_id == row.prompt_id and source.reward == row.reward
        assert row.onset_step <= row.step <= row.alert_step
        value = (
            dense.loc[(row.run, row.sample_id), row.feature]
            if row.feature in dense.columns
            else float((row.run, row.sample_id, row.feature) in sparse)
        )
        assert value == row.feature_value == 1
    unsupported = result.exemplars.query('kind == "continuous"')
    assert not unsupported.empty
    assert unsupported.status.eq("unsupported_kind").all()
    assert unsupported.sample_id.isna().all()
    assert set(
        result.exemplars[["run", "feature"]].itertuples(index=False, name=None)
    ) == set(
        result.sequential.alerts[["run", "feature"]].itertuples(index=False, name=None)
    )


def test_pipeline_has_stable_empty_schema_without_inventing_exemplars():
    expected = analyze(fixture()).exemplars.iloc[:0]
    for data in [
        fixture().iloc[:0],
        fixture().assign(response="constant", reward=0.0),
        fixture(single_prompt=True),
    ]:
        result = analyze(data)
        assert result.sequential.alerts.empty
        assert_frame_equal(result.exemplars, expected)
    falling = analyze(fixture(falling=True))
    assert (
        falling.evidence.set_index("feature").loc["def:solve", "sequential_status"]
        == "no_alert"
    )
    assert falling.exemplars.query('feature == "def:solve"').empty


def test_analysis_result_existing_constructor_remains_valid():
    result = analyze(fixture())
    arguments = {
        field.name: getattr(result, field.name)
        for field in fields(result)
        if field.name != "exemplars"
    }
    legacy = AnalysisResult(**arguments)
    another = AnalysisResult(**arguments)
    assert_frame_equal(legacy.exemplars, result.exemplars.iloc[:0])
    assert legacy.exemplars is not another.exemplars
