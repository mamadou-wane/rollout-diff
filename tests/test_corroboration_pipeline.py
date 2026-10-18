import zlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import fixture, row

from rolloutdiff import analyze_rollouts


def analyze(data, **kwargs):
    return analyze_rollouts(
        data,
        **(
            {
                "sequential_options": {"baseline_steps": 3},
                "retrospective_options": {"min_segment_steps": 3},
                "replication_tolerance": 2,
                "corroboration_radius": 2,
            }
            | kwargs
        ),
    )


def test_event_window_corroboration_is_separate_from_first_alert_replication():
    data = fixture(omit=[(0, 4)])
    result = analyze(data)
    target = row(result)
    assert target.replication_status == "unavailable"
    assert pd.isna(target.replicates)
    assert target.corroboration_status == "corroborated"
    assert target.corroborates
    assert (target.corroboration_window_start, target.corroboration_window_end) == (
        4,
        8,
    )
    assert target.corroboration_horizon_complete
    halves = result.corroboration_halves.query('feature == "def:solve"')
    assert len(halves) == 2
    assert halves.status.tolist() == ["corroborated", "corroborated"]


@pytest.mark.parametrize("feature", ["def:solve", "scalar:reward"])
def test_integrated_corroboration_matches_public_function(feature):
    from rolloutdiff import EventWindowEvidence, event_window_corroboration

    data = fixture(omit=[(0, 4)])
    saved = data.copy(deep=True)
    result = analyze(data)
    direct = event_window_corroboration(
        data, result.features, run="r", feature=feature, baseline_steps=3, radius=2
    )
    assert isinstance(direct, EventWindowEvidence)
    target = row(result, feature)
    assert target.corroboration_status == direct.status
    assert target.corroborates == direct.corroborates
    assert target.corroboration_reason == direct.reason
    assert target.corroboration_window_start == direct.window_start_step
    assert target.corroboration_window_end == direct.window_end_step
    assert target.corroboration_horizon_complete == direct.horizon_complete
    table = result.corroboration_halves.query("feature == @feature").drop(
        columns=["run", "feature"]
    )
    assert_frame_equal(table.reset_index(drop=True), direct.halves, check_dtype=False)
    assert_frame_equal(data, saved)


def test_observed_contradiction_and_unavailable_horizon_remain_distinct():
    complete = row(analyze(fixture(one_half=True)))
    assert complete.corroboration_status == "contradicted"
    assert not complete.corroborates
    incomplete = row(analyze(fixture(last=6, one_half=True)))
    assert incomplete.corroboration_status == "unavailable"
    assert pd.isna(incomplete.corroborates)
    assert not incomplete.corroboration_horizon_complete


@pytest.mark.parametrize(
    "data,reason",
    [
        (fixture(falling=True), "no_full_alert"),
        (fixture(single_prompt=True), "full_unscorable"),
    ],
)
def test_nonalerting_series_do_not_borrow_retrospective_change(data, reason):
    result = analyze(data)
    target = row(result)
    assert target.corroboration_status == "unavailable"
    assert target.corroboration_reason == reason
    assert pd.isna(target.corroborates)
    assert pd.isna(target.corroboration_window_start)
    assert pd.isna(target.corroboration_window_end)
    assert result.corroboration_halves.query('feature == "def:solve"').empty


def test_window_radius_changes_only_corroboration_evidence():
    data = fixture(omit=[(0, 4)])
    narrow = analyze(data)
    wider = analyze(data, corroboration_radius=6)
    for name in ["alerts", "diagnostics", "scores", "screening"]:
        assert_frame_equal(
            getattr(narrow.sequential, name), getattr(wider.sequential, name)
        )
    for name in ["changepoints", "diagnostics", "screening"]:
        assert_frame_equal(
            getattr(narrow.retrospective, name), getattr(wider.retrospective, name)
        )
    assert_frame_equal(narrow.half_assignments, wider.half_assignments)
    assert_frame_equal(narrow.replication_halves, wider.replication_halves)
    unchanged = [
        name
        for name in narrow.evidence
        if not name.startswith("corroboration_") and name != "corroborates"
    ]
    assert_frame_equal(narrow.evidence[unchanged], wider.evidence[unchanged])
    assert str(narrow.evidence.corroborates.dtype) == "boolean"


def test_default_radius_is_ten_independent_of_replication_tolerance():
    data = fixture(last=20)
    default = analyze_rollouts(
        data,
        sequential_options={"baseline_steps": 3},
        retrospective_options={"min_segment_steps": 3},
        replication_tolerance=0,
    )
    explicit = analyze(data, corroboration_radius=10, replication_tolerance=0)
    assert_frame_equal(default.evidence, explicit.evidence)
    assert_frame_equal(default.corroboration_halves, explicit.corroboration_halves)
    assert row(default).corroboration_window_end == 16


@pytest.mark.parametrize("radius", [-1, True, 1.5, None, "2"])
def test_invalid_radius_fails_before_feature_extraction(monkeypatch, radius):
    from rolloutdiff import pipeline

    def unexpected(*args, **kwargs):
        pytest.fail("invalid radius reached feature extraction")

    monkeypatch.setattr(pipeline, "featurize", unexpected)
    with pytest.raises(ValueError, match="corroboration_radius"):
        analyze(fixture(), corroboration_radius=radius)


def test_exact_prompt_variants_remain_in_whole_source_problem_halves():
    from rolloutdiff import event_window_corroboration

    original = fixture()
    copies, mappings = [], []
    for variant in (0, 1):
        copy = original.copy()
        copy["sample_id"] += f"-variant-{variant}"
        copy["prompt_id"] += f"-variant-{variant}"
        copy["prompt"] += f" variant {variant}"
        copies.append(copy)
        mapping = copy[["run", "prompt_id"]].drop_duplicates().copy()
        mapping["sampling_unit_id"] = mapping.prompt_id.str.removesuffix(
            f"-variant-{variant}"
        )
        mappings.append(mapping)
    data = pd.concat(copies, ignore_index=True)
    units = pd.concat(mappings, ignore_index=True)
    result = analyze(data, sampling_units=units)
    direct = event_window_corroboration(
        data,
        result.features,
        run="r",
        feature="def:solve",
        baseline_steps=3,
        radius=2,
        sampling_units=units,
    )
    assert (
        result.half_assignments.groupby(["run", "sampling_unit_id"])
        .half.nunique()
        .eq(1)
        .all()
    )
    assert (
        result.half_assignments.groupby(["run", "sampling_unit_id"])
        .prompt_id.nunique()
        .eq(2)
        .all()
    )
    assert row(result).corroboration_status == direct.status == "corroborated"
    table = result.corroboration_halves.query('feature == "def:solve"').drop(
        columns=["run", "feature"]
    )
    assert_frame_equal(table.reset_index(drop=True), direct.halves, check_dtype=False)


def test_event_window_step_ids_preserve_large_integers_with_missing_rows():
    data = fixture()
    base = 2**60
    data["step"] += base
    result = analyze(data)
    target = row(result)
    assert target.corroboration_window_start == base + 4
    assert target.corroboration_window_end == base + 8
    assert type(target.corroboration_window_start) is int
    assert result.evidence.corroboration_window_start.isna().any()


def test_short_fixed_window_is_unavailable_instead_of_negative():
    result = analyze(fixture(one_half=True), corroboration_radius=0)
    target = row(result)
    assert target.corroboration_status == "unavailable"
    detail = result.corroboration_halves.query('feature == "def:solve"')
    assert detail.reason.eq("insufficient_window_observations").all()
    assert pd.isna(target.corroborates)


def test_available_full_alert_does_not_supply_missing_half_uncertainty():
    data = fixture()
    units = data[["run", "prompt_id"]].drop_duplicates().copy()
    halves = units.prompt_id.map(lambda value: zlib.crc32(value.encode()) & 1)
    representatives = {
        half: units.loc[halves.eq(half), "prompt_id"].iloc[0] for half in (0, 1)
    }
    units["sampling_unit_id"] = halves.map(representatives)
    result = analyze(data, sampling_units=units)
    target = row(result)
    assert target.sequential_status == "alerted"
    assert target.corroboration_status == "unavailable"
    assert pd.isna(target.corroborates)
    detail = result.corroboration_halves.query('feature == "def:solve"')
    assert len(detail) == 2
    assert detail.status.eq("unavailable").all()
