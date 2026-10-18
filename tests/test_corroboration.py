import zlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_replication import fixture, replicate
from test_sparse_replication import remove_prompts_at

from rolloutdiff.features import FeatureTables


def corroborate(data, **kwargs):
    import rolloutdiff

    assert hasattr(rolloutdiff, "event_window_corroboration"), (
        "event-window API is missing"
    )
    return rolloutdiff.event_window_corroboration(
        *data,
        **({"run": "r", "feature": "flag", "baseline_steps": 3, "radius": 2} | kwargs),
    )


def test_broad_change_has_local_witnesses_separate_from_first_alerts():
    result = corroborate(fixture([[0] * 3 + [1] * 8] * 2))
    assert result.status == "corroborated" and result.corroborates is True
    assert result.window_start_step == 4 and result.window_end_step == 8
    assert result.horizon_complete
    assert result.halves.match_start_step.tolist() == [4, 4]
    assert result.halves.match_end_step.tolist() == [6, 6]
    assert not hasattr(result, "replicates")
    assert (
        "alert_step" not in result.halves
        and "first_alert_identified" not in result.halves
    )


def test_ambiguous_first_alert_can_have_later_event_corroboration():
    data = fixture([[0] * 3 + [1] * 12] * 2, omit=[(0, 5)])
    before = replicate(data, coverage_policy="fixed_window", tolerance=4)
    result = corroborate(data, radius=4)
    assert before.status == "unavailable"
    assert result.status == "corroborated"
    half = result.halves.iloc[0]
    assert half.match_start_step == 6 and half.match_end_step == 8
    assert half.missing_steps == [5] and half.n_unscorable == 1
    assert_frame_equal(
        before.halves,
        replicate(data, coverage_policy="fixed_window", tolerance=4).halves,
    )


@pytest.mark.parametrize("missing", ["absent", "se"])
def test_unknown_inside_streak_is_not_either_kind_of_evidence(missing):
    data = fixture(
        [[0] * 3 + [1] * 3] * 2, omit=[(0, 5)] if missing == "absent" else []
    )
    if missing == "se":
        data = remove_prompts_at(data, 0, 5)
    result = corroborate(data, radius=2)
    half = result.halves.iloc[0]
    assert result.status == "unavailable" and result.corroborates is None
    assert half.status == "unavailable" and pd.isna(half.match_end_step)
    assert half.n_scored == 2 and half.window_expected == 5 and half.future_steps == 2
    assert half.missing_steps == ([5] if missing == "absent" else [])
    assert half.unavailable_se_steps == ([5] if missing == "se" else [])


def test_complete_no_change_half_contradicts_broad_event():
    result = corroborate(fixture([[0] * 3 + [1] * 8, [0] * 11]))
    assert result.status == "contradicted" and result.corroborates is False
    assert result.halves.status.tolist() == ["corroborated", "contradicted"]
    assert result.halves.iloc[1].reason == "no_matching_streak_in_complete_window"


def test_even_isolated_gap_prevents_negative_absence_claim():
    data = fixture([[0] * 3 + [1] * 8, [0] * 11], omit=[(1, 5)])
    assert replicate(data, coverage_policy="fixed_window").status == "not_replicated"
    result = corroborate(data)
    assert result.status == "unavailable"
    assert result.halves.iloc[1].status == "unavailable"


def test_observed_opposite_does_not_disprove_match_hidden_by_gaps():
    values = [[0] * 3 + [-1] * 8, [0] * 3 + [6] * 8]
    complete = corroborate(fixture(values, kind="continuous"))
    assert complete.status == "contradicted"
    assert complete.halves.iloc[0].opposite_start_step == 4
    sparse = corroborate(fixture(values, kind="continuous", omit=[(0, 8)]))
    assert sparse.status == "unavailable"
    assert sparse.halves.iloc[0].opposite_end_step == 6


def test_mixed_direction_is_disclosed_without_changing_existence_question():
    values = [[0] * 3 + [-1] * 3 + [1] * 8, [0] * 3 + [6] * 11]
    result = corroborate(fixture(values, kind="continuous"), radius=5)
    assert result.full.alerts.iloc[0].direction == "up"
    assert result.status == "corroborated"
    half = result.halves.iloc[0]
    assert half.match_start_step == 7 and half.match_end_step == 9
    assert half.opposite_start_step == 4 and half.opposite_end_step == 6
    assert half.mixed_direction


def test_downward_continuous_event_uses_same_rule():
    result = corroborate(fixture([[0] * 3 + [-1] * 8] * 2, kind="continuous"))
    assert result.status == "corroborated"
    assert result.halves.match_end_step.tolist() == [6, 6]
    assert result.full.alerts.iloc[0].direction == "down"


def test_right_truncation_cannot_be_negative_but_can_contain_witnesses():
    matching = corroborate(fixture([[0] * 3 + [1] * 3] * 2))
    assert matching.status == "corroborated" and not matching.horizon_complete
    missing = corroborate(fixture([[0] * 3 + [1] * 3, [0] * 6]))
    assert missing.status == "unavailable" and missing.halves.iloc[1].future_steps == 2


def test_streak_must_fit_entirely_inside_window_and_be_long_enough():
    result = corroborate(fixture([[0] * 3 + [1] * 8] * 2), radius=0)
    assert result.status == "unavailable"
    assert result.halves.reason.eq("insufficient_window_observations").all()
    assert result.halves.match_end_step.isna().all()


def test_window_does_not_use_preceding_streak_points():
    values = [[0] * 3 + [1] * 3 + [0] * 5] * 2
    # Full alert at 6; window 5..7 has only two crossing points, 5 and 6.
    result = corroborate(fixture(values), radius=1)
    assert result.status == "contradicted"
    assert result.halves.match_end_step.isna().all()


def test_baseline_is_fixed_and_never_borrows_from_window():
    data = fixture([[0] * 5 + [1] * 10] * 2, omit=[(0, 1), (0, 2), (0, 3)])
    result = corroborate(data, baseline_steps=5)
    half = result.halves.iloc[0]
    assert (
        result.status == "unavailable" and half.reason == "insufficient_fixed_baseline"
    )
    assert half.baseline_used_steps == [4, 5] and half.baseline_required == 3
    assert half.n_scored == 0


def test_sparse_baseline_reuses_fixed_window_estimate_exactly():
    data = fixture([[0] * 5 + [1] * 10] * 2, omit=[(0, 1), (0, 3)])
    result = corroborate(data, baseline_steps=5)
    rep = replicate(data, baseline_steps=5, coverage_policy="fixed_window")
    fields = [
        "baseline_mean",
        "baseline_sd",
        "baseline_se_floor",
        "scale",
        "baseline_used_steps",
    ]
    assert_frame_equal(result.halves[fields], rep.halves[fields])
    assert result.status == "corroborated"


def test_single_unit_uncertainty_is_not_replaced_by_completion_count():
    result = corroborate(
        fixture([[0] * 3 + [1] * 8] * 2, prompts_per_half=1, copies=20)
    )
    assert result.status == "unavailable" and result.halves.baseline_used.eq(0).all()
    assert result.halves.match_end_step.isna().all()


def test_large_sparse_step_ids_preserved_with_missing_witness():
    steps = [2**53 + 7 * i for i in range(11)]
    result = corroborate(fixture([[0] * 3 + [1] * 8, [0] * 11], steps=steps))
    assert result.window_start_step == steps[3] and result.window_end_step == steps[7]
    assert result.halves.iloc[0].match_end_step == steps[5]
    assert pd.isna(result.halves.iloc[1].match_end_step)


def test_no_full_alert_is_unavailable_not_negative():
    result = corroborate(fixture([[0] * 11] * 2))
    assert result.status == "unavailable" and result.reason == "no_full_alert"
    assert result.corroborates is None and result.halves.empty


def test_thresholds_apply_to_halves_and_full():
    result = corroborate(
        fixture([[0] * 3 + [1] * 8] * 2), sustain=2, z_threshold=2, numerical_floor=0.4
    )
    assert result.full.alerts.iloc[0].alert_step == 5
    assert result.halves.match_end_step.tolist() == [5, 5]
    assert result.halves.scale.eq(0.4).all()


@pytest.mark.parametrize("radius", [-1, True, 1.5, np.nan])
def test_bad_radius_rejected_even_without_full_alert(radius):
    with pytest.raises(ValueError, match="radius"):
        corroborate(fixture([[0] * 11] * 2), radius=radius)


def test_complete_contradiction_survives_other_half_unavailable_baseline():
    rows, features = fixture([[0] * 11, [0] * 3 + [1] * 8])
    names = rows.loc[
        rows.prompt_id.map(lambda value: zlib.crc32(value.encode()) & 1).eq(1),
        "prompt_id",
    ].unique()
    rows = rows.loc[~rows.prompt_id.isin(names[1:])]
    features = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(rows.sample_id)],
        features.sparse,
        features.catalog,
    )
    result = corroborate((rows, features))
    assert result.status == "contradicted"
    assert result.halves.status.tolist() == ["contradicted", "unavailable"]
    assert result.halves.iloc[1].reason == "insufficient_fixed_baseline"


def test_matching_earlier_first_alert_is_not_required_to_be_in_event_window():
    values = [
        [-100.0] * 3 + [-99.0] * 4 + [900.0] * 5,
        [100.0] * 3 + [101.0] * 4 + [1100.0] * 5,
    ]
    data = fixture(values, kind="continuous")
    rep = replicate(data)
    assert rep.status == "not_replicated" and rep.halves.alert_step.tolist() == [6, 6]
    result = corroborate(data)
    assert result.status == "corroborated"
    assert result.window_start_step == 8
    assert result.halves.match_start_step.tolist() == [8, 8]
    assert result.halves.match_end_step.tolist() == [10, 10]


def test_changes_beyond_window_do_not_change_bounded_evidence():
    data = fixture([[0] * 3 + [1] * 12] * 2)
    before = corroborate(data)
    rows, features = data
    features.dense.loc[rows.step.gt(8).to_numpy(), "flag"] = 0
    after = corroborate((rows, features))
    assert before.status == after.status == "corroborated"
    assert_frame_equal(before.halves, after.halves)
    assert_frame_equal(before.assignments, after.assignments)


def test_full_unscorable_has_no_event_window():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2, prompts_per_half=1)
    selected = rows.prompt_id.eq(rows.prompt_id.iloc[0])
    result = corroborate(
        (
            rows.loc[selected],
            FeatureTables(
                features.dense.loc[selected], features.sparse, features.catalog
            ),
        )
    )
    assert result.status == "unavailable" and result.reason == "full_unscorable"
    assert result.window_start_step is None and result.window_end_step is None
    assert result.halves.empty


def test_counts_conserve_intended_window_even_with_missing_baseline_and_future():
    result = corroborate(
        fixture([[0] * 5 + [1] * 3] * 2, omit=[(0, 1), (0, 2), (0, 3), (0, 6)]),
        baseline_steps=5,
    )
    for half in result.halves.itertuples():
        assert half.n_scored + half.n_unscorable == half.window_expected
        assert (
            half.window_observed + len(half.missing_steps) + half.future_steps
            == half.window_expected
        )
        assert (
            half.window_available + len(half.unavailable_se_steps)
            == half.window_observed
        )
        assert (
            half.baseline_used + len(half.baseline_excluded_steps)
            == half.baseline_expected
        )


def test_empty_whole_unit_half_is_unavailable_without_row_fallback():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2, copies=8)
    selected = rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(0)
    data = (
        rows.loc[selected],
        FeatureTables(features.dense.loc[selected], features.sparse, features.catalog),
    )
    result = corroborate(data)
    assert result.status == "unavailable" and result.corroborates is None
    assert result.assignments.half.eq(0).all()
    half = result.halves.iloc[1]
    assert half.status == "unavailable" and half.reason == "insufficient_fixed_baseline"
    assert half.n_rollouts == half.n_sampling_units == half.window_observed == 0
    assert half.missing_steps == [4, 5, 6, 7, 8] and half.n_unscorable == 5
    fixed = replicate(data, coverage_policy="fixed_window")
    assert fixed.status == "unavailable"
    assert fixed.halves.iloc[1].reason == "insufficient_fixed_baseline"
