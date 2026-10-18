import zlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_replication import fixture, replicate

from rolloutdiff.features import FeatureTables
from rolloutdiff.replication import split_half_replication


def sparse(data, **kwargs):
    return replicate(data, coverage_policy="fixed_window", **kwargs)


def remove_prompts_at(data, half, step):
    rows, features = data
    names = rows.loc[
        rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(half), "prompt_id"
    ].unique()
    rows = rows.loc[~(rows.step.eq(step) & rows.prompt_id.isin(names[1:]))]
    features = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(rows.sample_id)],
        features.sparse,
        features.catalog,
    )
    return rows, features


def test_dense_fixed_window_matches_legacy():
    data = fixture([[0] * 3 + [1] * 8] * 2)
    old, new = replicate(data), sparse(data)
    assert old.coverage_policy == "contiguous_prefix"
    assert new.coverage_policy == "fixed_window"
    assert new.replicates is old.replicates is True
    assert_frame_equal(new.halves[old.halves.columns], old.halves)
    assert new.halves.first_alert_identified.all()


def test_baseline_gaps_use_fixed_window_and_retained_observations():
    data = fixture([[0] * 5 + [1] * 8] * 2, omit=[(0, 1), (0, 3)])
    result = sparse(data, baseline_steps=5)
    half = result.halves.iloc[0]
    assert result.replicates is True
    assert half.baseline_expected == 5 and half.baseline_required == 3
    assert half.baseline_used == 3 and half.baseline_used_steps == [2, 4, 5]
    assert half.baseline_excluded_steps == [1, 3]
    assert half.alert_step == 8 and half.onset_step == 6
    assert half.n_steps == 8 and half.missing_steps == [1, 3]
    assert half.baseline_mean == 0 and half.scale == 1e-9


def test_baseline_never_borrows_postbaseline_observations():
    data = fixture([[0] * 5 + [1] * 8] * 2, omit=[(0, 1), (0, 2), (0, 3)])
    result = sparse(data, baseline_steps=5)
    half = result.halves.iloc[0]
    assert result.status == "unavailable"
    assert half.reason == "insufficient_fixed_baseline"
    assert half.baseline_used == 2 and half.baseline_used_steps == [4, 5]
    assert pd.isna(half.alert_step) and half.n_scored == 0


def test_baseline_unavailable_se_is_excluded_without_zero_fill():
    data = remove_prompts_at(fixture([[0] * 3 + [1] * 8] * 2), 0, 2)
    result = sparse(data)
    half = result.halves.iloc[0]
    assert result.replicates is True
    assert half.baseline_used_steps == [1, 3]
    assert half.baseline_excluded_steps == [2]
    assert half.unavailable_se_steps == [2]
    assert half.missing_steps == []


def test_all_single_unit_half_baselines_remain_unavailable():
    result = sparse(fixture([[0] * 3 + [1] * 8] * 2, prompts_per_half=1, copies=8))
    assert result.status == "unavailable"
    assert result.halves.baseline_used.eq(0).all()
    assert result.halves.n_scored.eq(0).all()


@pytest.mark.parametrize("missing_type", ["absent", "unavailable_se"])
def test_monitoring_gaps_reset_streak_and_preserve_first_alert_ambiguity(missing_type):
    data = fixture(
        [[0] * 3 + [1] * 8] * 2, omit=[(0, 5)] if missing_type == "absent" else []
    )
    if missing_type == "unavailable_se":
        data = remove_prompts_at(data, 0, 5)
    result = sparse(data, tolerance=4)
    half = result.halves.iloc[0]
    assert half.status == "unavailable" and half.reason == "ambiguous_first_alert"
    assert half.first_possible_alert_step == 6
    assert half.observed_onset_step == 6 and half.observed_alert_step == 8
    assert half.observed_direction == "up" and not half.first_alert_identified
    assert pd.isna(half.alert_step) and result.replicates is None


def test_missing_step_cannot_supply_an_actual_alert():
    result = sparse(fixture([[0] * 3 + [1] * 3] * 2, omit=[(0, 5)]), tolerance=0)
    half = result.halves.iloc[0]
    assert half.reason == "ambiguous_first_alert" and pd.isna(half.observed_alert_step)
    assert half.first_possible_alert_step == 6
    assert result.replicates is None


def test_gap_after_identified_first_alert_is_irrelevant():
    result = sparse(fixture([[0] * 3 + [1] * 8] * 2, omit=[(0, 7)]))
    assert result.replicates is True
    assert result.halves.iloc[0].alert_step == 6
    assert result.halves.iloc[0].n_steps == 7
    assert result.halves.iloc[0].first_alert_identified


def test_isolated_gaps_blocked_by_observed_zeros_can_rule_out_alerts():
    result = sparse(fixture([[0] * 3 + [1] * 8, [0] * 11], omit=[(1, 5)]))
    half = result.halves.iloc[1]
    assert half.status == "no_alert" and pd.isna(half.first_possible_alert_step)
    assert result.replicates is False
    assert half.monitoring_expected == 5 and half.monitoring_available == 4
    assert half.n_scored == 4 and half.n_unscorable == 1
    assert half.n_scored + half.n_unscorable == half.monitoring_expected


def test_incomplete_horizon_without_possible_alert_is_pending():
    result = sparse(fixture([[0] * 3 + [1] * 3, [0] * 6], omit=[(1, 4)]))
    assert result.status == "pending"
    assert result.halves.iloc[1].status == "pending"
    assert not result.horizon_complete


def test_known_wrong_direction_is_failure_even_with_other_half_unavailable():
    data = fixture([[0.0] * 3 + [-1.0] * 8, [0.0] * 3 + [6.0] * 8], kind="continuous")
    for step in (1, 2):
        data = remove_prompts_at(data, 1, step)
    result = sparse(data, numerical_floor=0.01)
    assert result.halves.iloc[0].direction == "down"
    assert result.halves.iloc[0].first_alert_identified
    assert result.halves.iloc[1].status == "unavailable"
    assert result.status == "not_replicated"


def test_wrong_observed_direction_after_ambiguous_gap_is_not_known_failure():
    values = [[0.0] * 3 + [-1.0] * 8, [0.0] * 3 + [6.0] * 8]
    result = sparse(
        fixture(values, kind="continuous", omit=[(0, 4)]),
        numerical_floor=0.01,
        tolerance=4,
    )
    half = result.halves.iloc[0]
    assert half.observed_direction == "down"
    assert half.reason == "ambiguous_first_alert"
    assert pd.isna(half.direction)
    assert result.status == "unavailable"


def test_sparse_large_step_ids_preserve_full_grid_positions():
    steps = [2**53 + i * 10 for i in range(11)]
    data = fixture(
        [[0] * 3 + [1] * 8, [0] * 5 + [1] * 6], steps=steps, omit=[(0, steps[1])]
    )
    result = sparse(data)
    assert result.replicates is True
    assert result.halves.alert_step.tolist() == [steps[5], steps[7]]
    assert result.halves.alert_position.tolist() == [5, 7]
    assert result.halves.iloc[0].baseline_excluded_steps == [steps[1]]
    assert all(type(value) is int for value in result.halves.alert_step)


def test_appending_after_cutoff_cannot_change_fixed_evidence():
    data = fixture([[0] * 3 + [1] * 8] * 2, omit=[(0, 2)])
    old = sparse(data)
    rows, features = data
    extra = rows.loc[rows.step.eq(11)].assign(step=12)
    extra["sample_id"] += ":later"
    extra_features = extra[["run", "sample_id"]].assign(flag=0)
    extended = (
        pd.concat([rows, extra], ignore_index=True),
        FeatureTables(
            pd.concat([features.dense, extra_features], ignore_index=True),
            features.sparse,
            features.catalog,
        ),
    )
    new = sparse(extended)
    assert_frame_equal(old.halves, new.halves)
    assert_frame_equal(old.assignments, new.assignments)


def test_canonical_prompts_stay_intact_and_whole_units_do_not_cross_halves():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2)
    mapping = (
        rows[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=lambda d: d.prompt_id)
    )
    # Add a distinct exact variant with the same explicit unit as its source.
    copied = rows.loc[rows.prompt_id.eq(rows.prompt_id.iloc[0])].copy()
    old_prompt = copied.prompt_id.iloc[0]
    copied["prompt_id"] = "variant"
    copied["sample_id"] += ":variant"
    copied["prompt"] = "variant text"
    dense = copied[["run", "sample_id"]].assign(flag=copied.reward.to_numpy())
    rows = pd.concat([rows, copied], ignore_index=True)
    mapping = pd.concat(
        [
            mapping,
            pd.DataFrame(
                [{"run": "r", "prompt_id": "variant", "sampling_unit_id": old_prompt}]
            ),
        ],
        ignore_index=True,
    )
    before = rows.copy(deep=True)
    result = split_half_replication(
        rows,
        FeatureTables(
            pd.concat([features.dense, dense], ignore_index=True),
            features.sparse,
            features.catalog,
        ),
        run="r",
        feature="flag",
        baseline_steps=3,
        coverage_policy="fixed_window",
        sampling_units=mapping,
    )
    assert result.assignments.groupby("sampling_unit_id").half.nunique().eq(1).all()
    assert set(result.assignments.prompt_id) == set(mapping.prompt_id)
    assert_frame_equal(before, rows)


@pytest.mark.parametrize("policy", ["other", None, 0])
def test_invalid_policy_fails_even_without_full_alert(policy):
    with pytest.raises(ValueError, match="coverage_policy"):
        replicate(fixture([[0] * 10] * 2), coverage_policy=policy)


def test_observed_monitoring_can_resume_after_gap_that_cannot_hide_earlier_alert():
    values = [[0] * 6 + [1] * 5, [0] * 3 + [1] * 8]
    result = sparse(fixture(values, omit=[(0, 5)]), tolerance=4)
    half = result.halves.iloc[0]
    assert result.full.alerts.iloc[0].alert_step == 6
    assert (
        half.first_possible_alert_step
        == half.observed_alert_step
        == half.alert_step
        == 9
    )
    assert half.onset_step == 7 and half.first_alert_identified
    assert result.replicates is True


def test_fixed_window_no_full_alert_has_explicit_policy_and_coverage_schema():
    result = sparse(fixture([[0] * 10] * 2))
    assert result.coverage_policy == "fixed_window"
    assert result.status == "no_full_alert" and result.halves.empty
    assert "baseline_used" in result.halves


def test_prepared_fixed_window_resolves_omitted_detector_defaults():
    from rolloutdiff.detect import detect_sequential
    from rolloutdiff.replication import _replication_prepared

    steps = list(range(1, 26))
    series = pd.DataFrame(
        {
            "run": "r",
            "step": steps,
            "feature": "flag",
            "tier": "structural",
            "kind": "binary",
            "mean": [0.0] * 15 + [1.0] * 10,
            "se": 0.0,
            "n_rollouts": 2,
            "n_prompts": 2,
            "n_sampling_units": 2,
        }
    )
    schedule = pd.DataFrame(
        [
            {
                "step": step,
                "prompt_id": prompt,
                "sampling_unit_id": prompt,
                "n_rollouts": 1,
            }
            for step in steps
            for prompt in ("a", "b")
        ]
    )
    result = _replication_prepared(
        detect_sequential(series),
        {0: series, 1: series},
        {0: schedule, 1: schedule},
        pd.DataFrame(),
        run="r",
        feature="flag",
        tolerance=2,
        parameters={},
        coverage_policy="fixed_window",
    )
    assert result.replicates is True
    assert result.halves.baseline_expected.eq(15).all()
    assert result.halves.baseline_required.eq(8).all()
    assert result.halves.alert_step.eq(18).all()


def test_failed_baseline_counts_every_full_grid_monitoring_opportunity_as_unscorable():
    data = fixture([[0] * 5 + [1] * 8] * 2, omit=[(0, 1), (0, 2), (0, 3), (0, 6)])
    half = sparse(data, baseline_steps=5).halves.iloc[0]
    assert half.reason == "insufficient_fixed_baseline"
    assert half.monitoring_expected == 5 and half.monitoring_available == 4
    assert half.n_scored == 0 and half.n_unscorable == 5
    assert half.n_scored + half.n_unscorable == half.monitoring_expected
