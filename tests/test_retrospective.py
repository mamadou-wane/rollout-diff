import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_detect import series


def detect(frame, **kwargs):
    from rolloutdiff.retrospective import detect_retrospective

    return detect_retrospective(frame, **kwargs)


@pytest.mark.parametrize("before,after", [(0, 0.5), (0.8, 0.2)])
def test_exact_binary_shift_can_rise_or_fall_without_a_numerical_floor(before, after):
    result = detect(series([before] * 10 + [after] * 10, se=0, kind="binary"))
    cp = result.changepoints.iloc[0]
    assert (cp.previous_step, cp.changepoint_step, cp.n_before, cp.n_after) == (
        10,
        11,
        10,
        10,
    )
    assert cp.mean_before == pytest.approx(before)
    assert cp.mean_after == pytest.approx(after)
    assert cp.delta == pytest.approx(after - before)
    assert cp.direction == ("up" if after > before else "down")
    diag = result.diagnostics.iloc[0]
    assert diag.variance_scaled == diag.penalty_scaled == 0
    assert diag.gain_scaled > 0
    assert (
        "alert_step" not in result.changepoints
        and "onset_step" not in result.changepoints
    )


def test_retrospective_shift_need_not_exceed_four_within_segment_sds():
    values = [-0.1, 0.1] * 10 + [0.1, 0.3] * 10
    result = detect(series(values, se=0.05))
    assert len(result.changepoints) == 1
    # Alternating noise moves the earliest SSE optimum from planted k=20 to k=19:
    # left sum=-.1 and right sum=4.1.
    cp = result.changepoints.iloc[0]
    assert (cp.n_before, cp.n_after) == (19, 21)
    assert cp.delta == pytest.approx(0.1 / 19 + 4.1 / 21)
    diag = result.diagnostics.iloc[0]
    assert (
        cp.delta < 4 * np.sqrt(diag.residual_variance_scaled) * diag.normalization_scale
    )
    assert diag.gain_scaled > diag.penalty_scaled
    assert diag.variance_source == "residual"


@pytest.mark.parametrize("value", [0, 0.75])
def test_constant_series_selects_no_change(value):
    result = detect(series([value] * 40, se=0))
    assert result.changepoints.empty
    assert result.diagnostics.iloc[0].status == "no_change"
    assert result.diagnostics.iloc[0].reason == "constant"
    assert result.screening.iloc[0].n_screened == 1


def test_stationary_oscillation_does_not_force_a_split():
    result = detect(series([-1, 1] * 20, se=0.1))
    assert result.changepoints.empty
    diag = result.diagnostics.iloc[0]
    assert diag.status == "no_change"
    assert diag.reason == "penalty"
    assert diag.gain_scaled <= diag.penalty_scaled


def test_large_measured_se_can_reject_an_otherwise_exact_shift():
    result = detect(series([0] * 20 + [0.2] * 20, se=1, kind="binary"))
    assert result.changepoints.empty
    diag = result.diagnostics.iloc[0]
    assert diag.reason == "penalty"
    assert diag.variance_source == "across_prompt_se"
    assert diag.rms_se_before == diag.rms_se_after == 1


def test_binary_practical_effect_floor_applies_after_fitting_and_is_configurable():
    data = series([0] * 20 + [0.05] * 20, se=0, kind="binary")
    result = detect(data)
    assert result.changepoints.empty
    assert result.diagnostics.iloc[0].candidate_step == 21
    assert result.diagnostics.iloc[0].reason == "below_effect"
    assert len(detect(data, min_prevalence_change=0.01).changepoints) == 1


@pytest.mark.parametrize("before,after", [(0.25, 0.35), (0.5, 0.4)])
def test_inclusive_prevalence_threshold_tolerates_roundoff_only(before, after):
    assert (
        len(
            detect(
                series([before] * 20 + [after] * 20, se=0, kind="binary")
            ).changepoints
        )
        == 1
    )
    closer = after - np.sign(after - before) * 1e-10
    assert detect(
        series([before] * 20 + [closer] * 20, se=0, kind="binary")
    ).changepoints.empty


def test_continuous_domain_effect_threshold_is_explicit():
    data = series([0] * 20 + [0.05] * 20, se=0.01)
    assert len(detect(data).changepoints) == 1
    rejected = detect(data, min_continuous_change=0.1)
    assert rejected.changepoints.empty
    assert rejected.diagnostics.iloc[0].reason == "below_effect"


@pytest.mark.parametrize("k", [5, 15])
def test_minimum_segment_length_includes_both_valid_edges(k):
    result = detect(series([0] * k + [1] * (20 - k), se=0.01))
    cp = result.changepoints.iloc[0]
    assert cp.changepoint_step == k + 1
    assert min(cp.n_before, cp.n_after) == 5
    assert result.diagnostics.iloc[0].n_candidates == 11


def test_insufficient_length_is_distinct_from_no_change():
    result = detect(series([0] * 9, se=0.1))
    assert result.changepoints.empty
    diag = result.diagnostics.iloc[0]
    assert diag.status == "insufficient_data" and diag.n_candidates == 0
    assert result.screening.iloc[0].n_unscorable == 1


def test_missing_uncertainty_does_not_get_filled_or_deleted():
    data = series([0] * 10 + [1] * 10, se=[0.1] * 7 + [np.nan] + [0.1] * 12)
    saved = data.copy(deep=True)
    result = detect(data)
    assert result.changepoints.empty
    assert result.diagnostics.iloc[0].status == "unavailable_uncertainty"
    assert result.diagnostics.iloc[0].n_missing_se == 1
    assert result.diagnostics.iloc[0].n_observations == 20
    assert_frame_equal(data, saved)


def test_sparse_steps_use_first_right_observation_and_preserve_prior_boundary():
    result = detect(series([0] * 10 + [1] * 10, se=0.01, steps=list(range(0, 200, 10))))
    cp = result.changepoints.iloc[0]
    assert (cp.previous_step, cp.changepoint_step) == (90, 100)


def test_candidate_step_identity_is_not_coerced_to_float_by_missing_candidates():
    start = 2**53 + 1
    data = pd.concat(
        [
            series([0] * 10 + [1] * 10, se=0.1, steps=list(range(start, start + 20))),
            series([0], se=0, run="short"),
        ]
    )
    result = detect(data)
    diag = result.diagnostics.set_index("run").loc["r"]
    assert diag.candidate_step == start + 10
    assert diag.previous_step == start + 9


@pytest.mark.parametrize("factor", [1e-150, 1e150, -7])
def test_continuous_units_rescaling_preserves_decision(factor):
    values = np.array([-0.1, 0.1] * 10 + [0.4, 0.6] * 10)
    base = detect(series(values, se=0.1))
    scaled = detect(series(values * factor, se=0.1 * abs(factor)))
    assert (
        scaled.changepoints.iloc[0].changepoint_step
        == base.changepoints.iloc[0].changepoint_step
    )
    assert scaled.changepoints.iloc[0].delta == pytest.approx(
        base.changepoints.iloc[0].delta * factor
    )
    assert scaled.diagnostics.iloc[0].gain_scaled == pytest.approx(
        base.diagnostics.iloc[0].gain_scaled
    )
    assert scaled.diagnostics.iloc[0].penalty_scaled == pytest.approx(
        base.diagnostics.iloc[0].penalty_scaled
    )
    assert np.isfinite(scaled.diagnostics.iloc[0].gain_scaled)


def test_large_translation_does_not_destroy_small_signal_through_cancellation():
    result = detect(series(np.array([0.0] * 10 + [1.0] * 10) + 1e12, se=0.1))
    assert result.changepoints.iloc[0].changepoint_step == 11
    assert result.changepoints.iloc[0].delta == 1


def test_equal_cost_candidate_ties_choose_earliest_boundary():
    result = detect(series([0] * 5 + [1] * 5 + [0] * 5, se=0))
    assert result.diagnostics.iloc[0].candidate_step == 6


def test_isolated_interior_spike_does_not_imply_a_persistent_shift():
    values = [0.0] * 100
    values[50] = 10.0
    result = detect(series(values, se=0.1))
    assert result.changepoints.empty


def test_best_split_matches_independent_exhaustive_squared_error_reference():
    rng = np.random.default_rng(217)
    for _ in range(12):
        values = rng.normal(size=30)
        costs = [
            ((values[:k] - values[:k].mean()) ** 2).sum()
            + ((values[k:] - values[k:].mean()) ** 2).sum()
            for k in range(5, 26)
        ]
        k = 5 + int(np.argmin(costs))
        split_cost = min(costs)
        null_cost = ((values - values.mean()) ** 2).sum()
        penalty = 3 * np.log(30) * max(split_cost / 28, 0.2**2)
        result = detect(series(values, se=0.2), dependence="independent")
        diag = result.diagnostics.iloc[0]
        assert diag.candidate_step == k + 1
        assert diag.sse_split_scaled * diag.normalization_scale**2 == pytest.approx(
            split_cost
        )
        assert diag.gain_scaled * diag.normalization_scale**2 == pytest.approx(
            null_cost - split_cost
        )
        assert diag.penalty_scaled * diag.normalization_scale**2 == pytest.approx(
            penalty
        )
        assert len(result.changepoints) == int(null_cost - split_cost > penalty)


def test_run_feature_isolation_sorting_and_screening_counts():
    data = pd.concat(
        [
            series([0] * 10 + [1] * 10, se=0.1),
            series([0] * 20, feature="stable", se=0.1),
            series([0] * 9, run="short", se=0.1),
        ]
    )
    expected = detect(data)
    shuffled = data.sample(frac=1, random_state=5)
    shuffled.index = [17] * len(data)
    actual = detect(shuffled)
    assert_frame_equal(expected.changepoints, actual.changepoints)
    assert_frame_equal(expected.diagnostics, actual.diagnostics)
    counts = actual.screening.set_index("run")
    assert counts.loc[
        "r", ["n_series", "n_screened", "n_changes", "n_unscorable"]
    ].tolist() == [2, 2, 1, 0]
    assert counts.loc[
        "short", ["n_series", "n_screened", "n_changes", "n_unscorable"]
    ].tolist() == [1, 0, 0, 1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_segment_steps": 1},
        {"min_segment_steps": True},
        {"min_segment_steps": 2.5},
        {"penalty_multiplier": 0},
        {"penalty_multiplier": np.inf},
        {"min_prevalence_change": -0.1},
        {"min_prevalence_change": 1.1},
        {"min_continuous_change": -1},
        {"min_continuous_change": np.nan},
    ],
)
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        detect(series([0] * 20), **kwargs)


def test_input_validation_matches_existing_series_contract():
    for frame in [
        series([0] * 20).assign(se=-1),
        series([0] * 20).assign(n_prompts=1),
        pd.concat([series([0] * 20), series([0] * 20)]),
        series([0] * 20).assign(mean=np.inf),
    ]:
        with pytest.raises(ValueError):
            detect(frame)


def test_extreme_centering_overflow_is_explicit():
    with pytest.raises(ValueError, match="overflow"):
        detect(series([1e308] * 10 + [-1e308] * 10, se=0.1))


def test_empty_input_has_stable_output_columns():
    result = detect(series([]))
    assert (
        result.changepoints.empty
        and result.diagnostics.empty
        and result.screening.empty
    )
    assert {"run", "feature", "changepoint_step"} <= set(result.changepoints)


def test_appending_data_can_move_a_retrospective_boundary():
    short = detect(series([0.0] * 10 + [1.0] * 10, se=0.01))
    full = detect(series([0.0] * 10 + [1.0] * 10 + [4.0] * 20, se=0.01))
    assert short.changepoints.iloc[0].changepoint_step == 11
    assert full.changepoints.iloc[0].changepoint_step == 21


def test_unrepresentable_nonzero_effect_cannot_become_a_zero_downward_change():
    tiny = np.nextafter(0.0, 1.0)
    with pytest.raises(ValueError, match="underflow"):
        detect(series([0.0] * 100 + [tiny, 0.0] * 50, se=0))


@pytest.mark.parametrize("se,expected_changes", [(0.0, 1), (1e-100, 0)])
def test_finite_penalty_with_large_multiplier_avoids_intermediate_overflow(
    se, expected_changes
):
    result = detect(series([0.0] * 10 + [1.0] * 10, se=se), penalty_multiplier=1e308)
    assert len(result.changepoints) == expected_changes
    assert np.isfinite(result.diagnostics.iloc[0].penalty_scaled)
