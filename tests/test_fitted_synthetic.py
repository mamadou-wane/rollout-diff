"""Baseline-fitted simulation contracts."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rolloutdiff.features import FeatureTables


def fixture(*, n_steps=16, n_prompts=8, n_rollouts=8, constant=False):
    rows, dense = [], []
    for step in range(1, n_steps + 1):
        common = 0.7 * np.sin(step / 3)
        for prompt in range(n_prompts):
            for copy in range(n_rollouts):
                value = (
                    3.0 if constant else common + (prompt - 3.5) / 4 + (copy - 3.5) / 8
                )
                sample = f"{step}:{prompt}:{copy}"
                rows.append(
                    {
                        "run": "r",
                        "step": step,
                        "sample_id": sample,
                        "prompt_id": f"p{prompt}",
                        "prompt": f"p{prompt}",
                        "response": "",
                        "reward": float(value),
                    }
                )
                dense.append(
                    {
                        "run": "r",
                        "sample_id": sample,
                        "flag": int(copy < 1 + prompt % 5),
                        "absent": 0,
                        "value": value,
                    }
                )
    return pd.DataFrame(rows), FeatureTables(
        pd.DataFrame(dense),
        pd.DataFrame(columns=["run", "sample_id", "feature"]),
        pd.DataFrame(
            [
                {"feature": name, "kind": kind, "tier": "structural"}
                for name, kind in [
                    ("flag", "binary"),
                    ("absent", "binary"),
                    ("value", "continuous"),
                ]
            ]
        ),
    )


def fit(data=None, **options):
    from bench.synthetic import fit_baseline

    return fit_baseline(*(data or fixture()), run="r", window=(1, 15), **options)


def test_fit_uses_only_explicit_baseline_and_is_order_independent():
    rows, features = fixture()
    first = fit((rows, features))
    changed = rows.copy()
    changed.loc[changed.step == 16, "reward"] = 1e12
    dense = features.dense.copy()
    dense.loc[dense.sample_id.str.startswith("16:"), "value"] = -1e12
    second = fit(
        (
            changed.sample(frac=1, random_state=2),
            FeatureTables(
                dense.sample(frac=1, random_state=3), features.sparse, features.catalog
            ),
        )
    )
    assert_frame_equal(first, second)
    assert set(first.feature) == {"flag", "absent", "value", "scalar:reward"}
    assert first.n_steps.eq(15).all()
    assert first.n_rollouts.eq(15 * 8 * 8).all()
    assert all(row == tuple(range(1, 16)) for row in first.baseline_steps)
    assert all(row == ((8,) * 8,) * 15 for row in first.group_size_schedule)


def test_fitted_variances_match_known_prompt_and_within_prompt_moments():
    result = fit().set_index("feature").loc["value"]
    within = np.var((np.arange(8) - 3.5) / 8, ddof=1)
    prompt_observed = np.var((np.arange(8) - 3.5) / 4, ddof=1)
    common = 0.7 * np.sin(np.arange(1, 16) / 3)
    assert result.within_variance == pytest.approx(within)
    assert result.prompt_variance == pytest.approx(prompt_observed - within / 8)
    assert result.step_sampling_variance == pytest.approx(prompt_observed / 8)
    assert result.temporal_variance == pytest.approx(
        max(0, common.var(ddof=1) - prompt_observed / 8)
    )
    assert result["mean"] == pytest.approx(common.mean())
    assert result.n_distinct_prompts == 8
    assert "independent_prompt_groups_per_step" in result.assumptions


def test_degenerate_binary_has_no_invented_heterogeneity_fit():
    result = fit().set_index("feature").loc["absent"]
    assert result.fit_status == "degenerate_binary"
    assert result["mean"] == 0
    assert np.isnan(result.binary_rho)
    assert result.temporal_variance == 0


@pytest.mark.parametrize(
    "options, reason",
    [
        ({"n_steps": 5}, "too_few_baseline_steps"),
        ({"n_prompts": 1}, "insufficient_prompt_groups"),
        ({"n_rollouts": 1}, "insufficient_within_prompt_replication"),
    ],
)
def test_sparse_baselines_are_explicitly_unavailable(options, reason):
    result = fit(fixture(**options))
    assert result.fit_status.eq("unavailable").all()
    assert result.reason.eq(reason).all()


def test_unknown_feature_and_invalid_window_are_rejected():
    from bench.synthetic import fit_baseline

    with pytest.raises(ValueError, match="unknown"):
        fit(feature_names=["missing"])
    for window in [(4, 1), (-1, 15), (True, 15), (1.5, 15)]:
        with pytest.raises(ValueError, match="window"):
            fit_baseline(*fixture(), run="r", window=window)


def test_generated_cases_reproduce_seed_schedule_and_prompt_se():
    from bench.synthetic import generate_fitted

    fits = fit(feature_names=["flag", "value"])
    first = generate_fitted(fits, seed=123, n_replicates=2, n_steps=20, onset=10)
    second = generate_fitted(fits, seed=123, n_replicates=2, n_steps=20, onset=10)
    assert_frame_equal(first.series, second.series)
    assert_frame_equal(first.trials, second.trials)
    assert len(first.trials) == 4 and len(first.series) == 80
    assert first.trials.status.eq("generated").all()
    assert first.series.n_prompts.eq(8).all()
    assert first.series.n_rollouts.eq(64).all()
    assert first.series.se.notna().all()
    assert first.series.loc[first.series.kind.eq("binary"), "mean"].between(0, 1).all()
    assert first.settings["seed"] == 123 and first.settings["planted"] is False


def test_degenerate_binary_null_is_constant_and_plant_is_unavailable():
    from bench.synthetic import generate_fitted

    fits = fit(feature_names=["absent"])
    null = generate_fitted(fits, seed=12, n_replicates=2)
    planted = generate_fitted(fits, seed=13, n_replicates=2, planted=True)
    assert null.series["mean"].eq(0).all() and null.series.se.eq(0).all()
    assert planted.series.empty
    assert planted.trials.status.eq("unavailable").all()
    assert planted.trials.reason.eq("degenerate_binary_plant").all()


def test_out_of_range_binary_effect_and_degenerate_continuous_plant_are_unavailable():
    from bench.synthetic import generate_fitted

    binary = generate_fitted(
        fit(feature_names=["flag"]),
        seed=1,
        planted=True,
        binary_effect=0.9,
        n_replicates=1,
    )
    continuous = generate_fitted(
        fit(fixture(constant=True), feature_names=["value"]),
        seed=1,
        planted=True,
        n_replicates=1,
    )
    assert binary.trials.reason.tolist() == ["binary_target_out_of_range"]
    assert continuous.trials.reason.tolist() == ["zero_continuous_effect"]
    assert binary.series.empty and continuous.series.empty


def test_positive_effect_parameters_and_horizon_are_validated():
    from bench.synthetic import generate_fitted

    fits = fit(feature_names=["value"])
    for kwargs in [
        {"seed": -1},
        {"n_replicates": 0},
        {"n_steps": 4, "onset": 4},
        {"binary_effect": -0.2},
        {"continuous_effect_sd": 0},
        {"planted": 1},
    ]:
        with pytest.raises(ValueError):
            generate_fitted(fits, **({"seed": 1} | kwargs))


def test_both_detectors_evaluate_null_coverage_and_unavailable_plants():
    from bench.synthetic import evaluate_fitted, generate_fitted

    fits = fit(feature_names=["absent"])
    null = evaluate_fitted(generate_fitted(fits, seed=12, n_replicates=3))
    planted = evaluate_fitted(
        generate_fitted(fits, seed=13, n_replicates=3, planted=True)
    )
    assert set(null.summary.detector) == {"sequential", "retrospective"}
    assert null.summary.n_trials.eq(3).all()
    assert null.summary.n_scorable.eq(3).all()
    assert null.summary.n_flagged.eq(0).all()
    assert null.summary.false_alert_rate_all.eq(0).all()
    assert planted.summary.n_trials.eq(3).all()
    assert planted.summary.n_generated.eq(0).all()
    assert planted.summary.false_alert_rate_all.isna().all()
    assert planted.summary.detection_rate.isna().all()


def test_timing_and_early_alerts_are_different_from_retrospective_boundaries():
    from bench.synthetic import SyntheticBatch, evaluate_fitted

    rows = []
    trials = []
    for name, change_at in [
        ("matched", 40),
        ("early", 20),
        ("wrong", 40),
        ("missed", 99),
    ]:
        for step in range(80):
            rows.append(
                {
                    "run": name,
                    "step": step,
                    "feature": "value",
                    "tier": "structural",
                    "kind": "continuous",
                    "mean": float((step >= change_at) * (-1 if name == "wrong" else 1)),
                    "se": 0.01,
                    "n_rollouts": 16,
                    "n_prompts": 4,
                }
            )
        trials.append(
            {
                "run": name,
                "source_run": "r",
                "feature": "value",
                "kind": "continuous",
                "tier": "structural",
                "replicate": 0,
                "status": "generated",
                "reason": "",
                "onset": 40,
                "effect": 1.0,
                "planted": True,
            }
        )
    result = evaluate_fitted(
        SyntheticBatch(pd.DataFrame(rows), pd.DataFrame(trials), {})
    )
    sequential = result.trials.loc[result.trials.detector.eq("sequential")].set_index(
        "run"
    )
    retrospective = result.trials.loc[
        result.trials.detector.eq("retrospective")
    ].set_index("run")
    assert sequential.loc["matched", "alert_delay"] == 2
    assert retrospective.loc["matched", "boundary_error"] == 0
    assert retrospective.loc["early", "boundary_error"] == -20
    assert sequential.loc["early", "outcome"] == "early_alert"
    assert sequential.loc["wrong", "outcome"] == "wrong_direction"
    assert sequential.loc["missed", "outcome"] == "missed"
    assert retrospective.loc["early", "detected"]
    assert not retrospective.loc["early", "localized"]
    assert np.isnan(sequential.loc["early", "alert_delay"])
    summary = result.summary.set_index("detector")
    assert summary.loc["sequential", "detection_rate"] == 0.25
    assert summary.loc["sequential", "n_early"] == 1
    assert summary.loc["retrospective", "localization_rate"] == 0.25


def test_unavailable_detector_uncertainty_is_not_a_no_change_result():
    from bench.synthetic import evaluate_fitted, generate_fitted

    generated = generate_fitted(fit(feature_names=["flag"]), seed=3, n_replicates=2)
    generated.series["se"] = np.nan
    generated.series["n_prompts"] = 1
    result = evaluate_fitted(generated)
    assert result.trials.outcome.eq("unscorable").all()
    assert result.summary.n_scorable.eq(0).all()
    assert result.summary.n_unscorable.eq(2).all()
    assert result.summary.false_alert_rate_scorable.isna().all()


def test_gaussian_generated_moments_and_lag_covariance_match_fitted_model():
    from bench.synthetic import generate_fitted

    fits = fit(feature_names=["value"])
    profile = fits.iloc[0]
    batch = generate_fitted(fits, seed=901, n_replicates=128)
    grouped = batch.series.groupby("run", sort=True)
    expected_noise = (profile.prompt_variance + profile.within_variance / 8) / 8
    measurements = []
    for _, series in grouped:
        centered = series["mean"].to_numpy() - profile["mean"]
        measurements.append(
            [
                centered.mean(),
                np.mean(centered**2),
                np.mean(centered[:-1] * centered[1:]),
                np.mean(series.se**2),
            ]
        )
    measured = np.array(measurements)
    expected = np.array(
        [
            0,
            profile.temporal_variance + expected_noise,
            profile.ar1 * profile.temporal_variance,
            expected_noise,
        ]
    )
    # Independent trials supply uncertainty for this fixed-seed simulator check.
    assert (
        np.abs(measured.mean(axis=0) - expected)
        <= 6 * measured.std(axis=0, ddof=1) / np.sqrt(len(measured))
    ).all()


def test_beta_binomial_generation_recovers_fitted_mean_and_prompt_variance():
    from bench.synthetic import generate_fitted

    fits = fit(feature_names=["flag"])
    profile = fits.iloc[0]
    assert profile.temporal_variance == 0
    batch = generate_fitted(fits, seed=902, n_replicates=128)
    variance = (
        profile["mean"]
        * (1 - profile["mean"])
        * (profile.binary_rho + (1 - profile.binary_rho) / 8)
        / 8
    )
    measured = np.array(
        [
            [group["mean"].mean(), np.mean(group.se**2)]
            for _, group in batch.series.groupby("run", sort=True)
        ]
    )
    assert (
        np.abs(measured.mean(axis=0) - [profile["mean"], variance])
        <= 6 * measured.std(axis=0, ddof=1) / np.sqrt(len(measured))
    ).all()


def test_generation_stream_is_stable_when_other_profiles_are_removed():
    from bench.synthetic import generate_fitted

    profiles = fit(feature_names=["flag", "value"])
    all_profiles = generate_fitted(profiles, seed=50, n_replicates=2)
    selected = generate_fitted(
        profiles.loc[profiles.feature.eq("value")], seed=50, n_replicates=2
    )
    assert_frame_equal(
        all_profiles.series.loc[all_profiles.series.feature.eq("value")].reset_index(
            drop=True
        ),
        selected.series,
    )
    assert_frame_equal(
        all_profiles.trials.loc[all_profiles.trials.feature.eq("value")].reset_index(
            drop=True
        ),
        selected.trials,
    )


def test_probability_projection_is_counted_for_every_generated_trial():
    from bench.synthetic import generate_fitted

    profiles = fit(feature_names=["flag"])
    # High temporal variance exercises probability clipping.
    profiles["temporal_variance"] = 100.0
    profiles["ar1"] = 0.0
    batch = generate_fitted(profiles, seed=11, n_replicates=2)
    assert batch.trials.n_probability_clipped.gt(0).all()
    assert np.allclose(
        batch.trials.probability_clipped_fraction,
        batch.trials.n_probability_clipped / 80,
    )


def test_ar_projection_and_variance_truncation_remain_visible():
    profiles = fit().set_index("feature")
    assert profiles.loc["value", "ar1_projected"]
    assert profiles.loc["value", "ar1"] == 0.95
    assert profiles.loc["value", "ar1_raw"] > 0.95
    assert profiles.loc["flag", "temporal_variance_truncated"]
    assert profiles.loc["flag", "temporal_variance_raw"] < 0
    assert np.isnan(profiles.loc["flag", "ar1_raw"])


def test_generation_ids_do_not_collide_when_source_ids_contain_separators():
    from bench.synthetic import generate_fitted

    first = fit(feature_names=["value"])
    second = first.copy()
    first["source_run"], first["feature"] = "a:b", "c"
    second["source_run"], second["feature"] = "a", "b:c"
    batch = generate_fitted(
        pd.concat([first, second], ignore_index=True), seed=1, n_replicates=1
    )
    assert batch.trials.run.is_unique
    assert batch.series.groupby("run").size().eq(80).all()


def test_empty_profile_selection_has_a_clear_error():
    from bench.synthetic import generate_fitted

    with pytest.raises(ValueError, match="at least one"):
        fit(feature_names=[])
    profiles = fit(feature_names=["value"])
    with pytest.raises(ValueError, match="at least one"):
        generate_fitted(profiles.iloc[:0], seed=1)


def test_unanimous_binary_steps_with_different_levels_have_unidentified_heterogeneity():
    from bench.synthetic import generate_fitted

    rows, features = fixture()
    dense = features.dense.copy()
    dense["flag"] = dense.sample_id.str.split(":").str[0].astype(int).ge(8).astype(int)
    profiles = fit(
        (rows, FeatureTables(dense, features.sparse, features.catalog)),
        feature_names=["flag"],
    )
    profile = profiles.iloc[0]
    assert profile.fit_status == "unavailable"
    assert profile.reason == "binary_heterogeneity_unidentified"
    assert profile["mean"] == pytest.approx(8 / 15)
    assert profile.temporal_variance > 0
    assert profile.observed_prompt_variance == 0
    assert profile.within_variance == 0
    assert np.isnan(profile.binary_rho)
    assert np.isnan(profile.binary_rho_raw)
    generated = generate_fitted(profiles, seed=1, n_replicates=1)
    assert generated.series.empty
    assert generated.trials.reason.tolist() == ["binary_heterogeneity_unidentified"]


def test_nonfinite_fitted_moments_are_unavailable_not_generated():
    from bench.synthetic import generate_fitted

    rows, features = fixture()
    dense = features.dense.copy()
    dense["value"] *= 1e200
    profiles = fit(
        (rows, FeatureTables(dense, features.sparse, features.catalog)),
        feature_names=["value"],
    )
    profile = profiles.iloc[0]
    assert profile.fit_status == "unavailable"
    assert profile.reason == "nonfinite_baseline_moments"
    generated = generate_fitted(profiles, seed=1, n_replicates=1)
    assert generated.series.empty
    assert generated.trials.reason.tolist() == ["nonfinite_baseline_moments"]


def test_roundoff_temporal_variance_is_unidentified_not_projected():
    # One positive rollout in the whole baseline: the step variance equals the
    # sampling variance algebraically, so the remainder is roundoff.
    rows, features = fixture(n_prompts=16, n_rollouts=16)
    dense = features.dense.copy()
    dense["flag"] = 0.0
    dense.loc[dense.sample_id.eq("3:2:1"), "flag"] = 1.0
    profile = fit(
        (rows, FeatureTables(dense, features.sparse, features.catalog)),
        feature_names=["flag"],
    ).iloc[0]
    assert profile.fit_status == "fit"
    assert profile.temporal_variance == 0
    assert profile.ar1 == 0 and np.isnan(profile.ar1_raw)
    assert not profile.ar1_projected


def test_sequential_streak_starting_before_plant_is_early_not_detected():
    from bench.synthetic import SyntheticBatch, evaluate_fitted

    rows = [
        {
            "run": "early",
            "step": step,
            "feature": "value",
            "tier": "structural",
            "kind": "continuous",
            "mean": float(step >= 39),
            "se": 0.01,
            "n_rollouts": 16,
            "n_prompts": 4,
        }
        for step in range(80)
    ]
    trials = pd.DataFrame(
        [
            {
                "run": "early",
                "source_run": "r",
                "feature": "value",
                "tier": "structural",
                "kind": "continuous",
                "replicate": 0,
                "seed": 1,
                "planted": True,
                "onset": 40,
                "effect": 1.0,
                "status": "generated",
                "reason": "",
                "n_probability_clipped": 0,
                "probability_clipped_fraction": np.nan,
            }
        ]
    )
    evaluation = evaluate_fitted(SyntheticBatch(pd.DataFrame(rows), trials, {}))
    sequential = evaluation.trials.loc[
        evaluation.trials.detector.eq("sequential")
    ].iloc[0]
    # The alert at 41 completes a streak that began at 39, before the plant.
    assert sequential.event_step == 41
    assert sequential.outcome == "early_alert"
    assert not sequential.detected and not sequential.timely


def units(assignment):
    return pd.DataFrame(
        {
            "run": "r",
            "prompt_id": [f"p{i}" for i in range(8)],
            "sampling_unit_id": assignment,
        }
    )


EQUAL_UNITS = ["u0"] * 4 + ["u1"] * 4
UNEQUAL_UNITS = ["u0"] * 3 + ["u1"] * 5


def test_singleton_unit_map_reproduces_exact_prompt_fit():
    identity = fit(sampling_units=units([f"p{i}" for i in range(8)]))
    assert_frame_equal(identity, fit())
    assert identity.group_definition.eq("exact_prompt").all()
    assert identity.n_sampling_units.eq(8).all()
    assert identity.n_sampling_unit_groups.eq(identity.n_prompt_groups).all()


@pytest.mark.parametrize("assignment", [EQUAL_UNITS, UNEQUAL_UNITS])
def test_unit_fit_groups_by_unit_and_uses_cluster_se_of_prompt_mean(assignment):
    from rolloutdiff._sampling import cluster_standard_error

    rows, features = fixture()
    mapping = units(assignment)
    result = fit(sampling_units=mapping).set_index("feature").loc["value"]
    baseline = (
        rows.loc[rows.step.le(15)]
        .merge(features.dense, on=["run", "sample_id"])
        .merge(mapping, on=["run", "prompt_id"])
    )
    prompt = baseline.groupby(["step", "prompt_id"]).agg(
        value=("value", "mean"), unit=("sampling_unit_id", "first")
    )
    by_step = prompt.groupby(level="step")
    errors = [
        cluster_standard_error(step.value.to_numpy(), step.unit.to_numpy()) ** 2
        for _, step in by_step
    ]
    unit = baseline.groupby(["step", "sampling_unit_id"]).value
    sizes = (assignment.count("u0") * 8, assignment.count("u1") * 8)
    assert result.group_definition == "sampling_unit"
    assert (result.n_sampling_units, result.n_sampling_unit_groups) == (2, 30)
    assert (result.n_distinct_prompts, result.n_prompt_groups) == (8, 120)
    assert result.group_size_schedule == (sizes,) * 15
    assert result["mean"] == pytest.approx(by_step.value.mean().mean())
    assert result.step_sampling_variance == pytest.approx(np.mean(errors))
    assert result.temporal_variance_raw == pytest.approx(
        by_step.value.mean().var(ddof=1) - np.mean(errors)
    )
    assert result.within_variance == pytest.approx(
        (unit.var() * (unit.size() - 1)).sum() / (unit.size() - 1).sum()
    )
    assert result.observed_prompt_variance == pytest.approx(
        unit.mean().groupby(level="step").var(ddof=1).mean()
    )
    assert "equal_weight_sampling_unit_generation" in result.assumptions
    assert "independent_prompt_groups_per_step" not in result.assumptions


def test_unit_generation_counts_units_with_finite_se():
    from bench.synthetic import evaluate_fitted, generate_fitted

    fits = fit(feature_names=["flag", "value"], sampling_units=units(UNEQUAL_UNITS))
    batch = generate_fitted(fits, seed=123, n_replicates=2, n_steps=20, onset=10)
    assert batch.trials.status.eq("generated").all()
    assert batch.series.n_prompts.eq(2).all()
    assert batch.series.n_rollouts.eq(64).all()
    assert "n_sampling_units" not in batch.series
    assert np.isfinite(batch.series.se).all()
    assert "equal_weight_sampling_unit_generation" in batch.settings["assumptions"]
    assert evaluate_fitted(batch).trials.scorable.all()


def test_unit_fits_without_two_units_or_replicated_units_are_unavailable():
    single = fit(sampling_units=units(["u0"] * 8))
    assert single.fit_status.eq("unavailable").all()
    assert single.reason.eq("insufficient_prompt_groups").all()
    assert single.group_definition.eq("sampling_unit").all()
    sparse = fixture(n_rollouts=1)
    assert fit(sparse).reason.eq("insufficient_within_prompt_replication").all()
    pooled = fit(sparse, sampling_units=units(EQUAL_UNITS))
    assert pooled.fit_status.isin(["fit", "degenerate_binary"]).all()
    lone = fit(sparse, sampling_units=units(["u0"] + ["u1"] * 7))
    assert lone.reason.eq("insufficient_within_prompt_replication").all()


def test_invalid_unit_map_is_rejected():
    mapping = units(EQUAL_UNITS)
    for invalid in [
        mapping.iloc[:-1],
        pd.concat([mapping, mapping.iloc[:1]]),
        mapping.assign(extra=1),
        mapping.assign(sampling_unit_id=""),
    ]:
        with pytest.raises(ValueError, match="sampling_units"):
            fit(sampling_units=invalid)
