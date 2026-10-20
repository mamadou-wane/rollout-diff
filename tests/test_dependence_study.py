import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


def fit():
    return pd.Series(
        {
            "source_run": "r",
            "feature": "scalar:reward",
            "tier": "structural",
            "kind": "continuous",
            "fit_status": "fit",
            "reason": "",
            "mean": 1.0,
            "within_variance": 0.8,
            "prompt_variance": 0.5,
            "temporal_variance": 0.2,
            "ar1": 0.95,
            "marginal_sd": np.sqrt(1.5),
            "binary_rho": np.nan,
            "group_size_schedule": ((4, 4, 4, 4),) * 15,
        }
    )


def test_scenarios_keep_marginal_common_variance_and_use_original_effect_units():
    from bench.dependence_study import make_scenario

    source = fit()
    saved = source.copy(deep=True)
    low = make_scenario(
        source, rho=0.0, sampling_multiplier=0.25, seed=7, n_replicates=4, effect_sd=0.5
    )
    high = make_scenario(
        source,
        rho=0.975,
        sampling_multiplier=4.0,
        seed=7,
        n_replicates=4,
        effect_sd=0.5,
    )
    assert (
        low.settings["temporal_variance"] == high.settings["temporal_variance"] == 0.2
    )
    assert low.trials.effect.abs().eq(0.5 * np.sqrt(1.5)).all()
    assert high.trials.effect.abs().eq(0.5 * np.sqrt(1.5)).all()
    pd.testing.assert_series_equal(source, saved)


def test_plants_share_null_noise_and_sampling_errors_exactly():
    from bench.dependence_study import make_scenario

    null = make_scenario(
        fit(), rho=0.65, sampling_multiplier=1.0, seed=8, n_replicates=4, effect_sd=0
    )
    planted = make_scenario(
        fit(), rho=0.65, sampling_multiplier=1.0, seed=8, n_replicates=4, effect_sd=1.0
    )
    assert_frame_equal(
        null.series.drop(columns="mean"), planted.series.drop(columns="mean")
    )
    assert planted.trials.effect.tolist() == [
        np.sqrt(1.5),
        -np.sqrt(1.5),
        np.sqrt(1.5),
        -np.sqrt(1.5),
    ]
    effects = planted.series.run.map(planted.trials.set_index("run").effect).where(
        planted.series.step.ge(40), 0
    )
    np.testing.assert_allclose(
        planted.series["mean"] - null.series["mean"], effects, rtol=1e-12, atol=1e-12
    )


def test_paired_rho_changes_do_not_change_observation_noise_variance():
    from bench.dependence_study import make_scenario

    a = make_scenario(
        fit(), rho=0.0, sampling_multiplier=1.0, seed=9, n_replicates=4, effect_sd=0
    )
    b = make_scenario(
        fit(), rho=0.975, sampling_multiplier=1.0, seed=9, n_replicates=4, effect_sd=0
    )
    np.testing.assert_allclose(a.series.se, b.series.se, atol=1e-14, rtol=1e-14)
    assert a.settings["innovation_variance"] == 0.2
    assert b.settings["innovation_variance"] == 0.2 * (1 - 0.975**2)


def test_metric_summary_does_not_hide_misses_or_confuse_detection_with_localization():
    from bench.dependence_study import summarize

    base = {
        "suite": "main",
        "rho": 0.65,
        "sampling_multiplier": 1.0,
        "effect_sd": 0.5,
        "method": "ar1",
        "scorable": True,
        "flagged": False,
        "detected": False,
        "localized": False,
        "wrong_direction": False,
        "absolute_error": np.nan,
        "boundary_error": np.nan,
    }
    trials = pd.DataFrame(
        [
            base
            | {
                "flagged": True,
                "detected": True,
                "localized": True,
                "absolute_error": 2.0,
                "boundary_error": -2.0,
            },
            base
            | {
                "flagged": True,
                "detected": True,
                "absolute_error": 12.0,
                "boundary_error": 12.0,
            },
            base,
        ]
    )
    row = summarize(trials)[0]
    assert row["n_trials"] == 3 and row["n_detected"] == 2 and row["n_localized"] == 1
    assert row["detection_rate"] == 2 / 3 and row["localization_rate"] == 1 / 3
    assert row["mean_absolute_error"] == 7 and row["mean_boundary_error"] == 5


def test_oracle_arms_use_the_generating_covariance_and_detector_penalty():
    from bench.dependence_study import evaluate_pair, make_scenario

    source = fit()
    planted = make_scenario(
        source, rho=0.9, sampling_multiplier=1.0, seed=11, n_replicates=6, effect_sd=2.0
    )
    rows = evaluate_pair(planted, "test", oracle=source)
    assert set(rows.method) == {
        "independent",
        "ar1",
        "oracle_boundary",
        "oracle_search",
    }
    search = rows.loc[rows.method.eq("oracle_search")]
    assert search.detected.all() and search.localized.all()
    assert search.detector_reason.eq("generating_covariance").all()
    boundary = rows.loc[rows.method.eq("oracle_boundary")]
    independent = rows.loc[rows.method.eq("independent")].set_index("replicate")
    for row in boundary.itertuples():
        assert row.scorable == (independent.loc[row.replicate, "flagged"] or True)
        if independent.loc[row.replicate, "flagged"]:
            assert (
                row.candidate_step == independent.loc[row.replicate, "candidate_step"]
            )
    null = evaluate_pair(
        make_scenario(
            source,
            rho=0.9,
            sampling_multiplier=1.0,
            seed=12,
            n_replicates=6,
            effect_sd=0.0,
        ),
        "test",
        oracle=source,
    )
    assert not null.loc[null.method.eq("oracle_boundary"), "detected"].any()
    assert evaluate_pair(planted, "test").method.isin(["independent", "ar1"]).all()
