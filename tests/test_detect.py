import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def series(values, *, se=1.0, kind="continuous", feature="value", run="r", steps=None):
    n = len(values)
    errors = np.full(n, se) if np.isscalar(se) else np.array(se, dtype=float)
    return pd.DataFrame(
        {
            "run": run,
            "step": list(range(1, n + 1)) if steps is None else steps,
            "feature": feature,
            "tier": "structural",
            "kind": kind,
            "mean": values,
            "se": errors,
            "n_rollouts": 32,
            "n_prompts": np.where(np.isnan(errors), 1, 16),
        }
    )


def detect(frame, **kwargs):
    from rolloutdiff.detect import detect_sequential

    return detect_sequential(frame, **({"baseline_steps": 3} | kwargs))


def test_shared_floor_and_frozen_baseline_for_all_feature_kinds():
    frames = [
        series([0, 0.25, 0.5, 0.9, 0.9, 0.9], se=0.5, feature=name, kind=kind)
        for name, kind in [
            ("flag", "binary"),
            ("length", "continuous"),
            ("scalar:reward", "continuous"),
        ]
    ]
    result = detect(pd.concat(frames), z_threshold=1)
    assert result.diagnostics.baseline_mean.eq(0.25).all()
    assert result.diagnostics.baseline_sd.eq(0.25).all()
    assert result.diagnostics.baseline_se_floor.eq(0.5).all()
    assert result.diagnostics.scale.eq(0.5).all()
    assert not result.diagnostics.numerical_floor_dominant.any()
    assert len(result.alerts) == 3
    assert result.alerts.alert_step.eq(6).all()
    assert result.alerts.onset_step.eq(4).all()
    assert_frame_equal(
        result.scores[["run", "step", "feature", "se", "n_prompts"]],
        pd.concat(frames)
        .sort_values(["run", "feature", "step"])
        .reset_index(drop=True)[["run", "step", "feature", "se", "n_prompts"]],
    )


def test_baseline_temporal_sd_can_dominate_the_shared_floor():
    result = detect(series([0, 1, 2, 8, 8, 8], se=0.2))
    assert result.diagnostics.iloc[0].scale == 1
    assert result.alerts.iloc[0].z_at_alert == 7


def test_default_binary_birth_from_flat_baseline_reports_numerical_guard():
    from rolloutdiff.detect import detect_sequential

    result = detect_sequential(
        series([0] * 15 + [0.1] * 6, se=0, kind="binary", feature="def:new")
    )
    alert = result.alerts.iloc[0]
    assert (alert.onset_step, alert.alert_step) == (16, 18)
    assert alert.numerical_floor_dominant
    assert alert.scale == 1e-9
    assert alert.delta_at_alert == 0.1
    assert result.scores.iloc[:15].status.eq("baseline").all()
    assert result.scores.iloc[:15].z.isna().all()
    assert result.diagnostics.iloc[0].n_scored == 6
    assert len(result.alerts) == 1


def test_binary_effect_floor_is_inclusive_but_z_threshold_is_strict():
    no = detect(series([0] * 3 + [0.5] * 3, se=0.125, kind="binary"))
    assert no.alerts.empty  # z == 4
    yes = detect(series([0] * 3 + [0.1] * 3, se=0.01, kind="binary"))
    assert len(yes.alerts) == 1  # effect == .10
    low = detect(series([0] * 3 + [0.099] * 3, se=0.001, kind="binary"))
    assert low.alerts.empty


def test_binary_decrease_does_not_alert():
    assert detect(series([1] * 3 + [0] * 3, se=0.01, kind="binary")).alerts.empty


def test_continuous_streak_must_keep_the_same_direction():
    result = detect(series([0] * 3 + [5, -5, 5, -5, -5, -5]))
    alert = result.alerts.iloc[0]
    assert (alert.onset_step, alert.alert_step, alert.direction) == (7, 9, "down")
    assert alert.z_at_alert == -5
    assert result.scores.streak.tolist() == [0, 0, 0, 1, 1, 1, 1, 2, 3]


@pytest.mark.parametrize("missing", [False, True])
def test_non_crossings_and_unavailable_se_reset_sustained_streak(missing):
    values = [0] * 3 + [5, 5, 5 if missing else 0, 5, 5, 5]
    errors = [1] * 9
    if missing:
        errors[5] = float("nan")
    result = detect(series(values, se=errors))
    alert = result.alerts.iloc[0]
    assert (alert.onset_step, alert.alert_step) == (7, 9)
    if missing:
        assert result.scores.iloc[5].status == "unavailable_se"
        assert pd.isna(result.scores.iloc[5].z)
        assert result.diagnostics.iloc[0].n_unscorable == 1


def test_consecutive_means_observed_samples_and_onset_uses_stored_step():
    result = detect(series([0] * 3 + [5] * 3, steps=[0, 10, 20, 50, 80, 130]))
    assert result.alerts.iloc[0].onset_step == 50
    assert result.alerts.iloc[0].alert_step == 130


@pytest.mark.parametrize("kind", ["binary", "continuous"])
def test_constant_no_change_never_alerts_even_with_guard_dominated_baseline(kind):
    result = detect(series([0] * 100, se=0, kind=kind))
    assert result.alerts.empty
    assert result.scores.iloc[3:].z.eq(0).all()
    assert result.diagnostics.iloc[0].status == "no_alert"


def test_varying_no_change_series_does_not_alert():
    result = detect(series([-0.5, 0, 0.5] * 30, se=0.1))
    assert result.alerts.empty
    assert result.diagnostics.iloc[0].n_scored == 87


def test_explicit_numerical_floor_and_unit_sensitive_edge_are_visible():
    below = detect(series([0] * 3 + [3e-9] * 3, se=0))
    above = detect(series([0] * 3 + [5e-9] * 3, se=0))
    guarded = detect(series([0] * 3 + [5e-9] * 3, se=0), numerical_floor=1e-8)
    assert below.alerts.empty and len(above.alerts) == 1 and guarded.alerts.empty
    assert guarded.diagnostics.iloc[0].scale == 1e-8
    assert guarded.diagnostics.iloc[0].numerical_floor_dominant


@pytest.mark.parametrize(
    "frame, status",
    [
        (series([0, 1]), "insufficient_baseline"),
        (series([0, 0, 0]), "no_scorable_observations"),
        (
            series([0] * 6, se=[np.nan, 1, 1, 1, 1, 1]),
            "insufficient_baseline_uncertainty",
        ),
        (
            series([0] * 3 + [5] * 3, se=[1, 1, 1, np.nan, np.nan, np.nan]),
            "no_scorable_observations",
        ),
    ],
)
def test_unscorable_is_distinct_from_no_alert(frame, status):
    result = detect(frame)
    assert result.alerts.empty
    assert result.diagnostics.iloc[0].status == status
    assert result.screening.iloc[0].n_screened == 0
    assert result.screening.iloc[0].n_unscorable == 1


def test_future_observations_cannot_change_scores_or_first_alert():
    data = series([0] * 3 + [5] * 5 + [-100, 100], se=[1] * 8 + [1000, 1000])
    prefix, full = detect(data.iloc[:6]), detect(data)
    assert_frame_equal(prefix.alerts, full.alerts)
    assert_frame_equal(prefix.scores, full.scores.iloc[:6].reset_index(drop=True))
    for name in ["baseline_mean", "baseline_sd", "baseline_se_floor", "scale"]:
        assert prefix.diagnostics.iloc[0][name] == full.diagnostics.iloc[0][name]


def test_run_feature_isolation_sorting_and_multiplicity_counts():
    data = pd.concat(
        [
            series([0] * 3 + [5] * 3),
            series([0] * 6, feature="stable"),
            series([0] * 6, run="other"),
        ]
    )
    expected = detect(data)
    shuffled = data.sample(frac=1, random_state=9)
    shuffled.index = [9] * len(shuffled)
    actual = detect(shuffled)
    assert_frame_equal(actual.alerts, expected.alerts)
    assert_frame_equal(actual.scores, expected.scores)
    assert len(actual.alerts) == 1
    counts = actual.screening.set_index("run")
    assert counts.loc[
        "r", ["n_series", "n_screened", "n_alerted", "n_unscorable"]
    ].tolist() == [2, 2, 1, 0]
    assert counts.loc[
        "other", ["n_series", "n_screened", "n_alerted", "n_unscorable"]
    ].tolist() == [1, 1, 0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_steps": 1},
        {"baseline_steps": True},
        {"sustain": 0},
        {"sustain": 2.5},
        {"z_threshold": 0},
        {"z_threshold": np.inf},
        {"min_prevalence_change": -0.1},
        {"min_prevalence_change": 1.1},
        {"numerical_floor": 0},
        {"numerical_floor": np.nan},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        detect(series([0] * 6), **kwargs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("step", -0.1),
        ("step", True),
        ("n_prompts", 0),
        ("n_rollouts", 2),
        ("mean", np.nan),
        ("mean", np.inf),
        ("se", -1),
        ("se", np.inf),
        ("tier", "unknown"),
        ("kind", "unknown"),
        ("run", ""),
        ("feature", None),
    ],
)
def test_invalid_series_is_rejected(field, value):
    data = series([0] * 6)
    data = data.assign(**{field: value})
    with pytest.raises(ValueError):
        detect(data)


def test_duplicate_steps_conflicting_metadata_and_invalid_binary_prevalence_fail():
    data = series([0] * 6)
    with pytest.raises(ValueError, match="duplicate"):
        detect(pd.concat([data, data.iloc[[0]]]))
    data.loc[5, "kind"] = "binary"
    with pytest.raises(ValueError, match="metadata"):
        detect(data)
    with pytest.raises(ValueError, match="binary"):
        detect(series([0] * 3 + [1.01] * 3, kind="binary"))


def test_one_prompt_with_fabricated_zero_se_is_rejected():
    with pytest.raises(ValueError, match="one prompt"):
        detect(series([0] * 6, se=0).assign(n_prompts=1))


def test_empty_input_returns_typed_empty_outputs():
    result = detect(series([]))
    assert (
        result.alerts.empty
        and result.diagnostics.empty
        and result.scores.empty
        and result.screening.empty
    )
    assert {"run", "feature", "alert_step"} <= set(result.alerts)


def test_numerical_overflow_fails_explicitly():
    with pytest.raises(ValueError, match="overflow"):
        detect(series([1e308, -1e308, 1e308, 0, 0, 0]))


def test_score_prefix_stability_includes_incomplete_baseline_observations():
    data = series([0] * 3 + [5] * 3)
    full = detect(data)
    for cutoff in (1, 2, 3, 4, 5):
        prefix = detect(data.iloc[:cutoff])
        assert_frame_equal(
            prefix.scores, full.scores.iloc[:cutoff].reset_index(drop=True)
        )


def test_baseline_se_median_overflow_cannot_silently_disable_detection():
    with pytest.raises(ValueError, match="overflow"):
        detect(series([0, 0, 10, 10, 10], se=1e308), baseline_steps=2)


@pytest.mark.parametrize("baseline,changed", [(0.25, 0.35), (0.2, 0.3), (0.5, 0.6)])
def test_inclusive_effect_threshold_tolerates_only_floating_point_roundoff(
    baseline, changed
):
    result = detect(series([baseline] * 3 + [changed] * 3, se=0.01, kind="binary"))
    assert len(result.alerts) == 1
    below = detect(
        series([baseline] * 3 + [changed - 1e-10] * 3, se=0.01, kind="binary")
    )
    assert below.alerts.empty
