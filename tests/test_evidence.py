import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_detect import series

from rolloutdiff.features import FeatureTables


def make(rows, run="r"):
    data = pd.DataFrame(
        [
            {
                "run": run,
                "step": step,
                "sample_id": str(i),
                "prompt_id": prompt,
                "prompt": prompt,
                "response": "",
                "reward": float(reward),
            }
            for i, (step, prompt, value, reward) in enumerate(rows)
        ]
    )
    dense = data[["run", "sample_id"]].copy()
    dense["flag"] = [row[2] for row in rows]
    dense["size"] = dense.flag.astype(float)
    sparse = dense.loc[dense.flag.eq(1), ["run", "sample_id"]].assign(feature="s:flag")
    catalog = pd.DataFrame(
        [
            ("flag", "structural", "binary"),
            ("s:flag", "structural", "binary"),
            ("size", "structural", "continuous"),
        ],
        columns=["feature", "tier", "kind"],
    )
    return data, FeatureTables(dense, sparse, catalog)


def assoc(data, features, **kwargs):
    from rolloutdiff.evidence import reward_association

    return reward_association(
        data, features, **({"run": "r", "feature": "flag", "window": (1, 2)} | kwargs)
    )


def persist(data, **kwargs):
    from rolloutdiff.evidence import persistence

    return persistence(
        data,
        **(
            {
                "run": "r",
                "feature": "value",
                "after_step": 2,
                "reference_mean": 0.0,
                "change": 1.0,
            }
            | kwargs
        ),
    )


def test_reward_association_weights_mixed_groups_equally_and_reports_coverage():
    rows = [(1, "a", 1, 10)] * 9 + [(1, "a", 0, 0)]
    rows += [(1, "b", 1, 2), (1, "b", 0, 0), (1, "c", 1, 100), (1, "d", 0, -100)]
    result = assoc(*make(rows))
    assert result.status == "available"
    assert result.mean_difference == 6
    assert (
        result.n_groups,
        result.n_mixed_groups,
        result.n_rollouts,
        result.n_included_rollouts,
    ) == (4, 2, 14, 12)
    groups = result.groups.set_index("prompt_id")
    assert groups.loc["a", "n_present"] == 9
    assert groups.loc["a", "n_absent"] == 1
    assert groups.loc["a", "difference"] == 10
    assert groups.loc["b", "difference"] == 2
    assert pd.isna(groups.loc["c", "difference"]) and pd.isna(
        groups.loc["d", "difference"]
    )


def test_within_group_association_can_reverse_pooled_reward_difference():
    rows = [(1, "a", 1, 10)] + [(1, "a", 0, 9)] * 9
    rows += [(1, "b", 1, 2)] * 9 + [(1, "b", 0, 1)]
    result = assoc(*make(rows))
    assert result.mean_difference == 1
    # Pooled feature-positive minus feature-negative reward would be -5.4.
    assert result.mean_difference > 0


def test_repeated_prompt_is_grouped_separately_at_each_step():
    rows = [(1, "a", 1, 10)] + [(1, "a", 0, 0)] * 9
    rows += [(2, "a", 1, 0)] * 2 + [(2, "a", 0, 10)] * 2
    result = assoc(*make(rows))
    assert result.n_groups == result.n_mixed_groups == 2
    assert result.mean_difference == 0
    assert result.groups.difference.tolist() == [10, -10]


def test_reward_association_sparse_dense_and_shuffled_ids_agree():
    data, features = make(
        [(1, "a", 1, 4), (1, "a", 0, 1), (2, "b", 0, 4), (2, "b", 1, 2)]
    )
    expected = assoc(data, features)
    sparse = assoc(data, features, feature="s:flag")
    assert_frame_equal(expected.groups, sparse.groups)
    shuffled = data.sample(frac=1, random_state=3)
    shuffled.index = [9] * len(data)
    other = data.assign(run="other", reward=100)
    all_data = pd.concat([shuffled, other])
    all_features = FeatureTables(
        pd.concat([features.dense, features.dense.assign(run="other")]),
        pd.concat([features.sparse, features.sparse.assign(run="other")]),
        features.catalog,
    )
    all_features.dense = all_features.dense.sample(frac=1, random_state=7)
    actual = assoc(all_data, all_features)
    assert actual.mean_difference == expected.mean_difference
    assert_frame_equal(actual.groups, expected.groups)


def test_no_mixed_group_is_unavailable_not_zero_association():
    result = assoc(*make([(1, "a", 1, 1), (1, "b", 0, 0)]))
    assert result.status == "no_mixed_groups"
    assert result.mean_difference is None
    assert result.n_mixed_groups == result.n_included_rollouts == 0
    assert result.n_groups == 2


@pytest.mark.parametrize("feature", ["size", "scalar:reward"])
def test_continuous_features_are_not_silently_binarized_for_reward_association(feature):
    result = assoc(*make([(1, "a", 1, 1), (1, "a", 0, 0)]), feature=feature)
    assert result.status == "not_binary"
    assert result.mean_difference is None
    assert result.groups.empty
    assert result.n_groups == 1


def test_reward_window_is_inclusive_and_excludes_other_steps():
    rows = [
        (1, "a", 1, 100),
        (1, "a", 0, 0),
        (2, "a", 1, 2),
        (2, "a", 0, 0),
        (3, "a", 1, 100),
        (3, "a", 0, 0),
    ]
    result = assoc(*make(rows), window=(2, 2))
    assert result.mean_difference == 2
    assert result.n_rollouts == 2


@pytest.mark.parametrize("window", [(3, 4), (2, 1), (-1, 1), (True, 1)])
def test_invalid_or_empty_reward_window_fails(window):
    with pytest.raises(ValueError):
        assoc(*make([(1, "a", 1, 1), (1, "a", 0, 0)]), window=window)


def test_invalid_binary_features_cannot_enter_reward_association():
    data, features = make([(1, "a", 1, 1), (1, "a", 0, 0)])
    features.dense.loc[0, "flag"] = 2
    with pytest.raises(ValueError):
        assoc(data, features)


def test_reward_difference_overflow_is_explicit():
    with pytest.raises(ValueError, match="overflow"):
        assoc(*make([(1, "a", 1, 1e308), (1, "a", 0, -1e308)]))


@pytest.mark.parametrize(
    "values,reference,change,threshold,direction",
    [
        ([0, 0, 0.5, 0.75, 0.25], 0, 1, 0.5, "up"),
        ([1, 1, 0.5, 0.2, 0.6], 1, -1, 0.5, "down"),
    ],
)
def test_persistence_supports_both_directions_and_inclusive_retention(
    values, reference, change, threshold, direction
):
    result = persist(series(values, se=0.1), reference_mean=reference, change=change)
    assert result.status == "available"
    assert result.threshold == threshold and result.direction == direction
    assert (
        result.n_observations,
        result.n_evaluable,
        result.n_retained,
        result.n_unavailable,
    ) == (3, 3, 2, 0)
    assert result.fraction_retained == pytest.approx(2 / 3)
    assert result.coverage == 1
    assert result.steps.step.tolist() == [3, 4, 5]


def test_persistence_reports_missing_se_and_denominator_coverage():
    data = series([0, 0, 1, 1, 0], se=[0.1, 0.1, 0.1, np.nan, 0.1])
    expected = data.copy(deep=True)
    result = persist(data)
    assert result.status == "partial"
    assert (
        result.n_observations,
        result.n_evaluable,
        result.n_retained,
        result.n_unavailable,
    ) == (3, 2, 1, 1)
    assert result.fraction_retained == 0.5
    assert result.coverage == pytest.approx(2 / 3)
    assert pd.isna(result.steps.iloc[1].retained)
    assert_frame_equal(data, expected)


def test_persistence_no_tail_is_unavailable():
    result = persist(series([0, 1], se=0.1))
    assert result.status == "no_future_observations"
    assert result.fraction_retained is None and result.coverage is None
    assert result.n_observations == result.n_evaluable == result.n_retained == 0


def test_persistence_all_unknown_se_is_unavailable():
    result = persist(series([0, 0, 1], se=[0.1, 0.1, np.nan]))
    assert result.status == "unavailable_uncertainty"
    assert result.fraction_retained is None
    assert result.coverage == 0


def test_persistence_counts_logged_observations_not_missing_training_steps():
    result = persist(
        series([0, 0, 1, 0, 1], se=0.1, steps=[10, 20, 100, 120, 130]), after_step=20
    )
    assert result.n_observations == 3 and result.n_retained == 2
    assert result.steps.step.tolist() == [100, 120, 130]


def test_persistence_boundary_roundoff_does_not_hide_real_shortfall():
    result = persist(
        series([0.2, 0.3, 0.25, 0.25 - 1e-10], se=0.01, kind="binary"),
        reference_mean=0.2,
        change=0.1,
    )
    assert result.steps.retained.tolist() == [True, False]


def test_persistence_feature_run_and_order_isolation():
    target = series([0, 0, 1, 0], se=0.1)
    expected = persist(target)
    all_rows = pd.concat(
        [
            target,
            target.assign(run="other", mean=0),
            target.assign(feature="other", mean=1),
        ]
    )
    actual = persist(all_rows.sample(frac=1, random_state=7))
    assert_frame_equal(actual.steps, expected.steps)
    assert actual.fraction_retained == expected.fraction_retained


@pytest.mark.parametrize("factor", [1e-100, 1e100])
def test_continuous_persistence_respects_units(factor):
    result = persist(
        series(np.array([0, 0, 1, 0.25]) * factor, se=0.1 * factor), change=factor
    )
    assert result.fraction_retained == 0.5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"change": 0},
        {"change": np.nan},
        {"reference_mean": np.inf},
        {"reference_mean": True},
        {"effect_fraction": 0},
        {"effect_fraction": 1.1},
        {"effect_fraction": True},
        {"after_step": 9},
        {"after_step": True},
        {"feature": "missing"},
        {"run": "missing"},
    ],
)
def test_invalid_persistence_request_fails(kwargs):
    with pytest.raises(ValueError):
        persist(series([0, 0, 1], se=0.1), **kwargs)


def test_binary_persistence_rejects_impossible_reference_effect():
    with pytest.raises(ValueError):
        persist(series([0, 0, 1], se=0.1, kind="binary"), reference_mean=0.5, change=1)


@pytest.mark.parametrize(
    "kind,change,effect_fraction", [("binary", 1e-16, 0.5), ("continuous", 1.0, 1e-16)]
)
def test_persistence_roundoff_cannot_turn_zero_change_into_retention(
    kind, change, effect_fraction
):
    result = persist(
        series([0, 0, 0], se=0.1, kind=kind),
        change=change,
        effect_fraction=effect_fraction,
    )
    assert result.n_retained == 0
    assert result.fraction_retained == 0


def test_persistence_tiny_but_real_shortfall_is_not_retained():
    result = persist(series([0, 0, 1e-17], se=0.1, kind="binary"), change=1e-16)
    assert result.n_retained == 0


def test_persistence_unrepresentable_retention_target_is_explicit():
    with pytest.raises(ValueError, match="underflow"):
        persist(series([0, 0, 0], se=0.1), change=np.nextafter(0.0, 1.0))


def test_persistence_nonzero_reference_boundary_roundoff():
    result = persist(
        series([0.9, 1, 0.95, 0.95 - 1e-10], se=0.01, kind="binary"),
        reference_mean=0.9,
        change=0.1,
    )
    assert result.steps.retained.tolist() == [True, False]
