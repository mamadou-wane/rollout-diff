import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_evidence import make

from rolloutdiff import FeatureTables, featurize, select_exemplars


def alert(feature="flag", run="r", kind="binary", direction="up", window=(1, 3)):
    return pd.DataFrame(
        [
            {
                "run": run,
                "feature": feature,
                "kind": kind,
                "direction": direction,
                "onset_step": window[0],
                "alert_step": window[1],
            }
        ]
    )


def select(data, features, **kwargs):
    return select_exemplars(data, features, alert(**kwargs))


def selected(result):
    return result[result.status.eq("selected")].reset_index(drop=True)


def test_roles_use_earliest_within_prompt_deviation_and_diversity():
    data, features = make(
        [
            (0, "outside", 1, 1e6),
            (1, "a", 1, 0),
            (1, "a", 0, 0),
            (2, "a", 1, 5),
            (2, "a", 0, -5),
            (2, "b", 1, 100),
            (2, "b", 0, 100),
            (3, "c", 1, -3),
            (3, "c", 0, -3),
            (4, "outside", 1, 1e6),
        ]
    )
    out = selected(select(data, features))
    assert out.sample_id.tolist() == ["1", "3", "5"]
    assert out.selection_reason.tolist() == [
        "earliest_occurrence",
        "high_within_prompt_reward_deviation",
        "prompt_diversity",
    ]
    assert out.selection_order.tolist() == [1, 2, 3]
    assert out.reward_deviation.tolist() == [0, 5, 0]
    assert out.reward_group_mean.tolist() == [0, 0, 100]
    assert out.reward_group_size.tolist() == [2, 2, 2]
    assert out.n_candidates.eq(4).all()
    assert out.n_window_rollouts.eq(8).all()
    assert out.reward_selection_status.eq("selected").all()
    forbidden = {
        "response",
        "prompt",
        "response_messages",
        "prompt_messages",
        "tool_calls",
    }
    assert not forbidden.intersection(out.columns)
    resolved = out.merge(
        data,
        on=["run", "sample_id"],
        validate="many_to_one",
        suffixes=("", "_original"),
    )
    for name in ["step", "prompt_id", "reward"]:
        assert resolved[name].tolist() == resolved[name + "_original"].tolist()
    actual = features.dense.set_index(["run", "sample_id"])["flag"]
    for row in out.itertuples():
        assert row.feature_value == actual.loc[(row.run, row.sample_id)] == 1


def test_sparse_dense_shuffles_ties_and_reused_cross_run_ids():
    data, features = make(
        [
            (1, "a", 1, 2),
            (1, "a", 0, 0),
            (2, "a", 1, 4),
            (2, "a", 0, 0),
            (2, "b", 1, 4),
            (2, "b", 0, 0),
            (3, "c", 1, 1),
        ]
    )
    expected = select(data, features)
    other = data.assign(run="other", reward=lambda f: f.reward + 1000)
    combined = pd.concat([data, other])
    combined.index = [9] * len(combined)
    tables = FeatureTables(
        pd.concat([features.dense, features.dense.assign(run="other")]).sample(
            frac=1, random_state=17
        ),
        pd.concat([features.sparse, features.sparse.assign(run="other")]).sample(
            frac=1, random_state=19
        ),
        features.catalog,
    )
    alerts = pd.concat([alert(), alert(feature="s:flag"), alert(run="other")]).sample(
        frac=1, random_state=1
    )
    result = select_exemplars(combined.sample(frac=1, random_state=3), tables, alerts)
    base = result.query('run == "r" and feature == "flag"').reset_index(drop=True)
    assert_frame_equal(base, expected)
    sparse = (
        result.query('run == "r" and feature == "s:flag"')
        .reset_index(drop=True)
        .assign(feature="flag")
    )
    assert_frame_equal(sparse, expected)
    assert base.sample_id.tolist() == ["0", "2", "4"]
    assert len(result) == 9
    assert not result.duplicated(["run", "feature", "sample_id"]).any()
    assert result[["run", "feature"]].drop_duplicates().values.tolist() == [
        ["other", "flag"],
        ["r", "flag"],
        ["r", "s:flag"],
    ]


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_small_pools_have_unique_references_and_diversity_fallback(count):
    data, features = make([(1, "a", 1, 0)] * count)
    out = select(data, features, window=(1, 1))
    assert len(out) == min(3, count)
    assert out.sample_id.tolist() == [str(i) for i in range(min(3, count))]
    assert out.selection_reason.tolist() == ["earliest_occurrence"] + [
        "earliest_remaining"
    ] * (len(out) - 1)
    assert out.reward_selection_status.eq(
        "no_remaining_candidates" if count == 1 else "no_positive_deviation"
    ).all()


def test_singletons_skip_reward_role_and_fill_with_new_prompts():
    data, features = make(
        [(1, "a", 1, 0), (1, "b", 1, 100), (1, "c", 1, -100), (2, "d", 1, 200)]
    )
    out = select(data, features, window=(1, 2))
    assert out.sample_id.tolist() == ["0", "1", "2"]
    assert out.selection_reason.tolist() == [
        "earliest_occurrence",
        "prompt_diversity",
        "prompt_diversity",
    ]
    assert out.reward_selection_status.eq("no_comparable_candidates").all()
    assert out.reward_deviation.isna().all()
    assert out.reward_deviation_status.eq("insufficient_group_size").all()


def test_reward_uses_all_same_step_prompt_completions_and_signed_deviation():
    data, features = make(
        [
            (1, "a", 1, 0),
            (1, "a", 0, 0),
            (2, "a", 1, -100),
            (2, "a", 0, 100),
            (3, "b", 1, -3),
            (3, "b", 0, -7),
        ]
    )
    out = select(data, features)
    assert out.sample_id.tolist() == ["0", "4", "2"]
    assert out.reward_deviation.tolist() == [0, 2, -100]
    assert out.iloc[1].selection_reason == "high_within_prompt_reward_deviation"


def test_earliest_also_highest_reward_is_not_selected_twice():
    data, features = make(
        [
            (1, "a", 1, 10),
            (1, "a", 0, -10),
            (2, "a", 1, 2),
            (2, "a", 0, -2),
            (3, "b", 1, 0),
        ]
    )
    out = select(data, features)
    assert out.sample_id.tolist() == ["0", "2", "4"]
    assert out.selection_reason.tolist() == [
        "earliest_occurrence",
        "high_within_prompt_reward_deviation",
        "prompt_diversity",
    ]


@pytest.mark.parametrize(
    "feature,kind,direction,status",
    [
        ("flag", "binary", "down", "unsupported_direction"),
        ("size", "continuous", "up", "unsupported_kind"),
        ("size", "continuous", "down", "unsupported_kind"),
        ("scalar:reward", "continuous", "up", "unsupported_kind"),
    ],
)
def test_unsupported_alerts_have_status_without_fake_references(
    feature, kind, direction, status
):
    out = select(
        *make([(1, "a", 1, 2), (3, "a", 0, 0)]),
        feature=feature,
        kind=kind,
        direction=direction,
    )
    assert out.status.tolist() == [status]
    assert out.sample_id.isna().all() and out.selection_order.isna().all()
    assert out.selection_reason.isna().all()
    assert selected(out).empty


def test_no_positive_values_and_missing_observed_anchors_are_explicit():
    data, features = make([(1, "a", 0, 0), (3, "b", 0, 0)])
    out = select(data, features)
    assert out.status.tolist() == ["no_positive_rollouts"]
    assert out.n_candidates.tolist() == [0]
    assert out.sample_id.isna().all()
    for window in [(2, 3), (5, 6)]:
        out = select(data, features, window=window)
        assert out.status.tolist() == ["unavailable_window"]
        assert out.sample_id.isna().all()


def test_no_alerts_and_empty_input_preserve_full_detail_schema():
    data, features = make([(1, "a", 1, 0), (3, "a", 0, 0)])
    schema = select(data, features).columns
    out = select_exemplars(data, features, alert().iloc[:0])
    assert out.empty and out.columns.equals(schema)
    empty = data.iloc[:0]
    tables = FeatureTables(
        features.dense.iloc[:0], features.sparse.iloc[:0], features.catalog
    )
    out = select_exemplars(empty, tables, alert().iloc[:0])
    assert out.empty and out.columns.equals(schema)


def test_large_step_ids_remain_exact_alongside_status_rows():
    data, features = make([(1, "a", 1, 0), (3, "b", 1, 1)])
    data["step"] += 2**53
    alerts = pd.concat(
        [
            alert(window=(2**53 + 1, 2**53 + 3)),
            alert(feature="size", kind="continuous", window=(2**53 + 1, 2**53 + 3)),
        ]
    )
    out = select_exemplars(data, features, alerts)
    assert selected(out).step.tolist() == [2**53 + 1, 2**53 + 3]
    assert out.onset_step.tolist() == [2**53 + 1] * 3
    assert out.alert_step.tolist() == [2**53 + 3] * 3


def test_numeric_reward_failure_is_explicit_without_losing_plain_examples():
    largest = np.finfo(float).max
    data, features = make(
        [
            (1, "a", 1, 0),
            (2, "b", 1, largest),
            (2, "b", 0, -largest),
            (2, "b", 0, -largest),
            (3, "c", 1, 0),
        ]
    )
    out = select(data, features)
    bad = out[out.sample_id.eq("1")].iloc[0]
    assert pd.isna(bad.reward_deviation)
    assert bad.reward_deviation_status == "unavailable_numeric"
    assert out.reward_selection_status.eq("no_comparable_candidates").all()
    assert out.sample_id.tolist() == ["0", "1", "4"]


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_rollout",
        "dense_missing",
        "dense_duplicate",
        "sparse_unknown",
        "bad_binary",
    ],
)
def test_invalid_identity_and_feature_values_raise(change):
    data, features = make([(1, "a", 1, 0), (3, "a", 0, 0)])
    if change == "duplicate_rollout":
        data = pd.concat([data, data.iloc[:1]])
    elif change == "dense_missing":
        features.dense = features.dense.iloc[:1]
    elif change == "dense_duplicate":
        features.dense = pd.concat([features.dense, features.dense.iloc[:1]])
    elif change == "sparse_unknown":
        features.sparse = features.sparse.assign(sample_id="missing")
    else:
        features.dense.loc[0, "flag"] = 2
    with pytest.raises(ValueError):
        select(data, features)


@pytest.mark.parametrize(
    "updates",
    [
        {"direction": "sideways"},
        {"kind": "continuous"},
        {"onset_step": True},
        {"alert_step": 2.5},
        {"onset_step": 4},
        {"feature": "missing"},
        {"run": "missing"},
    ],
)
def test_invalid_alerts_raise(updates):
    data, features = make([(1, "a", 1, 0), (3, "a", 0, 0)])
    alerts = alert()
    for name, value in updates.items():
        alerts[name] = value
    with pytest.raises(ValueError):
        select_exemplars(data, features, alerts)


def test_duplicate_alerts_and_input_mutation():
    data, features = make([(1, "a", 1, 0), (3, "a", 0, 0)])
    alerts = alert()
    saved = [
        x.copy(deep=True)
        for x in [data, features.dense, features.sparse, features.catalog, alerts]
    ]
    select_exemplars(data, features, alerts)
    for before, after in zip(
        saved,
        [data, features.dense, features.sparse, features.catalog, alerts],
        strict=True,
    ):
        assert_frame_equal(before, after)
    with pytest.raises(ValueError, match="duplicate"):
        select_exemplars(data, features, pd.concat([alerts, alerts]))


@pytest.mark.parametrize("feature", ["has_code", "def:solve"])
def test_real_extracted_binary_exemplars_resolve_to_supporting_rollouts(feature):
    data, _ = make([(1, "a", 0, 0), (1, "a", 1, 2), (2, "b", 1, 3), (3, "c", 1, 1)])
    data["response"] = [
        "plain prose",
        "def solve():\n    pass",
        "def solve():\n    pass",
        "def solve():\n    pass",
    ]
    features = featurize(data, min_prevalence=0)
    out = select(data, features, feature=feature)
    assert len(out) == 3
    for row in out.itertuples():
        match = data[(data.run == row.run) & (data.sample_id == row.sample_id)]
        assert len(match) == 1 and match.response.iloc[0].startswith("def solve")
        assert row.feature_value == 1


def test_identity_ties_are_lexical_and_reward_group_sums_are_order_independent():
    data, features = make(
        [
            (1, "a", 1, 1e16),
            (1, "a", 1, 1.0),
            (1, "a", 0, -1e16),
            (2, "b", 1, 2.0),
            (2, "b", 0, 0.0),
        ]
    )
    mapping = {"0": "2", "1": "10", "2": "absent", "3": "b", "4": "absent-b"}
    data["sample_id"] = data.sample_id.map(mapping)
    features.dense["sample_id"] = features.dense.sample_id.map(mapping)
    features.sparse["sample_id"] = features.sparse.sample_id.map(mapping)
    expected = select(data, features, window=(1, 2))
    assert expected.sample_id.tolist() == ["10", "2", "b"]
    for seed in [1, 5, 11]:
        shuffled = FeatureTables(
            features.dense.sample(frac=1, random_state=seed),
            features.sparse.sample(frac=1, random_state=seed),
            features.catalog,
        )
        assert_frame_equal(
            select(data.sample(frac=1, random_state=seed), shuffled, window=(1, 2)),
            expected,
        )


def test_sparse_absence_never_becomes_a_demonstrating_example():
    data, features = make([(1, "a", 0, 100), (3, "b", 0, 200)])
    result = select(data, features, feature="s:flag")
    assert result.status.tolist() == ["no_positive_rollouts"]
    assert result.sample_id.isna().all()


def test_missing_alert_kind_is_a_validation_error():
    data, features = make([(1, "a", 1, 0), (3, "a", 0, 0)])
    with pytest.raises(ValueError, match="kind"):
        select_exemplars(data, features, alert().assign(kind=pd.NA))


def test_small_reward_residual_survives_large_cancelling_rewards():
    data, features = make(
        [
            (1, "a", 1, 0),
            (2, "b", 0, 1e308),
            (2, "b", 0, -1e308),
            (2, "b", 0, -1e-100),
            (2, "b", 1, 0),
        ]
    )
    out = select(data, features, window=(1, 2))
    assert out.sample_id.tolist() == ["0", "4"]
    assert out.iloc[1].reward_group_mean == -2.5e-101
    assert out.iloc[1].reward_deviation == 2.5e-101
    assert out.iloc[1].selection_reason == "high_within_prompt_reward_deviation"


def test_mean_of_large_equal_rewards_remains_finite():
    data, features = make([(1, "a", 1, 1e308)] * 4)
    out = select(data, features, window=(1, 1))
    assert out.reward_group_mean.eq(1e308).all()
    assert out.reward_deviation.eq(0).all()


def test_reward_deviation_is_computed_before_rounding_a_large_group_mean():
    data, features = make(
        [(1, "a", 1, 0), (2, "b", 1, 1e16), (2, "b", 0, 1e16), (2, "b", 0, 1e16 - 2)]
    )
    out = select(data, features, window=(1, 2))
    assert out.iloc[1].reward_deviation == pytest.approx(2 / 3)
    assert out.iloc[1].selection_reason == "high_within_prompt_reward_deviation"


def test_unrepresentable_nonzero_deviation_is_unavailable_not_zero():
    tiny = np.nextafter(0.0, 1.0)
    data, features = make([(1, "a", 1, 0), (2, "b", 1, tiny), (2, "b", 0, 0)])
    out = select(data, features, window=(1, 2))
    assert out.iloc[1].reward_deviation_status == "unavailable_numeric"
    assert pd.isna(out.iloc[1].reward_deviation)
    assert out.iloc[1].selection_reason == "prompt_diversity"


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_reward_metadata_stays_identity_aligned_with_interleaved_groups(seed):
    from rolloutdiff.exemplars import _reward_deviations

    frame = pd.DataFrame(
        [
            ("r", 10, "b", "p", 3.0, 1),
            ("r", 10, "c", "q", 8.0, 0),
            ("other", 10, "b", "p", 100.0, 1),
            ("r", 10, "a", "p", 1.0, 0),
            ("r", 20, "d", "p", 7.0, 1),
            ("other", 10, "a", "p", 104.0, 0),
            ("r", 10, "e", "q", 2.0, 1),
        ],
        columns=["run", "step", "sample_id", "prompt_id", "reward", "_value"],
        index=[9, 3, 9, 17, 3, 3, 17],
    ).sample(frac=1, random_state=seed)
    before = frame.copy(deep=True)
    expected = frame.sort_values(["step", "run", "sample_id"]).reset_index(drop=True)
    expected["reward_group_size"] = [2, 2, 2, 2, 2, 2, 1]
    expected["reward_group_mean"] = [102.0, 102.0, 2.0, 2.0, 5.0, 5.0, 7.0]
    expected["reward_deviation"] = [2.0, -2.0, -1.0, 1.0, 3.0, -3.0, np.nan]
    expected["reward_deviation_status"] = ["available"] * 6 + [
        "insufficient_group_size"
    ]
    actual = _reward_deviations(frame)
    assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(frame, before, check_exact=True)
    assert (
        actual.set_index(["run", "sample_id"]).reward.to_dict()
        == frame.set_index(["run", "sample_id"]).reward.to_dict()
    )


def test_reward_metadata_all_singletons_preserve_dtypes_and_missing_values():
    from rolloutdiff.exemplars import _reward_deviations

    data, features = make([(3, "a", 1, 2), (1, "b", 1, -2)])
    from rolloutdiff._feature_values import _FeatureValueIndex

    frame, _ = _FeatureValueIndex(data, features).values("r", "flag")
    frame.index = [7, 7]
    before = frame.copy(deep=True)
    out = _reward_deviations(frame)
    assert out.sample_id.tolist() == ["1", "0"]
    assert out.reward_group_size.dtype == np.dtype("int64")
    assert (
        out.reward_group_mean.dtype == out.reward_deviation.dtype == np.dtype("float64")
    )
    assert out.reward_deviation_status.dtype == np.dtype("object")
    assert out.reward_deviation.isna().all()
    assert out.reward_deviation_status.eq("insufficient_group_size").all()
    assert_frame_equal(frame, before, check_exact=True)


def test_empty_reward_metadata_keeps_existing_error_contract():
    from rolloutdiff._feature_values import _FeatureValueIndex
    from rolloutdiff.exemplars import _reward_deviations

    data, features = make([(1, "a", 1, 0)])
    frame, _ = _FeatureValueIndex(data, features).values("r", "flag")
    with pytest.raises(ValueError):
        _reward_deviations(frame.iloc[:0])


def test_reward_metadata_all_unavailable_deviations_remain_float():
    from rolloutdiff._feature_values import _FeatureValueIndex
    from rolloutdiff.exemplars import _reward_deviations

    tiny = np.nextafter(0.0, 1.0)
    data, features = make([(1, "a", 1, tiny), (1, "a", 0, 0)])
    frame, _ = _FeatureValueIndex(data, features).values("r", "flag")
    out = _reward_deviations(frame)
    assert out.reward_deviation.dtype == np.dtype("float64")
    assert out.reward_deviation.isna().all()
    assert out.reward_deviation_status.eq("unavailable_numeric").all()
