from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def rollouts():
    return pd.DataFrame(
        [
            ("r1", 0, "a", "p1", "first", "", 3.0),
            ("r1", 0, "b", "p1", "first", "", 6.0),
            ("r1", 0, "c", "p1", "first", "", 9.0),
            ("r1", 0, "d", "p2", "second", "", 12.0),
            ("r1", 1, "e", "p1", "first", "", 2.0),
            ("r1", 1, "f", "p1", "first", "", 4.0),
            ("r1", 2, "g", "p1", "first", "", 1.0),
            ("r1", 2, "h", "p2", "second", "", 1.0),
        ],
        columns=[
            "run",
            "step",
            "sample_id",
            "prompt_id",
            "prompt",
            "response",
            "reward",
        ],
    )


def feature_tables(df=None):
    df = rollouts() if df is None else df
    dense = df[["run", "sample_id"]].copy()
    dense["flag"] = df["sample_id"].isin(["a", "b"]).astype(float)
    dense["size"] = df["reward"].to_numpy()
    sparse = dense.loc[dense["flag"].eq(1), ["run", "sample_id"]].assign(
        feature="s:flag"
    )
    catalog = pd.DataFrame(
        [
            ("flag", "structural", "binary"),
            ("size", "structural", "continuous"),
            ("s:flag", "structural", "binary"),
            ("w:absent", "prose", "binary"),
        ],
        columns=["feature", "tier", "kind"],
    )
    return SimpleNamespace(dense=dense, sparse=sparse, catalog=catalog)


def run_aggregate(df, features):
    from rolloutdiff.series import aggregate

    return aggregate(df, features)


def test_prompt_means_weight_unequal_groups_equally_and_reward_uses_same_method():
    result = run_aggregate(rollouts(), feature_tables())
    step = result.series.set_index(["run", "step", "feature"])
    binary = step.loc[("r1", 0, "flag")]
    assert binary["mean"] == pytest.approx(1 / 3)
    assert binary["se"] == pytest.approx(1 / 3)
    assert binary["n_rollouts"] == 4
    assert binary["n_prompts"] == 2
    reward = step.loc[("r1", 0, "scalar:reward")]
    assert reward["mean"] == 9
    assert reward["se"] == pytest.approx(3)
    assert reward["tier"] == "structural"
    assert reward["kind"] == "continuous"
    assert_frame_equal(
        result.series.query("feature == 'size'")
        .drop(columns="feature")
        .reset_index(drop=True),
        result.series.query("feature == 'scalar:reward'")
        .drop(columns="feature")
        .reset_index(drop=True),
    )


def test_sparse_matches_dense_including_entirely_absent_steps_and_catalog_tokens():
    result = run_aggregate(rollouts(), feature_tables()).series
    assert_frame_equal(
        result.query("feature == 'flag'")
        .drop(columns="feature")
        .reset_index(drop=True),
        result.query("feature == 's:flag'")
        .drop(columns="feature")
        .reset_index(drop=True),
    )
    absent = result.query("feature == 'w:absent'").set_index("step")
    assert absent["mean"].tolist() == [0, 0, 0]
    assert absent.loc[0, "se"] == 0
    assert np.isnan(absent.loc[1, "se"])
    assert absent.loc[2, "se"] == 0
    assert len(result) == 15


def test_single_prompt_uncertainty_is_unknown_for_dense_sparse_and_reward():
    result = run_aggregate(rollouts(), feature_tables()).series
    single = result.query("step == 1")
    assert len(single) == 5
    assert single["se"].isna().all()
    assert single["n_prompts"].eq(1).all()
    assert single["n_rollouts"].eq(2).all()


def test_joins_use_sample_ids_and_ignore_shuffled_or_duplicate_dataframe_indices():
    df = rollouts()
    features = feature_tables()
    expected = run_aggregate(df, features)
    df = df.sample(frac=1, random_state=4)
    df.index = [7] * len(df)
    features.dense = features.dense.sample(frac=1, random_state=2)
    features.dense.index = [31] * len(features.dense)
    features.sparse = features.sparse.iloc[::-1]
    actual = run_aggregate(df, features)
    assert_frame_equal(actual.series, expected.series)
    assert_frame_equal(actual.prompt_schedule, expected.prompt_schedule)


def test_runs_with_shared_sample_ids_prompts_and_steps_are_never_pooled():
    one = rollouts().query("step == 0").copy()
    two = one.assign(run="r2", reward=30.0)
    df = pd.concat([one, two], ignore_index=True)
    features = feature_tables(df)
    features.dense.loc[features.dense["run"].eq("r2"), "flag"] = 1
    features.sparse = pd.concat(
        [
            features.sparse.query("run == 'r1'"),
            two[["run", "sample_id"]].assign(feature="s:flag"),
        ],
        ignore_index=True,
    )
    result = run_aggregate(df, features).series.set_index(["run", "feature"])
    assert result.loc[("r1", "flag"), "mean"] == pytest.approx(1 / 3)
    assert result.loc[("r2", "flag"), "mean"] == 1
    assert result.loc[("r2", "s:flag"), "mean"] == 1
    assert result.loc[("r2", "scalar:reward"), "mean"] == 30
    assert result["n_rollouts"].eq(4).all()
    assert result["n_prompts"].eq(2).all()


def test_prompt_schedule_has_counts_and_prompt_lengths_for_each_run_step_prompt():
    result = run_aggregate(rollouts(), feature_tables())
    assert result.prompt_schedule.to_dict("records") == [
        {
            "run": "r1",
            "step": 0,
            "prompt_id": "p1",
            "sampling_unit_id": "p1",
            "n_rollouts": 3,
            "prompt_chars": 5,
        },
        {
            "run": "r1",
            "step": 0,
            "prompt_id": "p2",
            "sampling_unit_id": "p2",
            "n_rollouts": 1,
            "prompt_chars": 6,
        },
        {
            "run": "r1",
            "step": 1,
            "prompt_id": "p1",
            "sampling_unit_id": "p1",
            "n_rollouts": 2,
            "prompt_chars": 5,
        },
        {
            "run": "r1",
            "step": 2,
            "prompt_id": "p1",
            "sampling_unit_id": "p1",
            "n_rollouts": 1,
            "prompt_chars": 5,
        },
        {
            "run": "r1",
            "step": 2,
            "prompt_id": "p2",
            "sampling_unit_id": "p2",
            "n_rollouts": 1,
            "prompt_chars": 6,
        },
    ]


def test_empty_inputs_preserve_output_schemas_without_inventing_steps():
    df = rollouts().iloc[:0]
    result = run_aggregate(df, feature_tables(df))
    assert result.series.empty
    assert result.prompt_schedule.empty
    assert list(result.series.columns) == [
        "run",
        "step",
        "feature",
        "tier",
        "kind",
        "mean",
        "se",
        "n_rollouts",
        "n_prompts",
        "n_sampling_units",
    ]
    assert list(result.prompt_schedule.columns) == [
        "run",
        "step",
        "prompt_id",
        "n_rollouts",
        "prompt_chars",
        "sampling_unit_id",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "dense_missing_key",
        "dense_unknown_key",
        "dense_duplicate_key",
        "dense_nan",
        "dense_inf",
        "sparse_unknown_key",
        "sparse_duplicate",
        "sparse_unknown_feature",
        "sparse_dense_feature",
        "catalog_duplicate",
        "catalog_missing_dense",
        "catalog_invalid_tier",
        "catalog_invalid_kind",
        "catalog_sparse_continuous",
        "dense_nonbinary",
        "dense_nonnumeric",
        "dense_complex",
        "catalog_reward_collision",
    ],
)
def test_invalid_feature_tables_fail_closed(mutation):
    features = feature_tables()
    if mutation == "dense_missing_key":
        features.dense = features.dense.iloc[1:]
    elif mutation == "dense_unknown_key":
        features.dense.loc[0, "sample_id"] = "missing"
    elif mutation == "dense_duplicate_key":
        features.dense = pd.concat(
            [features.dense, features.dense.iloc[:1]], ignore_index=True
        )
    elif mutation in {"dense_nan", "dense_inf"}:
        features.dense.loc[0, "size"] = np.nan if mutation == "dense_nan" else np.inf
    elif mutation == "sparse_unknown_key":
        features.sparse.loc[0, "sample_id"] = "missing"
    elif mutation == "sparse_duplicate":
        features.sparse = pd.concat(
            [features.sparse, features.sparse.iloc[:1]], ignore_index=True
        )
    elif mutation == "sparse_unknown_feature":
        features.sparse.loc[0, "feature"] = "unknown"
    elif mutation == "sparse_dense_feature":
        features.sparse.loc[0, "feature"] = "flag"
    elif mutation == "catalog_duplicate":
        features.catalog = pd.concat(
            [features.catalog, features.catalog.iloc[:1]], ignore_index=True
        )
    elif mutation == "catalog_missing_dense":
        features.catalog = features.catalog.query("feature != 'size'")
    elif mutation == "catalog_invalid_tier":
        features.catalog.loc[0, "tier"] = "other"
    elif mutation == "catalog_invalid_kind":
        features.catalog.loc[0, "kind"] = "other"
    elif mutation == "catalog_sparse_continuous":
        features.catalog.loc[2, "kind"] = "continuous"
    elif mutation == "dense_nonbinary":
        features.dense.loc[0, "flag"] = 0.5
    elif mutation == "dense_nonnumeric":
        features.dense["size"] = "value"
    elif mutation == "dense_complex":
        features.dense["size"] = 1 + 2j
    elif mutation == "catalog_reward_collision":
        features.catalog.loc[2, "feature"] = "scalar:reward"
    with pytest.raises(ValueError):
        run_aggregate(rollouts(), features)


def test_missing_prompt_id_is_rejected_instead_of_silently_dropped():
    df = rollouts()
    df.loc[0, "prompt_id"] = None
    with pytest.raises(ValueError):
        run_aggregate(df, feature_tables())


def test_empty_feature_extraction_can_be_aggregated():
    from rolloutdiff.features import featurize

    df = rollouts().iloc[:0]
    result = run_aggregate(df, featurize(df))
    assert result.series.empty
    assert result.prompt_schedule.empty


def test_identical_prompt_proportions_have_exactly_zero_sparse_se():
    # Every prompt mean is 1/3 for both the dense and sparse feature; a one-pass
    # variance would leave roundoff above the detector's 1e-9 numerical floor.
    rows, dense_rows, sparse_rows = [], [], []
    for prompt in range(13):
        for copy in range(3):
            sample = f"{prompt}:{copy}"
            rows.append(("r", 0, sample, f"p{prompt}", "q", "", 0.0))
            dense_rows.append(
                {"run": "r", "sample_id": sample, "flag": float(copy == 0)}
            )
            if copy == 0:
                sparse_rows.append({"run": "r", "sample_id": sample, "feature": "s"})
    df = pd.DataFrame(rows, columns=rollouts().columns)
    features = SimpleNamespace(
        dense=pd.DataFrame(dense_rows),
        sparse=pd.DataFrame(sparse_rows),
        catalog=pd.DataFrame(
            [("flag", "structural", "binary"), ("s", "structural", "binary")],
            columns=["feature", "tier", "kind"],
        ),
    )
    series = run_aggregate(df, features).series.set_index("feature")
    assert series.loc["flag", "mean"] == series.loc["s", "mean"]
    assert series.loc["flag", "se"] == 0.0
    assert series.loc["s", "se"] == 0.0
