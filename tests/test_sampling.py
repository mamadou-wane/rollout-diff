import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rolloutdiff.features import FeatureTables


def fixture():
    rows = []
    for step in (0, 1):
        for prompt, copies, value in [
            ("p0", 4, 0.0),
            ("p1", 1, 0.0),
            ("p2", 2, 0.0),
            ("p3", 1, 1.0),
        ]:
            for copy in range(copies):
                rows.append(
                    {
                        "run": "r",
                        "step": step,
                        "sample_id": f"{step}:{prompt}:{copy}",
                        "prompt_id": prompt,
                        "prompt": prompt,
                        "response": "",
                        "reward": 2.0 + 4.0 * value,
                        "value": value,
                    }
                )
    frame = pd.DataFrame(rows)
    dense = frame[["run", "sample_id", "value"]].rename(columns={"value": "flag"})
    dense["size"] = frame.reward
    sparse = dense.loc[dense.flag.eq(1), ["run", "sample_id"]].assign(feature="s:flag")
    catalog = pd.DataFrame(
        [
            ("flag", "structural", "binary"),
            ("size", "structural", "continuous"),
            ("s:flag", "structural", "binary"),
            ("w:absent", "prose", "binary"),
        ],
        columns=["feature", "tier", "kind"],
    )
    mapping = pd.DataFrame(
        [("r", "p0", "a"), ("r", "p1", "a"), ("r", "p2", "a"), ("r", "p3", "b")],
        columns=["run", "prompt_id", "sampling_unit_id"],
    )
    return frame.drop(columns="value"), FeatureTables(dense, sparse, catalog), mapping


def test_cluster_se_preserves_exact_prompt_estimator_with_unequal_units():
    from rolloutdiff.series import aggregate

    rows, features, units = fixture()
    original = rows.copy(deep=True)
    result = aggregate(rows, features, sampling_units=units)
    step = result.series.loc[result.series.step.eq(0)].set_index("feature")
    assert (
        step.loc["flag", "mean"] == 0.25
    )  # Exact prompt weights yield .25; source-unit weights would yield .5.
    assert step.loc["flag", "se"] == 0.375
    assert step.loc["scalar:reward", "mean"] == 3.0
    assert step.loc["scalar:reward", "se"] == 1.5
    assert step.n_prompts.eq(4).all() and step.n_sampling_units.eq(2).all()
    assert step.n_rollouts.eq(8).all()
    assert step.loc["w:absent", "mean"] == step.loc["w:absent", "se"] == 0
    assert_frame_equal(rows, original)
    assert_frame_equal(
        step.loc[["flag"]].reset_index(drop=True),
        step.loc[["s:flag"]].reset_index(drop=True),
    )
    assert_frame_equal(
        step.loc[["size"]].reset_index(drop=True),
        step.loc[["scalar:reward"]].reset_index(drop=True),
    )
    assert result.prompt_schedule.prompt_id.nunique() == 4
    assert result.prompt_schedule.sampling_unit_id.nunique() == 2


def test_many_prompts_in_one_unit_have_unavailable_uncertainty():
    from rolloutdiff._series_validation import validate_series
    from rolloutdiff.series import aggregate

    rows, features, units = fixture()
    result = aggregate(
        rows, features, sampling_units=units.assign(sampling_unit_id="only")
    )
    assert result.series.n_prompts.eq(4).all()
    assert result.series.n_sampling_units.eq(1).all()
    assert result.series.se.isna().all()
    validated = validate_series(result.series)
    assert validated.se.isna().all() and validated.n_sampling_units.eq(1).all()


def test_identity_mapping_keeps_existing_results_and_actual_counts():
    from rolloutdiff.series import aggregate

    rows, features, _ = fixture()
    mapping = (
        rows[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=lambda x: x.prompt_id)
    )
    default = aggregate(rows, features)
    explicit = aggregate(rows, features, sampling_units=mapping)
    assert_frame_equal(default.series, explicit.series, check_exact=True)
    assert_frame_equal(
        default.prompt_schedule, explicit.prompt_schedule, check_exact=True
    )
    assert default.series.n_sampling_units.equals(default.series.n_prompts)
    assert default.series.loc[default.series.feature.eq("flag"), "se"].eq(0.25).all()


def test_mapping_and_rows_reordering_unit_renaming_and_run_isolation():
    from rolloutdiff.series import aggregate

    rows, features, mapping = fixture()
    expected = aggregate(rows, features, sampling_units=mapping)
    changed = mapping.sample(frac=1, random_state=1).replace(
        {"sampling_unit_id": {"a": "z", "b": "y"}}
    )
    actual = aggregate(
        rows.sample(frac=1, random_state=2), features, sampling_units=changed
    )
    assert_frame_equal(actual.series, expected.series)
    both = pd.concat([rows, rows.assign(run="r2")], ignore_index=True)
    both_features = FeatureTables(
        pd.concat([features.dense, features.dense.assign(run="r2")]),
        pd.concat([features.sparse, features.sparse.assign(run="r2")]),
        features.catalog,
    )
    both_mapping = pd.concat(
        [mapping, mapping.assign(run="r2", sampling_unit_id="only")]
    )
    separate = aggregate(both, both_features, sampling_units=both_mapping)
    assert_frame_equal(
        separate.series.loc[separate.series.run.eq("r")].reset_index(drop=True),
        expected.series,
    )
    assert separate.series.loc[separate.series.run.eq("r2"), "se"].isna().all()


def test_resolve_and_subset_use_keys_and_preserve_complete_mapping():
    from rolloutdiff._sampling import resolve_sampling_units, subset_sampling_units

    rows, _, mapping = fixture()
    resolved = resolve_sampling_units(rows, mapping.iloc[::-1])
    assert_frame_equal(resolved, mapping)
    subset = rows.loc[rows.prompt_id.isin(["p0", "p3"])]
    assert_frame_equal(
        subset_sampling_units(resolved, subset),
        mapping.iloc[[0, 3]].reset_index(drop=True),
    )
    assert_frame_equal(
        resolve_sampling_units(rows),
        rows[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=lambda x: x.prompt_id)
        .reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "unknown",
        "duplicate",
        "conflicting",
        "blank",
        "null",
        "numeric",
        "extra_column",
        "missing_column",
        "duplicate_column",
        "not_frame",
    ],
)
def test_invalid_mapping_fails_closed(mutation):
    from rolloutdiff.series import aggregate

    rows, features, mapping = fixture()
    if mutation == "missing":
        mapping = mapping.iloc[:-1]
    elif mutation == "unknown":
        mapping.loc[0, "prompt_id"] = "unknown"
    elif mutation == "duplicate":
        mapping = pd.concat([mapping, mapping.iloc[:1]])
    elif mutation == "conflicting":
        mapping = pd.concat(
            [mapping, mapping.iloc[:1].assign(sampling_unit_id="other")]
        )
    elif mutation == "blank":
        mapping.loc[0, "sampling_unit_id"] = " "
    elif mutation == "null":
        mapping.loc[0, "sampling_unit_id"] = None
    elif mutation == "numeric":
        mapping.loc[0, "sampling_unit_id"] = 1
    elif mutation == "extra_column":
        mapping["step"] = 0
    elif mutation == "missing_column":
        mapping = mapping.drop(columns="sampling_unit_id")
    elif mutation == "duplicate_column":
        mapping.columns = ["run", "prompt_id", "prompt_id"]
    elif mutation == "not_frame":
        mapping = {}
    with pytest.raises(ValueError):
        aggregate(rows, features, sampling_units=mapping)


def test_cluster_standard_error_matches_explicit_cluster_residuals():
    from rolloutdiff._sampling import cluster_standard_error

    rng = np.random.default_rng(20261018)
    for n in (4, 10, 30):
        values = rng.normal(size=n)
        units = np.array([f"u{i % 3}" for i in range(n)])
        expected = np.sqrt(
            3
            / 2
            * sum(
                np.sum(values[units == u] - values.mean()) ** 2
                for u in np.unique(units)
            )
            / n**2
        )
        assert cluster_standard_error(values, units) == pytest.approx(expected)
        assert cluster_standard_error(values + 100, units) == pytest.approx(expected)
        assert cluster_standard_error(values * 1e200, units) == pytest.approx(
            expected * 1e200
        )
    assert cluster_standard_error([0, 0, 0, 1], ["a", "a", "a", "b"]) == 0.375
    assert cluster_standard_error([1, 2, 3], ["a"] * 3) is None
    assert cluster_standard_error([], []) is None
    assert cluster_standard_error([1, 1], ["a", "b"]) == 0


@pytest.mark.parametrize(
    "values, units",
    [
        ([1, float("nan")], ["a", "b"]),
        ([1, float("inf")], ["a", "b"]),
        ([1 + 2j, 3], ["a", "b"]),
        ([1], ["a", "b"]),
        ([[1, 2]], ["a"]),
        ([1, 2], ["a", None]),
    ],
)
def test_cluster_standard_error_rejects_invalid_data(values, units):
    from rolloutdiff._sampling import cluster_standard_error

    with pytest.raises(ValueError):
        cluster_standard_error(values, units)


def test_validation_accepts_legacy_series_but_preserves_explicit_units():
    from rolloutdiff._series_validation import validate_series
    from rolloutdiff.series import aggregate

    rows, features, units = fixture()
    result = aggregate(rows, features, sampling_units=units).series
    assert validate_series(result).n_sampling_units.eq(2).all()
    legacy = result.drop(columns="n_sampling_units")
    assert validate_series(legacy).n_sampling_units.eq(4).all()
    for count in (0, 5, 1.5):
        with pytest.raises(ValueError):
            validate_series(result.assign(n_sampling_units=count))
    with pytest.raises(ValueError):
        validate_series(result.assign(n_sampling_units=1))


def test_cluster_standard_error_retains_small_variation_under_large_offset():
    from rolloutdiff._sampling import cluster_standard_error

    values = np.array([0.0, 0.25, 0.5, 1.0])
    units = ["a", "a", "b", "b"]
    expected = cluster_standard_error(values, units)
    assert cluster_standard_error(values + 2.0**40, units) == expected
    assert np.isfinite(cluster_standard_error([-1e308, 1e308], ["a", "b"]))


def test_sparse_cluster_errors_match_dense_oracle_with_absent_units_and_steps():
    from rolloutdiff._sampling import cluster_standard_error
    from rolloutdiff.series import aggregate

    rng = np.random.default_rng(127)
    rows, values = [], []
    for step in range(4):
        for prompt in range(9):
            for copy in range(1 + prompt % 4):
                sample = f"{step}:{prompt}:{copy}"
                value = float(rng.random() < 0.4) if step else 0.0
                rows.append(
                    {
                        "run": "r",
                        "step": step,
                        "sample_id": sample,
                        "prompt_id": f"p{prompt}",
                        "prompt": f"p{prompt}",
                        "response": "",
                        "reward": value,
                    }
                )
                values.append({"run": "r", "sample_id": sample, "flag": value})
    rows, dense = pd.DataFrame(rows), pd.DataFrame(values)
    sparse = dense.loc[dense.flag.eq(1), ["run", "sample_id"]].assign(feature="s:flag")
    features = FeatureTables(
        dense,
        sparse,
        pd.DataFrame(
            [
                ("flag", "structural", "binary"),
                ("s:flag", "structural", "binary"),
                ("s:never", "structural", "binary"),
            ],
            columns=["feature", "tier", "kind"],
        ),
    )
    mapping = (
        rows[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=["a"] * 5 + ["b"] * 3 + ["c"])
    )
    result = aggregate(rows, features, sampling_units=mapping).series
    actual = result.set_index(["step", "feature"])
    for step, group in rows.groupby("step"):
        prompt_means = group.groupby("prompt_id").reward.mean()
        units = mapping.set_index("prompt_id").sampling_unit_id.loc[prompt_means.index]
        se = cluster_standard_error(prompt_means.to_numpy(), units.to_numpy())
        for feature in ("flag", "s:flag", "scalar:reward"):
            assert actual.loc[(step, feature), "mean"] == pytest.approx(
                prompt_means.mean()
            )
            assert actual.loc[(step, feature), "se"] == pytest.approx(se)
        assert actual.loc[(step, "s:never"), "se"] == 0.0


def test_explicit_mapping_for_empty_rollouts_must_be_empty():
    from rolloutdiff.series import aggregate

    rows, features, mapping = fixture()
    empty_features = FeatureTables(
        features.dense.iloc[:0], features.sparse.iloc[:0], features.catalog
    )
    result = aggregate(rows.iloc[:0], empty_features, sampling_units=mapping.iloc[:0])
    assert result.series.empty and "n_sampling_units" in result.series
    with pytest.raises(ValueError):
        aggregate(rows.iloc[:0], empty_features, sampling_units=mapping)


def test_non_dyadic_sparse_point_means_remain_bit_identical_with_unit_mapping():
    from rolloutdiff.series import aggregate

    denominators = [5, 3, 12, 3, 13, 3, 2, 12, 2, 10]
    numerators = [4, 1, 1, 2, 10, 2, 2, 8, 1, 10]
    rows, dense = [], []
    for prompt, (total, positive) in enumerate(
        zip(denominators, numerators, strict=False)
    ):
        for copy in range(total):
            sample = f"{prompt}:{copy}"
            rows.append(
                {
                    "run": "r",
                    "step": 0,
                    "sample_id": sample,
                    "prompt_id": f"p{prompt}",
                    "prompt": f"p{prompt}",
                    "response": "",
                    "reward": float(copy < positive),
                }
            )
            dense.append(
                {"run": "r", "sample_id": sample, "flag": float(copy < positive)}
            )
    rows, dense = pd.DataFrame(rows), pd.DataFrame(dense)
    features = FeatureTables(
        dense,
        dense.loc[dense.flag.eq(1), ["run", "sample_id"]].assign(feature="s:flag"),
        pd.DataFrame(
            [("flag", "structural", "binary"), ("s:flag", "structural", "binary")],
            columns=["feature", "tier", "kind"],
        ),
    )
    units = (
        rows[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=["a"] * 5 + ["b"] * 3 + ["c"] * 2)
    )
    original = aggregate(rows, features).series
    grouped = aggregate(rows, features, sampling_units=units).series
    assert_frame_equal(
        original.drop(columns=["se", "n_sampling_units"]),
        grouped.drop(columns=["se", "n_sampling_units"]),
        check_exact=True,
    )
