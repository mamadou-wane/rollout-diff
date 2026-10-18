"""Exact-prompt-balanced series with sampling-unit uncertainty."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_numeric_dtype

from ._sampling import cluster_standard_error, resolve_sampling_units
from .features import FeatureTables
from .schema import validate_rollouts

_KEYS = ["run", "sample_id"]
_GROUPS = ["run", "step", "prompt_id"]
_STEPS = ["run", "step"]
_SERIES_COLUMNS = [
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
_SCHEDULE_COLUMNS = [
    "run",
    "step",
    "prompt_id",
    "n_rollouts",
    "prompt_chars",
    "sampling_unit_id",
]


@dataclass(frozen=True)
class Aggregation:
    series: pd.DataFrame
    prompt_schedule: pd.DataFrame


def _require_columns(table: pd.DataFrame, columns: list[str], name: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise ValueError(f"{name} must be a DataFrame")
    if not table.columns.is_unique:
        raise ValueError(f"{name} contains duplicate columns")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing columns: {sorted(missing)}")


def _validate_features(rollouts: pd.DataFrame, features: FeatureTables) -> list[str]:
    dense, sparse, catalog = features.dense, features.sparse, features.catalog
    _require_columns(dense, _KEYS, "dense features")
    _require_columns(sparse, [*_KEYS, "feature"], "sparse features")
    _require_columns(catalog, ["feature", "tier", "kind"], "feature catalog")
    if (
        not catalog["feature"]
        .map(lambda name: isinstance(name, str) and bool(name))
        .all()
    ):
        raise ValueError("catalog feature names must be nonempty strings")
    if catalog["feature"].duplicated().any():
        raise ValueError("catalog contains duplicate feature names")
    if not catalog["tier"].isin(["structural", "prose"]).all():
        raise ValueError("catalog tier must be structural or prose")
    if not catalog["kind"].isin(["binary", "continuous"]).all():
        raise ValueError("catalog kind must be binary or continuous")
    if catalog["feature"].eq("scalar:reward").any():
        raise ValueError("scalar:reward is reserved for the rollout reward")

    dense_names = [name for name in dense.columns if name not in _KEYS]
    metadata = catalog.set_index("feature")
    if not set(dense_names).issubset(metadata.index):
        raise ValueError("dense feature columns are missing from the catalog")
    sparse_names = catalog.loc[
        ~catalog["feature"].isin(dense_names), "feature"
    ].tolist()
    if not metadata.loc[sparse_names, "kind"].eq("binary").all():
        raise ValueError("sparse features must be binary")
    if not sparse["feature"].isin(sparse_names).all():
        raise ValueError("sparse feature names must belong to the sparse catalog")

    expected_keys = pd.MultiIndex.from_frame(rollouts[_KEYS])
    dense_keys = pd.MultiIndex.from_frame(dense[_KEYS])
    if not dense_keys.is_unique:
        raise ValueError("dense features contain duplicate rollout keys")
    if (
        len(dense_keys) != len(expected_keys)
        or not dense_keys.isin(expected_keys).all()
    ):
        raise ValueError("dense feature keys must exactly match rollout keys")
    if sparse.duplicated([*_KEYS, "feature"]).any():
        raise ValueError("sparse features contain duplicate rollout/feature rows")
    if not pd.MultiIndex.from_frame(sparse[_KEYS]).isin(expected_keys).all():
        raise ValueError("sparse features contain unknown rollout keys")
    for name in dense_names:
        values = dense[name]
        if (
            not is_numeric_dtype(values.dtype)
            or is_complex_dtype(values.dtype)
            or not np.isfinite(values.to_numpy(dtype=float)).all()
        ):
            raise ValueError(
                f"dense feature {name!r} must contain finite real numeric values"
            )
        if metadata.loc[name, "kind"] == "binary" and not values.isin([0, 1]).all():
            raise ValueError(
                f"binary dense feature {name!r} must contain only zero or one"
            )
    return dense_names


def _aggregate_dense(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    dense_names: list[str],
    units: pd.DataFrame,
    counts: pd.DataFrame,
    singleton_units: bool,
) -> list[pd.DataFrame]:
    # Internal names keep caller feature names from colliding with group keys.
    internal_names = {name: f"__value_{i}" for i, name in enumerate(dense_names)}
    dense = (
        rollouts[[*_KEYS, "step", "prompt_id", "reward"]]
        .rename(columns={"reward": "__reward"})
        .merge(
            features.dense.rename(columns=internal_names),
            on=_KEYS,
            validate="one_to_one",
        )
    )
    value_names = {**internal_names, "scalar:reward": "__reward"}
    prompt_means = dense.groupby(_GROUPS, sort=True, observed=True)[
        list(value_names.values())
    ].mean()
    step_means = prompt_means.groupby(_STEPS, sort=True, observed=True)
    means, deviations = step_means.mean(), step_means.std(ddof=1)
    clustered_errors = None
    if not singleton_units:
        grouped = prompt_means.reset_index().merge(
            units, on=["run", "prompt_id"], validate="many_to_one"
        )
        errors = []
        for (run, step), group in grouped.groupby(_STEPS, sort=True, observed=True):
            errors.append(
                {
                    "run": run,
                    "step": step,
                    **{
                        internal: cluster_standard_error(
                            group[internal].to_numpy(),
                            group.sampling_unit_id.to_numpy(),
                        )
                        for internal in value_names.values()
                    },
                }
            )
        clustered_errors = pd.DataFrame(errors).set_index(_STEPS)
    parts = []
    for name, internal in value_names.items():
        part = pd.DataFrame(
            {
                "mean": means[internal],
                "se": deviations[internal],
            }
        ).reset_index()
        part = part.merge(counts, on=_STEPS, validate="one_to_one")
        if singleton_units:
            part["se"] /= np.sqrt(part["n_prompts"])
        else:
            part["se"] = clustered_errors[internal].to_numpy(dtype=float)
        part["feature"] = name
        parts.append(part)

    return parts


def _aggregate_sparse(
    rollouts: pd.DataFrame,
    sparse: pd.DataFrame,
    sparse_catalog: pd.DataFrame,
    schedule: pd.DataFrame,
    counts: pd.DataFrame,
    singleton_units: bool,
) -> pd.DataFrame:
    sparse_result = counts.merge(sparse_catalog[["feature"]], how="cross")
    prompt_counts = pd.DataFrame()
    if not sparse.empty:
        present = sparse[[*_KEYS, "feature"]].merge(
            rollouts[[*_KEYS, "step", "prompt_id"]], on=_KEYS, validate="many_to_one"
        )
        prompt_counts = (
            present.groupby([*_GROUPS, "feature"], observed=True, sort=False)
            .size()
            .rename("present")
            .reset_index()
        ).merge(
            schedule[[*_GROUPS, "n_rollouts", "sampling_unit_id"]],
            on=_GROUPS,
            validate="many_to_one",
        )
        prompt_counts["p"] = prompt_counts["present"] / prompt_counts["n_rollouts"]
    if singleton_units:
        return _sparse_prompt_errors(sparse_result, prompt_counts)
    return _sparse_cluster_errors(sparse_result, prompt_counts, schedule)


def _sparse_prompt_errors(
    sparse_result: pd.DataFrame, prompt_counts: pd.DataFrame
) -> pd.DataFrame:
    if prompt_counts.empty:
        sparse_result["sum_p"] = sparse_result["sum_p2"] = 0.0
    else:
        prompt_counts["p2"] = prompt_counts["p"] ** 2
        moments = (
            prompt_counts.groupby([*_STEPS, "feature"], observed=True, sort=False)
            .agg(sum_p=("p", "sum"), sum_p2=("p2", "sum"))
            .reset_index()
        )
        sparse_result = sparse_result.merge(
            moments, on=[*_STEPS, "feature"], how="left", validate="one_to_one"
        )
        sparse_result[["sum_p", "sum_p2"]] = sparse_result[["sum_p", "sum_p2"]].fillna(
            0.0
        )
    groups = sparse_result["n_prompts"].to_numpy(dtype=float)
    first, second = (
        sparse_result["sum_p"].to_numpy(),
        sparse_result["sum_p2"].to_numpy(),
    )
    sparse_result["mean"] = first / groups
    variance = np.maximum(second - first**2 / groups, 0.0)
    sparse_result["se"] = np.sqrt(
        np.divide(
            variance,
            groups * (groups - 1),
            out=np.full(len(groups), np.nan),
            where=groups > 1,
        )
    )
    return sparse_result


def _sparse_cluster_errors(
    sparse_result: pd.DataFrame, prompt_counts: pd.DataFrame, schedule: pd.DataFrame
) -> pd.DataFrame:
    # Sum exact-prompt proportions within units. Only units where a
    # feature occurs are materialized; absent-unit residuals use their
    # squared prompt counts, including entirely absent features.
    unit_keys = [*_STEPS, "sampling_unit_id"]
    sizes = (
        schedule.groupby(unit_keys, sort=False, observed=True)
        .size()
        .rename("unit_prompts")
        .reset_index()
    )
    sizes["unit_prompts2"] = sizes.unit_prompts.astype(float) ** 2
    totals = (
        sizes.groupby(_STEPS, sort=False)
        .unit_prompts2.sum()
        .rename("total_prompts2")
        .reset_index()
    )
    sparse_result = sparse_result.merge(totals, on=_STEPS, validate="many_to_one")
    if prompt_counts.empty:
        sparse_result["mean"] = 0.0
        sparse_result["residual2"] = 0.0
    else:
        sums = (
            prompt_counts.groupby([*unit_keys, "feature"], sort=False, observed=True)
            .p.sum()
            .rename("sum_p")
            .reset_index()
        )
        # Keep exact-prompt summation order for stable point estimates.
        first = (
            prompt_counts.groupby([*_STEPS, "feature"], sort=False, observed=True)
            .p.sum()
            .rename("sum_p")
            .reset_index()
        )
        sparse_result = sparse_result.merge(
            first, on=[*_STEPS, "feature"], how="left", validate="one_to_one"
        )
        sparse_result["mean"] = (
            sparse_result.sum_p.fillna(0.0) / sparse_result.n_prompts
        )
        sums = sums.merge(sizes, on=unit_keys, validate="many_to_one").merge(
            sparse_result[[*_STEPS, "feature", "mean"]],
            on=[*_STEPS, "feature"],
            validate="many_to_one",
        )
        sums["active_residual2"] = (sums.sum_p - sums.unit_prompts * sums["mean"]) ** 2
        residuals = (
            sums.groupby([*_STEPS, "feature"], sort=False)
            .agg(
                active_residual2=("active_residual2", "sum"),
                active_prompts2=("unit_prompts2", "sum"),
            )
            .reset_index()
        )
        sparse_result = sparse_result.merge(
            residuals, on=[*_STEPS, "feature"], how="left", validate="one_to_one"
        )
        sparse_result[["active_residual2", "active_prompts2"]] = sparse_result[
            ["active_residual2", "active_prompts2"]
        ].fillna(0.0)
        sparse_result["residual2"] = (
            sparse_result.active_residual2
            + (sparse_result.total_prompts2 - sparse_result.active_prompts2)
            * sparse_result["mean"] ** 2
        )
    groups = sparse_result.n_sampling_units.to_numpy(dtype=float)
    variance = (
        sparse_result.residual2.to_numpy()
        / sparse_result.n_prompts.to_numpy(dtype=float) ** 2
    )
    sparse_result["se"] = np.sqrt(
        np.divide(
            groups * variance,
            groups - 1,
            out=np.full(len(groups), np.nan),
            where=groups > 1,
        )
    )
    return sparse_result


def aggregate(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    sampling_units: pd.DataFrame | None = None,
) -> Aggregation:
    """Aggregate dense, sparse, and reward values by run and step.

    Weight exact-prompt means equally. Default SE is SD / sqrt(prompt count);
    explicit units use CR1 SE of the same mean. Fewer than two units yields NaN;
    measured zero variance stays zero. SE excludes detector floors. Small-unit
    inference remains uncalibrated. Align features by (run, sample_id).
    """
    rollouts = validate_rollouts(rollouts)
    dense_names = _validate_features(rollouts, features)
    units = resolve_sampling_units(rollouts, sampling_units)
    singleton_units = not units.duplicated(["run", "sampling_unit_id"]).any()
    if rollouts.empty:
        return Aggregation(
            pd.DataFrame(columns=_SERIES_COLUMNS),
            pd.DataFrame(columns=_SCHEDULE_COLUMNS),
        )

    schedule = (
        rollouts.assign(prompt_chars=rollouts["prompt"].str.len())
        .groupby(_GROUPS, sort=True, observed=True)
        .agg(n_rollouts=("sample_id", "size"), prompt_chars=("prompt_chars", "first"))
        .reset_index()
        .merge(units, on=["run", "prompt_id"], validate="many_to_one")
    )
    counts = (
        schedule.groupby(_STEPS, sort=True, observed=True)
        .agg(
            n_rollouts=("n_rollouts", "sum"),
            n_prompts=("prompt_id", "size"),
            n_sampling_units=("sampling_unit_id", "nunique"),
        )
        .reset_index()
    )
    parts = _aggregate_dense(
        rollouts, features, dense_names, units, counts, singleton_units
    )

    sparse_catalog = features.catalog.loc[
        ~features.catalog["feature"].isin(dense_names)
    ]
    if not sparse_catalog.empty:
        parts.append(
            _aggregate_sparse(
                rollouts,
                features.sparse,
                sparse_catalog,
                schedule,
                counts,
                singleton_units,
            )
        )

    catalog = pd.concat(
        [
            features.catalog[["feature", "tier", "kind"]],
            pd.DataFrame(
                [
                    {
                        "feature": "scalar:reward",
                        "tier": "structural",
                        "kind": "continuous",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    series = pd.concat(parts, ignore_index=True).merge(
        catalog, on="feature", validate="many_to_one"
    )
    series = (
        series[_SERIES_COLUMNS].sort_values([*_STEPS, "feature"]).reset_index(drop=True)
    )
    return Aggregation(series, schedule[_SCHEDULE_COLUMNS])
