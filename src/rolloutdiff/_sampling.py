"""Explicit dependence units, separate from canonical exact prompt identities."""

from __future__ import annotations

import numpy as np
import pandas as pd

_KEYS = ["run", "prompt_id"]
_COLUMNS = [*_KEYS, "sampling_unit_id"]


def _prompt_keys(rollouts: pd.DataFrame) -> pd.DataFrame:
    if (
        not isinstance(rollouts, pd.DataFrame)
        or not rollouts.columns.is_unique
        or not set(_KEYS).issubset(rollouts.columns)
    ):
        raise ValueError("rollouts must contain unique run and prompt_id columns")
    keys = rollouts[_KEYS].drop_duplicates()
    for name in _KEYS:
        if (
            not keys[name]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"{name} must contain nonblank string identifiers")
    return keys.sort_values(_KEYS).reset_index(drop=True)


def resolve_sampling_units(
    rollouts: pd.DataFrame,
    mapping: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return a stable unit per (run, exact prompt ID).

    Supplied maps must exactly cover observed keys. Units are scoped by run and
    may group prompts.
    """
    keys = _prompt_keys(rollouts)
    if mapping is None:
        return keys.assign(sampling_unit_id=keys.prompt_id)
    if (
        not isinstance(mapping, pd.DataFrame)
        or not mapping.columns.is_unique
        or set(mapping.columns) != set(_COLUMNS)
    ):
        raise ValueError(f"sampling_units must contain exactly {_COLUMNS}")
    units = mapping[_COLUMNS].copy()
    for name in _COLUMNS:
        if (
            not units[name]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(
                f"sampling_units {name} must contain nonblank string identifiers"
            )
    if units.duplicated(_KEYS).any():
        raise ValueError("sampling_units contains duplicate (run, prompt_id) keys")
    expected = pd.MultiIndex.from_frame(keys)
    actual = pd.MultiIndex.from_frame(units[_KEYS])
    if len(actual) != len(expected) or not actual.isin(expected).all():
        raise ValueError("sampling_units keys must exactly match rollout prompt keys")
    return units.sort_values(_KEYS).reset_index(drop=True)


def subset_sampling_units(
    mapping: pd.DataFrame, rollouts: pd.DataFrame
) -> pd.DataFrame:
    """Restrict a validated map to a rollout subset, checking full subset coverage."""
    keys = _prompt_keys(rollouts)
    if (
        not isinstance(mapping, pd.DataFrame)
        or not mapping.columns.is_unique
        or set(mapping.columns) != set(_COLUMNS)
    ):
        raise ValueError(f"sampling_units must contain exactly {_COLUMNS}")
    selected = pd.MultiIndex.from_frame(mapping[_KEYS]).isin(
        pd.MultiIndex.from_frame(keys)
    )
    return resolve_sampling_units(keys, mapping.loc[selected])


def cluster_standard_error(values, unit_ids) -> float | None:
    """CR1 SE of an equal-prompt mean over explicit sampling units.

    For P prompts and U units: sqrt(U/(U-1) * sum(unit residual sums**2)) / P.
    Unequal unit sizes retain prompt weights. Fewer than two units returns None;
    small-cluster inference remains uncalibrated.
    """
    data = np.asarray(values)
    units = np.asarray(unit_ids, dtype=object)
    if (
        data.ndim != 1
        or units.ndim != 1
        or len(data) != len(units)
        or data.dtype.kind not in "biuf"
        or not np.isfinite(data).all()
    ):
        raise ValueError(
            "cluster SE requires aligned finite real values and unit identifiers"
        )
    if not all(isinstance(value, str) and bool(value.strip()) for value in units):
        raise ValueError("cluster SE unit identifiers must be nonblank strings")
    if not len(data):
        return None
    codes, names = pd.factorize(units, sort=True)
    count = len(names)
    if count < 2:
        return None
    # Preserve small differences by shifting before scaling. If subtraction
    # overflows, scale raw values first. Neither path squares raw data.
    data = data.astype(float)
    with np.errstate(over="ignore", invalid="ignore"):
        shifted = data - data[0]
    centered_data = shifted if np.isfinite(shifted).all() else data
    scale = float(np.max(np.abs(centered_data)))
    if scale == 0:
        return 0.0
    normalized = centered_data / scale
    residuals = normalized - normalized.mean()
    sums = np.bincount(codes, weights=residuals, minlength=count) / len(data)
    normalized_se = np.sqrt(count / (count - 1) * np.dot(sums, sums))
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(normalized_se * scale)
    if not np.isfinite(result):
        raise ValueError("numerical overflow computing cluster standard error")
    return result
