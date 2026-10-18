"""Shared validation of step-series identities, values, and measured uncertainty."""

import numpy as np
import pandas as pd
from pandas.api.types import is_complex_dtype, is_integer_dtype, is_numeric_dtype

_REQUIRED = [
    "run",
    "step",
    "feature",
    "tier",
    "kind",
    "mean",
    "se",
    "n_rollouts",
    "n_prompts",
]


def validate_series(series: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(series, pd.DataFrame) or not series.columns.is_unique:
        raise ValueError("series must be a DataFrame with unique columns")
    if set(_REQUIRED) - set(series.columns):
        raise ValueError("series is missing required aggregate columns")
    frame = series[_REQUIRED].copy()
    frame["n_sampling_units"] = series.get("n_sampling_units", frame.n_prompts)
    if frame.empty:
        return frame.reset_index(drop=True)
    for name in ("run", "feature"):
        if not frame[name].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"{name} must contain nonempty strings")
    for name in ("step", "n_rollouts", "n_prompts", "n_sampling_units"):
        if not is_integer_dtype(frame[name].dtype) or frame[name].isna().any():
            raise ValueError(f"{name} must contain integers")
        if (frame[name] < (0 if name == "step" else 1)).any():
            raise ValueError(f"invalid {name}")
    if (frame.n_sampling_units > frame.n_prompts).any():
        raise ValueError("n_sampling_units cannot exceed n_prompts")
    if (frame.n_prompts > frame.n_rollouts).any():
        raise ValueError("n_prompts cannot exceed n_rollouts")
    for name in ("mean", "se"):
        if not is_numeric_dtype(frame[name].dtype) or is_complex_dtype(
            frame[name].dtype
        ):
            raise ValueError(f"{name} must contain real numbers")
        frame[name] = frame[name].to_numpy(dtype=float, na_value=np.nan)
    if not np.isfinite(frame["mean"]).all():
        raise ValueError("mean must be finite")
    if np.isinf(frame.se).any() or frame.se.lt(0).any():
        raise ValueError("se must be nonnegative or unavailable (NaN)")
    if (frame.n_prompts.eq(1) & frame.se.notna()).any():
        raise ValueError("across-prompt SE must be unavailable with one prompt")
    if (frame.n_sampling_units.eq(1) & frame.se.notna()).any():
        raise ValueError("across-unit SE must be unavailable with one sampling unit")
    if (
        not frame.tier.isin(["structural", "prose"]).all()
        or not frame.kind.isin(["binary", "continuous"]).all()
    ):
        raise ValueError("invalid feature metadata")
    if frame.duplicated(["run", "feature", "step"]).any():
        raise ValueError("duplicate (run, feature, step) series rows")
    if frame.groupby(["run", "feature"])[["tier", "kind"]].nunique().gt(1).any().any():
        raise ValueError("feature metadata changes within a series")
    binary = frame.loc[frame.kind.eq("binary"), "mean"]
    if not binary.between(0, 1).all():
        raise ValueError("binary prevalence must be between zero and one")
    return frame.sort_values(["run", "feature", "step"]).reset_index(drop=True)
