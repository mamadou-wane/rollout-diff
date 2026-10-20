"""Identity-aligned values for evidence about one retained feature."""

from __future__ import annotations

import pandas as pd

from .features import FeatureTables
from .schema import validate_rollouts
from .series import _validate_features


def feature_values(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    feature: str,
) -> tuple[pd.DataFrame, str]:
    """Return canonical rows for one run plus `_value`, and the feature kind."""
    rollouts = validate_rollouts(rollouts)
    dense_names = _validate_features(rollouts, features)
    frame = rollouts.loc[rollouts.run.eq(run)].copy()
    if frame.empty:
        raise ValueError("run has no rollouts")
    if feature == "scalar:reward":
        frame["_value"] = frame.reward
        return frame.reset_index(drop=True), "continuous"
    metadata = features.catalog.set_index("feature")
    if feature not in metadata.index:
        raise ValueError("feature is not in the retained catalog")
    kind = metadata.loc[feature, "kind"]
    keys = ["run", "sample_id"]
    if feature in dense_names:
        values = features.dense[[*keys, feature]].rename(columns={feature: "_value"})
        frame = frame.merge(values, on=keys, validate="one_to_one")
    else:
        values = features.sparse.loc[features.sparse.feature.eq(feature), keys].assign(
            _value=1.0
        )
        frame = frame.merge(values, on=keys, how="left", validate="one_to_one")
        frame["_value"] = frame["_value"].fillna(0.0)
    return frame, kind


class _FeatureValueIndex:
    """Index validated identities and sparse presence for repeated evidence.

    Avoids a dense rollout-by-vocabulary matrix.
    """

    def __init__(self, rollouts: pd.DataFrame, features: FeatureTables):
        self.runs = {
            run: frame[
                ["run", "sample_id", "step", "prompt_id", "prompt", "reward"]
            ].reset_index(drop=True)
            for run, frame in rollouts.groupby("run", sort=False)
        }
        self.dense = {
            run: frame.set_index("sample_id").drop(columns="run")
            for run, frame in features.dense.groupby("run", sort=False)
        }
        self.sparse = {
            key: pd.Index(frame.sample_id)
            for key, frame in features.sparse.groupby(["run", "feature"], sort=False)
        }
        self.metadata = features.catalog.set_index("feature")

    def values(self, run: str, feature: str) -> tuple[pd.DataFrame, str]:
        frame = self.runs[run].copy()
        if feature == "scalar:reward":
            frame["_value"] = frame.reward
            return frame, "continuous"
        kind = self.metadata.loc[feature, "kind"]
        if feature in self.dense[run].columns:
            frame["_value"] = frame.sample_id.map(self.dense[run][feature])
        else:
            present = self.sparse.get((run, feature), pd.Index([]))
            frame["_value"] = frame.sample_id.isin(present).astype(float)
        return frame, kind
