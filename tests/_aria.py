"""Local Aria fixtures for named-feature regressions."""

import os

import pandas as pd
import pytest

from bench.aria import load_aria_rollouts
from rolloutdiff.features import FeatureTables, extract

_FEATURES = ("call:bisect.bisect_left", "def:run_tests")


def load_aria(windows):
    source = os.environ.get("ROLLOUTDIFF_ARIA_PATH")
    if not source:
        pytest.skip("set ROLLOUTDIFF_ARIA_PATH to the supplied aria.parquet")
    rollouts = load_aria_rollouts(source, windows)
    # Keep the fixed feature catalog, including absent tokens, across windows.
    sparse_rows = []
    for run, sample_id, response in rollouts[
        ["run", "sample_id", "response"]
    ].itertuples(index=False, name=None):
        found = extract(response).sparse.intersection(_FEATURES)
        sparse_rows.extend((run, sample_id, feature) for feature in sorted(found))
    features = FeatureTables(
        dense=rollouts[["run", "sample_id"]].copy(),
        sparse=pd.DataFrame(sparse_rows, columns=["run", "sample_id", "feature"]),
        catalog=pd.DataFrame(
            [(feature, "structural", "binary") for feature in _FEATURES],
            columns=["feature", "tier", "kind"],
        ),
    )
    return rollouts, features
