import hashlib
import os
from pathlib import Path

import pandas as pd
import pytest

from rolloutdiff import aggregate, featurize, from_table, load_trl


def test_ingestion_to_prompt_balanced_evidence(tmp_path):
    raw = pd.DataFrame(
        {
            "step": [1, 1, 1, 1, 2, 2],
            "prompt": ["a", "a", "a", "b", "a", "b"],
            "response": ["def run_tests():\n    pass"] * 3 + ["A prose response."] * 3,
            "reward": [1, 1, 1, 0, 0, 1],
            "label_hack": [1] * 6,
        }
    )
    rollouts = from_table(raw, run="synthetic")
    features = featurize(rollouts, min_prevalence=0)
    result = aggregate(rollouts, features)
    definitions = result.series[result.series.feature.eq("def:run_tests")]
    assert definitions["mean"].tolist() == [0.5, 0.0]
    assert definitions.se.tolist() == [0.5, 0.0]
    assert definitions.n_rollouts.tolist() == [4, 2]
    assert definitions.n_prompts.tolist() == [2, 2]
    assert result.prompt_schedule.n_rollouts.tolist() == [3, 1, 1, 1]
    assert "label_hack" not in features.catalog.feature.tolist()
    path = tmp_path / "series.parquet"
    result.series.to_parquet(path, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(path), result.series)


# Checksums for the supplied local TRL fixtures.
_LOCAL_FILES = {
    "completions_00001.parquet": "6bf165a220acec288a4b682ed1c7d1b3a760f9ecd25c053598bb62735e6804bb",
    "completions_00058.parquet": "c7174dc860ba097a88b1893b3adbc960e889e106c524622039702280a8d8dccf",
    "completions_00115.parquet": "b365e74988c0674a7d19a2f7b559e73bc8622d41c5b2246420d9c1cb6781e9d6",
}


@pytest.mark.local_data
def test_local_trl_fixed_files():
    root = os.environ.get("ROLLOUTDIFF_TRL_DIR")
    if not root:
        pytest.skip("set ROLLOUTDIFF_TRL_DIR to the supplied trl_sunshine dump")
    parts = []
    for name, digest in _LOCAL_FILES.items():
        path = Path(root) / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, (
            f"local fixture changed: {name}"
        )
        parts.append(load_trl(path, run="trl", reward_column="training_passed"))
    rollouts = pd.concat(parts, ignore_index=True)
    assert len(rollouts) == 96
    result = aggregate(rollouts, featurize(rollouts))
    rewards = result.series[result.series.feature.eq("scalar:reward")]
    assert rewards.step.tolist() == [1, 58, 115]
    assert rewards["mean"].tolist() == [0.03125, 0.0, 0.96875]
    assert rewards.n_prompts.tolist() == [1, 1, 1]
    assert result.series.se.isna().all()
    assert result.prompt_schedule.n_rollouts.tolist() == [32, 32, 32]


def test_sequential_public_api_from_rollouts_to_alert():
    from rolloutdiff import detect_sequential

    rows = [
        {
            "step": step,
            "prompt": prompt,
            "response": "words here." if step <= 15 else "def solve():\n    pass",
            "reward": 0 if step <= 15 else 1,
        }
        for step in range(1, 19)
        for prompt in ["a", "b"]
    ]
    rollouts = from_table(pd.DataFrame(rows), run="synthetic")
    features = featurize(rollouts)
    result = detect_sequential(aggregate(rollouts, features).series)
    alert = result.alerts.set_index("feature").loc["def:solve"]
    assert (alert.onset_step, alert.alert_step) == (16, 18)
    assert alert.numerical_floor_dominant
    assert result.screening.iloc[0].n_screened == len(features.catalog) + 1


def test_retrospective_public_api_reports_boundary_separately_from_alert():
    from rolloutdiff import detect_retrospective, detect_sequential

    rows = [
        {
            "step": step,
            "prompt": prompt,
            "response": "words here." if step <= 15 else "def solve():\n    pass",
            "reward": 0 if step <= 15 else 1,
        }
        for step in range(1, 31)
        for prompt in ["a", "b"]
    ]
    rollouts = from_table(pd.DataFrame(rows), run="synthetic")
    step_series = aggregate(rollouts, featurize(rollouts)).series
    retrospective = detect_retrospective(step_series)
    sequential = detect_sequential(step_series)
    cp = retrospective.changepoints.set_index("feature").loc["def:solve"]
    alert = sequential.alerts.set_index("feature").loc["def:solve"]
    assert cp.changepoint_step == 16
    assert cp.previous_step == 15
    assert alert.alert_step == 18
    assert "alert_step" not in retrospective.changepoints


def test_public_validation_evidence_from_extracted_rollouts():
    from rolloutdiff import persistence, reward_association, split_half_replication

    rows = []
    for step in range(1, 31):
        for prompt in range(8):
            for completion in range(2):
                present = step > 15 and completion == 0
                rows.append(
                    {
                        "step": step,
                        "prompt_id": f"p{prompt}",
                        "prompt": f"question {prompt}",
                        "response": "def solve():\n    pass"
                        if present
                        else "plain prose.",
                        "reward": 2 if present else 0,
                    }
                )
    rollouts = from_table(pd.DataFrame(rows), run="r")
    features = featurize(rollouts)
    rep = split_half_replication(rollouts, features, run="r", feature="def:solve")
    assert rep.replicates is True
    assert rep.halves.alert_step.eq(18).all()
    assert rep.halves.numerical_floor_dominant.all()
    assert rep.halves.scale.eq(1e-9).all()
    alert = rep.full.alerts.iloc[0]
    retained = persistence(
        rep.full.scores,
        run="r",
        feature="def:solve",
        after_step=int(alert.alert_step),
        reference_mean=float(alert.baseline_mean),
        change=float(alert.delta_at_alert),
    )
    assert retained.n_observations == retained.n_retained == 12
    assert retained.fraction_retained == retained.coverage == 1
    association = reward_association(
        rollouts,
        features,
        run="r",
        feature="def:solve",
        window=(int(alert.onset_step), int(alert.alert_step)),
    )
    assert association.mean_difference == 2
    assert association.n_mixed_groups == association.n_groups == 24
    assert association.n_included_rollouts == association.n_rollouts == 48
