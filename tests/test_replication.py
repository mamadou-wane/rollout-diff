import zlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rolloutdiff.features import FeatureTables


def fixture(
    values, *, kind="binary", steps=None, prompts_per_half=3, copies=1, omit=()
):
    """Build two prompt groups with independently specified step values."""
    names = {
        half: [
            f"prompt-{i}"
            for i in range(100)
            if zlib.crc32(f"prompt-{i}".encode()) & 1 == half
        ][:prompts_per_half]
        for half in (0, 1)
    }
    if steps is None:
        steps = list(range(1, len(values[0]) + 1))
    rows, dense = [], []
    for pos, step in enumerate(steps):
        for half in (0, 1):
            if (half, step) in omit:
                continue
            for prompt in names[half]:
                for copy in range(copies):
                    sample = f"{step}:{prompt}:{copy}"
                    value = values[half][pos]
                    rows.append(
                        {
                            "run": "r",
                            "step": step,
                            "sample_id": sample,
                            "prompt_id": prompt,
                            "prompt": prompt,
                            "response": "text",
                            "reward": float(value),
                        }
                    )
                    dense.append({"run": "r", "sample_id": sample, "flag": value})
    features = FeatureTables(
        pd.DataFrame(dense),
        pd.DataFrame(columns=["run", "sample_id", "feature"]),
        pd.DataFrame([{"feature": "flag", "tier": "structural", "kind": kind}]),
    )
    return pd.DataFrame(rows), features


def replicate(data, **kwargs):
    from rolloutdiff.replication import split_half_replication

    return split_half_replication(
        *data,
        **(
            {"run": "r", "feature": "flag", "baseline_steps": 3, "tolerance": 2}
            | kwargs
        ),
    )


def test_broad_planted_change_replicates_with_same_sequential_rule():
    result = replicate(fixture([[0] * 3 + [1] * 7] * 2))
    assert result.status == "replicated" and result.replicates is True
    assert result.horizon_complete and result.cutoff_step == 8
    assert result.full.alerts.iloc[0].alert_step == 6
    assert result.halves.alert_step.tolist() == [6, 6]
    assert result.halves.onset_step.tolist() == [4, 4]
    assert result.halves.status.tolist() == ["alerted", "alerted"]
    assert result.halves.n_scored.tolist() == [5, 5]


def test_change_confined_to_one_half_fails_replication():
    result = replicate(fixture([[0] * 3 + [1] * 7, [0] * 10]))
    assert result.status == "not_replicated" and result.replicates is False
    assert result.halves.status.tolist() == ["alerted", "no_alert"]


def test_no_full_alert_does_not_treat_two_stable_halves_as_replication():
    result = replicate(fixture([[0] * 10] * 2))
    assert result.status == "no_full_alert" and result.replicates is None
    assert result.halves.empty and result.full.alerts.empty
    assert result.cutoff_step is None and not result.horizon_complete


def test_full_unscorable_is_distinct_from_no_alert():
    rows, features = fixture([[0] * 3 + [1] * 7] * 2, prompts_per_half=1)
    selected = rows.prompt_id.eq(rows.prompt_id.iloc[0])
    rows = rows.loc[selected]
    features = FeatureTables(
        features.dense.loc[selected], features.sparse, features.catalog
    )
    result = replicate((rows, features))
    assert result.status == "full_unscorable" and result.replicates is None
    assert result.full.diagnostics.iloc[0].status == "insufficient_baseline_uncertainty"
    assert result.halves.empty


def test_prompt_partition_is_stable_for_order_and_prefixes():
    data = fixture([[0] * 3 + [1] * 7] * 2, copies=3)
    original = replicate(data)
    rows, features = data
    shuffled = rows.sample(frac=1, random_state=19)
    mixed = FeatureTables(
        features.dense.sample(frac=1, random_state=41),
        features.sparse,
        features.catalog,
    )
    reordered = replicate((shuffled, mixed))
    assert_frame_equal(original.assignments, reordered.assignments)
    assert_frame_equal(original.halves, reordered.halves)
    assert original.assignments.prompt_id.is_unique
    assert original.assignments.half.tolist() == [
        zlib.crc32(name.encode()) & 1 for name in original.assignments.prompt_id
    ]
    prefix = rows.loc[rows.step <= 6]
    subset = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(prefix.sample_id)],
        features.sparse,
        features.catalog,
    )
    assert_frame_equal(original.assignments, replicate((prefix, subset)).assignments)


def test_appended_prompts_cannot_change_existing_assignments_or_bounded_evidence():
    rows, features = fixture([[0] * 3 + [1] * 7] * 2)
    original = replicate((rows, features))
    extra = pd.DataFrame(
        [
            {
                "run": "r",
                "step": 11,
                "sample_id": f"later-{i}",
                "prompt_id": f"later-{i}",
                "prompt": f"later-{i}",
                "response": "text",
                "reward": 0.0,
            }
            for i in range(7)
        ]
    )
    extended = pd.concat([rows, extra], ignore_index=True)
    extended_features = FeatureTables(
        pd.concat(
            [features.dense, extra[["run", "sample_id"]].assign(flag=0)],
            ignore_index=True,
        ),
        features.sparse,
        features.catalog,
    )
    result = replicate((extended, extended_features))
    existing = result.assignments.loc[result.assignments.prompt_id.isin(rows.prompt_id)]
    assert_frame_equal(original.assignments, existing.reset_index(drop=True))
    assert_frame_equal(original.halves, result.halves)
    assert result.replicates is original.replicates


def test_same_half_prompts_never_fall_back_to_splitting_rollout_rows():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2, copies=8)
    selected = rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(0)
    rows = rows.loc[selected]
    features = FeatureTables(
        features.dense.loc[selected], features.sparse, features.catalog
    )
    result = replicate((rows, features))
    assert result.full.alerts.iloc[0].alert_step == 6
    assert result.assignments.half.eq(0).all()
    assert result.halves.iloc[1].n_rollouts == 0
    assert result.halves.iloc[1].status == "unavailable"
    assert result.halves.iloc[1].missing_steps == list(range(1, 9))
    assert result.replicates is None


def test_sparse_step_ids_use_logged_positions_for_tolerance():
    values = [[0] * 3 + [1] * 7, [0] * 5 + [1] * 5]
    steps = [0, 10, 30, 100, 500, 1000, 3000, 8000, 15000, 20000]
    result = replicate(fixture(values, steps=steps))
    assert result.replicates is True
    assert result.halves.alert_step.tolist() == [1000, 8000]
    assert result.halves.alert_position.tolist() == [5, 7]
    assert result.cutoff_step == 8000


def test_optional_half_alert_steps_preserve_large_integer_identity():
    steps = [2**53 + position for position in range(11)]
    result = replicate(fixture([[0] * 3 + [1] * 8, [0] * 11], steps=steps))
    alerted = result.halves.iloc[0]
    assert int(alerted.alert_step) == steps[5]
    assert int(alerted.onset_step) == steps[3]
    assert pd.isna(result.halves.iloc[1].alert_step)
    assert pd.isna(result.halves.iloc[1].onset_step)
    assert result.cutoff_step == steps[7]
    assert result.replicates is False


def test_half_detection_stops_at_full_alert_comparison_horizon():
    # Opposite temporal fluctuations cancel in the full baseline but keep both
    # half thresholds high. The early moderate rise triggers only the full run.
    values = [
        [0.0, 1.0, -1.0] + [1.0] * 4 + [10.0] * 5,
        [0.0, -1.0, 1.0] + [1.0] * 4 + [10.0] * 5,
    ]
    result = replicate(
        fixture(values, kind="continuous", prompts_per_half=20), tolerance=2
    )
    assert result.full.alerts.iloc[0].alert_step == 6
    assert result.cutoff_step == 8
    assert result.halves.status.eq("no_alert").all()
    assert result.replicates is False


def test_halves_agreeing_far_before_full_alert_do_not_replicate():
    # Different constant levels make the full baseline SE much larger than
    # either half's scale. The first modest rise alerts only the halves.
    values = [
        [-100.0] * 3 + [-99.0] * 4 + [900.0] * 5,
        [100.0] * 3 + [101.0] * 4 + [1100.0] * 5,
    ]
    result = replicate(fixture(values, kind="continuous"))
    assert result.full.alerts.iloc[0].alert_step == 10
    assert result.halves.alert_step.tolist() == [6, 6]
    assert result.halves.direction.eq("up").all()
    assert result.replicates is False and result.status == "not_replicated"


def test_first_wrong_direction_is_failure_even_if_other_half_is_unavailable():
    values = [[0.0] * 3 + [-1.0] * 8, [0.0] * 3 + [6.0] * 8]
    rows, features = fixture(values, kind="continuous")
    # One prompt in half 1 leaves its baseline SE unavailable while full SE exists.
    prompts = rows.loc[
        rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(1), "prompt_id"
    ].unique()
    rows = rows.loc[~rows.prompt_id.isin(prompts[1:])]
    features = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(rows.sample_id)],
        features.sparse,
        features.catalog,
    )
    result = replicate((rows, features), numerical_floor=0.01)
    assert result.full.alerts.iloc[0].direction == "up"
    assert result.halves.iloc[0].direction == "down"
    assert result.halves.iloc[1].status == "unavailable"
    assert result.replicates is False


@pytest.mark.parametrize("missing_step", [1, 2, 4, 5])
def test_missing_half_steps_cannot_shift_baseline_or_bridge_streaks(missing_step):
    result = replicate(fixture([[0] * 3 + [1] * 8] * 2, omit=[(0, missing_step)]))
    half = result.halves.iloc[0]
    assert half.status == "unavailable"
    assert half.missing_steps == [missing_step]
    assert half.n_steps == missing_step - 1
    assert pd.isna(half.alert_step)
    assert result.status == "unavailable" and result.replicates is None


def test_alert_established_before_later_gap_stays_usable():
    result = replicate(fixture([[0] * 3 + [1] * 8] * 2, omit=[(0, 7)]))
    assert result.replicates is True
    assert result.halves.iloc[0].alert_step == 6
    assert result.halves.iloc[0].missing_steps == [7]
    assert result.halves.iloc[0].n_steps == 6


def test_a_single_prompt_in_each_half_has_no_row_fallback():
    result = replicate(fixture([[0] * 3 + [1] * 8] * 2, prompts_per_half=1, copies=10))
    assert result.full.alerts.iloc[0].alert_step == 6
    assert result.replicates is None and result.status == "unavailable"
    assert result.halves.status.eq("unavailable").all()
    assert result.halves.n_prompts.tolist() == [1, 1]
    assert result.assignments.prompt_id.nunique() == 2


def test_unavailable_monitoring_se_is_not_definitive_negative_evidence():
    rows, features = fixture([[0] * 3 + [1] * 8, [0] * 11])
    names = rows.loc[
        rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(1), "prompt_id"
    ].unique()
    rows = rows.loc[~(rows.step.eq(5) & rows.prompt_id.isin(names[1:]))]
    features = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(rows.sample_id)],
        features.sparse,
        features.catalog,
    )
    result = replicate((rows, features))
    half = result.halves.iloc[1]
    assert half.n_unscorable == 1 and half.n_scored == 4
    assert half.status == "unavailable"
    assert half.reason == "unavailable_monitoring_uncertainty"
    assert half.missing_steps == []
    assert result.status == "unavailable" and result.replicates is None


def test_no_alert_yet_is_pending_until_full_tolerance_horizon():
    values = [[0] * 3 + [1] * 3, [0] * 6]
    result = replicate(fixture(values))
    assert not result.horizon_complete and result.cutoff_step == 6
    assert result.replicates is None and result.status == "pending"
    assert result.halves.status.tolist() == ["alerted", "pending"]


def test_success_before_end_does_not_require_remaining_horizon():
    result = replicate(fixture([[0] * 3 + [1] * 3] * 2))
    assert not result.horizon_complete
    assert result.replicates is True and result.status == "replicated"


@pytest.mark.parametrize(
    "kwargs, expected_step",
    [
        ({"baseline_steps": 2, "sustain": 1}, 4),
        ({"sustain": 2}, 5),
        ({"z_threshold": 2, "numerical_floor": 0.4}, 6),
    ],
)
def test_detector_parameters_apply_to_full_and_both_halves(kwargs, expected_step):
    result = replicate(fixture([[0] * 3 + [1] * 8] * 2), **kwargs)
    assert result.full.alerts.alert_step.tolist() == [expected_step]
    assert result.halves.alert_step.tolist() == [expected_step, expected_step]


def test_binary_effect_floor_applies_identically_to_halves():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2, prompts_per_half=3)
    features.dense.loc[features.dense.sample_id.str.contains("prompt-"), "flag"] = 0
    selected = rows.groupby("step").prompt_id.transform(
        lambda s: s.isin(s.unique()[::3])
    )
    changed = rows.step.gt(3) & selected
    features.dense.loc[changed, "flag"] = 1
    result = replicate((rows, features), min_prevalence_change=0.5)
    assert result.full.alerts.empty and result.status == "no_full_alert"


def test_reward_uses_the_same_full_and_half_evidence_path():
    result = replicate(
        fixture([[0.0] * 3 + [-1.0] * 8] * 2, kind="continuous"),
        feature="scalar:reward",
    )
    assert result.replicates is True
    assert result.full.alerts.iloc[0].direction == "down"
    assert result.halves.direction.eq("down").all()


def test_inputs_are_not_mutated_and_unselected_runs_are_excluded():
    rows, features = fixture([[0] * 3 + [1] * 8] * 2)
    other = rows.assign(run="other", reward=100.0)
    rows = pd.concat([rows, other], ignore_index=True)
    features = FeatureTables(
        pd.concat(
            [features.dense, features.dense.assign(run="other", flag=0)],
            ignore_index=True,
        ),
        features.sparse,
        features.catalog,
    )
    before = [
        table.copy(deep=True)
        for table in [rows, features.dense, features.sparse, features.catalog]
    ]
    result = replicate((rows, features))
    assert result.replicates is True
    assert result.assignments.run.eq("r").all()
    for actual, original in zip(
        [rows, features.dense, features.sparse, features.catalog], before, strict=False
    ):
        assert_frame_equal(actual, original)


@pytest.mark.parametrize("tolerance", [-1, 1.5, True, np.nan])
def test_invalid_tolerance_fails(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        replicate(fixture([[0] * 3 + [1] * 8] * 2), tolerance=tolerance)


def test_unavailable_se_before_half_alert_makes_first_alert_ambiguous():
    # Half 0 rises at steps 4-5 with SE unavailable at step 6, where it could have
    # alerted five positions before the full alert at 11, outside tolerance 2.
    rows, features = fixture([[0] * 3 + [1, 1, 0, 0, 0, 1, 1, 1], [0] * 8 + [1] * 3])
    names = rows.loc[
        rows.prompt_id.map(lambda p: zlib.crc32(p.encode()) & 1).eq(0), "prompt_id"
    ].unique()
    rows = rows.loc[~(rows.step.eq(6) & rows.prompt_id.isin(names[1:]))]
    features = FeatureTables(
        features.dense.loc[features.dense.sample_id.isin(rows.sample_id)],
        features.sparse,
        features.catalog,
    )
    for policy in ("contiguous_prefix", "fixed_window"):
        result = replicate((rows, features), coverage_policy=policy)
        assert result.full.alerts.iloc[0].alert_step == 11
        half = result.halves.iloc[0]
        assert half.status == "unavailable"
        assert half.reason == "ambiguous_first_alert"
        assert pd.isna(half.alert_step) and half.n_unscorable == 1
        assert result.halves.iloc[1].alert_step == 11
        assert result.status == "unavailable" and result.replicates is None
