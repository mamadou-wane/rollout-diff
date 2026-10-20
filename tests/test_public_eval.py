import numpy as np
import pandas as pd
import pytest

from rolloutdiff import featurize, from_table


def test_attribution_joins_ids_and_keeps_unknown_labels_out_of_confusion_counts():
    from bench.public_eval import attribution

    values = pd.DataFrame(
        {"run": ["r"] * 4, "sample_id": ["a", "b", "c", "d"], "_value": [1, 0, 1, 1]}
    )
    labels = pd.DataFrame(
        {
            "run": ["r"] * 4,
            "sample_id": ["d", "b", "a", "c"],
            "step": [3, 2, 1, 2],
            "label_hack": pd.array([1, 1, 0, None], dtype="Int64"),
        }
    )
    result = attribution(values, labels, "label_hack", start_step=2)
    assert (
        result["tp"] == 1
        and result["fn"] == 1
        and result["fp"] == 0
        and result["tn"] == 0
    )
    assert (
        result["n_window"] == 3 and result["n_known"] == 2 and result["n_unknown"] == 1
    )
    assert result["precision"] == 1 and result["recall"] == 0.5


def test_no_labels_or_no_window_is_unavailable_not_negative_evidence():
    from bench.public_eval import attribution

    values = pd.DataFrame({"run": ["r"], "sample_id": ["a"], "_value": [1]})
    labels = pd.DataFrame(
        {"run": ["r"], "sample_id": ["a"], "step": [1], "label_attempt": [pd.NA]}
    )
    assert (
        attribution(values, labels, "label_attempt", start_step=1)["status"]
        == "unavailable_labels"
    )
    assert (
        attribution(values, labels, "label_attempt", start_step=None)["status"]
        == "unavailable_window"
    )
    with pytest.raises(ValueError, match="identit"):
        attribution(values, labels.iloc[:0], "label_attempt", start_step=1)


def test_onset_requires_three_complete_observations_and_reports_first_not_alert_time():
    from bench.public_eval import label_onsets

    labels = pd.DataFrame(
        {"step": [1, 2, 3, 4, 5, 6, 7], "label_hack": [1, 1, pd.NA, 1, 1, 1, 0]}
    )
    result = label_onsets(labels, columns=["label_hack"])
    assert result["onsets"]["label_hack"]["0.01"] == 4
    assert result["rates"][2]["rate"] is None and result["rates"][2]["n_unknown"] == 1


def test_scalar_baselines_use_prompt_weighting_and_measured_uncertainty():
    from bench.public_eval import scalar_baselines

    raw = pd.DataFrame(
        {
            "step": [1] * 6,
            "prompt": ["a"] * 4 + ["b"] * 2,
            "response": ["x"] * 6,
            "reward": [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
        }
    )
    rollouts = from_table(raw, run="r")
    result = scalar_baselines(rollouts, featurize(rollouts))
    rows = result["series"].set_index("feature")
    assert rows.loc["scalar:reward", "mean"] == 0.5
    assert rows.loc["scalar:reward_max_frac", "mean"] == 0.5
    assert rows.loc["scalar:reward_max_frac", "se"] == 0.5
    assert rows.loc["scalar:zero_var_group_frac", "mean"] == 1
    assert rows.loc["scalar:zero_var_group_frac", "se"] == 0
    assert result["maximum_reward"] == 1
    assert result["maximum_policy"] == "observed_full_run_maximum"


def test_single_prompt_scalar_baselines_do_not_invent_independent_groups():
    from bench.public_eval import scalar_baselines

    raw = pd.DataFrame(
        {
            "step": [1, 1],
            "prompt": ["a", "a"],
            "response": ["x", "y"],
            "reward": [0.0, 1.0],
        }
    )
    rollouts = from_table(raw, run="r")
    result = scalar_baselines(rollouts, featurize(rollouts))
    assert result["series"].se.isna().all()


def test_unknown_group_variance_is_not_filled_with_zero():
    from bench.public_eval import scalar_baselines

    raw = pd.DataFrame(
        {
            "step": [1, 1, 1],
            "prompt": ["a", "b", "b"],
            "response": ["x"] * 3,
            "reward": [0.0, 1.0, 0.0],
        }
    )
    rollouts = from_table(raw, run="r")
    result = scalar_baselines(rollouts, featurize(rollouts))
    assert "scalar:zero_var_group_frac" not in set(result["series"].feature)
    assert (
        result["unavailable"]["scalar:zero_var_group_frac"] == "singleton_prompt_group"
    )


def test_integrated_benchmark_keeps_labels_out_of_detection_and_compares_both_modes():
    from types import SimpleNamespace

    from bench.public_benchmark import evaluate_loaded

    raw = pd.DataFrame(
        [
            {
                "step": step,
                "prompt": prompt,
                "response": "def run_tests():\n    pass"
                if step >= 16
                else "plain words",
                "reward": float(step >= 16),
            }
            for step in range(1, 21)
            for prompt in ["a", "b"]
            for _ in range(2)
        ]
    )
    rollouts = from_table(raw, run="aria")
    labels = rollouts[["run", "sample_id", "step", "prompt_id"]].copy()
    labels["label_hack"] = rollouts.step.ge(16).astype(int)
    labels["label_attempt"] = labels.label_hack
    labels["label_correct"] = 0
    data = SimpleNamespace(rollouts=rollouts, labels=labels, provenance={})
    result = evaluate_loaded(data, "aria")
    mechanism = next(
        row for row in result["mechanisms"] if row["feature"] == "def:run_tests"
    )
    assert mechanism["evidence"]["alert_step"] == 18
    assert mechanism["retrospective"]["independent"]["candidate_step"] == 16
    assert mechanism["retrospective"]["ar1"]["candidate_step"] == 16
    assert mechanism["post_onset"]["label_hack"]["precision"] == 1
    data.labels = labels.sample(frac=1, random_state=7).assign(
        label_hack=0, label_attempt=0
    )
    changed = evaluate_loaded(data, "aria")
    other = next(
        row for row in changed["mechanisms"] if row["feature"] == "def:run_tests"
    )
    assert mechanism["evidence"] == other["evidence"]
    assert mechanism["retrospective"] == other["retrospective"]
    assert other["post_onset"]["label_hack"]["status"] == "unavailable_window"


def test_reward_effect_comparison_keeps_continuous_scale_for_rewards_above_one():
    from bench.public_eval import reward_effect_comparison
    from rolloutdiff import detect_sequential

    frame = pd.DataFrame(
        {
            "run": "r",
            "feature": "scalar:reward",
            "tier": "structural",
            "kind": "continuous",
            "step": range(1, 22),
            "mean": [2.0] * 15 + [2.05] * 3 + [2.15] * 3,
            "se": 0.001,
            "n_prompts": 4,
            "n_rollouts": 64,
        }
    )
    standard = detect_sequential(frame)
    assert standard.alerts.alert_step.tolist() == [18]
    matched = reward_effect_comparison(frame)
    assert matched["onset_step"] == 19 and matched["alert_step"] == 21
    assert matched["scale"] == standard.diagnostics.iloc[0].scale
    assert matched["minimum_increase"] == 0.1
    frame["se"] = np.nan
    assert (
        reward_effect_comparison(frame)["status"] == "insufficient_baseline_uncertainty"
    )


def test_benchmark_uses_supplied_units_even_when_labels_offer_no_problem_grouping():
    from types import SimpleNamespace

    from bench.public_benchmark import evaluate_loaded

    raw = pd.DataFrame(
        [
            {
                "step": step,
                "prompt": f"variant{variant}",
                "response": "def run_tests():\n    pass"
                if step >= 16
                else "plain words",
                "reward": float(step >= 16),
            }
            for step in range(1, 21)
            for variant in range(4)
        ]
    )
    rollouts = from_table(raw, run="aria")
    labels = rollouts[["run", "sample_id", "step", "prompt_id"]].copy()
    labels["label_hack"] = labels.step.ge(16).astype(int)
    labels["label_attempt"] = labels.label_hack
    labels["label_correct"] = 0
    units = (
        rollouts[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id="one-problem")
    )
    data = SimpleNamespace(
        rollouts=rollouts, labels=labels, provenance={}, sampling_units=units
    )
    saved = rollouts.copy(deep=True)
    result = evaluate_loaded(data, "aria")
    target = next(
        row for row in result["mechanisms"] if row["feature"] == "def:run_tests"
    )
    assert (
        target["evidence"]["sequential_status"] == "insufficient_baseline_uncertainty"
    )
    assert target["evidence"]["replication_status"] == "full_unscorable"
    assert result["sampling_unit_comparison"]["units_crossing_halves"] == 0
    assert result["sampling_unit_comparison"]["point_estimates_equal"]
    old = result["sampling_unit_comparison"]["exact_prompt_detectors"][
        "sequential_alerts"
    ]
    assert (
        next(row["alert_step"] for row in old if row["feature"] == "def:run_tests")
        == 18
    )
    pd.testing.assert_frame_equal(rollouts, saved)


def test_scalar_baseline_means_preserved_but_uncertainty_uses_units():
    from bench.public_eval import scalar_baselines

    raw = pd.DataFrame(
        {
            "step": [1] * 3,
            "prompt": ["a", "b", "c"],
            "response": ["x"] * 3,
            "reward": [1.0, 1.0, 0.0],
        }
    )
    rollouts = from_table(raw, run="r")
    units = rollouts[["run", "prompt_id"]].assign(sampling_unit_id=["u", "u", "v"])
    result = scalar_baselines(rollouts, featurize(rollouts), sampling_units=units)[
        "series"
    ]
    row = result.set_index("feature").loc["scalar:reward"]
    assert row["mean"] == pytest.approx(2 / 3)
    assert row["se"] == pytest.approx(4 / 9)
    assert row.n_prompts == 3 and row.n_sampling_units == 2


def test_benchmark_reports_steps_to_onset_floor_dominance_and_pre_onset_list():
    from types import SimpleNamespace

    from bench.public_benchmark import evaluate_loaded

    raw = pd.DataFrame(
        [
            {
                "step": step,
                "prompt": prompt,
                "response": "def run_tests():\n    pass"
                if step >= 16
                else "plain words",
                "reward": float(step >= 16),
            }
            for step in range(1, 21)
            for prompt in ["a", "b"]
            for _ in range(2)
        ]
    )
    rollouts = from_table(raw, run="aria")
    labels = rollouts[["run", "sample_id", "step", "prompt_id"]].copy()
    labels["label_hack"] = rollouts.step.ge(16).astype(int)
    labels["label_attempt"] = labels.label_hack
    labels["label_correct"] = 0
    result = evaluate_loaded(
        SimpleNamespace(rollouts=rollouts, labels=labels, provenance={}), "aria"
    )
    pipeline = result["pipeline"]
    # Every baseline series is constant, so every alert is scaled by the numerical floor.
    assert (
        pipeline["n_floor_dominated_alerts"]
        == pipeline["sequential_screening"][0]["n_alerted"]
        > 0
    )
    assert pipeline["pre_reference_alert_status"] == "available"
    assert pipeline["pre_reference_alerts"] == 0
    assert pipeline["pre_reference_alert_list"] == []
    mechanism = next(
        row for row in result["mechanisms"] if row["feature"] == "def:run_tests"
    )
    assert mechanism["steps_to_onset"] == {"0.01": 2, "0.05": 2, "0.1": 2}
    data = SimpleNamespace(
        rollouts=rollouts,
        labels=labels.assign(label_hack=0, label_attempt=0),
        provenance={},
    )
    other = next(
        row
        for row in evaluate_loaded(data, "aria")["mechanisms"]
        if row["feature"] == "def:run_tests"
    )
    assert other["steps_to_onset"] == {"0.01": None, "0.05": None, "0.1": None}


def test_pre_onset_alert_list_names_features_and_marks_unscorable_runs():
    from types import SimpleNamespace

    from bench.public_benchmark import evaluate_loaded

    rows = []
    for step in range(1, 33):
        for prompt in ["a", "b"]:
            early = step >= 18
            late = step >= 26
            rows.append(
                {
                    "step": step,
                    "prompt": prompt,
                    "response": ("import os\n" if early else "")
                    + ("def run_tests():\n    pass" if late else "plain words"),
                    "reward": float(late),
                }
            )
    rollouts = from_table(pd.DataFrame(rows), run="aria")
    labels = rollouts[["run", "sample_id", "step", "prompt_id"]].copy()
    labels["label_hack"] = rollouts.step.ge(26).astype(int)
    labels["label_attempt"] = labels.label_hack
    labels["label_correct"] = 0
    result = evaluate_loaded(
        SimpleNamespace(rollouts=rollouts, labels=labels, provenance={}), "aria"
    )
    listed = result["pipeline"]["pre_reference_alert_list"]
    assert result["pipeline"]["pre_reference_alerts"] == len(listed) > 0
    assert all(entry["alert_step"] < 26 for entry in listed)
    assert all(
        isinstance(entry["feature"], str) and entry["target"] is False
        for entry in listed
    )
    assert {entry["tier"] for entry in listed} <= {"structural", "prose"}
    single = from_table(pd.DataFrame(rows).loc[lambda t: t.prompt.eq("a")], run="aria")
    labels = single[["run", "sample_id", "step", "prompt_id"]].copy()
    labels["label_hack"] = single.step.ge(26).astype(int)
    labels["label_attempt"] = labels.label_hack
    labels["label_correct"] = 0
    unscorable = evaluate_loaded(
        SimpleNamespace(rollouts=single, labels=labels, provenance={}), "aria"
    )["pipeline"]
    assert unscorable["sequential_screening"][0]["n_screened"] == 0
    assert unscorable["pre_reference_alert_status"] == "unscorable"
    assert unscorable["pre_reference_alerts"] is None
    assert unscorable["pre_reference_alert_list"] == []


def test_source_group_zero_variance_is_separate_from_exact_prompt_metric():
    from bench.public_eval import scalar_baselines

    raw = pd.DataFrame(
        {
            "step": [1] * 6,
            "prompt": ["a", "a", "b", "b", "c", "c"],
            "response": ["x"] * 6,
            "reward": [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        }
    )
    rollouts = from_table(raw, run="r")
    prompts = rollouts[["run", "prompt_id"]].drop_duplicates()
    units = prompts.assign(sampling_unit_id=["u", "u", "v"])
    exact = scalar_baselines(rollouts, featurize(rollouts))
    assert (
        exact["unavailable"]["scalar:zero_var_source_group_frac"]
        == "sampling_units_not_supplied"
    )
    assert "scalar:zero_var_source_group_frac" not in set(exact["series"].feature)
    result = scalar_baselines(rollouts, featurize(rollouts), sampling_units=units)
    rows = result["series"].set_index("feature")
    # Every exact prompt is constant; source group u mixes rewards 1 and 0.
    assert rows.loc["scalar:zero_var_group_frac", "mean"] == 1
    assert rows.loc["scalar:zero_var_source_group_frac", "mean"] == pytest.approx(1 / 3)
    assert "scalar:zero_var_source_group_frac" not in result["unavailable"]
    lone = rollouts.iloc[:-1]
    unavailable = scalar_baselines(lone, featurize(lone), sampling_units=units)[
        "unavailable"
    ]
    assert unavailable["scalar:zero_var_source_group_frac"] == "singleton_source_group"
