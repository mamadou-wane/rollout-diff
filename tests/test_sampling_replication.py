"""Exact prompt identities stay descriptive; source units govern uncertainty."""

import zlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rolloutdiff import analyze_rollouts, from_table, prompt_mix, split_half_replication


def fixture(*, units_per_half=2, variants=2, omit=()):
    units = {
        half: [
            f"problem-{i}"
            for i in range(100)
            if zlib.crc32(f"problem-{i}".encode()) & 1 == half
        ][:units_per_half]
        for half in (0, 1)
    }
    rows, mapping = [], []
    for half, names in units.items():
        for unit in names:
            prompts = [
                next(
                    f"{unit}:variant-{i}"
                    for i in range(100)
                    if zlib.crc32(f"{unit}:variant-{i}".encode()) & 1 == parity
                )
                for parity in range(variants)
            ]
            for prompt in prompts:
                mapping.append(
                    {"run": "r", "prompt_id": prompt, "sampling_unit_id": unit}
                )
                for step in range(1, 12):
                    if (half, step) in omit:
                        continue
                    for completion in range(2):
                        present = step > 3 and completion == 0
                        rows.append(
                            {
                                "step": step,
                                "prompt_id": prompt,
                                "prompt": prompt,
                                "response": "def solve():\n    pass"
                                if present
                                else "plain words.",
                                "reward": 2.0 if present else 0.0,
                            }
                        )
    return from_table(pd.DataFrame(rows), run="r"), pd.DataFrame(mapping)


def analyze(data, **kwargs):
    return analyze_rollouts(
        data,
        sequential_options={"baseline_steps": 3},
        retrospective_options={"min_segment_steps": 3},
        replication_tolerance=2,
        **kwargs,
    )


def target(result):
    return result.evidence.set_index("feature").loc["def:solve"]


def test_variants_share_whole_unit_halves_preserving_canonical_identity():
    data, mapping = fixture()
    before = data.copy(deep=True)
    legacy = analyze(data)
    grouped = analyze(data, sampling_units=mapping)
    old = legacy.half_assignments.merge(
        mapping, on=["run", "prompt_id"], suffixes=("_old", "")
    )
    assert old.groupby("sampling_unit_id").half.nunique().eq(2).all()
    assert (
        grouped.half_assignments.groupby("sampling_unit_id").half.nunique().eq(1).all()
    )
    assert set(grouped.half_assignments.prompt_id) == set(data.prompt_id)
    assert grouped.half_assignments.half.tolist() == [
        zlib.crc32(unit.encode()) & 1
        for unit in grouped.half_assignments.sampling_unit_id
    ]
    assert grouped.aggregation.series.n_prompts.eq(8).all()
    assert grouped.aggregation.series.n_sampling_units.eq(4).all()
    assert target(grouped).replication_status == "replicated"
    assert_frame_equal(data, before)
    assert_frame_equal(
        legacy.aggregation.series.drop(columns=["se", "n_sampling_units"]),
        grouped.aggregation.series.drop(columns=["se", "n_sampling_units"]),
    )
    alert = grouped.sequential.alerts.query('feature == "def:solve"').iloc[0]
    assert alert.n_prompts_at_alert == 8 and alert.n_sampling_units_at_alert == 4


def test_integrated_unit_evidence_agrees_with_public_helpers():
    data, mapping = fixture()
    result = analyze(data, sampling_units=mapping)
    rep = split_half_replication(
        data,
        result.features,
        run="r",
        feature="def:solve",
        baseline_steps=3,
        tolerance=2,
        sampling_units=mapping,
    )
    halves = result.replication_halves.query('feature == "def:solve"').drop(
        columns=["run", "feature"]
    )
    assert_frame_equal(halves.reset_index(drop=True), rep.halves, check_dtype=False)
    assert_frame_equal(result.half_assignments, rep.assignments)
    assert rep.halves.n_sampling_units.tolist() == [2, 2]
    mix = prompt_mix(
        data,
        result.features,
        run="r",
        feature="def:solve",
        before=(1, 3),
        after=(4, 6),
        sampling_units=mapping,
    )
    assert mix.shared_prompts == 8 and mix.shared_sampling_units == 4
    assert target(result).prompt_shared_sampling_units == mix.shared_sampling_units
    assert target(result).prompt_shared_se == mix.shared_se
    assert set(mix.per_prompt.sampling_unit_id) == set(mapping.sampling_unit_id)


def test_many_variants_of_one_unit_cannot_supply_uncertainty():
    data, mapping = fixture()
    mapping["sampling_unit_id"] = "one-problem"
    result = analyze(data, sampling_units=mapping)
    assert target(result).sequential_status == "insufficient_baseline_uncertainty"
    assert target(result).retrospective_status == "unavailable_uncertainty"
    assert target(result).replication_status == "full_unscorable"
    assert result.aggregation.series.se.isna().all()
    rep = split_half_replication(
        data,
        result.features,
        run="r",
        feature="def:solve",
        baseline_steps=3,
        sampling_units=mapping,
    )
    assert rep.status == "full_unscorable" and rep.replicates is None
    mix = prompt_mix(
        data,
        result.features,
        run="r",
        feature="def:solve",
        before=(1, 3),
        after=(4, 6),
        sampling_units=mapping,
    )
    assert mix.shared_prompts == 8 and mix.shared_sampling_units == 1
    assert mix.shared_change == 0.5 and mix.shared_se is None


def test_one_unit_in_each_half_is_unavailable_without_row_fallback():
    data, mapping = fixture(units_per_half=1)
    result = analyze(data, sampling_units=mapping)
    assert target(result).sequential_status == "alerted"
    assert target(result).replication_status == "unavailable"
    halves = result.replication_halves.query('feature == "def:solve"')
    assert halves.n_prompts.tolist() == [2, 2]
    assert halves.n_sampling_units.tolist() == [1, 1]
    assert halves.reason.eq("insufficient_baseline_uncertainty").all()


def test_all_units_in_one_half_do_not_rebalance_or_split_variants():
    data, mapping = fixture()
    same_half = [
        f"source-{i}" for i in range(100) if zlib.crc32(f"source-{i}".encode()) & 1 == 0
    ][:4]
    translation = dict(zip(mapping.sampling_unit_id.unique(), same_half, strict=False))
    mapping["sampling_unit_id"] = mapping.sampling_unit_id.map(translation)
    result = analyze(data, sampling_units=mapping)
    assert target(result).replication_status == "unavailable"
    assert result.half_assignments.half.eq(0).all()
    half = result.replication_halves.query('feature == "def:solve" and half == 1').iloc[
        0
    ]
    assert half.n_sampling_units == half.n_prompts == half.n_rollouts == 0
    assert half.missing_steps == list(range(1, 9))


def test_missing_unit_half_baseline_stops_at_first_gap():
    data, mapping = fixture(omit=[(0, 2)])
    result = analyze(data, sampling_units=mapping)
    assert target(result).replication_status == "unavailable"
    half = result.replication_halves.query('feature == "def:solve" and half == 0').iloc[
        0
    ]
    assert half.n_steps == 1 and half.missing_steps == [2]
    assert half.reason == "missing_step_grid"


def test_default_mapping_is_exact_prompt_and_order_stable():
    data, _ = fixture()
    explicit = (
        data[["run", "prompt_id"]]
        .drop_duplicates()
        .assign(sampling_unit_id=lambda frame: frame.prompt_id)
    )
    original = analyze(data)
    result = analyze(
        data.sample(frac=1, random_state=7),
        sampling_units=explicit.sample(frac=1, random_state=9),
    )
    for field in [
        "evidence",
        "half_assignments",
        "replication_halves",
        "prompt_values",
    ]:
        assert_frame_equal(getattr(original, field), getattr(result, field))
    assert_frame_equal(original.aggregation.series, result.aggregation.series)


def test_units_do_not_create_shared_exact_prompts():
    data, mapping = fixture()
    later = data.step.gt(3)
    data.loc[later, "prompt_id"] += ":later"
    data.loc[later, "prompt"] += ":later"
    mapping = pd.concat(
        [mapping, mapping.assign(prompt_id=lambda x: x.prompt_id + ":later")]
    )
    result = analyze(data, sampling_units=mapping)
    assert target(result).prompt_shared_prompts == 0
    assert target(result).prompt_shared_sampling_units == 0
    assert target(result).promptmix_status == "no_shared_prompts"
    assert pd.isna(target(result).prompt_shared_change)


def test_sampling_assignment_keys_include_run():
    data, mapping = fixture()
    both = pd.concat([data, data.assign(run="s")], ignore_index=True)
    other = mapping.assign(run="s", sampling_unit_id=lambda frame: frame.prompt_id)
    result = analyze(both, sampling_units=pd.concat([mapping, other]))
    assignments = result.half_assignments
    assert (
        assignments.query('run == "r"')
        .groupby("sampling_unit_id")
        .half.nunique()
        .eq(1)
        .all()
    )
    assert len(assignments) == len(mapping) * 2
    assert result.aggregation.series.query('run == "r"').n_sampling_units.eq(4).all()
    assert result.aggregation.series.query('run == "s"').n_sampling_units.eq(8).all()


def test_invalid_mapping_fails_before_featurization(monkeypatch):
    from rolloutdiff import pipeline

    data, mapping = fixture()

    def unexpected(*args, **kwargs):
        pytest.fail("invalid sampling map reached featurize")

    monkeypatch.setattr(pipeline, "featurize", unexpected)
    with pytest.raises(ValueError):
        analyze(data, sampling_units=mapping.iloc[1:])


def test_shared_change_keeps_prompt_weights_but_clusters_correlated_variants():
    data, mapping = fixture(units_per_half=1)
    group = mapping.sampling_unit_id.iloc[0]
    changed = mapping.loc[mapping.sampling_unit_id.eq(group), "prompt_id"]
    data["response"] = "plain words."
    data.loc[data.step.gt(3) & data.prompt_id.isin(changed), "response"] = (
        "def solve():\n    pass"
    )
    result = analyze(data, sampling_units=mapping)
    kwargs = {"run": "r", "feature": "def:solve", "before": (1, 3), "after": (4, 6)}
    independent = prompt_mix(data, result.features, **kwargs)
    grouped = prompt_mix(data, result.features, sampling_units=mapping, **kwargs)
    assert independent.shared_change == grouped.shared_change == 0.5
    assert independent.shared_prompts == grouped.shared_prompts == 4
    assert independent.shared_se == pytest.approx((1 / 12) ** 0.5)
    assert grouped.shared_se == pytest.approx(0.5)
    assert grouped.shared_sampling_units == 2
    assert grouped.positive_prompts == independent.positive_prompts == 2
    assert_frame_equal(
        independent.per_prompt.drop(columns="sampling_unit_id"),
        grouped.per_prompt.drop(columns="sampling_unit_id"),
    )


def test_single_unit_monitoring_uncertainty_is_unavailable_even_with_variants():
    data, mapping = fixture()
    half_one = mapping.loc[
        mapping.sampling_unit_id.map(lambda unit: zlib.crc32(unit.encode()) & 1).eq(1)
    ]
    data.loc[data.prompt_id.isin(half_one.prompt_id), "response"] = "plain words."
    absent_unit = half_one.sampling_unit_id.iloc[-1]
    absent_prompts = half_one.loc[
        half_one.sampling_unit_id.eq(absent_unit), "prompt_id"
    ]
    data = data.loc[~(data.step.eq(5) & data.prompt_id.isin(absent_prompts))]
    result = analyze(data, sampling_units=mapping)
    assert target(result).sequential_status == "alerted"
    assert target(result).replication_status == "unavailable"
    half = result.replication_halves.query('feature == "def:solve" and half == 1').iloc[
        0
    ]
    assert half.n_unscorable == 1
    assert half.reason == "unavailable_monitoring_uncertainty"
    assert half.status == "unavailable"
    assert half.missing_steps == []
