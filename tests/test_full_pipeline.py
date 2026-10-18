import zlib

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from rolloutdiff import (
    from_table,
    persistence,
    prompt_mix,
    reward_association,
    split_half_replication,
)


def fixture(
    *,
    last=14,
    one_half=False,
    single_prompt=False,
    switched_prompts=False,
    omit=(),
    falling=False,
):
    names = {
        half: [f"p{i}" for i in range(100) if zlib.crc32(f"p{i}".encode()) & 1 == half][
            :3
        ]
        for half in (0, 1)
    }
    rows = []
    for step in range(1, last + 1):
        for half in (0, 1):
            if (half, step) in omit:
                continue
            for prompt in names[half][: 1 if single_prompt else 3]:
                if single_prompt and half == 1:
                    continue
                prompt = f"later-{prompt}" if switched_prompts and step > 3 else prompt
                for completion in range(2):
                    changed = (step > 3) != falling
                    present = (
                        changed and completion == 0 and (not one_half or half == 0)
                    )
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
    return from_table(pd.DataFrame(rows), run="r")


def analyze(data, **kwargs):
    from rolloutdiff.pipeline import analyze_rollouts

    return analyze_rollouts(
        data,
        **(
            {
                "sequential_options": {"baseline_steps": 3},
                "retrospective_options": {"min_segment_steps": 3},
                "replication_tolerance": 2,
            }
            | kwargs
        ),
    )


def row(result, feature="def:solve", run="r"):
    return result.evidence.set_index(["run", "feature"]).loc[(run, feature)]


def test_all_retained_features_receive_both_detectors_and_evidence_rows():
    result = analyze(fixture())
    expected = set(result.features.catalog.feature) | {"scalar:reward"}
    assert set(result.evidence.feature) == expected
    assert len(result.evidence) == len(expected)
    assert set(result.sequential.diagnostics.feature) == expected
    assert set(result.retrospective.diagnostics.feature) == expected
    target = row(result)
    assert (target.onset_step, target.alert_step, target.changepoint_step) == (4, 6, 4)
    assert target.replication_status == "replicated"
    assert target.persistence_fraction == target.persistence_coverage == 1
    assert target.reward_mean_difference == 2
    assert (target.prompt_before_start, target.prompt_before_end) == (1, 3)
    assert (target.prompt_after_start, target.prompt_after_end) == (4, 6)
    assert target.prompt_shared_change == 0.5
    assert target.prompt_positive_prompts == target.prompt_total_prompts == 6
    assert target.prompt_length_distance == 0


@pytest.mark.parametrize("feature", ["def:solve", "scalar:reward", "has_code"])
def test_integrated_evidence_agrees_with_existing_public_functions(feature):
    data = fixture()
    result = analyze(data)
    if feature == "has_code":
        feature = next(
            name
            for name in result.features.dense.columns
            if name not in ["run", "sample_id"]
            and row(result, name).sequential_status == "alerted"
        )
    target = row(result, feature)
    rep = split_half_replication(
        data, result.features, run="r", feature=feature, baseline_steps=3, tolerance=2
    )
    halves = result.replication_halves.query('run == "r" and feature == @feature').drop(
        columns=["run", "feature"]
    )
    assert_frame_equal(halves.reset_index(drop=True), rep.halves, check_dtype=False)
    assert target.replication_status == rep.status
    alert = rep.full.alerts.iloc[0]
    retained = persistence(
        rep.full.scores,
        run="r",
        feature=feature,
        after_step=int(alert.alert_step),
        reference_mean=float(alert.baseline_mean),
        change=float(alert.delta_at_alert),
    )
    assert target.persistence_status == retained.status
    assert target.persistence_fraction == retained.fraction_retained
    window = (int(alert.onset_step), int(alert.alert_step))
    assoc = reward_association(
        data, result.features, run="r", feature=feature, window=window
    )
    assert target.reward_status == assoc.status
    if assoc.mean_difference is not None:
        assert target.reward_mean_difference == assoc.mean_difference
    mixed = prompt_mix(
        data, result.features, run="r", feature=feature, before=(1, 3), after=window
    )
    assert target.prompt_shared_change == mixed.shared_change
    assert target.prompt_shared_se == mixed.shared_se


def test_absent_feature_in_half_is_zero_and_keeps_full_catalog():
    result = analyze(fixture(one_half=True))
    target = row(result)
    assert target.replication_status == "not_replicated"
    halves = result.replication_halves.query('feature == "def:solve"')
    assert halves.status.tolist() == ["alerted", "no_alert"]
    assert halves.baseline_mean.tolist() == [0, 0]


def test_retrospective_only_does_not_borrow_boundary_as_alert():
    result = analyze(fixture(falling=True))
    target = row(result)
    assert target.sequential_status == "no_alert"
    assert target.retrospective_status == "change"
    assert target.changepoint_step == 4
    assert pd.isna(target.alert_step) and pd.isna(target.onset_step)
    for field in [
        "replication_status",
        "persistence_status",
        "reward_status",
        "promptmix_status",
    ]:
        assert target[field] == "not_applicable_no_sequential_alert"


def test_full_uncertainty_failure_stays_unavailable():
    result = analyze(fixture(single_prompt=True))
    target = row(result)
    assert target.sequential_status == "insufficient_baseline_uncertainty"
    assert target.retrospective_status == "unavailable_uncertainty"
    assert target.replication_status == "full_unscorable"
    assert target.persistence_status == "not_applicable_full_unscorable"
    assert pd.isna(target.persistence_fraction) and pd.isna(target.replicates)


def test_missing_half_step_stops_before_gap_and_preserves_earlier_alert():
    for missing_step, expected in [(4, "unavailable"), (7, "replicated")]:
        result = analyze(fixture(omit=[(0, missing_step)]))
        assert row(result).replication_status == expected
        half = result.replication_halves.query(
            'feature == "def:solve" and half == 0'
        ).iloc[0]
        assert half.missing_steps == [missing_step]
        assert half.n_steps == missing_step - 1


def test_pending_no_tail_and_no_shared_prompts_are_not_negative_claims():
    result = analyze(fixture(last=6, one_half=True, switched_prompts=True))
    target = row(result)
    assert target.replication_status == "pending"
    assert target.persistence_status == "no_future_observations"
    assert pd.isna(target.persistence_fraction)
    assert target.prompt_shared_prompts == 0 and pd.isna(target.prompt_shared_change)
    assert target.promptmix_status == "no_shared_prompts"
    assert row(result, "scalar:reward").reward_status == "not_binary"


def test_no_mixed_groups_remain_explicit():
    data = fixture()
    data = data.loc[data.response.str.startswith("def") | data.step.le(3)].copy()
    result = analyze(data)
    assert row(result).reward_status == "no_mixed_groups"
    assert row(result).reward_n_included_rollouts == 0
    assert pd.isna(row(result).reward_mean_difference)


def test_multiple_runs_with_reused_sample_ids_and_input_order_are_isolated():
    data = fixture()
    other = data.assign(run="other", response="constant words.", reward=0.0)
    combined = pd.concat([data, other], ignore_index=True).sample(
        frac=1, random_state=17
    )
    saved = combined.copy(deep=True)
    result = analyze(combined)
    assert row(result).sequential_status == "alerted"
    assert row(result, run="other").sequential_status == "no_alert"
    assert len(result.evidence) == 2 * (len(result.features.catalog) + 1)
    assert_frame_equal(combined, saved)


def test_aggregation_runs_once_for_full_data_and_each_prompt_half(monkeypatch):
    from rolloutdiff import pipeline

    original, calls = pipeline.aggregate, []

    def counted(data, features, **kwargs):
        calls.append((len(data), set(features.catalog.feature)))
        return original(data, features, **kwargs)

    monkeypatch.setattr(pipeline, "aggregate", counted)
    result = analyze(fixture())
    assert len(calls) == 3
    assert calls[0][0] == calls[1][0] + calls[2][0]
    assert all(catalog == set(result.features.catalog.feature) for _, catalog in calls)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sequential_options": {"unknown": 3}},
        {"retrospective_options": {"unknown": 3}},
        {"replication_tolerance": True},
        {"persistence_effect_fraction": 0},
        {"min_prevalence": 2},
    ],
)
def test_invalid_configuration_fails_even_without_alerts(kwargs):
    data = fixture().assign(response="constant", reward=0.0)
    with pytest.raises((ValueError, TypeError)):
        analyze(data, **kwargs)


def test_halves_without_measured_uncertainty_stay_unavailable():
    data = fixture()
    selected_prompts = (
        data[["prompt_id"]]
        .drop_duplicates()
        .assign(
            half=lambda frame: frame.prompt_id.map(
                lambda prompt: zlib.crc32(prompt.encode()) & 1
            )
        )
    )
    names = selected_prompts.groupby("half").prompt_id.first()
    data = data.loc[data.prompt_id.isin(names)]
    result = analyze(data)
    target = row(result)
    assert target.sequential_status == "alerted"
    assert target.replication_status == "unavailable" and pd.isna(target.replicates)
    halves = result.replication_halves.query('feature == "def:solve"')
    assert halves.reason.eq("insufficient_baseline_uncertainty").all()


def test_missing_post_alert_uncertainty_reduces_persistence_coverage():
    data = fixture()
    data = data.loc[~data.step.eq(8) | data.prompt_id.eq(data.prompt_id.iloc[0])]
    result = analyze(data)
    target = row(result)
    assert target.sequential_status == "alerted"
    assert target.retrospective_status == "unavailable_uncertainty"
    assert target.persistence_status == "partial"
    assert target.persistence_n_observations == 8
    assert target.persistence_n_unavailable == 1
    assert target.persistence_coverage == pytest.approx(7 / 8)
    assert target.persistence_fraction == 1


def test_empty_input_has_no_invented_run_or_evidence_rows():
    data = fixture().iloc[:0]
    result = analyze(data)
    assert result.evidence.empty and result.aggregation.series.empty
    assert (
        result.sequential.diagnostics.empty and result.retrospective.diagnostics.empty
    )
    assert result.half_assignments.empty


def test_large_integer_step_identity_is_preserved_with_missing_result_steps():
    data = fixture()
    data["step"] += 2**53
    result = analyze(data)
    target = row(result)
    assert target.onset_step == 2**53 + 4
    assert target.alert_step == 2**53 + 6
    assert target.changepoint_step == 2**53 + 4
    assert target.prompt_before_start == 2**53 + 1
    assert target.replication_cutoff_step == 2**53 + 8
    half = result.replication_halves.query('feature == "def:solve"').iloc[0]
    assert half.alert_step == 2**53 + 6


def test_extraction_and_detector_options_are_forwarded_without_global_mutation():
    options = {"baseline_steps": 2, "sustain": 1}
    result = analyze(
        fixture(),
        max_sparse_per_tier=0,
        sequential_options=options,
        persistence_effect_fraction=0.75,
    )
    assert result.features.sparse.empty
    assert set(result.features.catalog.feature) == set(
        result.features.dense.columns
    ) - {"run", "sample_id"}
    assert row(result, "scalar:reward").alert_step == 4
    assert result.persistence_steps.effect_fraction.eq(0.75).all()
    assert options == {"baseline_steps": 2, "sustain": 1}


def test_detail_concat_preserves_all_missing_columns_and_large_optional_steps():
    from rolloutdiff.pipeline import _details

    missing = pd.DataFrame(
        {
            "run": ["first"],
            "feature": ["flag"],
            "scale": [None],
            "alert_step": pd.Series([None], dtype=object),
            "reason": ["missing_uncertainty"],
            "always_missing": [None],
        }
    )
    available = pd.DataFrame(
        {
            "run": ["second"],
            "feature": ["flag"],
            "scale": [0.5],
            "alert_step": pd.Series([2**53 + 1], dtype=object),
            "reason": [None],
            "always_missing": [None],
        }
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = _details([missing, available])
    assert result.columns.tolist() == missing.columns.tolist()
    assert pd.isna(result.loc[0, "scale"]) and result.loc[1, "scale"] == 0.5
    assert pd.isna(result.loc[0, "alert_step"])
    assert result.loc[1, "alert_step"] == 2**53 + 1
    assert result.loc[0, "reason"] == "missing_uncertainty"
    assert pd.isna(result.loc[1, "reason"])
    assert result.always_missing.isna().all()
