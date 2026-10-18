"""Opt-in evidence checks on three fixed Aria features."""

import math
import zlib

import pandas as pd
import pytest
from _aria import load_aria

from rolloutdiff.detect import detect_sequential
from rolloutdiff.evidence import persistence, reward_association
from rolloutdiff.features import FeatureTables
from rolloutdiff.replication import split_half_replication
from rolloutdiff.series import aggregate

pytestmark = pytest.mark.local_data

# Policy: whole-prompt CRC32 halves; the default
# sequential detector in each half; at most ten full-run logged observations
# after the full alert; all three first alerts agree in direction and fall
# within ten observations. Persistence starts strictly after the full alert,
# retaining half its signed baseline departure. Reward differences use only
# mixed (step, prompt) groups over onset through alert, with equal group weight.
# These named-series checks leave error calibration unmeasured.
_EXPECTED = {
    "def:run_tests": {
        "onset_step": 110,
        "alert_step": 112,
        "cutoff_step": 122,
        "halves": [
            {
                "half": 0,
                "onset_step": 110,
                "alert_step": 112,
                "n_prompts": 509,
                "n_rollouts": 16064,
                "n_steps": 122,
                "n_scored": 107,
            },
            {
                "half": 1,
                "onset_step": 113,
                "alert_step": 115,
                "n_prompts": 483,
                "n_rollouts": 15168,
                "n_steps": 122,
                "n_scored": 107,
            },
        ],
        "n_tail": 88,
        "threshold": 0.08203125,
    },
    "scalar:reward": {
        "onset_step": 118,
        "alert_step": 120,
        "cutoff_step": 130,
        "halves": [
            {
                "half": 0,
                "onset_step": 127,
                "alert_step": 129,
                "n_prompts": 509,
                "n_rollouts": 17056,
                "n_steps": 130,
                "n_scored": 115,
            },
            {
                "half": 1,
                "onset_step": 119,
                "alert_step": 121,
                "n_prompts": 483,
                "n_rollouts": 16224,
                "n_steps": 130,
                "n_scored": 115,
            },
        ],
        "n_tail": 80,
        "threshold": 2.082486979166667,
    },
}


def _direct_halves(rollouts, features, assignments, steps, cutoff, feature):
    """Partition canonical rows directly and aggregate each half."""
    expected_steps = steps[: steps.index(cutoff) + 1]
    halves = []
    for half in (0, 1):
        prompts = set(assignments.loc[assignments.half.eq(half), "prompt_id"])
        rows = rollouts.loc[rollouts.prompt_id.isin(prompts) & rollouts.step.le(cutoff)]
        missing = sorted(set(expected_steps) - set(rows.step))
        if missing:
            rows = rows.loc[rows.step.lt(missing[0])]
        sample_ids = set(rows.sample_id)
        half_features = FeatureTables(
            dense=features.dense.loc[features.dense.sample_id.isin(sample_ids)],
            sparse=features.sparse.loc[features.sparse.sample_id.isin(sample_ids)],
            catalog=features.catalog,
        )
        series = aggregate(rows, half_features).series
        detected = detect_sequential(series.loc[series.feature.eq(feature)])
        assert len(detected.alerts) == 1
        alert = detected.alerts.iloc[0]
        diagnostic = detected.diagnostics.iloc[0]
        halves.append(
            {
                "half": half,
                "onset_step": int(alert.onset_step),
                "alert_step": int(alert.alert_step),
                "direction": alert.direction,
                "alert_position": steps.index(alert.alert_step),
                "n_prompts": rows.prompt_id.nunique(),
                "n_rollouts": len(rows),
                "n_steps": rows.step.nunique(),
                "n_scored": int(diagnostic.n_scored),
                "missing_steps": missing,
            }
        )
    return halves


def _direct_association(rollouts, features, feature, start, end):
    """Compute per-group reward means directly from completion membership."""
    present_ids = set(
        features.sparse.loc[features.sparse.feature.eq(feature), "sample_id"]
    )
    window = rollouts.loc[rollouts.step.between(start, end)].copy()
    window["present"] = window.sample_id.isin(present_ids)
    mixed = []
    for (step, prompt_id), group in window.groupby(["step", "prompt_id"], sort=True):
        present = group.loc[group.present, "reward"].tolist()
        absent = group.loc[~group.present, "reward"].tolist()
        if present and absent:
            difference = math.fsum(present) / len(present) - math.fsum(absent) / len(
                absent
            )
            mixed.append(
                {
                    "step": int(step),
                    "prompt_id": prompt_id,
                    "n_present": len(present),
                    "n_absent": len(absent),
                    "difference": difference,
                }
            )
    return window, pd.DataFrame(mixed)


@pytest.fixture(scope="module")
def aria_data():
    # Read and extract the entire pinned local fixture only once in this module.
    rollouts, features = load_aria([(1, 200)])
    assert len(rollouts) == 51200
    assert rollouts.step.nunique() == 200
    series = aggregate(rollouts, features).series
    full = detect_sequential(series)
    assignments = rollouts[["run", "prompt_id"]].drop_duplicates().copy()
    assignments["sampling_unit_id"] = assignments.prompt_id
    assignments["half"] = assignments.prompt_id.map(
        lambda value: zlib.crc32(value.encode("utf-8")) & 1
    )
    assignments = assignments.sort_values(["run", "prompt_id"]).reset_index(drop=True)
    references = {}
    for feature in _EXPECTED:
        table = series.loc[series.feature.eq(feature)].sort_values("step")
        steps = table.step.tolist()
        alert = full.alerts.loc[full.alerts.feature.eq(feature)].iloc[0]
        cutoff = steps[min(steps.index(alert.alert_step) + 10, len(steps) - 1)]
        tail = table.loc[table.step.gt(alert.alert_step)]
        threshold = 0.5 * abs(alert.delta_at_alert)
        sign = 1 if alert.direction == "up" else -1
        evaluable = tail.se.notna()
        retained = evaluable & (sign * (tail["mean"] - alert.baseline_mean)).ge(
            threshold
        )
        references[feature] = {
            "halves": _direct_halves(
                rollouts, features, assignments, steps, cutoff, feature
            ),
            "cutoff": cutoff,
            "tail": tail,
            "n_evaluable": int(evaluable.sum()),
            "n_retained": int(retained.sum()),
            "threshold": alert.baseline_mean + 0.5 * alert.delta_at_alert,
        }
    window, groups = _direct_association(rollouts, features, "def:run_tests", 110, 112)
    return {
        "rollouts": rollouts,
        "features": features,
        "series": series,
        "full": full,
        "assignments": assignments,
        "references": references,
        "association_window": window,
        "association_groups": groups,
    }


@pytest.fixture(scope="module")
def aria_evidence(aria_data):
    rollouts, features = aria_data["rollouts"], aria_data["features"]
    sources = [
        rollouts,
        features.dense,
        features.sparse,
        features.catalog,
        aria_data["series"],
    ]
    originals = [table.copy(deep=True) for table in sources]
    results = {}
    for feature in _EXPECTED:
        alert = (
            aria_data["full"]
            .alerts.loc[aria_data["full"].alerts.feature.eq(feature)]
            .iloc[0]
        )
        results[feature] = {
            "replication": split_half_replication(
                rollouts, features, run="aria", feature=feature
            ),
            "persistence": persistence(
                aria_data["series"],
                run="aria",
                feature=feature,
                after_step=int(alert.alert_step),
                reference_mean=float(alert.baseline_mean),
                change=float(alert.delta_at_alert),
            ),
            "association": reward_association(
                rollouts,
                features,
                run="aria",
                feature=feature,
                window=(int(alert.onset_step), int(alert.alert_step)),
            ),
        }
    results["call:bisect.bisect_left"] = split_half_replication(
        rollouts,
        features,
        run="aria",
        feature="call:bisect.bisect_left",
    )
    for actual, original in zip(sources, originals, strict=False):
        pd.testing.assert_frame_equal(actual, original)
    return results


def test_local_aria_fixed_policy_references(aria_data):
    assert aria_data["assignments"].groupby("half").size().to_dict() == {0: 509, 1: 483}
    for feature, expected in _EXPECTED.items():
        alert = (
            aria_data["full"]
            .alerts.loc[aria_data["full"].alerts.feature.eq(feature)]
            .iloc[0]
        )
        assert alert.onset_step == expected["onset_step"]
        assert alert.alert_step == expected["alert_step"]
        reference = aria_data["references"][feature]
        assert reference["cutoff"] == expected["cutoff_step"]
        for actual, literal in zip(
            reference["halves"], expected["halves"], strict=False
        ):
            for name, value in literal.items():
                assert actual[name] == value
            assert actual["direction"] == "up"
            assert actual["missing_steps"] == []
        assert len(reference["tail"]) == expected["n_tail"]
        assert reference["n_evaluable"] == reference["n_retained"] == expected["n_tail"]
        assert reference["threshold"] == pytest.approx(expected["threshold"])
    groups = aria_data["association_groups"]
    window = aria_data["association_window"]
    assert len(window) == 768
    assert window.groupby(["step", "prompt_id"]).ngroups == 48
    assert len(groups) == 29
    assert (groups.n_present + groups.n_absent).sum() == 464
    assert math.fsum(groups.difference) / len(groups) == pytest.approx(
        0.7442781356574459
    )


@pytest.mark.parametrize("feature", _EXPECTED)
def test_local_aria_split_halves_match_direct_reference(
    aria_data, aria_evidence, feature
):
    result = aria_evidence[feature]["replication"]
    expected = _EXPECTED[feature]
    assert result.status == "replicated"
    assert result.replicates
    assert result.horizon_complete
    assert result.cutoff_step == expected["cutoff_step"]
    pd.testing.assert_frame_equal(result.assignments, aria_data["assignments"])
    for actual, reference in zip(
        result.halves.sort_values("half").to_dict("records"),
        aria_data["references"][feature]["halves"],
        strict=False,
    ):
        for name, value in reference.items():
            if name == "missing_steps":
                assert list(actual[name]) == value
            else:
                assert actual[name] == value
        assert actual["n_unscorable"] == 0
    assert len(result.full.alerts) == 1
    assert result.full.alerts.iloc[0].alert_step == expected["alert_step"]
    assert result.full.alerts.iloc[0].onset_step == expected["onset_step"]

    # Each prompt's completions must stay together and preserve step counts.
    assigned = aria_data["rollouts"].merge(
        result.assignments, on=["run", "prompt_id"], validate="many_to_one"
    )
    assert assigned.groupby(["run", "prompt_id"]).half.nunique().eq(1).all()
    halves = [set(assigned.loc[assigned.half.eq(half), "prompt_id"]) for half in (0, 1)]
    assert halves[0].isdisjoint(halves[1])
    assert halves[0] | halves[1] == set(aria_data["rollouts"].prompt_id)
    assert assigned.groupby(["step", "prompt_id"]).size().eq(16).all()
    histogram = assigned.groupby(["step", "half"]).size().unstack(fill_value=0)
    assert histogram.sum(axis=1).eq(256).all()
    assert (histogram % 16).eq(0).all().all()
    for half, reference in enumerate(aria_data["references"][feature]["halves"]):
        assert (
            histogram.loc[: expected["cutoff_step"], half].sum()
            == reference["n_rollouts"]
        )


@pytest.mark.parametrize("feature", _EXPECTED)
def test_local_aria_persistence_matches_direct_reference(
    aria_data, aria_evidence, feature
):
    result = aria_evidence[feature]["persistence"]
    expected = _EXPECTED[feature]
    reference = aria_data["references"][feature]
    assert result.status == "available"
    assert result.n_observations == expected["n_tail"]
    assert result.n_evaluable == reference["n_evaluable"]
    assert result.n_retained == reference["n_retained"]
    assert result.n_unavailable == 0
    assert result.fraction_retained == 1.0
    assert result.coverage == 1.0
    assert result.threshold == pytest.approx(expected["threshold"])
    assert result.direction == "up"
    assert result.steps.step.tolist() == reference["tail"].step.tolist()
    assert result.steps.step.min() > expected["alert_step"]


def test_local_aria_reward_association_matches_direct_reference(
    aria_data, aria_evidence
):
    result = aria_evidence["def:run_tests"]["association"]
    groups = aria_data["association_groups"]
    assert result.status == "available"
    assert result.n_groups == 48
    assert result.n_mixed_groups == 29
    assert result.n_rollouts == 768
    assert result.n_included_rollouts == 464
    assert result.mean_difference == pytest.approx(
        math.fsum(groups.difference) / len(groups)
    )
    assert result.mean_difference == pytest.approx(0.7442781356574459)
    mixed = result.groups.loc[
        result.groups.n_present.gt(0) & result.groups.n_absent.gt(0)
    ]
    pd.testing.assert_frame_equal(
        mixed[groups.columns].reset_index(drop=True),
        groups.reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )
    assert len(result.groups) == 48
    assert result.groups.run.eq("aria").all()
    assert result.groups.n_rollouts.eq(
        result.groups.n_present + result.groups.n_absent
    ).all()
    assert (
        result.groups.loc[~result.groups.index.isin(mixed.index), "difference"]
        .isna()
        .all()
    )
    # Dependent training observations support a descriptive reward difference,
    # without an inferential interval, p-value, or causal interpretation.
    assert not {"pvalue", "p_value", "ci_low", "ci_high"}.intersection(
        result.groups.columns
    )


def test_local_aria_unavailable_evidence_is_explicit(aria_evidence):
    absent = aria_evidence["call:bisect.bisect_left"]
    assert absent.status == "no_full_alert"
    assert absent.replicates is None
    assert absent.full.alerts.empty
    continuous = aria_evidence["scalar:reward"]["association"]
    assert continuous.status == "not_binary"
    assert continuous.mean_difference is None
    assert continuous.n_included_rollouts == 0
    assert continuous.groups.empty
