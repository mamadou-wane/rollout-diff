"""Opt-in retrospective checks on three fixed Aria series."""

import math

import pytest
from _aria import load_aria

from rolloutdiff.detect import detect_sequential
from rolloutdiff.retrospective import detect_retrospective
from rolloutdiff.series import aggregate

pytestmark = pytest.mark.local_data

# Policy: one globally SSE-minimizing mean split,
# at least five observations per side, variance=max(split SSE/(n-2), mean(SE^2)),
# SSE gain > 3*log(n)*variance, and binary absolute mean change >= .10.
# These literal results come from exhaustive direct residual calculations below.
# Detection accuracy and calibration are unmeasured by these named-series checks.
_EXPECTED = {
    "call:bisect.bisect_left": {
        "status": "no_change",
        "candidate_step": 196,
        "previous_step": 195,
        "n_before": 195,
        "n_after": 5,
        "mean_before": 0.009635416666666667,
        "mean_after": 0.0296875,
        "delta": 0.02005208333333333,
        "rms_se_before": 0.017957714467071693,
        "rms_se_after": 0.03261972139608696,
        "sse_gain": 0.00196016947428386,
        "penalty": 0.005925923326602894,
    },
    "def:run_tests": {
        "status": "change",
        "candidate_step": 117,
        "previous_step": 116,
        "n_before": 116,
        "n_after": 84,
        "mean_before": 0.022191540948275863,
        "mean_after": 0.9633556547619048,
        "delta": 0.9411641138136289,
        "rms_se_before": 0.01816402629074997,
        "rms_se_after": 0.023459191351241557,
        "sse_gain": 43.15568339844251,
        "penalty": 0.07132692354649886,
    },
    "scalar:reward": {
        "status": "change",
        "candidate_step": 118,
        "previous_step": 117,
        "n_before": 117,
        "n_after": 83,
        "mean_before": 1.5514823717948718,
        "mean_after": 3.2222091490963853,
        "delta": 1.6707267773015135,
        "rms_se_before": 0.2643338041419041,
        "rms_se_after": 0.11093946371342359,
        "sse_gain": 135.53292931106822,
        "penalty": 1.3733568541582504,
    },
}


def _direct_reference(table):
    """Calculate exhaustive residual SSE independently of detector helpers."""
    table = table.sort_values("step")
    values = table["mean"].tolist()
    standard_errors = table["se"].tolist()
    assert all(math.isfinite(value) for value in standard_errors)
    n = len(values)
    overall_mean = math.fsum(values) / n
    unsplit_sse = math.fsum((value - overall_mean) ** 2 for value in values)
    candidates = []
    for split in range(5, n - 4):
        before, after = values[:split], values[split:]
        mean_before = math.fsum(before) / len(before)
        mean_after = math.fsum(after) / len(after)
        split_sse = math.fsum(
            (value - mean_before) ** 2 for value in before
        ) + math.fsum((value - mean_after) ** 2 for value in after)
        candidates.append((split_sse, split, mean_before, mean_after))
    split_sse, split, mean_before, mean_after = min(candidates)
    variance = max(
        split_sse / (n - 2),
        math.fsum(value**2 for value in standard_errors) / n,
    )
    gain = unsplit_sse - split_sse
    penalty = 3 * math.log(n) * variance
    delta = mean_after - mean_before
    minimum = 0.10 if table.kind.iloc[0] == "binary" else 0
    accepted = gain > penalty and abs(delta) >= minimum
    return {
        "status": "change" if accepted else "no_change",
        "candidate_step": int(table.step.iloc[split]),
        "previous_step": int(table.step.iloc[split - 1]),
        "n_before": split,
        "n_after": n - split,
        "mean_before": mean_before,
        "mean_after": mean_after,
        "delta": delta,
        "rms_se_before": math.sqrt(
            math.fsum(value**2 for value in standard_errors[:split]) / split
        ),
        "rms_se_after": math.sqrt(
            math.fsum(value**2 for value in standard_errors[split:]) / (n - split)
        ),
        "sse_gain": gain,
        "penalty": penalty,
    }


@pytest.fixture(scope="module")
def aria_series():
    rollouts, features = load_aria([(1, 200)])
    assert len(rollouts) == 51200
    assert rollouts.step.nunique() == 200
    series = aggregate(rollouts, features).series
    assert set(series.feature) == set(_EXPECTED)
    assert len(series) == 600
    assert series.n_rollouts.eq(256).all()
    assert series.n_prompts.eq(16).all()
    assert series.se.notna().all()
    return series


@pytest.fixture(scope="module")
def aria_reference(aria_series):
    return {
        feature: _direct_reference(table)
        for feature, table in aria_series.groupby("feature", sort=True)
    }


@pytest.fixture(scope="module")
def aria_retrospective(aria_series):
    return detect_retrospective(aria_series)


def test_local_aria_direct_reference(aria_reference):
    for feature, expected in _EXPECTED.items():
        for field, value in expected.items():
            actual = aria_reference[feature][field]
            if isinstance(value, float):
                assert actual == pytest.approx(value)
            else:
                assert actual == value
    # The weak candidate fails both the fit-improvement and magnitude policies.
    weak = aria_reference["call:bisect.bisect_left"]
    assert weak["sse_gain"] < weak["penalty"]
    assert abs(weak["delta"]) < 0.10


def test_local_aria_retrospective_matches_direct_reference(
    aria_retrospective, aria_reference
):
    diagnostics = aria_retrospective.diagnostics.set_index("feature")
    assert set(diagnostics.index) == set(_EXPECTED)
    assert diagnostics.run.eq("aria").all()
    assert diagnostics.n_observations.eq(200).all()
    assert diagnostics.n_candidates.eq(191).all()
    assert diagnostics.n_missing_se.eq(0).all()
    assert diagnostics.min_segment_steps.eq(5).all()
    assert diagnostics.penalty_multiplier.eq(3).all()
    for feature, expected in aria_reference.items():
        row = diagnostics.loc[feature]
        assert row.status == expected["status"]
        assert row.reason == (
            "penalized_fit" if expected["status"] == "change" else "below_effect"
        )
        assert row.min_effect == (0.10 if row.kind == "binary" else 0)
        assert row.normalization_scale > 0
        assert row.gain_scaled * row.normalization_scale**2 == pytest.approx(
            expected["sse_gain"]
        )
        assert row.penalty_scaled * row.normalization_scale**2 == pytest.approx(
            expected["penalty"]
        )
        assert row.candidate_step == expected["candidate_step"]
        assert row.previous_step == expected["previous_step"]
        assert row.n_before == expected["n_before"]
        assert row.n_after == expected["n_after"]
        for field in ["mean_before", "mean_after", "delta"]:
            assert row[field] == pytest.approx(expected[field])

    changes = aria_retrospective.changepoints.set_index("feature")
    assert set(changes.index) == {"def:run_tests", "scalar:reward"}
    assert changes.run.eq("aria").all()
    assert changes.direction.eq("up").all()
    for feature, row in changes.iterrows():
        expected = aria_reference[feature]
        assert row.changepoint_step == expected["candidate_step"]
        for field in ["previous_step", "n_before", "n_after"]:
            assert row[field] == expected[field]
        for field in [
            "mean_before",
            "mean_after",
            "delta",
            "rms_se_before",
            "rms_se_after",
        ]:
            assert row[field] == pytest.approx(expected[field])
    assert aria_retrospective.screening.to_dict("records") == [
        {
            "run": "aria",
            "n_series": 3,
            "n_screened": 3,
            "n_changes": 2,
            "n_unscorable": 0,
        }
    ]
    assert not {"onset_step", "alert_step"}.intersection(
        aria_retrospective.changepoints.columns
    )


def test_local_aria_retrospective_is_distinct_from_sequential(
    aria_series, aria_retrospective
):
    alerts = detect_sequential(aria_series).alerts.set_index("feature")
    assert alerts[["onset_step", "alert_step"]].to_dict("index") == {
        "def:run_tests": {"onset_step": 110, "alert_step": 112},
        "scalar:reward": {"onset_step": 118, "alert_step": 120},
    }
    changes = aria_retrospective.changepoints.set_index("feature")
    assert changes.loc["def:run_tests", "changepoint_step"] == 117
    assert changes.loc["scalar:reward", "changepoint_step"] == 118
    # Full-series boundaries and causal monitoring times answer different questions.
    assert (
        changes.loc["def:run_tests", "changepoint_step"]
        != alerts.loc["def:run_tests", "onset_step"]
    )
    for feature in changes.index:
        assert (
            changes.loc[feature, "changepoint_step"]
            != alerts.loc[feature, "alert_step"]
        )
