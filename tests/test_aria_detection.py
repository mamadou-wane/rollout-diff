"""Opt-in sequential checks on three named Aria series."""

import pandas as pd
import pytest
from _aria import load_aria

from rolloutdiff.detect import detect_sequential
from rolloutdiff.series import aggregate

pytestmark = pytest.mark.local_data

_BASELINES = {
    "call:bisect.bisect_left": {
        "baseline_mean": 0.004427083333333333,
        "baseline_sd": 0.007932486347781388,
        "baseline_se_floor": 0.0,
        "scale": 0.007932486347781388,
        "numerical_floor_dominant": False,
        "status": "no_alert",
    },
    "def:run_tests": {
        "baseline_mean": 0.0,
        "baseline_sd": 0.0,
        "baseline_se_floor": 0.0,
        "scale": 1e-9,
        "numerical_floor_dominant": True,
        "status": "alerted",
    },
    "scalar:reward": {
        "baseline_mean": 1.2196614583333334,
        "baseline_sd": 0.29167858281089637,
        "baseline_se_floor": 0.24956780268048526,
        "scale": 0.29167858281089637,
        "numerical_floor_dominant": False,
        "status": "alerted",
    },
}
_ALERTS = {
    "def:run_tests": {
        "onset_step": 110,
        "alert_step": 112,
        "direction": "up",
        "value_at_alert": 0.1640625,
        "delta_at_alert": 0.1640625,
        "z_at_alert": 164062500.0,
        "se_at_alert": 0.049038608832735045,
    },
    "scalar:reward": {
        "onset_step": 118,
        "alert_step": 120,
        "direction": "up",
        "value_at_alert": 2.9453125,
        "delta_at_alert": 1.7256510416666666,
        "z_at_alert": 5.916276145600501,
        "se_at_alert": 0.1470604989947448,
    },
}


@pytest.fixture(scope="module")
def aria_series():
    rollouts, features = load_aria([(1, 200)])
    assert len(rollouts) == 51200
    assert rollouts.step.nunique() == 200
    result = aggregate(rollouts, features)
    assert set(result.series.feature) == set(_BASELINES)
    assert len(result.series) == 600
    assert result.series.n_rollouts.eq(256).all()
    assert result.series.n_prompts.eq(16).all()
    return result.series


@pytest.fixture(scope="module")
def aria_detection(aria_series):
    return detect_sequential(aria_series)


def test_local_aria_named_series(aria_detection):
    result = aria_detection
    diagnostics = result.diagnostics.set_index("feature")
    assert set(diagnostics.index) == set(_BASELINES)
    assert diagnostics.run.eq("aria").all()
    assert diagnostics.baseline_n.eq(15).all()
    assert diagnostics.baseline_start.eq(1).all()
    assert diagnostics.baseline_end.eq(15).all()
    assert diagnostics.n_scored.eq(185).all()
    assert diagnostics.n_unscorable.eq(0).all()
    assert diagnostics.numerical_floor.eq(1e-9).all()
    for feature, expected in _BASELINES.items():
        row = diagnostics.loc[feature]
        for field, value in expected.items():
            if isinstance(value, float):
                assert row[field] == pytest.approx(value)
            else:
                assert row[field] == value

    alerts = result.alerts.set_index("feature")
    assert set(alerts.index) == set(_ALERTS)
    assert alerts.n_prompts_at_alert.eq(16).all()
    assert alerts.n_rollouts_at_alert.eq(256).all()
    for feature, expected in _ALERTS.items():
        row = alerts.loc[feature]
        for field, value in expected.items():
            if isinstance(value, float):
                assert row[field] == pytest.approx(value)
            else:
                assert row[field] == value
        assert row.scale == pytest.approx(_BASELINES[feature]["scale"])
        assert (
            row.numerical_floor_dominant
            == _BASELINES[feature]["numerical_floor_dominant"]
        )

    scores = result.scores
    assert len(scores) == 600
    baseline = scores.loc[scores.step.le(15)]
    assert len(baseline) == 45
    assert baseline.status.eq("baseline").all()
    assert baseline.z.isna().all()
    assert baseline.delta.isna().all()
    assert not baseline.qualifies.any()
    assert scores.loc[scores.step.gt(15), "status"].eq("scored").all()
    assert result.screening.to_dict("records") == [
        {
            "run": "aria",
            "n_series": 3,
            "n_screened": 3,
            "n_alerted": 2,
            "n_unscorable": 0,
        }
    ]


@pytest.mark.parametrize("cutoff", [112, 120])
def test_local_aria_detection_is_prefix_invariant(aria_series, aria_detection, cutoff):
    prefix = detect_sequential(aria_series.loc[aria_series.step.le(cutoff)])
    expected_alerts = aria_detection.alerts.loc[
        aria_detection.alerts.alert_step.le(cutoff)
    ]
    expected_scores = aria_detection.scores.loc[aria_detection.scores.step.le(cutoff)]
    pd.testing.assert_frame_equal(
        prefix.alerts.reset_index(drop=True),
        expected_alerts.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        prefix.scores.reset_index(drop=True),
        expected_scores.reset_index(drop=True),
    )

    changed = aria_series.copy()
    future = changed.step.gt(cutoff)
    changed.loc[future & changed.kind.eq("binary"), "mean"] = 0.5
    changed.loc[future & changed.kind.eq("continuous"), "mean"] = -1.0
    changed.loc[future, "se"] = 0.0
    future_result = detect_sequential(changed)
    pd.testing.assert_frame_equal(
        future_result.scores.loc[future_result.scores.step.le(cutoff)].reset_index(
            drop=True
        ),
        expected_scores.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        future_result.alerts.loc[
            future_result.alerts.alert_step.le(cutoff)
        ].reset_index(drop=True),
        expected_alerts.reset_index(drop=True),
    )
    frozen_fields = [
        "run",
        "feature",
        "baseline_n",
        "baseline_start",
        "baseline_end",
        "baseline_mean",
        "baseline_sd",
        "baseline_se_floor",
        "numerical_floor",
        "scale",
        "numerical_floor_dominant",
    ]
    pd.testing.assert_frame_equal(
        future_result.diagnostics[frozen_fields],
        aria_detection.diagnostics[frozen_fields],
    )
