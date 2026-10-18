"""Uncalibrated segmentation checks using synthetic baseline parameters.

Reuse the sequential detector's large-shift generator with the same seeds
and parameters. Measure retrospective localization against the planted
boundary, independently of sequential alert delays, with a tolerance of
two observations.
"""

import pytest
from test_synthetic_detection import N_CASES, ONSET, controlled_series

MAX_LOCALIZATION_ERROR = 2


@pytest.fixture(scope="module")
def planted_series():
    return controlled_series(planted=True)


def test_planted_binary_and_reward_boundaries_are_localized(planted_series):
    from rolloutdiff.retrospective import detect_retrospective

    result = detect_retrospective(planted_series)
    changepoints = result.changepoints
    assert len(changepoints) == N_CASES
    assert changepoints["run"].is_unique
    assert set(changepoints["run"]) == set(planted_series["run"])
    assert (
        (changepoints["changepoint_step"] - ONSET)
        .abs()
        .le(MAX_LOCALIZATION_ERROR)
        .all()
    )
    assert changepoints["previous_step"].eq(changepoints["changepoint_step"] - 1).all()
    assert changepoints["run"].str.startswith("binary-").sum() == N_CASES // 2
    assert changepoints["run"].str.startswith("continuous-").sum() == N_CASES // 2
    for row in changepoints.itertuples():
        downward = (
            row.run.startswith("continuous-") and int(row.run.rsplit("-", 1)[1]) % 2
        )
        assert row.direction == ("down" if downward else "up")
        assert row.delta < 0 if downward else row.delta > 0
    assert result.diagnostics["status"].eq("change").all()


def test_unchanged_binary_and_reward_series_allow_no_changepoint():
    from rolloutdiff.retrospective import detect_retrospective

    result = detect_retrospective(controlled_series(planted=False))
    assert result.changepoints.empty
    assert len(result.diagnostics) == N_CASES
    assert result.diagnostics["status"].eq("no_change").all()
    assert result.screening["n_screened"].sum() == N_CASES
    assert result.screening["n_changes"].sum() == 0
    assert result.screening["n_unscorable"].sum() == 0


def test_downward_binary_changes_are_localized(planted_series):
    from rolloutdiff.retrospective import detect_retrospective

    # Complementing the observed prevalence reverses the planted effect and
    # preserves its observed SE and the generator parameters.
    series = planted_series.loc[planted_series["kind"].eq("binary")].copy()
    series["mean"] = 1.0 - series["mean"]
    result = detect_retrospective(series)
    changepoints = result.changepoints
    assert len(changepoints) == N_CASES // 2
    assert changepoints["run"].is_unique
    assert (
        (changepoints["changepoint_step"] - ONSET)
        .abs()
        .le(MAX_LOCALIZATION_ERROR)
        .all()
    )
    assert changepoints["direction"].eq("down").all()
    assert changepoints["delta"].lt(-0.1).all()
    assert result.diagnostics["status"].eq("change").all()
