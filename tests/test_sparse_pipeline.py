import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import fixture

from rolloutdiff import analyze_rollouts, split_half_replication


def test_fixed_window_pipeline_restores_baseline_gap_without_changing_full_analysis():
    data = fixture(omit=[(0, 2)])
    saved = data.copy(deep=True)
    options = {
        "sequential_options": {"baseline_steps": 3},
        "retrospective_options": {"min_segment_steps": 3},
        "replication_tolerance": 2,
    }
    prefix = analyze_rollouts(data, **options)
    fixed = analyze_rollouts(
        data, **options, replication_coverage_policy="fixed_window"
    )
    old = prefix.evidence.set_index("feature").loc["def:solve"]
    new = fixed.evidence.set_index("feature").loc["def:solve"]
    assert old.replication_status == "unavailable"
    assert new.replication_status == "replicated" and new.replicates
    assert new.replication_coverage_policy == "fixed_window"
    assert_frame_equal(prefix.sequential.alerts, fixed.sequential.alerts)
    assert_frame_equal(
        prefix.retrospective.changepoints, fixed.retrospective.changepoints
    )
    assert_frame_equal(prefix.half_assignments, fixed.half_assignments)
    direct = split_half_replication(
        data,
        fixed.features,
        run="r",
        feature="def:solve",
        baseline_steps=3,
        tolerance=2,
        coverage_policy="fixed_window",
    )
    table = fixed.replication_halves.query('feature == "def:solve"').drop(
        columns=["run", "feature"]
    )
    assert_frame_equal(table.reset_index(drop=True), direct.halves, check_dtype=False)
    assert_frame_equal(data, saved)


@pytest.mark.parametrize("policy", ["fill_gaps", None, 1])
def test_bad_replication_policy_fails_even_without_full_alert(policy):
    with pytest.raises(ValueError, match="coverage_policy"):
        analyze_rollouts(
            fixture().assign(response="constant", reward=0.0),
            replication_coverage_policy=policy,
        )
