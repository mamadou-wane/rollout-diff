from types import SimpleNamespace

import pandas as pd
from test_full_pipeline import fixture


def test_benchmark_keeps_event_corroboration_separate_from_first_alert_replication():
    from bench.sparse_replication import compare_loaded

    # The earlier half gap leaves the first alert ambiguous, while the fixed
    # event window has a fully observed sustained same-direction witness.
    rollouts = fixture(omit=[(0, 4)]).assign(run="aria")
    rollouts["response"] = rollouts.response.str.replace(
        "def solve()", "def run_tests()", regex=False
    )
    saved = rollouts.copy(deep=True)
    data = SimpleNamespace(rollouts=rollouts, sampling_units=None, provenance={})
    report = compare_loaded(
        data,
        "aria",
        sequential_options={"baseline_steps": 3},
        tolerance=2,
        corroboration_radius=2,
    )
    rep = {
        row["coverage_policy"]: row
        for row in report["comparisons"]
        if row["feature"] == "def:run_tests"
    }
    event = next(
        row
        for row in report["corroboration_comparisons"]
        if row["feature"] == "def:run_tests"
    )
    assert (
        rep["contiguous_prefix"]["status"]
        == rep["fixed_window"]["status"]
        == "unavailable"
    )
    assert event["status"] == "corroborated" and event["corroborates"] is True
    assert event["evidence_kind"] == "event_window_corroboration"
    assert "replicates" not in event and "coverage_policy" not in event
    assert event["radius"] == 2
    assert event["window_start_step"] == 4 and event["window_end_step"] == 8
    assert event["full_alert_step"] == rep["contiguous_prefix"]["full_alert_step"] == 6
    half = next(row for row in event["halves"] if row["half"] == 0)
    assert half["missing_steps"] == [4]
    assert half["window_expected"] == 5 and half["window_available"] == 4
    assert half["match_start_step"] == 5 and half["match_end_step"] == 7
    assert (
        report["corroboration_summary"]["n_candidates"] == report["full"]["n_alerted"]
    )
    assert report["policy"]["event_window_corroboration"]["radius"] == 2
    assert report["unit_leakage"] == 0
    pd.testing.assert_frame_equal(rollouts, saved)


def test_corroboration_summary_preserves_unknowns_and_reports_coverage():
    from bench.sparse_replication import summarize_corroboration

    half = {
        "status": "unavailable",
        "reason": "incomplete_window",
        "window_expected": 5,
        "window_observed": 3,
        "window_available": 2,
        "n_scored": 2,
        "n_unscorable": 3,
        "future_steps": 1,
        "missing_steps": [2],
        "unavailable_se_steps": [3],
        "match_start_step": None,
        "opposite_start_step": None,
        "mixed_direction": False,
    }
    rows = [
        {"status": status, "corroborates": value, "halves": [half]}
        for status, value in [
            ("corroborated", True),
            ("contradicted", False),
            ("unavailable", None),
        ]
    ]
    result = summarize_corroboration(rows)
    assert result["n_candidates"] == 3
    assert (
        result["n_corroborated"]
        == result["n_contradicted"]
        == result["n_unavailable"]
        == 1
    )
    assert result["coverage_totals"] == {
        "window_expected": 15,
        "window_observed": 9,
        "window_available": 6,
        "n_scored": 6,
        "n_unscorable": 9,
        "future_steps": 3,
    }
    assert result["n_observed_matching_half_windows"] == 0
    assert result["n_observed_opposite_half_windows"] == 0
    assert result["n_mixed_direction_halves"] == 0


def test_benchmark_corroboration_keeps_non_target_feature_names_private():
    from bench.sparse_replication import compare_loaded

    data = SimpleNamespace(rollouts=fixture(last=6), sampling_units=None, provenance={})
    report = compare_loaded(
        data,
        "not-a-target-run",
        sequential_options={"baseline_steps": 3},
        tolerance=2,
        corroboration_radius=2,
    )
    assert report["corroboration_comparisons"]
    for row in report["corroboration_comparisons"]:
        assert row["feature"] is None and len(row["feature_sha256"]) == 64
        assert row["horizon_complete"] is False
    assert report["policy"]["no_labels_used"] is True


def test_corroboration_window_does_not_change_existing_replication_results():
    from bench.sparse_replication import compare_loaded

    class UnlabeledInput:
        def __init__(self):
            self.rollouts = fixture(one_half=True).assign(run="aria")
            self.sampling_units = None
            self.provenance = {}

        @property
        def labels(self):
            raise AssertionError("corroboration comparison must not read labels")

    data = UnlabeledInput()
    data.rollouts["response"] = data.rollouts.response.str.replace(
        "def solve()", "def run_tests()", regex=False
    )
    reports = [
        compare_loaded(
            data,
            "aria",
            sequential_options={"baseline_steps": 3},
            tolerance=2,
            corroboration_radius=radius,
        )
        for radius in (1, 2)
    ]
    assert reports[0]["summary"] == reports[1]["summary"]
    assert reports[0]["comparisons"] == reports[1]["comparisons"]
    for report in reports:
        event = next(
            row
            for row in report["corroboration_comparisons"]
            if row["feature"] == "def:run_tests"
        )
        assert event["status"] == "contradicted" and event["corroborates"] is False
        assert (
            next(half for half in event["halves"] if half["half"] == 1)[
                "window_available"
            ]
            == 2 * event["radius"] + 1
        )
