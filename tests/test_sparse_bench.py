from types import SimpleNamespace

import pandas as pd
from test_full_pipeline import fixture


def test_benchmark_compares_identical_units_and_full_alerts_without_labels():
    from bench.sparse_replication import compare_loaded

    rollouts = fixture(omit=[(0, 2)]).assign(run="aria")
    rollouts["response"] = rollouts.response.str.replace(
        "def solve()", "def run_tests()", regex=False
    )
    data = SimpleNamespace(rollouts=rollouts, sampling_units=None, provenance={})
    saved = rollouts.copy(deep=True)
    report = compare_loaded(
        data, "aria", sequential_options={"baseline_steps": 3}, tolerance=2
    )
    target = {
        row["coverage_policy"]: row
        for row in report["comparisons"]
        if row["feature"] == "def:run_tests"
    }
    assert target["contiguous_prefix"]["status"] == "unavailable"
    assert target["fixed_window"]["status"] == "replicated"
    assert (
        target["fixed_window"]["full_alert_step"]
        == target["contiguous_prefix"]["full_alert_step"]
        == 6
    )
    assert report["unit_leakage"] == 0
    assert (
        report["summary"]["fixed_window"]["n_candidates"] == report["full"]["n_alerted"]
    )
    pd.testing.assert_frame_equal(rollouts, saved)


def test_summary_keeps_unknowns_out_of_negative_counts():
    from bench.sparse_replication import summarize

    rows = [
        {"coverage_policy": "fixed_window", "status": s, "replicates": v, "halves": []}
        for s, v in [
            ("replicated", True),
            ("not_replicated", False),
            ("unavailable", None),
            ("pending", None),
        ]
    ]
    result = summarize(rows)["fixed_window"]
    assert result["n_candidates"] == 4 and result["n_replicated"] == 1
    assert (
        result["n_not_replicated"] == 1
        and result["n_unavailable"] == 1
        and result["n_pending"] == 1
    )
