"""Opt-in full vocabulary and baseline-fitted evaluation on local pinned Aria."""

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.local_data


@pytest.fixture(scope="module")
def local_report():
    if os.environ.get("ROLLOUTDIFF_FULL_ARIA") != "1":
        pytest.skip(
            "set ROLLOUTDIFF_FULL_ARIA=1 for the full vocabulary/fitted evaluation"
        )
    source = os.environ.get("ROLLOUTDIFF_ARIA_PATH")
    if not source:
        pytest.skip("set ROLLOUTDIFF_ARIA_PATH to the supplied aria.parquet")
    from bench.run_local import run_local

    report = run_local(source)
    output = os.environ.get("ROLLOUTDIFF_METRICS_PATH")
    if output:
        Path(output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def test_local_aria_full_vocabulary_preserves_every_series_and_status(local_report):
    source, pipeline = local_report["source"], local_report["pipeline"]
    assert source["rows"] == 51200 and source["steps"] == 200
    assert pipeline["feature_count"] > 2000
    assert pipeline["series_count"] == pipeline["feature_count"] + 1
    assert pipeline["series_rows"] == pipeline["series_count"] * 200
    assert pipeline["evidence_rows"] == pipeline["series_count"]
    assert {row["tier"] for row in pipeline["catalog_counts"]} == {
        "structural",
        "prose",
    }
    for name in ("sequential_screening", "retrospective_screening"):
        assert pipeline[name][0]["n_series"] == pipeline["series_count"]
    for counts in pipeline["status_counts"].values():
        assert sum(counts.values()) == pipeline["series_count"]


def test_local_aria_integrated_named_evidence_matches_existing_regressions(
    local_report,
):
    rows = {row["feature"]: row for row in local_report["pipeline"]["named_evidence"]}
    mechanism, reward, bisect = (
        rows["def:run_tests"],
        rows["scalar:reward"],
        rows["call:bisect.bisect_left"],
    )
    assert (
        mechanism["onset_step"],
        mechanism["alert_step"],
        mechanism["changepoint_step"],
    ) == (110, 112, 117)
    assert mechanism["replication_status"] == "replicated" and mechanism["replicates"]
    assert mechanism["persistence_fraction"] == mechanism["persistence_coverage"] == 1
    assert mechanism["persistence_n_retained"] == 88
    assert mechanism["reward_mean_difference"] == pytest.approx(0.7442781356574459)
    assert (
        mechanism["reward_n_mixed_groups"] == 29 and mechanism["reward_n_groups"] == 48
    )
    assert (
        mechanism["prompt_before_start"] == 1 and mechanism["prompt_before_end"] == 15
    )
    assert (
        mechanism["prompt_after_start"] == 110 and mechanism["prompt_after_end"] == 112
    )
    assert mechanism["prompt_shared_prompts"] == 12
    assert (reward["onset_step"], reward["alert_step"], reward["changepoint_step"]) == (
        118,
        120,
        118,
    )
    assert (
        reward["reward_status"] == "not_binary"
        and reward["reward_mean_difference"] is None
    )
    assert reward["persistence_n_retained"] == 80
    assert (
        bisect["sequential_status"] == "no_alert"
        and bisect["retrospective_status"] == "no_change"
    )
    assert bisect["replicates"] is None and bisect["reward_mean_difference"] is None


def test_local_aria_evidence_unavailability_is_not_filled_with_zero(local_report):
    for row in local_report["pipeline"]["status_examples"]:
        if row["replication_status"] in (
            "pending",
            "unavailable",
            "full_unscorable",
        ) or row["replication_status"].startswith("not_applicable"):
            assert row["replicates"] is None
        if row["persistence_status"] in (
            "no_future_observations",
            "unavailable_uncertainty",
        ) or row["persistence_status"].startswith("not_applicable"):
            assert row["persistence_fraction"] is None
        if row["reward_status"] in ("not_binary", "no_mixed_groups") or row[
            "reward_status"
        ].startswith("not_applicable"):
            assert row["reward_mean_difference"] is None
        if row["promptmix_status"] == "no_shared_prompts":
            assert (
                row["prompt_shared_change"] is None
                and row["prompt_shared_prompts"] == 0
            )


def test_local_fitted_evaluation_reports_denominators_and_distinct_timing(local_report):
    report = local_report
    assert report["source"]["baseline_window"] == [1, 15]
    profiles = report["baseline_fit"]["selected_profiles"]
    assert profiles and all(
        profile["baseline_steps"] == list(range(1, 16)) for profile in profiles
    )
    assert all(profile["n_rollouts"] == 3840 for profile in profiles)
    assert (
        sum(row["n"] for row in report["baseline_fit"]["coverage"])
        == report["pipeline"]["series_count"]
    )
    assert len(report["synthetic"]["settings"]) == 2
    assert {settings["seed"] for settings in report["synthetic"]["settings"]} == {
        20261017,
        20261018,
    }
    for summary in report["synthetic"]["overall"]:
        assert (
            summary["n_trials"] == summary["n_generated"] + summary["n_unavailable_fit"]
        )
        assert summary["n_generated"] == summary["n_scorable"] + summary["n_unscorable"]
        if summary["planted"] and summary["n_generated"]:
            assert (
                summary["detection_rate"]
                == summary["n_detected"] / summary["n_generated"]
            )
        elif not summary["planted"] and summary["n_generated"]:
            assert (
                summary["false_alert_rate"]
                == summary["n_flagged"] / summary["n_generated"]
            )
        if summary["detector"] == "sequential":
            assert summary["mean_absolute_boundary_error"] is None
        else:
            assert summary["mean_alert_delay"] is None
    # Report fitted-model quality for the fixed profiles and seeds.
