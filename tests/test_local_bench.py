import pandas as pd
from pandas.testing import assert_frame_equal

from bench.run_local import select_profiles, summarize_trials


def test_profile_selection_uses_only_baseline_ranks_and_keeps_degenerate_checks_separate():
    fits = pd.DataFrame(
        [
            {
                "source_run": "r",
                "feature": f"f{i}",
                "tier": "structural",
                "kind": "binary",
                "fit_status": "fit",
                "marginal_sd": 1.0,
                "mean": 0.2,
                "prompt_variance": i,
            }
            for i in range(5)
        ]
        + [
            {
                "source_run": "r",
                "feature": "zero",
                "tier": "structural",
                "kind": "binary",
                "fit_status": "degenerate_binary",
                "marginal_sd": 0.0,
                "mean": 0.0,
                "prompt_variance": 0.0,
            },
            {
                "source_run": "r",
                "feature": "unplantable",
                "tier": "structural",
                "kind": "binary",
                "fit_status": "fit",
                "marginal_sd": 0.1,
                "mean": 0.99,
                "prompt_variance": 10.0,
            },
        ]
    )
    selected = select_profiles(fits)
    assert selected.feature.tolist() == ["f1", "f3", "zero"]
    assert selected.evaluation_group.tolist() == [
        "primary",
        "primary",
        "degenerate_null_check",
    ]
    assert_frame_equal(selected, select_profiles(fits.sample(frac=1, random_state=3)))


def test_overall_metrics_count_early_and_unavailable_without_filling_timing():
    base = {
        "detector": "sequential",
        "planted": True,
        "evaluation_group": "primary",
        "status": "generated",
        "scorable": True,
        "flagged": True,
        "detected": False,
        "timely": False,
        "localized": False,
        "alert_delay": float("nan"),
        "boundary_error": float("nan"),
        "absolute_boundary_error": float("nan"),
    }
    trials = pd.DataFrame(
        [
            base
            | {
                "outcome": "detected",
                "detected": True,
                "timely": True,
                "alert_delay": 2,
            },
            base | {"outcome": "early_alert"},
            base | {"outcome": "missed", "flagged": False},
            base
            | {
                "outcome": "unavailable_fit",
                "status": "unavailable",
                "scorable": False,
                "flagged": False,
            },
        ]
    )
    summary = summarize_trials(trials)[0]
    assert summary["n_trials"] == 4 and summary["n_generated"] == 3
    assert (
        summary["n_unavailable_fit"] == 1
        and summary["n_early"] == 1
        and summary["n_missed"] == 1
    )
    assert summary["detection_rate"] == 1 / 3
    assert (
        summary["mean_alert_delay"] == 2
        and summary["mean_absolute_boundary_error"] is None
    )
