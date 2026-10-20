import pandas as pd
from pandas.testing import assert_frame_equal

from bench.run_local import acceptance, select_profiles, summarize_trials


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


def test_run_local_refuses_output_inside_repository(monkeypatch):
    import sys
    from pathlib import Path

    import pytest

    from bench import run_local

    root = Path(run_local.__file__).resolve().parents[1]
    monkeypatch.setattr(
        run_local,
        "run_local",
        lambda path: pytest.fail("analysis ran before the output path was checked"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_local", "--aria", "missing.parquet", "--output", str(root / "f.json")],
    )
    with pytest.raises(SystemExit) as failure:
        run_local.main()
    assert failure.value.code == 2


def test_public_suite_reports_source_units_and_fitted_structure():
    from types import SimpleNamespace

    from bench.run_local import public_suite
    from rolloutdiff import from_table

    raw = pd.DataFrame(
        [
            {
                "step": step,
                "prompt": f"problem {problem} variant {variant}",
                "response": "def f():\n    return 1" if copy else "plain words",
                "reward": float((step + problem + copy) % 3 == 0),
            }
            for step in range(1, 21)
            for problem in range(2)
            for variant in range(2)
            for copy in range(2)
        ]
    )
    rollouts = from_table(raw, run="r")
    units = (
        rollouts[["run", "prompt_id", "prompt"]]
        .drop_duplicates("prompt_id")
        .assign(sampling_unit_id=lambda t: t.prompt.str.split().str[1])
        .drop(columns="prompt")
    )
    data = SimpleNamespace(
        rollouts=rollouts,
        sampling_units=units,
        provenance={
            "retained_rows": len(rollouts),
            "retained_steps": 20,
            "sampling_unit_definition": "source problem_id",
            "n_sampling_units": 2,
        },
    )
    suite = public_suite(data, "r")
    assert set(suite) == {
        "source",
        "baseline_fit",
        "synthetic",
        "acceptance",
        "runtime",
    }
    assert suite["source"] == {
        "rows": 160,
        "steps": 20,
        "sampling_unit_definition": "source problem_id",
        "n_units": 2,
    }
    profiles = suite["baseline_fit"]["selected_profiles"]
    assert "scalar:reward" in {profile["feature"] for profile in profiles}
    assert all(profile["group_definition"] == "sampling_unit" for profile in profiles)
    assert all(profile["n_sampling_units"] == 2 for profile in profiles)
    assert {settings["seed"] for settings in suite["synthetic"]["settings"]} == {
        20261017,
        20261018,
    }
    assert {row["detector"] for row in suite["synthetic"]["overall"]} == {
        "sequential",
        "retrospective",
    }


def test_acceptance_scores_primary_profiles_per_detector_with_distinct_timing():
    def trial(detector, planted, group="primary", status="generated", **flags):
        row = {
            "detector": detector,
            "planted": planted,
            "evaluation_group": group,
            "status": status,
            "flagged": False,
            "timely": False,
            "localized": False,
        }
        return row | flags

    rows = (
        [trial("sequential", True, flagged=True, timely=True)] * 19
        + [trial("sequential", True, flagged=True, timely=False)]
        + [trial("sequential", False)] * 200
        + [trial("retrospective", True, flagged=True, localized=True)] * 20
        + [trial("retrospective", False)] * 198
        + [trial("retrospective", False, flagged=True)] * 2
        + [trial("retrospective", False, group="degenerate_null_check", flagged=True)]
        + [trial("sequential", True, status="unavailable")]
    )
    result = acceptance(pd.DataFrame(rows))
    sequential, retrospective = result["sequential"], result["retrospective"]
    assert sequential["timing"] == "alert_delay"
    assert (sequential["n_planted"], sequential["n_within_tolerance"]) == (20, 19)
    assert sequential["detection_within_tolerance"] == 0.95
    assert sequential["null_false_alert_rate"] == 0 and sequential["met"]
    assert retrospective["timing"] == "boundary_error"
    assert retrospective["detection_within_tolerance"] == 1.0
    assert (retrospective["n_null"], retrospective["n_false_alerts"]) == (200, 2)
    assert retrospective["null_false_alert_rate"] == 0.01
    assert retrospective["detection_met"] and not retrospective["null_met"]
    assert not retrospective["met"]
