"""Run the bounded full-vocabulary Aria and baseline-fitted synthetic checks."""

from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rolloutdiff.features import FeatureTables, featurize
from rolloutdiff.pipeline import analyze_rollouts

from .aria import ARIA_SHA256, load_aria_rollouts
from .public_data import PublicRun, load_public_run
from .synthetic import evaluate_fitted, fit_baseline, generate_fitted

BASELINE_WINDOW = (1, 15)
VOCABULARY = {"max_sparse_per_tier": 2000, "min_prevalence": 0.05}
PUBLIC_RUNS = ("aisi0", "aisi2")
SEQUENTIAL = {
    "baseline_steps": 15,
    "z_threshold": 4.0,
    "sustain": 3,
    "min_prevalence_change": 0.10,
    "numerical_floor": 1e-9,
}
RETROSPECTIVE = {
    "min_segment_steps": 5,
    "penalty_multiplier": 3.0,
    "min_prevalence_change": 0.10,
    "min_continuous_change": 0.0,
    "dependence": "ar1",
}
GENERATION = {
    "n_replicates": 32,
    "n_steps": 80,
    "onset": 40,
    "binary_effect": 0.20,
    "continuous_effect_sd": 2.0,
}
ACCEPTANCE = {"detection_within_tolerance": 0.95, "null_false_alert_rate": 0.005}


def select_profiles(fits: pd.DataFrame) -> pd.DataFrame:
    """Select profiles using baseline moments only."""
    eligible = fits.loc[
        fits.fit_status.eq("fit")
        & fits.marginal_sd.gt(0)
        & (~fits.kind.eq("binary") | fits["mean"].le(0.8))
    ]
    names = set()
    for _, group in eligible.groupby(["tier", "kind"], sort=True):
        ordered = group.sort_values(["prompt_variance", "feature"])
        for quantile in (0.25, 0.75):
            names.add(ordered.iloc[math.floor((len(ordered) - 1) * quantile)].feature)
    if eligible.feature.eq("scalar:reward").any():
        names.add("scalar:reward")
    for endpoint in (0.0, 1.0):
        degenerate = fits.loc[
            fits.fit_status.eq("degenerate_binary") & fits["mean"].eq(endpoint)
        ]
        if len(degenerate):
            names.add(degenerate.sort_values("feature").iloc[0].feature)
    selected = (
        fits.loc[fits.feature.isin(names)]
        .sort_values(["tier", "kind", "feature"])
        .copy()
    )
    selected["evaluation_group"] = np.where(
        selected.fit_status.eq("degenerate_binary"), "degenerate_null_check", "primary"
    )
    return selected.reset_index(drop=True)


def summarize_trials(trials: pd.DataFrame) -> list[dict]:
    """Pool trial counts and compute timing summaries from individual trials."""
    output = []
    for (detector, planted, evaluation_group), group in trials.groupby(
        ["detector", "planted", "evaluation_group"], sort=True
    ):
        generated = int(group.status.eq("generated").sum())
        scorable = int(group.scorable.sum())
        flagged, detected = int(group.flagged.sum()), int(group.detected.sum())
        row = {
            "detector": detector,
            "planted": bool(planted),
            "evaluation_group": evaluation_group,
            "n_trials": len(group),
            "n_generated": generated,
            "n_unavailable_fit": len(group) - generated,
            "n_scorable": scorable,
            "n_unscorable": generated - scorable,
            "n_flagged": flagged,
            "n_detected": detected,
            "n_early": int(group.outcome.eq("early_alert").sum()),
            "n_wrong_direction": int(group.outcome.eq("wrong_direction").sum()),
            "n_missed": int(group.outcome.eq("missed").sum()),
            "n_timely": int(group.timely.sum()),
            "n_localized": int(group.localized.sum()),
            "detection_rate": detected / generated if planted and generated else None,
            "false_alert_rate": flagged / generated
            if not planted and generated
            else None,
            "false_alert_rate_scorable": flagged / scorable
            if not planted and scorable
            else None,
        }
        for field in ("alert_delay", "boundary_error", "absolute_boundary_error"):
            values = group[field].dropna()
            row[f"mean_{field}"] = float(values.mean()) if len(values) else None
            row[f"median_{field}"] = float(values.median()) if len(values) else None
            row[f"max_{field}"] = float(values.max()) if len(values) else None
        output.append(row)
    return output


def acceptance(trials: pd.DataFrame) -> dict:
    """Score each detector's pooled primary trials against the fixed thresholds."""
    primary = trials.loc[
        trials.evaluation_group.eq("primary") & trials.status.eq("generated")
    ]
    result = {}
    for detector, group in primary.groupby("detector", sort=True):
        planted, null = group.loc[group.planted], group.loc[~group.planted]
        within = planted.timely if detector == "sequential" else planted.localized
        detection = within.sum() / len(planted) if len(planted) else None
        false_alerts = null.flagged.sum() / len(null) if len(null) else None
        detection_met = (
            detection is not None
            and detection >= ACCEPTANCE["detection_within_tolerance"]
        )
        null_met = (
            false_alerts is not None
            and false_alerts <= ACCEPTANCE["null_false_alert_rate"]
        )
        result[detector] = {
            "timing": "alert_delay" if detector == "sequential" else "boundary_error",
            "n_planted": len(planted),
            "n_within_tolerance": int(within.sum()),
            "detection_within_tolerance": detection,
            "n_null": len(null),
            "n_false_alerts": int(null.flagged.sum()),
            "null_false_alert_rate": false_alerts,
            "thresholds": ACCEPTANCE,
            "detection_met": detection_met,
            "null_met": null_met,
            "met": detection_met and null_met,
        }
    return result


def _clean(value):
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def fitted_suite(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    sampling_units: pd.DataFrame | None = None,
) -> dict:
    """Fit baseline 1\u201315, select profiles, and evaluate null and planted trials."""
    start = time.perf_counter()
    fits = fit_baseline(
        rollouts,
        features,
        run=run,
        window=BASELINE_WINDOW,
        sampling_units=sampling_units,
    )
    selected = select_profiles(fits)
    fitted = time.perf_counter()
    print(
        f"Selected {run} baseline profiles: " + ", ".join(selected.feature), flush=True
    )
    evaluations, batches = [], []
    for planted, seed in [(False, 20261017), (True, 20261018)]:
        batch = generate_fitted(selected, seed=seed, planted=planted, **GENERATION)
        evaluation = evaluate_fitted(
            batch,
            sequential_options=SEQUENTIAL,
            retrospective_options=RETROSPECTIVE,
            timing_tolerance=5,
        )
        evaluations.append(evaluation)
        batches.append(
            dict(
                **evaluation.settings,
                generation_status_counts=batch.trials.status.value_counts().to_dict(),
            )
        )
    finished = time.perf_counter()
    trials = pd.concat(
        [evaluation.trials for evaluation in evaluations], ignore_index=True
    )
    trials = trials.merge(
        selected[["source_run", "feature", "evaluation_group"]],
        on=["source_run", "feature"],
        validate="many_to_one",
    )
    summaries = pd.concat(
        [evaluation.summary for evaluation in evaluations], ignore_index=True
    )
    fit_coverage = (
        fits.groupby(["tier", "kind", "fit_status"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
    )
    return {
        "baseline_fit": {
            "coverage": fit_coverage.to_dict("records"),
            "n_temporal_unidentified": int(fits.ar1_raw.isna().sum()),
            "n_ar_projected": int(fits.ar1_projected.sum()),
            "n_prompt_variance_truncated": int(fits.prompt_variance_truncated.sum()),
            "n_temporal_variance_truncated": int(
                fits.temporal_variance_truncated.sum()
            ),
            "selected_profiles": selected.to_dict("records"),
        },
        "synthetic": {
            "overall": summarize_trials(trials),
            "per_profile": summaries.to_dict("records"),
            "trials": trials.to_dict("records"),
            "settings": batches,
        },
        "acceptance": acceptance(trials),
        "runtime": {
            "fit_seconds": fitted - start,
            "simulation_seconds": finished - fitted,
        },
    }


def public_suite(data: PublicRun, run: str) -> dict:
    """Fitted suite for one loaded public run, using its source-problem units."""
    features = featurize(data.rollouts, **VOCABULARY)
    provenance = data.provenance
    return {
        "source": {
            "rows": provenance["retained_rows"],
            "steps": provenance["retained_steps"],
            "sampling_unit_definition": provenance["sampling_unit_definition"],
            "n_units": provenance["n_sampling_units"],
        },
        **fitted_suite(
            data.rollouts, features, run=run, sampling_units=data.sampling_units
        ),
    }


def run_local(path: str | Path, public_root: str | Path | None = None) -> dict:
    start = time.perf_counter()
    rollouts = load_aria_rollouts(path)
    loaded = time.perf_counter()
    print(
        "Loaded pinned Aria: 51,200 rollouts; extracting the full retained vocabulary.",
        flush=True,
    )
    analysis = analyze_rollouts(
        rollouts,
        **VOCABULARY,
        sequential_options=SEQUENTIAL,
        retrospective_options=RETROSPECTIVE,
    )
    analyzed = time.perf_counter()
    print(
        f"Integrated {len(analysis.features.catalog) + 1} series in {analyzed - loaded:.2f}s; fitting baseline 1\u201315.",
        flush=True,
    )
    suite = fitted_suite(rollouts, analysis.features, run="aria")
    finished = time.perf_counter()
    status_columns = [
        "sequential_status",
        "retrospective_status",
        "replication_status",
        "persistence_status",
        "reward_status",
        "promptmix_status",
    ]
    evidence = analysis.evidence
    examples = evidence.loc[
        evidence.feature.isin(
            ["def:run_tests", "call:bisect.bisect_left", "scalar:reward"]
        )
    ]
    unavailable_examples = pd.concat(
        [
            evidence.groupby(column, sort=True, dropna=False).head(1)
            for column in status_columns
        ]
    ).drop_duplicates(["run", "feature"])
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_rss / (1024**2 if platform.system() == "Darwin" else 1024)
    report = {
        "schema_version": 1,
        "source": {
            "path": str(Path(path).resolve()),
            "sha256": ARIA_SHA256,
            "rows": len(rollouts),
            "steps": rollouts.step.nunique(),
            "baseline_window": BASELINE_WINDOW,
        },
        "policy": {
            "vocabulary": VOCABULARY,
            "sequential": SEQUENTIAL,
            "retrospective": RETROSPECTIVE,
            "replication_tolerance": 10,
            "persistence_effect_fraction": 0.5,
            "profile_selection": "baseline-only 25th/75th variance ranks per tier/kind + reward; separate degenerate checks",
            "generation": GENERATION,
            "timing_tolerance": 5,
            "acceptance": ACCEPTANCE,
        },
        "pipeline": {
            "feature_count": len(analysis.features.catalog),
            "series_count": len(analysis.sequential.diagnostics),
            "series_rows": len(analysis.aggregation.series),
            "catalog_counts": analysis.features.catalog.groupby(["tier", "kind"])
            .size()
            .rename("n")
            .reset_index()
            .to_dict("records"),
            "sequential_screening": analysis.sequential.screening.to_dict("records"),
            "n_floor_dominated_alerts": int(
                analysis.sequential.alerts.numerical_floor_dominant.sum()
            ),
            "n_floor_dominated_series": int(
                analysis.sequential.diagnostics.numerical_floor_dominant.sum()
            ),
            "retrospective_screening": analysis.retrospective.screening.to_dict(
                "records"
            ),
            "evidence_rows": len(evidence),
            "status_counts": {
                column: evidence[column].value_counts(dropna=False).to_dict()
                for column in status_columns
            },
            "named_evidence": examples.to_dict("records"),
            "status_examples": unavailable_examples.to_dict("records"),
        },
        "baseline_fit": suite["baseline_fit"],
        "synthetic": suite["synthetic"],
        "acceptance": suite["acceptance"],
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "load_seconds": loaded - start,
            "pipeline_seconds": analyzed - loaded,
            **suite["runtime"],
            "total_seconds": finished - start,
            "peak_rss_mb": peak_mb,
        },
        "limitations": [
            "Synthetic rates are per generated single-series trial, not full-run error guarantees.",
            "Aria baseline 1\u201315 is not verified clean and only supplies 15 observations for temporal fitting.",
            "Stationarity, independent fresh prompt groups, Gaussian continuous values, and fixed fitted parameters are assumptions.",
            "Continuous surrogates do not enforce count or nonnegative support; binary common probabilities may be clipped.",
            "Full-run vocabulary selection is retrospective; no full-run ranking, label attribution or causal claim is tested.",
        ],
    }
    if public_root is not None:
        report["additional_runs"] = {
            run: public_suite(load_public_run(public_root, run), run)
            for run in PUBLIC_RUNS
        }
    return _clean(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aria", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--public-root", help="pinned public data root for AISI fits")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output is not None and args.output.resolve().is_relative_to(root):
        parser.error("generated results must stay outside the repository")
    report = run_local(args.aria, args.public_root)
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.write_text(text + "\n")
        print(f"Metrics written to {args.output}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    main()
