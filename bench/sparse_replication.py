"""Compare first-alert replication and event corroboration on pinned AISI inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import pandas as pd

from rolloutdiff import analyze_rollouts
from rolloutdiff._sampling import subset_sampling_units
from rolloutdiff.corroboration import _corroboration_prepared
from rolloutdiff.detect import DetectionResult
from rolloutdiff.features import FeatureTables
from rolloutdiff.pipeline import _groups, _subset_features
from rolloutdiff.replication import _replication_prepared
from rolloutdiff.series import aggregate

from .public_benchmark import TARGETS
from .public_data import load_public_run
from .run_local import RETROSPECTIVE, SEQUENTIAL, _clean

POLICIES = ("contiguous_prefix", "fixed_window")


def summarize(rows: list[dict]) -> dict:
    output = {}
    for policy in POLICIES:
        selected = [row for row in rows if row["coverage_policy"] == policy]
        counts = Counter(row["status"] for row in selected)
        halves = [half for row in selected for half in row["halves"]]
        output[policy] = dict(
            n_candidates=len(selected),
            **{
                f"n_{status}": counts[status]
                for status in ("replicated", "not_replicated", "unavailable", "pending")
            },
            half_status_counts=dict(Counter(half["status"] for half in halves)),
            half_reason_counts=dict(Counter(half["reason"] for half in halves)),
            n_observed_half_alerts=sum(
                half.get("observed_alert_step") is not None for half in halves
            )
            if policy == "fixed_window"
            else sum(half["alert_step"] is not None for half in halves),
            n_identified_first_half_alerts=sum(
                half["status"] == "alerted" for half in halves
            ),
        )
    return output


def summarize_corroboration(rows: list[dict]) -> dict:
    counts = Counter(row["status"] for row in rows)
    halves = [half for row in rows for half in row["halves"]]
    return dict(
        n_candidates=len(rows),
        **{
            f"n_{status}": counts[status]
            for status in ("corroborated", "contradicted", "unavailable")
        },
        half_status_counts=dict(Counter(half["status"] for half in halves)),
        half_reason_counts=dict(Counter(half["reason"] for half in halves)),
        coverage_totals={
            name: sum(half[name] for half in halves)
            for name in (
                "window_expected",
                "window_observed",
                "window_available",
                "n_scored",
                "n_unscorable",
                "future_steps",
            )
        },
        n_observed_matching_half_windows=sum(
            half["match_start_step"] is not None for half in halves
        ),
        n_observed_opposite_half_windows=sum(
            half["opposite_start_step"] is not None for half in halves
        ),
        n_mixed_direction_halves=sum(bool(half["mixed_direction"]) for half in halves),
    )


def _half_aggregations(analysis, rollouts, assignments, names):
    # Halves retain the exact vocabulary used by the full detection.
    # Materialize only alerted feature values to bound memory use.
    features = FeatureTables(
        analysis.features.dense[
            [
                name
                for name in analysis.features.dense
                if name in names or name in ("run", "sample_id")
            ]
        ],
        analysis.features.sparse.loc[analysis.features.sparse.feature.isin(names)],
        analysis.features.catalog.loc[analysis.features.catalog.feature.isin(names)],
    )
    half_ids = (
        assignments.set_index(["run", "prompt_id"])
        .half.reindex(pd.MultiIndex.from_frame(rollouts[["run", "prompt_id"]]))
        .to_numpy()
    )
    half_series, half_schedules = {}, {}
    units = assignments[["run", "prompt_id", "sampling_unit_id"]]
    for half in (0, 1):
        subset = rollouts.loc[half_ids == half]
        aggregation = aggregate(
            subset,
            _subset_features(features, subset),
            sampling_units=subset_sampling_units(units, subset),
        )
        half_series[half] = _groups(aggregation.series)
        half_schedules[half] = aggregation.prompt_schedule
    return half_series, half_schedules


def _check_corroboration(analysis, corroboration, selected_run, feature, expected):
    expected_halves = analysis.corroboration_halves.loc[
        analysis.corroboration_halves.run.eq(selected_run)
        & analysis.corroboration_halves.feature.eq(feature)
    ].drop(columns=["run", "feature"])
    if _clean(corroboration.halves.to_dict("records")) != _clean(
        expected_halves.to_dict("records")
    ):
        raise AssertionError(
            "event-window half evidence differs from integrated pipeline"
        )
    actual = {
        "status": corroboration.status,
        "corroborates": corroboration.corroborates,
        "reason": corroboration.reason,
        "window_start_step": corroboration.window_start_step,
        "window_end_step": corroboration.window_end_step,
        "horizon_complete": corroboration.horizon_complete,
    }
    expected_values = {
        "status": expected.corroboration_status,
        "corroborates": expected.corroborates,
        "reason": expected.corroboration_reason,
        "window_start_step": expected.corroboration_window_start,
        "window_end_step": expected.corroboration_window_end,
        "horizon_complete": expected.corroboration_horizon_complete,
    }
    if _clean(actual) != _clean(expected_values):
        raise AssertionError("event-window result differs from integrated pipeline")
    return actual


def _alert_metadata(alert, run, feature):
    return {
        "feature": feature if feature in TARGETS.get(run, []) else None,
        "feature_sha256": hashlib.sha256(feature.encode()).hexdigest(),
        "tier": alert.tier,
        "kind": alert.kind,
        "full_onset_step": int(alert.onset_step),
        "full_alert_step": int(alert.alert_step),
        "full_direction": alert.direction,
    }


def _comparison_policy(parameters, tolerance, corroboration_radius):
    return {
        "coverage_policies": POLICIES,
        "default_unchanged": "contiguous_prefix",
        "sequential": parameters,
        "tolerance": tolerance,
        "baseline_window": "first B full-grid positions only",
        "minimum_usable_baseline": max(2, (parameters["baseline_steps"] + 1) // 2),
        "gaps": "absent rows and unavailable SE break empirical streaks; observed data may resume",
        "first_alert": "earlier compatible unknown alert makes first-alert evidence unavailable",
        "assumptions": [
            "baseline estimate conditional on availability",
            "no calibrated error-rate claim",
        ],
        "event_window_corroboration": {
            "radius": corroboration_radius,
            "window": "full alert plus or minus radius full-grid observations, after baseline",
            "positive": "fully observed same-direction sustained streak wholly inside window in both halves",
            "negative": "at least one complete, fully scorable window without a matching streak",
            "unavailable": "no decisive observed evidence; missing data and uncertainty are not negative evidence",
            "opposite": "reported independently; does not veto a matching witness",
            "first_alert_replication": False,
        },
        "no_labels_used": True,
    }


def compare_loaded(
    data,
    run: str,
    *,
    sequential_options: dict | None = None,
    tolerance: int = 10,
    corroboration_radius: int = 10,
) -> dict:
    started = time.perf_counter()
    parameters = dict(SEQUENTIAL, **(sequential_options or {}))
    rollouts = data.rollouts
    saved_keys = rollouts[["run", "sample_id", "prompt_id"]].copy()
    analysis = analyze_rollouts(
        rollouts,
        sampling_units=getattr(data, "sampling_units", None),
        sequential_options=parameters,
        retrospective_options=RETROSPECTIVE,
        replication_tolerance=tolerance,
        corroboration_radius=corroboration_radius,
    )
    analyzed = time.perf_counter()
    assignments = analysis.half_assignments
    leakage = int(
        assignments.groupby(["run", "sampling_unit_id"]).half.nunique().gt(1).sum()
    )
    if leakage:
        raise AssertionError("sampling unit crosses halves")
    names = set(analysis.sequential.alerts.feature)
    half_series, half_schedules = _half_aggregations(
        analysis, rollouts, assignments, names
    )
    full_scores = _groups(
        analysis.sequential.scores.loc[analysis.sequential.scores.feature.isin(names)]
    )
    full_alerts = _groups(analysis.sequential.alerts)
    diagnostics = analysis.sequential.diagnostics.set_index(
        ["run", "feature"], drop=False
    )
    rows, corroboration_rows = [], []
    evidence = analysis.evidence.set_index(["run", "feature"])
    empty = analysis.aggregation.series.iloc[:0]
    for key, alerts in full_alerts.items():
        selected_run, feature = key
        full = DetectionResult(
            alerts,
            pd.DataFrame([diagnostics.loc[key]]),
            full_scores[key],
            analysis.sequential.screening.iloc[:0],
        )
        metadata = _alert_metadata(alerts.iloc[0], run, feature)
        feature_halves = {half: half_series[half].get(key, empty) for half in (0, 1)}
        for policy in POLICIES:
            rep = _replication_prepared(
                full,
                feature_halves,
                half_schedules,
                assignments,
                run=selected_run,
                feature=feature,
                tolerance=tolerance,
                parameters=parameters,
                coverage_policy=policy,
            )
            if policy == "contiguous_prefix":
                expected = analysis.replication_halves.loc[
                    analysis.replication_halves.run.eq(selected_run)
                    & analysis.replication_halves.feature.eq(feature)
                ].drop(columns=["run", "feature"])
                if _clean(rep.halves.to_dict("records")) != _clean(
                    expected.to_dict("records")
                ):
                    raise AssertionError(
                        "reference half evidence differs from integrated pipeline"
                    )
            rows.append(
                dict(
                    coverage_policy=policy,
                    **metadata,
                    status=rep.status,
                    replicates=rep.replicates,
                    cutoff_step=rep.cutoff_step,
                    horizon_complete=rep.horizon_complete,
                    halves=rep.halves.to_dict("records"),
                )
            )
        corroboration = _corroboration_prepared(
            full,
            feature_halves,
            half_schedules,
            assignments,
            run=selected_run,
            feature=feature,
            radius=corroboration_radius,
            parameters=parameters,
        )
        actual = _check_corroboration(
            analysis, corroboration, selected_run, feature, evidence.loc[key]
        )
        corroboration_rows.append(
            dict(
                evidence_kind="event_window_corroboration",
                **metadata,
                radius=corroboration_radius,
                **actual,
                halves=corroboration.halves.to_dict("records"),
            )
        )
    if not saved_keys.equals(rollouts[["run", "sample_id", "prompt_id"]]):
        raise AssertionError("canonical identities changed")
    finished = time.perf_counter()
    screened = analysis.sequential.screening.iloc[0].to_dict()
    return _clean(
        {
            "run": run,
            "provenance": data.provenance,
            "unit_leakage": leakage,
            "policy": _comparison_policy(parameters, tolerance, corroboration_radius),
            "full": screened,
            "n_retained_features": len(analysis.features.catalog),
            "summary": summarize(_clean(rows)),
            "comparisons": rows,
            "corroboration_summary": summarize_corroboration(
                _clean(corroboration_rows)
            ),
            "corroboration_comparisons": corroboration_rows,
            "timing": {
                "analysis_seconds": analyzed - started,
                "comparison_seconds": finished - analyzed,
                "total_seconds": finished - started,
            },
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--run", choices=["aisi0", "aisi2"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("generated comparison results must remain outside the repository")
    data = load_public_run(args.data_root, args.run)
    print(
        f"{args.run}: pinned input verified; comparing first-alert replication and event corroboration.",
        flush=True,
    )
    report = compare_loaded(data, args.run)
    report["code_sha256"] = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for pattern in ("src/rolloutdiff/*.py", "bench/*.py", "bench/*manifest.json")
        for path in sorted(root.glob(pattern))
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Metrics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
