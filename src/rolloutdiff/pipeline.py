"""Analyze every retained feature with shared aggregation and evidence policies."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd

from ._feature_values import _FeatureValueIndex
from ._sampling import resolve_sampling_units, subset_sampling_units
from .corroboration import _corroboration_prepared
from .detect import DetectionResult, detect_sequential
from .evidence import _persistence_prepared, _reward_association_prepared
from .features import FeatureTables, featurize
from .prompt_mix import _prompt_mix_prepared
from .replication import _replication_prepared, _validate_coverage_policy
from .retrospective import RetrospectiveResult, detect_retrospective
from .schema import validate_rollouts
from .series import Aggregation, aggregate

_EVIDENCE_COLUMNS = [
    "run",
    "feature",
    "tier",
    "kind",
    "sequential_status",
    "scale_source",
    "numerical_floor_dominant",
    "retrospective_status",
    "onset_step",
    "alert_step",
    "changepoint_step",
    "replication_status",
    "replication_coverage_policy",
    "replicates",
    "replication_horizon_complete",
    "replication_cutoff_step",
    "corroboration_status",
    "corroborates",
    "corroboration_reason",
    "corroboration_window_start",
    "corroboration_window_end",
    "corroboration_horizon_complete",
    "persistence_status",
    "persistence_fraction",
    "persistence_coverage",
    "persistence_n_observations",
    "persistence_n_evaluable",
    "persistence_n_retained",
    "persistence_n_unavailable",
    "reward_status",
    "reward_mean_difference",
    "reward_n_groups",
    "reward_n_mixed_groups",
    "reward_n_rollouts",
    "reward_n_included_rollouts",
    "reward_window_start",
    "reward_window_end",
    "promptmix_status",
    "prompt_before_start",
    "prompt_before_end",
    "prompt_after_start",
    "prompt_after_end",
    "prompt_positive_prompts",
    "prompt_total_prompts",
    "prompt_shared_prompts",
    "prompt_shared_sampling_units",
    "prompt_shared_change",
    "prompt_shared_se",
    "prompt_length_distance",
    "prompt_length_delta",
]


@dataclass(frozen=True)
class AnalysisResult:
    features: FeatureTables
    aggregation: Aggregation
    sequential: DetectionResult
    retrospective: RetrospectiveResult
    evidence: pd.DataFrame
    half_assignments: pd.DataFrame
    replication_halves: pd.DataFrame
    corroboration_halves: pd.DataFrame
    persistence_steps: pd.DataFrame
    reward_groups: pd.DataFrame
    prompt_values: pd.DataFrame
    prompt_shifts: pd.DataFrame


def _options(value: Mapping | None, name: str) -> dict:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping of detector keyword arguments")
    return dict(value)


def _groups(frame: pd.DataFrame) -> dict:
    return dict(tuple(frame.groupby(["run", "feature"], sort=False)))


def _subset_features(features: FeatureTables, rollouts: pd.DataFrame) -> FeatureTables:
    keys = ["run", "sample_id"]
    selected = pd.MultiIndex.from_frame(rollouts[keys])
    return FeatureTables(
        features.dense.loc[
            pd.MultiIndex.from_frame(features.dense[keys]).isin(selected)
        ],
        features.sparse.loc[
            pd.MultiIndex.from_frame(features.sparse[keys]).isin(selected)
        ],
        features.catalog,
    )


def _details(parts: list[pd.DataFrame]) -> pd.DataFrame:
    if not parts:
        return pd.DataFrame(columns=["run", "feature"])
    columns = list(dict.fromkeys(column for part in parts for column in part.columns))
    nonempty = [part for part in parts if not part.empty]
    if not nonempty:
        return parts[0].iloc[:0].reindex(columns=columns).copy()
    # Series concat preserves all-NA object dtypes and exact optional integer IDs.
    data = {}
    for column in columns:
        column_parts = [
            part[column]
            if column in part
            else pd.Series([pd.NA] * len(part), dtype=object)
            for part in nonempty
        ]
        data[column] = pd.concat(column_parts, ignore_index=True)
    return pd.DataFrame(data)


def _aggregate_halves(
    rollouts: pd.DataFrame, features: FeatureTables, units: pd.DataFrame
) -> tuple[pd.DataFrame, dict, dict]:
    assignments = units.copy()
    assignments["half"] = assignments.sampling_unit_id.map(
        lambda unit: zlib.crc32(unit.encode("utf-8")) & 1
    )
    half_ids = (
        assignments.set_index(["run", "prompt_id"])
        .half.reindex(pd.MultiIndex.from_frame(rollouts[["run", "prompt_id"]]))
        .to_numpy()
    )
    half_series, half_schedules = {}, {}
    for half in (0, 1):
        subset = rollouts.loc[half_ids == half]
        result = aggregate(
            subset,
            _subset_features(features, subset),
            sampling_units=subset_sampling_units(units, subset),
        )
        half_series[half] = _groups(result.series)
        half_schedules[half] = dict(
            tuple(result.prompt_schedule.groupby("run", sort=False))
        )

    return assignments, half_series, half_schedules


def _alert_evidence(
    diagnostic,
    selected_full: DetectionResult,
    half_series: dict,
    half_schedules: dict,
    assignments: pd.DataFrame,
    values: _FeatureValueIndex,
    units: pd.DataFrame,
    *,
    replication_tolerance: int,
    replication_coverage_policy: str,
    corroboration_radius: int,
    persistence_effect_fraction: float,
    sequential_parameters: dict,
) -> tuple[dict, dict[str, pd.DataFrame]]:
    run, feature = diagnostic.run, diagnostic.feature
    scores, alerts = selected_full.scores, selected_full.alerts
    alert = alerts.iloc[0]
    onset, alert_step = int(alert.onset_step), int(alert.alert_step)
    before = (int(diagnostic.baseline_start), int(diagnostic.baseline_end))
    after = (onset, alert_step)
    rep = _replication_prepared(
        selected_full,
        half_series,
        half_schedules,
        assignments,
        run=run,
        feature=feature,
        tolerance=replication_tolerance,
        parameters=sequential_parameters,
        coverage_policy=replication_coverage_policy,
    )
    corroboration = _corroboration_prepared(
        selected_full,
        half_series,
        half_schedules,
        assignments,
        run=run,
        feature=feature,
        radius=corroboration_radius,
        parameters=sequential_parameters,
    )
    retained = _persistence_prepared(
        scores,
        run=run,
        feature=feature,
        after_step=alert_step,
        reference_mean=float(alert.baseline_mean),
        change=float(alert.delta_at_alert),
        effect_fraction=persistence_effect_fraction,
    )
    frame, kind = values.values(run, feature)
    association = _reward_association_prepared(
        frame, kind, run=run, feature=feature, window=after
    )
    mix = _prompt_mix_prepared(
        frame,
        kind,
        run=run,
        feature=feature,
        before=before,
        after=after,
        sampling_units=units,
    )
    length = mix.prompt_shift.loc[mix.prompt_shift.descriptor.eq("prompt_chars")].iloc[
        0
    ]
    if not mix.shared_prompts:
        mix_status = "no_shared_prompts"
    elif mix.shared_se is None:
        mix_status = "shared_uncertainty_unavailable"
    else:
        mix_status = "available"
    row = {"onset_step": onset, "alert_step": alert_step}
    row.update(
        replication_status=rep.status,
        replicates=rep.replicates,
        replication_horizon_complete=rep.horizon_complete,
        replication_cutoff_step=rep.cutoff_step,
    )
    row.update(
        corroboration_status=corroboration.status,
        corroborates=corroboration.corroborates,
        corroboration_reason=corroboration.reason,
        corroboration_window_start=corroboration.window_start_step,
        corroboration_window_end=corroboration.window_end_step,
        corroboration_horizon_complete=corroboration.horizon_complete,
    )
    row.update(
        persistence_status=retained.status,
        persistence_fraction=retained.fraction_retained,
        persistence_coverage=retained.coverage,
        persistence_n_observations=retained.n_observations,
        persistence_n_evaluable=retained.n_evaluable,
        persistence_n_retained=retained.n_retained,
        persistence_n_unavailable=retained.n_unavailable,
    )
    row.update(
        reward_status=association.status,
        reward_mean_difference=association.mean_difference,
        reward_n_groups=association.n_groups,
        reward_n_mixed_groups=association.n_mixed_groups,
        reward_n_rollouts=association.n_rollouts,
        reward_n_included_rollouts=association.n_included_rollouts,
        reward_window_start=after[0],
        reward_window_end=after[1],
    )
    row.update(
        promptmix_status=mix_status,
        prompt_before_start=before[0],
        prompt_before_end=before[1],
        prompt_after_start=after[0],
        prompt_after_end=after[1],
        prompt_positive_prompts=mix.positive_prompts,
        prompt_total_prompts=mix.total_prompts,
        prompt_shared_prompts=mix.shared_prompts,
        prompt_shared_sampling_units=mix.shared_sampling_units,
        prompt_shared_change=mix.shared_change,
        prompt_shared_se=mix.shared_se,
        prompt_length_distance=length.distance,
        prompt_length_delta=length.delta,
    )
    details = {
        "replication_halves": rep.halves.assign(run=run, feature=feature),
        "corroboration_halves": corroboration.halves.assign(run=run, feature=feature),
        "persistence_steps": retained.steps.assign(
            after_step=alert_step,
            reference_mean=retained.reference_mean,
            reference_change=retained.change,
            effect_fraction=retained.effect_fraction,
            threshold=retained.threshold,
        ),
        "reward_groups": association.groups.assign(
            feature=feature, window_start=after[0], window_end=after[1]
        ),
    }
    for name, detail in (
        ("prompt_values", mix.per_prompt),
        ("prompt_shifts", mix.prompt_shift),
    ):
        details[name] = detail.assign(
            run=run,
            feature=feature,
            before_start=before[0],
            before_end=before[1],
            after_start=after[0],
            after_end=after[1],
        )
    return row, details


def analyze_rollouts(
    rollouts: pd.DataFrame,
    *,
    max_sparse_per_tier: int = 2000,
    min_prevalence: float = 0.05,
    sequential_options: Mapping | None = None,
    retrospective_options: Mapping | None = None,
    replication_tolerance: int = 10,
    replication_coverage_policy: str = "contiguous_prefix",
    corroboration_radius: int = 10,
    persistence_effect_fraction: float = 0.5,
    sampling_units: pd.DataFrame | None = None,
) -> AnalysisResult:
    """Screen all retained features and reward, then attach sequential evidence.

    Select vocabulary once for every run and both CRC32 whole-unit halves.
    Anchor reward and prompt mix to onset through alert, with initial-baseline
    prompt comparison and post-alert persistence.
    """
    _validate_coverage_policy(replication_coverage_policy)
    if type(corroboration_radius) is not int or corroboration_radius < 0:
        raise ValueError("corroboration_radius must be a nonnegative integer")
    if type(replication_tolerance) is not int or replication_tolerance < 0:
        raise ValueError("replication_tolerance must be a nonnegative integer")
    if (
        isinstance(persistence_effect_fraction, bool)
        or not isinstance(persistence_effect_fraction, Real)
        or not np.isfinite(persistence_effect_fraction)
        or not 0 < persistence_effect_fraction <= 1
    ):
        raise ValueError("persistence_effect_fraction must be in (0, 1]")
    sequential_parameters = _options(sequential_options, "sequential_options")
    retrospective_parameters = _options(retrospective_options, "retrospective_options")
    rollouts = validate_rollouts(rollouts)
    units = resolve_sampling_units(rollouts, sampling_units)
    features = featurize(
        rollouts, max_sparse_per_tier=max_sparse_per_tier, min_prevalence=min_prevalence
    )
    aggregation = aggregate(rollouts, features, sampling_units=units)
    sequential = detect_sequential(aggregation.series, **sequential_parameters)
    retrospective = detect_retrospective(aggregation.series, **retrospective_parameters)

    assignments, half_series, half_schedules = _aggregate_halves(
        rollouts, features, units
    )

    full_scores = _groups(sequential.scores)
    full_alerts = _groups(sequential.alerts)
    retro_diagnostics = retrospective.diagnostics.set_index(["run", "feature"])
    changes = retrospective.changepoints.set_index(["run", "feature"])
    assignments_by_run = dict(tuple(assignments.groupby("run", sort=False)))
    units_by_run = dict(tuple(units.groupby("run", sort=False)))
    values = _FeatureValueIndex(rollouts, features)
    rows = []
    detail_parts = {
        "replication_halves": [],
        "corroboration_halves": [],
        "persistence_steps": [],
        "reward_groups": [],
        "prompt_values": [],
        "prompt_shifts": [],
    }
    empty_series = aggregation.series.iloc[:0]
    empty_schedule = aggregation.prompt_schedule.iloc[:0]
    for diagnostic in sequential.diagnostics.itertuples(index=False):
        run, feature = diagnostic.run, diagnostic.feature
        key = (run, feature)
        row = dict.fromkeys(_EVIDENCE_COLUMNS)
        row.update(
            run=run,
            feature=feature,
            tier=diagnostic.tier,
            kind=diagnostic.kind,
            sequential_status=diagnostic.status,
            scale_source=diagnostic.scale_source,
            numerical_floor_dominant=bool(diagnostic.numerical_floor_dominant),
            retrospective_status=retro_diagnostics.loc[key, "status"],
            replication_coverage_policy=replication_coverage_policy,
        )
        if key in changes.index:
            row["changepoint_step"] = int(changes.loc[key, "changepoint_step"])
        if key not in full_alerts:
            unscorable = diagnostic.n_scored == 0
            unavailable = (
                "not_applicable_full_unscorable"
                if unscorable
                else "not_applicable_no_sequential_alert"
            )
            row.update(
                replication_status="full_unscorable" if unscorable else unavailable,
                corroboration_status="unavailable",
                corroboration_reason="full_unscorable"
                if unscorable
                else "no_full_alert",
                persistence_status=unavailable,
                reward_status=unavailable,
                promptmix_status=unavailable,
            )
            rows.append(row)
            continue

        selected_full = DetectionResult(
            full_alerts[key],
            pd.DataFrame([diagnostic._asdict()]),
            full_scores[key],
            sequential.screening.iloc[:0],
        )
        alert_row, details = _alert_evidence(
            diagnostic,
            selected_full,
            {half: half_series[half].get(key, empty_series) for half in (0, 1)},
            {half: half_schedules[half].get(run, empty_schedule) for half in (0, 1)},
            assignments_by_run[run],
            values,
            units_by_run[run],
            replication_tolerance=replication_tolerance,
            replication_coverage_policy=replication_coverage_policy,
            corroboration_radius=corroboration_radius,
            persistence_effect_fraction=persistence_effect_fraction,
            sequential_parameters=sequential_parameters,
        )
        row.update(alert_row)
        rows.append(row)
        for name, detail in details.items():
            detail_parts[name].append(detail)

    evidence = pd.DataFrame(rows, columns=_EVIDENCE_COLUMNS)
    # Keep integer step IDs exact when identities are missing.
    for name in [
        "onset_step",
        "alert_step",
        "changepoint_step",
        "replication_cutoff_step",
        "corroboration_window_start",
        "corroboration_window_end",
        "reward_window_start",
        "reward_window_end",
        "prompt_before_start",
        "prompt_before_end",
        "prompt_after_start",
        "prompt_after_end",
    ]:
        evidence[name] = pd.Series([row[name] for row in rows], dtype=object)
    evidence["replicates"] = pd.array(evidence.replicates, dtype="boolean")
    evidence["corroborates"] = pd.array(evidence.corroborates, dtype="boolean")
    evidence["numerical_floor_dominant"] = evidence.numerical_floor_dominant.astype(
        bool
    )
    return AnalysisResult(
        features=features,
        aggregation=aggregation,
        sequential=sequential,
        retrospective=retrospective,
        evidence=evidence,
        half_assignments=assignments,
        **{name: _details(parts) for name, parts in detail_parts.items()},
    )
