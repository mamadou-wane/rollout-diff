"""Small synthetic checks fitted to baseline observations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from inspect import signature
from numbers import Real

import numpy as np
import pandas as pd

from rolloutdiff.detect import detect_sequential
from rolloutdiff.features import FeatureTables
from rolloutdiff.retrospective import detect_retrospective
from rolloutdiff.schema import validate_rollouts
from rolloutdiff.series import _validate_features

_SERIES_COLUMNS = [
    "run",
    "step",
    "feature",
    "tier",
    "kind",
    "mean",
    "se",
    "n_rollouts",
    "n_prompts",
]
_ASSUMPTIONS = (
    "stationary_baseline_moments",
    "independent_prompt_groups_per_step",
    "gaussian_ar1_common_effect_in_observation_time",
    "fixed_fitted_parameters",
    "independent_features",
    "empirical_step_group_size_distribution",
    "binary_common_probability_clipped_to_unit_interval",
    "continuous_gaussian_prompt_means_without_support_constraints",
)
_AR_LIMIT = 0.95
_MIN_BASELINE_STEPS = 8


@dataclass(frozen=True)
class SyntheticBatch:
    series: pd.DataFrame
    trials: pd.DataFrame
    settings: dict


@dataclass(frozen=True)
class SyntheticEvaluation:
    trials: pd.DataFrame
    summary: pd.DataFrame
    settings: dict


def _baseline_moments(baseline, values):
    # Preserve nonfinite moments as unavailable: max(0, NaN) would yield zero
    # variance, and infinite values cannot define a generated series.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        prompt = (
            baseline[["step", "prompt_id"]]
            .assign(value=values)
            .groupby(["step", "prompt_id"], sort=True)
            .value.agg(["mean", "var", "size"])
        )
        prompt["sampling_variance"] = prompt["var"] / prompt["size"]
        step_group = prompt.groupby(level="step", sort=True)
        means = step_group["mean"].mean().to_numpy()
        prompt_variances = step_group["mean"].var(ddof=1).to_numpy()
        sampling = step_group.sampling_variance.mean().to_numpy()
        n_prompts = step_group.size().to_numpy()
        errors_squared = prompt_variances / n_prompts
        mean = float(means.mean())
        within = float(
            (prompt["var"] * (prompt["size"] - 1)).sum() / (prompt["size"] - 1).sum()
        )
        observed_prompt_variance = float(prompt_variances.mean())
        prompt_raw = float(np.mean(prompt_variances - sampling))
        step_variance = float(means.var(ddof=1))
        step_sampling = float(errors_squared.mean())
        temporal_raw = step_variance - step_sampling
        centered = means - mean
        covariance = float(np.mean(centered[:-1] * centered[1:]))
    moments = {
        "mean": mean,
        "within_variance": within,
        "observed_prompt_variance": observed_prompt_variance,
        "prompt_variance_raw": prompt_raw,
        "observed_step_variance": step_variance,
        "step_sampling_variance": step_sampling,
        "temporal_variance_raw": temporal_raw,
        "lag1_covariance": covariance,
    }
    return moments, means, errors_squared


def _fit_profile(
    metadata, common, unavailable, baseline, dense, dense_names, sparse_ids
):
    row = dict(
        **common,
        **metadata,
        fit_status="unavailable",
        reason=unavailable,
        mean=np.nan,
        within_variance=np.nan,
        observed_prompt_variance=np.nan,
        prompt_variance_raw=np.nan,
        prompt_variance=np.nan,
        observed_step_variance=np.nan,
        step_sampling_variance=np.nan,
        temporal_variance_raw=np.nan,
        temporal_variance=np.nan,
        lag1_covariance=np.nan,
        ar1_raw=np.nan,
        ar1=np.nan,
        binary_variance_denominator=np.nan,
        binary_rho_raw=np.nan,
        binary_rho=np.nan,
        marginal_sd=np.nan,
        ar1_projected=False,
        prompt_variance_truncated=False,
        temporal_variance_truncated=False,
        binary_rho_projected=False,
    )
    if unavailable:
        return row
    name = metadata["feature"]
    if name == "scalar:reward":
        values = baseline.reward.to_numpy(dtype=float)
    elif name in dense_names:
        values = dense[name].to_numpy(dtype=float)
    else:
        values = baseline.sample_id.isin(sparse_ids.get(name, set())).to_numpy(
            dtype=float
        )
    moments, means, errors_squared = _baseline_moments(baseline, values)
    row.update(moments)
    if not np.isfinite(list(moments.values())).all():
        row["reason"] = "nonfinite_baseline_moments"
        return row
    prompt_variance = max(0.0, moments["prompt_variance_raw"])
    temporal = max(0.0, moments["temporal_variance_raw"])
    ar_raw = moments["lag1_covariance"] / temporal if temporal > 0 else np.nan
    ar1 = float(np.clip(ar_raw, -_AR_LIMIT, _AR_LIMIT)) if temporal > 0 else 0.0
    degenerate = metadata["kind"] == "binary" and (values == values[0]).all()
    rho_raw, denominator = np.nan, np.nan
    unidentified = False
    if metadata["kind"] == "binary" and not degenerate:
        denominator = float(np.mean(means * (1 - means) + errors_squared))
        unidentified = denominator <= 0
        if not unidentified:
            rho_raw = prompt_variance / denominator
    rho = float(np.clip(rho_raw, 0, 1)) if np.isfinite(rho_raw) else np.nan
    marginal_sd = float(
        np.sqrt(moments["within_variance"] + prompt_variance + temporal)
    )
    row.update(
        fit_status=(
            "unavailable"
            if unidentified
            else "degenerate_binary"
            if degenerate
            else "fit"
        ),
        reason=(
            "binary_heterogeneity_unidentified"
            if unidentified
            else "unidentified_heterogeneity_away_from_boundary"
            if degenerate
            else ""
        ),
        prompt_variance=prompt_variance,
        temporal_variance=temporal,
        ar1_raw=ar_raw,
        ar1=ar1,
        binary_variance_denominator=denominator,
        binary_rho_raw=rho_raw,
        binary_rho=rho,
        marginal_sd=marginal_sd,
        ar1_projected=bool(temporal > 0 and ar1 != ar_raw),
        prompt_variance_truncated=moments["prompt_variance_raw"] < 0,
        temporal_variance_truncated=moments["temporal_variance_raw"] < 0,
        binary_rho_projected=bool(np.isfinite(rho_raw) and rho != rho_raw),
    )
    if not np.isfinite(marginal_sd) or np.isinf(ar_raw) or np.isinf(rho_raw):
        row.update(fit_status="unavailable", reason="nonfinite_baseline_moments")

    return row


def _generate_trial(fit, trial, entropy, replicate, n_steps, onset, effect):
    rows = []
    rng = np.random.default_rng(np.random.SeedSequence([*entropy, replicate]))
    schedule = fit["group_size_schedule"]
    common = rng.normal(0, np.sqrt(fit["temporal_variance"]))
    innovation_sd = np.sqrt(fit["temporal_variance"] * (1 - fit["ar1"] ** 2))
    for step in range(n_steps):
        if step:
            common = fit["ar1"] * common + rng.normal(0, innovation_sd)
        counts = np.asarray(schedule[int(rng.integers(len(schedule)))], dtype=int)
        center = fit["mean"] + common + effect * (step >= onset)
        if fit["kind"] == "binary":
            probability = float(np.clip(center, 0, 1))
            trial["n_probability_clipped"] += int(probability != center)
            rho = fit["binary_rho"]
            if (
                probability in (0, 1)
                or rho == 0
                or fit["fit_status"] == "degenerate_binary"
            ):
                probabilities = np.full(len(counts), probability)
            elif rho == 1:
                probabilities = rng.binomial(1, probability, size=len(counts))
            else:
                concentration = (1 - rho) / rho
                probabilities = rng.beta(
                    probability * concentration,
                    (1 - probability) * concentration,
                    size=len(counts),
                )
            prompt_means = rng.binomial(counts, probabilities) / counts
        else:
            prompt_means = rng.normal(
                center,
                np.sqrt(fit["prompt_variance"] + fit["within_variance"] / counts),
            )
        rows.append(
            {
                "run": trial["run"],
                "feature": fit["feature"],
                "tier": fit["tier"],
                "kind": fit["kind"],
                "step": step,
                "mean": float(prompt_means.mean()),
                "se": float(prompt_means.std(ddof=1) / np.sqrt(len(counts))),
                "n_prompts": len(counts),
                "n_rollouts": int(counts.sum()),
            }
        )
    trial["probability_clipped_fraction"] = (
        trial["n_probability_clipped"] / n_steps if fit["kind"] == "binary" else np.nan
    )
    return rows


def _evaluate_trial(trial, detector, diagnostics, events, step_name, timing_tolerance):
    row = dict(
        **trial,
        detector=detector,
        outcome="unavailable_fit",
        detector_status=None,
        scorable=False,
        flagged=False,
        detected=False,
        localized=False,
        timely=False,
        event_step=np.nan,
        alert_delay=np.nan,
        boundary_error=np.nan,
        absolute_boundary_error=np.nan,
    )
    if trial["status"] != "generated":
        return row
    key = (trial["run"], trial["feature"])
    diagnostic = diagnostics.loc[key]
    scorable = (
        diagnostic.n_scored > 0
        if detector == "sequential"
        else diagnostic.status in ("change", "no_change")
    )
    row.update(detector_status=diagnostic.status, scorable=bool(scorable))
    if not scorable:
        row["outcome"] = "unscorable"
    elif key not in events.index:
        row["outcome"] = "missed" if trial["planted"] else "no_change"
    else:
        event = events.loc[key]
        event_step = int(event[step_name])
        row.update(flagged=True, event_step=event_step)
        if not trial["planted"]:
            row["outcome"] = "false_alert"
        elif detector == "sequential" and event_step < trial["onset"]:
            row["outcome"] = "early_alert"
        elif event.direction != ("up" if trial["effect"] > 0 else "down"):
            row["outcome"] = "wrong_direction"
        else:
            row.update(outcome="detected", detected=True)
            error = event_step - trial["onset"]
            if detector == "sequential":
                row.update(alert_delay=error, timely=error <= timing_tolerance)
            else:
                row.update(
                    boundary_error=error,
                    absolute_boundary_error=abs(error),
                    localized=abs(error) <= timing_tolerance,
                )
    return row


def _summarize_trials(trials):
    summaries = []
    keys = ["detector", "source_run", "feature", "tier", "kind", "planted"]
    for identity, group in trials.groupby(keys, sort=True):
        row = dict(zip(keys, identity, strict=False))
        generated = int(group.status.eq("generated").sum())
        scorable = int(group.scorable.sum())
        flagged = int(group.flagged.sum())
        detected = int(group.detected.sum())
        planted = row["planted"]
        row.update(
            n_trials=len(group),
            n_generated=generated,
            n_unavailable_fit=len(group) - generated,
            n_scorable=scorable,
            n_unscorable=generated - scorable,
            n_flagged=flagged,
            n_detected=detected,
            n_early=int(group.outcome.eq("early_alert").sum()),
            n_wrong_direction=int(group.outcome.eq("wrong_direction").sum()),
            n_missed=int(group.outcome.eq("missed").sum()),
            n_localized=int(group.localized.sum()),
            n_timely=int(group.timely.sum()),
            detection_rate=detected / generated if planted and generated else np.nan,
            false_alert_rate_all=flagged / generated
            if not planted and generated
            else np.nan,
            false_alert_rate_scorable=flagged / scorable
            if not planted and scorable
            else np.nan,
            localization_rate=float(group.localized.sum()) / generated
            if planted and generated and row["detector"] == "retrospective"
            else np.nan,
            timely_detection_rate=float(group.timely.sum()) / generated
            if planted and generated and row["detector"] == "sequential"
            else np.nan,
            mean_alert_delay=float(group.alert_delay.mean()),
            median_alert_delay=float(group.alert_delay.median())
            if group.alert_delay.notna().any()
            else np.nan,
            mean_boundary_error=float(group.boundary_error.mean()),
            mean_absolute_boundary_error=float(group.absolute_boundary_error.mean()),
            max_absolute_boundary_error=float(group.absolute_boundary_error.max()),
        )
        summaries.append(row)
    return pd.DataFrame(summaries)


def fit_baseline(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    window: tuple[int, int],
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Fit baseline-only moments, retaining unavailable and degenerate profiles."""
    if (
        not isinstance(window, tuple)
        or len(window) != 2
        or any(type(value) is not int for value in window)
        or not 0 <= window[0] <= window[1]
    ):
        raise ValueError("window must be an ordered pair of nonnegative integer steps")
    rollouts = validate_rollouts(rollouts)
    dense_names = _validate_features(rollouts, features)
    if not rollouts.run.eq(run).any():
        raise ValueError("run has no rollouts")
    catalog = pd.concat(
        [
            features.catalog[["feature", "tier", "kind"]],
            pd.DataFrame(
                [
                    {
                        "feature": "scalar:reward",
                        "tier": "structural",
                        "kind": "continuous",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    if feature_names is not None:
        if not feature_names:
            raise ValueError("feature_names must contain at least one profile")
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("feature_names contains duplicates")
        if set(feature_names) - set(catalog.feature):
            raise ValueError("unknown feature in feature_names")
        catalog = catalog.loc[catalog.feature.isin(feature_names)]
    catalog = catalog.sort_values("feature")
    baseline = (
        rollouts.loc[rollouts.run.eq(run) & rollouts.step.between(*window)]
        .sort_values(["step", "prompt_id", "sample_id"])
        .reset_index(drop=True)
    )
    keys = ["run", "sample_id"]
    dense = baseline[keys].merge(features.dense, on=keys, validate="one_to_one")
    sparse = features.sparse.merge(baseline[keys], on=keys, validate="many_to_one")
    sparse_ids = {
        name: set(group.sample_id)
        for name, group in sparse.groupby("feature", sort=False)
    }
    groups = baseline.groupby(["step", "prompt_id"], sort=True).size()
    schedule = tuple(
        tuple(int(v) for v in step.to_numpy())
        for _, step in groups.groupby(level="step")
    )
    steps = tuple(int(v) for v in sorted(baseline.step.unique()))
    common = {
        "source_run": run,
        "window": window,
        "baseline_steps": steps,
        "group_size_schedule": schedule,
        "n_steps": len(steps),
        "n_rollouts": len(baseline),
        "n_prompt_groups": len(groups),
        "n_distinct_prompts": baseline.prompt_id.nunique(),
        "assumptions": _ASSUMPTIONS,
        "ar1_limit": _AR_LIMIT,
        "minimum_baseline_steps": _MIN_BASELINE_STEPS,
    }
    unavailable = ""
    if len(steps) < _MIN_BASELINE_STEPS:
        unavailable = "too_few_baseline_steps"
    elif any(len(s) < 2 for s in schedule):
        unavailable = "insufficient_prompt_groups"
    elif any(min(s) < 2 for s in schedule):
        unavailable = "insufficient_within_prompt_replication"
    return pd.DataFrame(
        [
            _fit_profile(
                metadata, common, unavailable, baseline, dense, dense_names, sparse_ids
            )
            for metadata in catalog.to_dict("records")
        ]
    )


def generate_fitted(
    fits: pd.DataFrame,
    *,
    seed: int,
    n_replicates: int = 32,
    n_steps: int = 80,
    onset: int = 40,
    binary_effect: float = 0.20,
    continuous_effect_sd: float = 2.0,
    planted: bool = False,
) -> SyntheticBatch:
    """Generate independent trials with recorded seed, schedule, and effects."""
    for name, value, minimum in [
        ("seed", seed, 0),
        ("n_replicates", n_replicates, 1),
        ("n_steps", n_steps, 2),
        ("onset", onset, 1),
    ]:
        if type(value) is not int or value < minimum:
            raise ValueError(f"invalid {name}")
    if onset >= n_steps or type(planted) is not bool:
        raise ValueError("onset must precede horizon and planted must be boolean")
    for name, value in [
        ("binary_effect", binary_effect),
        ("continuous_effect_sd", continuous_effect_sd),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if binary_effect > 1:
        raise ValueError("binary_effect cannot exceed one")
    if fits.empty:
        raise ValueError("fits must contain at least one profile")
    if fits.duplicated(["source_run", "feature"]).any():
        raise ValueError("duplicate fitted profile")
    settings = {
        "seed": seed,
        "n_replicates": n_replicates,
        "n_steps": n_steps,
        "onset": onset,
        "binary_effect": float(binary_effect),
        "continuous_effect_sd": float(continuous_effect_sd),
        "planted": planted,
        "assumptions": _ASSUMPTIONS,
    }
    rows, trials = [], []
    for fit in fits.sort_values(["source_run", "feature"]).to_dict("records"):
        effect = 0.0
        if planted:
            effect = (
                float(binary_effect)
                if fit["kind"] == "binary"
                else float(continuous_effect_sd * fit["marginal_sd"])
            )
        reason = fit["reason"] if fit["fit_status"] == "unavailable" else ""
        if planted and not reason:
            if fit["fit_status"] == "degenerate_binary":
                reason = "degenerate_binary_plant"
            elif fit["kind"] == "binary" and fit["mean"] + effect > 1:
                reason = "binary_target_out_of_range"
            elif fit["kind"] == "continuous" and effect <= 0:
                reason = "zero_continuous_effect"
        # Stable keyed seeds make selection, order, and added profiles irrelevant
        # to every existing trial's random stream.
        profile_key = hashlib.sha256(
            (fit["source_run"] + "\0" + fit["feature"]).encode()
        ).digest()
        entropy = [seed, *np.frombuffer(profile_key[:16], dtype="<u4").tolist()]
        for replicate in range(n_replicates):
            run = json.dumps(
                [
                    fit["source_run"],
                    fit["feature"],
                    "planted" if planted else "null",
                    replicate,
                ],
                separators=(",", ":"),
            )
            trial = {
                "run": run,
                "source_run": fit["source_run"],
                "feature": fit["feature"],
                "tier": fit["tier"],
                "kind": fit["kind"],
                "replicate": replicate,
                "seed": seed,
                "planted": planted,
                "onset": onset,
                "effect": effect,
                "status": "unavailable" if reason else "generated",
                "reason": reason,
                "n_probability_clipped": 0,
                "probability_clipped_fraction": np.nan,
            }
            if reason:
                trials.append(trial)
                continue
            rows.extend(
                _generate_trial(fit, trial, entropy, replicate, n_steps, onset, effect)
            )
            trials.append(trial)
    return SyntheticBatch(
        pd.DataFrame(rows, columns=_SERIES_COLUMNS), pd.DataFrame(trials), settings
    )


def _options(function, options: dict | None) -> dict:
    bound = signature(function).bind(pd.DataFrame(), **(options or {}))
    bound.apply_defaults()
    return {key: value for key, value in bound.arguments.items() if key != "series"}


def evaluate_fitted(
    batch: SyntheticBatch,
    *,
    sequential_options: dict | None = None,
    retrospective_options: dict | None = None,
    timing_tolerance: int = 5,
) -> SyntheticEvaluation:
    """Evaluate first alarms and retrospective boundaries with separate timing."""
    if type(timing_tolerance) is not int or timing_tolerance < 0:
        raise ValueError("timing_tolerance must be a nonnegative integer")
    seq_options = _options(detect_sequential, sequential_options)
    retro_options = _options(detect_retrospective, retrospective_options)
    sequential = detect_sequential(batch.series, **seq_options)
    retrospective = detect_retrospective(batch.series, **retro_options)
    settings = dict(
        **batch.settings,
        sequential_options=seq_options,
        retrospective_options=retro_options,
        timing_tolerance=timing_tolerance,
    )
    records = []
    for detector, result, changes, step_name in [
        ("sequential", sequential, sequential.alerts, "alert_step"),
        (
            "retrospective",
            retrospective,
            retrospective.changepoints,
            "changepoint_step",
        ),
    ]:
        diagnostics = result.diagnostics.set_index(["run", "feature"])
        events = changes.set_index(["run", "feature"])
        for trial in batch.trials.to_dict("records"):
            records.append(
                _evaluate_trial(
                    trial, detector, diagnostics, events, step_name, timing_tolerance
                )
            )
    trials = pd.DataFrame(records)
    return SyntheticEvaluation(trials, _summarize_trials(trials), settings)
