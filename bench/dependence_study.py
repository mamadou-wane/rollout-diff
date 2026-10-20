"""Paired temporal-dependence study, fitted to pinned Aria reward baseline 1\u201315."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from rolloutdiff.features import FeatureTables
from rolloutdiff.retrospective import detect_retrospective

from .aria import ARIA_SHA256, load_aria_rollouts
from .run_local import _clean
from .synthetic import SyntheticBatch, fit_baseline, generate_fitted

SEEDS = (20261019, 20261020)
RHOS = (-0.5, 0.0, 0.2, 0.65, 0.9, 0.975)
EFFECTS = (0.0, 0.25, 0.5, 1.0, 2.0)
N_REPLICATES = 64
HORIZON, ONSET, TOLERANCE = 80, 40, 5


def fit_reward(path: str | Path) -> pd.Series:
    rollouts = load_aria_rollouts(path, windows=((1, 15),))
    features = FeatureTables(
        dense=rollouts[["run", "sample_id"]].copy(),
        sparse=pd.DataFrame(columns=["run", "sample_id", "feature"]),
        catalog=pd.DataFrame(columns=["feature", "tier", "kind"]),
    )
    return fit_baseline(
        rollouts, features, run="aria", window=(1, 15), feature_names=["scalar:reward"]
    ).iloc[0]


def make_scenario(
    source: pd.Series,
    *,
    rho: float,
    sampling_multiplier: float,
    seed: int,
    n_replicates: int,
    effect_sd: float,
) -> SyntheticBatch:
    """Hold latent marginal variance and effect units fixed; alternate +/- plants."""
    if source.kind != "continuous" or source.fit_status != "fit":
        raise ValueError("study requires an available continuous baseline fit")
    if not np.isfinite(rho) or abs(rho) >= 1:
        raise ValueError("rho must be stationary")
    if not np.isfinite(sampling_multiplier) or sampling_multiplier <= 0:
        raise ValueError("sampling_multiplier must be positive")
    if not np.isfinite(effect_sd) or effect_sd < 0:
        raise ValueError("effect_sd must be nonnegative")
    fit = source.copy(deep=True)
    fit["ar1"] = rho
    for field in ("within_variance", "prompt_variance"):
        fit[field] *= sampling_multiplier
    batch = generate_fitted(
        pd.DataFrame([fit]),
        seed=seed,
        n_replicates=n_replicates,
        n_steps=HORIZON,
        onset=ONSET,
        planted=False,
    )
    trials, series = batch.trials.copy(), batch.series.copy()
    if not trials.status.eq("generated").all():
        raise ValueError("study unexpectedly contains unavailable generated trials")
    effects = float(effect_sd * source.marginal_sd) * np.where(
        trials.replicate % 2 == 0, 1.0, -1.0
    )
    trials["effect"] = effects
    trials["planted"] = effect_sd > 0
    shifts = series.run.map(trials.set_index("run").effect).where(
        series.step.ge(ONSET), 0.0
    )
    series["mean"] += shifts
    settings = dict(
        batch.settings,
        planted=effect_sd > 0,
        rho=rho,
        sampling_multiplier=sampling_multiplier,
        effect_sd=effect_sd,
        effect_unit=float(source.marginal_sd),
        effect_sign="alternating_by_replicate",
        temporal_variance=float(fit.temporal_variance),
        innovation_variance=float(fit.temporal_variance * (1 - rho**2)),
    )
    return SyntheticBatch(series, trials, settings)


def _oracle_gains(series: pd.DataFrame, settings: dict, fit: pd.Series, minimum: int):
    """GLS gains at every boundary under the generating covariance.

    Isolates covariance estimation from boundary search and penalty: the gain
    at the L2 boundary tests the selected split with the true covariance, and
    the maximum over boundaries tests the search itself.
    """
    x = series["mean"].to_numpy()
    n = len(x)
    counts = series.n_rollouts.to_numpy() / series.n_prompts.to_numpy()
    multiplier = settings["sampling_multiplier"]
    measured = (
        multiplier
        * (fit.prompt_variance + fit.within_variance / counts)
        / series.n_prompts
    )
    index = np.arange(n)
    covariance = settings["temporal_variance"] * settings["rho"] ** np.abs(
        index[:, None] - index[None, :]
    ) + np.diag(measured.to_numpy(dtype=float))
    root = np.linalg.cholesky(covariance)
    logdet = 2 * np.log(np.diag(root)).sum()
    whitened_y = np.linalg.solve(root, x)
    whitened_one = np.linalg.solve(root, np.ones(n))

    def cost(design):
        coefficient, *_ = np.linalg.lstsq(design, whitened_y, rcond=None)
        residual = whitened_y - design @ coefficient
        return logdet + residual @ residual, coefficient

    null, _ = cost(whitened_one[:, None])
    gains = {}
    for boundary in range(minimum, n - minimum + 1):
        step = np.linalg.solve(root, (index >= boundary).astype(float))
        alternative, coefficient = cost(np.column_stack([whitened_one, step]))
        gains[boundary] = (null - alternative, coefficient[1])
    return gains


def evaluate_pair(
    batch: SyntheticBatch, suite: str, *, oracle: pd.Series | None = None
) -> pd.DataFrame:
    """Score both detector modes and, with the scenario's fit, two oracle arms."""
    records, selected = [], {}
    oracle_rows = []
    for method in ("independent", "ar1"):
        result = detect_retrospective(batch.series, dependence=method)
        diagnostics = result.diagnostics.set_index(["run", "feature"])
        changes = result.changepoints.set_index(["run", "feature"])
        selected[method] = changes
        for trial in batch.trials.to_dict("records"):
            key = trial["run"], trial["feature"]
            diagnostic = diagnostics.loc[key]
            flagged = key in changes.index
            scorable = diagnostic.status in ("change", "no_change")
            detected, wrong, error = False, False, np.nan
            if flagged and trial["planted"]:
                event = changes.loc[key]
                detected = event.direction == ("up" if trial["effect"] > 0 else "down")
                wrong = not detected
                if detected:
                    error = int(event.changepoint_step) - trial["onset"]
            row = {
                "suite": suite,
                "seed": trial["seed"],
                "replicate": trial["replicate"],
                "rho": batch.settings["rho"],
                "sampling_multiplier": batch.settings["sampling_multiplier"],
                "effect_sd": batch.settings["effect_sd"],
                "effect": trial["effect"],
                "method": method,
                "scorable": scorable,
                "flagged": flagged,
                "detected": detected,
                "wrong_direction": wrong,
                "localized": detected and abs(error) <= TOLERANCE,
                "boundary_error": error,
                "absolute_error": abs(error),
                "candidate_step": diagnostic.candidate_step,
                "detector_status": diagnostic.status,
                "detector_reason": diagnostic.reason,
            }
            row.update(
                {
                    column: diagnostic[column]
                    for column in diagnostic.index
                    if column.startswith("covariance_")
                }
            )
            records.append(row)
            if method == "independent" and oracle is not None:
                oracle_rows.append((trial, row, diagnostic))
    minimum = 5
    for trial, row, diagnostic in oracle_rows:
        series = batch.series.loc[batch.series.run.eq(trial["run"])].sort_values("step")
        gains = _oracle_gains(series, batch.settings, oracle, minimum)
        penalty = 3.0 * np.log(len(series))
        arms = {}
        if diagnostic.status == "change":
            boundary = int(diagnostic.n_before)
            arms["oracle_boundary"] = (boundary, *gains[boundary])
        best = max(gains, key=lambda k: gains[k][0])
        arms["oracle_search"] = (best, *gains[best])
        for method in ("oracle_boundary", "oracle_search"):
            flagged, detected, error = False, False, np.nan
            if method in arms:
                boundary, gain, effect = arms[method]
                flagged = gain > penalty
                if flagged and trial["planted"]:
                    detected = np.sign(effect) == np.sign(trial["effect"])
                    if detected:
                        error = int(series.step.to_numpy()[boundary]) - trial["onset"]
            records.append(
                {
                    **{
                        key: row[key]
                        for key in row
                        if not key.startswith("covariance_")
                    },
                    "method": method,
                    "scorable": method in arms or diagnostic.status != "change",
                    "flagged": flagged,
                    "detected": detected,
                    "wrong_direction": flagged and trial["planted"] and not detected,
                    "localized": detected and abs(error) <= TOLERANCE,
                    "boundary_error": error,
                    "absolute_error": abs(error),
                    "candidate_step": int(series.step.to_numpy()[arms[method][0]])
                    if method in arms
                    else None,
                    "detector_status": "oracle",
                    "detector_reason": "generating_covariance",
                }
            )
    old, new = selected["independent"], selected["ar1"]
    if not new.index.isin(old.index).all():
        raise AssertionError("confirmation introduced a new boundary")
    if len(new) and not new[["changepoint_step", "direction"]].equals(
        old.loc[new.index, ["changepoint_step", "direction"]]
    ):
        raise AssertionError("confirmation moved an accepted boundary")
    return pd.DataFrame(records)


def summarize(trials: pd.DataFrame, *, per_seed: bool = False) -> list[dict]:
    keys = ["suite", "rho", "sampling_multiplier", "effect_sd", "method"]
    if per_seed:
        keys.append("seed")
    rows = []
    for identity, group in trials.groupby(keys, sort=True):
        row = dict(zip(keys, identity, strict=False))
        total = len(group)
        planted = row["effect_sd"] > 0
        row.update(
            n_trials=total,
            n_unscorable=int((~group.scorable).sum()),
            n_flagged=int(group.flagged.sum()),
            n_detected=int(group.detected.sum()),
            n_wrong_direction=int(group.wrong_direction.sum()),
            n_localized=int(group.localized.sum()),
            n_missed=int((group.scorable & ~group.flagged).sum()) if planted else None,
            false_alert_rate=float(group.flagged.mean()) if not planted else None,
            detection_rate=float(group.detected.mean()) if planted else None,
            localization_rate=float(group.localized.mean()) if planted else None,
        )
        for field in ("absolute_error", "boundary_error"):
            values = group[field].dropna()
            row[f"mean_{field}"] = float(values.mean()) if len(values) else None
            row[f"median_{field}"] = float(values.median()) if len(values) else None
        if "covariance_status" in group:
            row["covariance_status_counts"] = (
                group.covariance_status.value_counts().to_dict()
            )
            for field in (
                "null_rho_bound",
                "alternative_rho_bound",
                "null_variance_bound",
                "alternative_variance_bound",
            ):
                row[f"n_covariance_{field}"] = int(
                    group[f"covariance_{field}"].eq(True).sum()
                )
        rows.append(row)
    return rows


def run_study(path: str | Path) -> dict:
    started = time.perf_counter()
    fit = fit_reward(path)
    print(
        f"Reward baseline: raw rho={fit.ar1_raw:.6f}, projected={fit.ar1}, temporal variance={fit.temporal_variance:.8f}",
        flush=True,
    )
    evaluations = []
    # Keep the diagnostic seed separate from validation summaries.
    for rho in (0.0, 0.5, 0.8, 0.95):
        batch = make_scenario(
            fit,
            rho=rho,
            sampling_multiplier=1.0,
            seed=20261017,
            n_replicates=32,
            effect_sd=0.0,
        )
        evaluations.append(evaluate_pair(batch, "historical_diagnosis", oracle=fit))
    for seed in SEEDS:
        for rho in RHOS:
            for effect in EFFECTS:
                batch = make_scenario(
                    fit,
                    rho=rho,
                    sampling_multiplier=1.0,
                    seed=seed,
                    n_replicates=N_REPLICATES,
                    effect_sd=effect,
                )
                evaluations.append(evaluate_pair(batch, "fresh_dependence", oracle=fit))
            print(f"Completed seed={seed}, rho={rho}", flush=True)
        for sampling in (0.25, 1.0, 4.0):
            for effect in (0.0, 0.5, 2.0):
                batch = make_scenario(
                    fit,
                    rho=0.95,
                    sampling_multiplier=sampling,
                    seed=seed,
                    n_replicates=N_REPLICATES,
                    effect_sd=effect,
                )
                evaluations.append(evaluate_pair(batch, "fresh_sampling", oracle=fit))
            print(f"Completed seed={seed}, sampling multiplier={sampling}", flush=True)
    trials = pd.concat(evaluations, ignore_index=True)
    root = Path(__file__).resolve().parents[1]
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "src/rolloutdiff/retrospective.py",
            "src/rolloutdiff/_dependence.py",
            "bench/dependence_study.py",
            "bench/synthetic.py",
        )
    }
    return _clean(
        {
            "schema_version": 1,
            "source_sha256": ARIA_SHA256,
            "baseline": fit.to_dict(),
            "protocol": {
                "seeds": SEEDS,
                "n_replicates_per_seed": N_REPLICATES,
                "horizon": HORIZON,
                "onset": ONSET,
                "tolerance": TOLERANCE,
                "rhos": RHOS,
                "effects_sd": EFFECTS,
                "sampling_rho": 0.95,
                "sampling_multipliers": (0.25, 1.0, 4.0),
                "sampling_effects_sd": (0.0, 0.5, 2.0),
                "effect_unit": "original fitted rollout marginal SD",
                "positive_negative": "alternating replicate parity",
                "pairing": "same random draws and SE for each effect; same standard-normal draws across rho and sampling",
                "methods": ("independent", "ar1", "oracle_boundary", "oracle_search"),
                "min_segment_steps": 5,
                "penalty_multiplier": 3.0,
                "frozen_before_fresh_outcomes": True,
            },
            "summary": summarize(trials),
            "per_seed": summarize(trials, per_seed=True),
            "trials": trials.to_dict("records"),
            "source_code_sha256": hashes,
            "elapsed_seconds": time.perf_counter() - started,
            "limitations": [
                "Short projected baseline fit is a scenario source, not verified Aria dependence.",
                "Fixed-parameter stationary Gaussian latent AR1 plus independent prompt noise only.",
                "No multiple-feature false-selection guarantee or parameter-uncertainty bootstrap.",
                "Covariance confirmation evaluates only the original L2 boundary on a fixed nuisance grid.",
                "Oracle arms use the generating covariance and are unavailable to the detector.",
                "Conditional boundary-error changes reflect selection, never improved boundary estimation.",
            ],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aria", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.output.resolve().is_relative_to(root):
        parser.error("write generated study artifacts outside the repository")
    report = run_study(args.aria)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Metrics written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
