"""Optional comparisons using verified, unmodified external Python sources."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from contextlib import contextmanager
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path

import numpy as np
import pandas as pd

from rolloutdiff._sampling import resolve_sampling_units
from rolloutdiff.schema import validate_rollouts

_MANIFEST = Path(__file__).with_name("external_manifest.json")
_KEYS = ["run", "sample_id"]


def _prepare(
    rollouts: pd.DataFrame,
    dense_features: pd.DataFrame,
    *,
    reward_groups: pd.DataFrame | None = None,
) -> pd.DataFrame:
    frame = validate_rollouts(rollouts)
    if frame.empty or frame.run.nunique() != 1:
        raise ValueError("external baselines require one nonempty run")
    required = {*_KEYS, "resp_chars"}
    if not dense_features.columns.is_unique or not required.issubset(
        dense_features.columns
    ):
        raise ValueError(
            "dense features require unique columns run/sample_id/resp_chars"
        )
    expected = pd.MultiIndex.from_frame(frame[_KEYS])
    actual = pd.MultiIndex.from_frame(dense_features[_KEYS])
    if (
        not actual.is_unique
        or len(actual) != len(expected)
        or not actual.isin(expected).all()
    ):
        raise ValueError("dense feature identities must exactly match rollouts")
    lengths = dense_features.resp_chars.to_numpy(dtype=float)
    if not np.isfinite(lengths).all() or (lengths < 0).any():
        raise ValueError("resp_chars must be finite and nonnegative")
    if reward_groups is not None:
        if (
            not isinstance(reward_groups, pd.DataFrame)
            or not reward_groups.columns.is_unique
            or set(reward_groups.columns) != {"run", "prompt_id", "group_id"}
        ):
            raise ValueError(
                "reward_groups must contain exactly run/prompt_id/group_id"
            )
        mapping = reward_groups.rename(columns={"group_id": "sampling_unit_id"})
    else:
        mapping = None
    # Reuse complete-key validation, without equating descriptive reward groups
    # with the independent units used by the core uncertainty estimator.
    groups = resolve_sampling_units(frame, mapping).rename(
        columns={"sampling_unit_id": "group_id"}
    )
    frame = frame.merge(groups, on=["run", "prompt_id"], validate="many_to_one")
    return (
        frame.merge(
            dense_features[[*_KEYS, "resp_chars"]], on=_KEYS, validate="one_to_one"
        )
        .sort_values(["step", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def _scalar_records(frame: pd.DataFrame) -> list[dict]:
    records = []
    for step, group in frame.groupby("step", sort=True):
        reward = group.reward.to_numpy(dtype=float)
        length = group.resp_chars.to_numpy(dtype=float) / 4.0
        by_group = group.groupby("group_id").reward
        deviations = by_group.std(ddof=1)
        group_variance_available = bool(deviations.notna().all())
        advantage = reward - by_group.transform("mean").to_numpy()
        records.append(
            {
                "step": int(step),
                "reward_mean": float(reward.mean()),
                "reward_std": float(reward.std()),
                "reward_min": float(reward.min()),
                "reward_max": float(reward.max()),
                "response_len_mean": float(length.mean()),
                "response_len_p95": float(np.percentile(length, 95)),
                "response_len_max": float(length.max()),
                "group_reward_std_mean": float(deviations.mean())
                if group_variance_available
                else None,
                "zero_var_group_frac": float(deviations.eq(0).mean())
                if group_variance_available
                else None,
                "advantage_mean": float(advantage.mean()),
                "advantage_std": float(advantage.std()),
            }
        )
    return records


def _verify_source(source_root: Path, metadata: dict) -> Path:
    directory = Path(source_root) / metadata["directory"]
    if not directory.is_dir():
        raise FileNotFoundError(f"external source directory missing: {directory}")
    expected = metadata["files"]
    # Bytecode and native modules can shadow source. Reject unmanifested
    # executable files throughout each import root, including .pth files.
    executable_suffixes = {".py", ".pyc", ".pyo", ".so", ".pyd", ".dylib", ".pth"}
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"external source symlink is not allowed: {path}")
        if path.is_file() and path.suffix.lower() in executable_suffixes:
            relative = path.relative_to(directory).as_posix()
            if relative not in expected:
                raise ValueError(f"unmanifested external executable file: {relative}")
    for relative, digest in expected.items():
        path = directory / relative
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(
                f"external source checksum mismatch: {metadata['directory']}/{relative}"
            )
    return directory / metadata["import_path"]


@contextmanager
def _source_context(source_root: Path, metadata: dict):
    import_root = _verify_source(source_root, metadata).resolve()
    prefix = metadata["module_prefix"]
    matches = lambda name: name == prefix or name.startswith(prefix + ".")
    if any(matches(name) for name in sys.modules):
        raise ValueError(
            f"external package already imported: {prefix}; use a fresh process"
        )
    original_path = sys.path[:]
    original_bytecode = sys.dont_write_bytecode
    try:
        sys.path.insert(0, str(import_root))
        sys.dont_write_bytecode = True
        importlib.invalidate_caches()
        # Namespace packages lose to a regular package later on sys.path.
        # Resolve the root package only within the verified import directory,
        # then bind a fixed search path before any package code can execute.
        spec = PathFinder.find_spec(prefix, [str(import_root)])
        if spec is None or spec.submodule_search_locations is None:
            raise ValueError(f"verified source does not define package: {prefix}")
        locations = [
            str(Path(path).resolve()) for path in spec.submodule_search_locations
        ]
        if not locations or any(
            not Path(path).is_relative_to(import_root) for path in locations
        ):
            raise ValueError(
                f"external package path is outside verified source: {prefix}"
            )
        spec.submodule_search_locations = locations
        module = module_from_spec(spec)
        sys.modules[prefix] = module
        if spec.origin is not None:
            if not Path(spec.origin).resolve().is_relative_to(import_root):
                raise ValueError(
                    f"external package origin is outside verified source: {prefix}"
                )
            spec.loader.exec_module(module)
        yield
    finally:
        # Remove imported modules so repeated calls use their own source snapshots.
        for name in list(sys.modules):
            if matches(name):
                del sys.modules[name]
        sys.path[:] = original_path
        sys.dont_write_bytecode = original_bytecode
        importlib.invalidate_caches()


def _rlprobe_finding(finding: dict) -> dict:
    # Pinned fields contain fixed templates and scalar evidence. Reduce sample
    # links to counts to keep row identifiers and completion text private.
    fields = (
        "signature_id",
        "severity",
        "first_step",
        "confidence",
        "explanation",
        "evidence",
        "suggested_actions",
    )
    result = {key: finding[key] for key in fields if key in finding}
    result["n_linked_samples"] = len(finding.get("links_to_samples", []))
    return result


def _rldoctor_finding(finding: dict) -> dict:
    # This API receives scalar records only. Explicit fields retain its
    # code-authored templates/metrics and exclude any unexpected text payload.
    fields = (
        "detector",
        "title",
        "severity",
        "summary",
        "evidence",
        "prescription",
        "references",
        "metrics",
        "wasted_fraction",
        "missing_fields",
    )
    return {key: finding[key] for key in fields if key in finding}


def _run_rlprobe(frame: pd.DataFrame, records: list[dict]) -> dict:
    from rlprobe.doctor.base import REGISTRY
    from rlprobe.doctor.engine import diagnose_view
    from rlprobe.doctor.runview import RunView
    from rlprobe.recorder.sampling import SamplingConfig, select_samples
    from rlprobe.schemas import RunManifest, SampleRecord, StepRecord

    rename = {
        "response_len_mean": "completion_len_mean",
        "response_len_p95": "completion_len_p95",
        "response_len_max": "completion_len_max",
        "zero_var_group_frac": "zero_variance_group_frac",
        "advantage_mean": "adv_mean",
        "advantage_std": "adv_std",
    }
    steps = [
        StepRecord(**{rename.get(key, key): value for key, value in record.items()})
        for record in records
    ]
    rng = np.random.default_rng(0)
    samples = []
    config = SamplingConfig()
    for step, group in frame.groupby("step", sort=True):
        advantage = group.reward - group.groupby("group_id").reward.transform("mean")
        candidates = [
            SampleRecord(
                step=int(step),
                sample_id=row.sample_id,
                group_id=row.group_id,
                prompt_id=row.prompt_id,
                completion_text=row.response[:2000],
                reward_total=float(row.reward),
                advantage=float(adv),
                completion_len=int(row.resp_chars / 4.0),
            )
            for row, adv in zip(group.itertuples(index=False), advantage, strict=False)
        ]
        samples.extend(select_samples(candidates, config, rng))
    view = RunView(
        RunManifest(
            run_id=str(frame.run.iloc[0]),
            created_at="2026-10-17T00:00:00",
            algorithm="grpo",
        ),
        steps,
        samples_provider=lambda: samples,
    )
    findings = [
        _rlprobe_finding(finding.model_dump()) for finding in diagnose_view(view)
    ]
    return {
        "status": "executed",
        "findings": findings,
        "coverage": {
            "catalogue_size": len(REGISTRY),
            "catalogue": [detector.id for detector in REGISTRY],
            "n_step_records": len(steps),
            "n_sample_records": len(samples),
            "provided_scalar_fields": sorted(
                set().union(*(record.model_fields_set for record in steps))
            ),
            "sampling": {
                "top_k": config.top_k,
                "bottom_k": config.bottom_k,
                "random_k": config.random_k,
                "seed": 0,
            },
            "missingness": "The API returns findings only; zero findings does not establish every detector had its required inputs.",
        },
    }


def _run_rldoctor(records: list[dict], run: str) -> dict:
    from rldoctor import diagnose, run_from_records

    diagnosis = diagnose(run_from_records(records, name=run))
    findings = [_rldoctor_finding(finding.to_dict()) for finding in diagnosis.findings]
    return {
        "status": "executed",
        "findings": findings,
        "headline": diagnosis.headline,
        "coverage": {
            "catalogue_size": len(findings),
            "n_step_records": len(records),
            "n_skipped": len(diagnosis.skipped),
            "n_passed": len(diagnosis.passed),
            "available_fields": diagnosis.run.available_fields,
            "unmapped_keys": diagnosis.run.unmapped_keys,
        },
    }


def _run_rhda(frame: pd.DataFrame) -> dict:
    from detection.rhda.common.surface_stats import compute_surface_stats

    baseline = None
    rows = []
    for step, group in frame.groupby("step", sort=True):
        entries = [
            {"output": row.response, "score": float(row.reward)}
            for row in group.itertuples(index=False)
        ]
        summary, new_baseline = compute_surface_stats(entries, baseline, int(step))
        if new_baseline is not None:
            baseline = new_baseline
        n_drift_lines = sum(
            line.strip().startswith('"') and "% of responses" in line
            for line in summary.splitlines()
        )
        rows.append({"step": int(step), "n_drift_lines": n_drift_lines})
    return {
        "status": "executed",
        "steps": rows,
        "coverage": {
            "layer": "deterministic compute_surface_stats only",
            "agent_executed": False,
            "baseline_step": baseline.step if baseline else None,
            "n_steps": len(rows),
            "n_steps_with_drift": sum(row["n_drift_lines"] > 0 for row in rows),
            "interpretation": "Drift summary lines are evidence supplied to RHDA, not RHDA agent alerts.",
        },
    }


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, set):
        return [_json_safe(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if value is pd.NA or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def run_external(
    rollouts: pd.DataFrame,
    dense_features: pd.DataFrame,
    source_root: str | Path | None = None,
    *,
    reward_groups: pd.DataFrame | None = None,
) -> dict:
    """Run pinned third-party APIs with explicit provenance and input coverage."""
    frame = _prepare(rollouts, dense_features, reward_groups=reward_groups)
    records = _scalar_records(frame)
    group_coverage = []
    for step, group in frame.groupby("step", sort=True):
        counts = group.groupby("group_id").size()
        singletons = int(counts.eq(1).sum())
        group_coverage.append(
            {
                "step": int(step),
                "n_groups": len(counts),
                "n_exact_prompts": int(group.prompt_id.nunique()),
                "n_reward_groups": len(counts),
                "n_singleton_groups": singletons,
                "n_variance_groups": len(counts) - singletons,
                "group_variance_available": singletons == 0,
            }
        )
    manifest = json.loads(_MANIFEST.read_text())["tools"]
    # Validate all available snapshots before running any comparison, so an
    # integrity failure cannot leave a partially executed benchmark.
    if source_root is not None:
        for metadata in manifest.values():
            if (Path(source_root) / metadata["directory"]).is_dir():
                _verify_source(Path(source_root), metadata)
    runners = {
        "rlprobe": lambda: _run_rlprobe(frame, records),
        "rldoctor": lambda: _run_rldoctor(records, str(frame.run.iloc[0])),
        "rhda_surface": lambda: _run_rhda(frame),
    }
    output = {}
    for name, metadata in manifest.items():
        provenance = {
            key: metadata[key]
            for key in (
                "repository",
                "revision",
                "license",
                "license_url",
                "comparison",
            )
        }
        provenance["source_files_verified"] = len(metadata["files"])
        if (
            source_root is None
            or not (Path(source_root) / metadata["directory"]).is_dir()
        ):
            result = {
                "status": "unavailable",
                "reason": "pinned external source snapshot not supplied",
                "coverage": {"executed": False},
            }
            provenance["source_files_verified"] = 0
        else:
            with _source_context(Path(source_root), metadata):
                result = runners[name]()
        result["provenance"] = provenance
        result["coverage"]["prompt_groups"] = group_coverage
        result["input_policy"] = {
            "scalar_weighting": "rollout-weighted means, matching archived external wrappers",
            "approximate_tokens": "resp_chars / 4; not tokenizer measurements",
            "grouping": (
                "explicit descriptive reward groups"
                if reward_groups is not None
                else "exact prompt identity (default reward group)"
            ),
            "advantage": "computed reward minus descriptive group mean; not logged trainer advantages",
            "group_variance": "sample SD within reward group; whole-step group variance summaries unavailable if any group is singleton",
            "response": "Canonical raw responses include thinking; RHDA strips thinking internally. Archived wrappers stripped thinking for every tool.",
            "unavailable_metrics": [
                "entropy",
                "KL",
                "clipping",
                "gradients",
                "held-out evaluation",
                "reward components",
            ],
            "artifact_policy": "Aggregate metrics and code-authored findings only; no response ngrams, prompt/completion excerpts, or individual sample links.",
        }
        output[name] = result
    return _json_safe(output)
