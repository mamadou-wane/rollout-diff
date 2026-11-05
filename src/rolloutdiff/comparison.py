"""Prompt-balanced comparisons of selected windows or two rollout dumps."""

from __future__ import annotations

import hashlib
import json
import math
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

from ._feature_values import _FeatureValueIndex
from ._sampling import cluster_standard_error, resolve_sampling_units
from .features import extract, featurize
from .prompt_mix import _keywords, _prompt_shift, _window
from .schema import validate_rollouts
from .series import _validate_features

_Z95 = NormalDist().inv_cdf(0.975)


def _canonical_json(value) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError):
        raise ValueError("canonical metadata must contain finite JSON values") from None


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _select_run(rollouts: pd.DataFrame, run: str | None) -> pd.DataFrame:
    if run is None:
        runs = rollouts.run.unique()
        if len(runs) != 1:
            raise ValueError("select exactly one nonempty run")
        run = runs[0]
    if not isinstance(run, str) or not run.strip():
        raise ValueError("run must be a nonblank string")
    result = rollouts.loc[rollouts.run.eq(run)].sort_values(["step", "sample_id"])
    if result.empty:
        raise ValueError("selected run has no rollouts")
    return result.reset_index(drop=True)


def _fingerprint(frame: pd.DataFrame) -> str:
    return _sha256(_canonical_json(frame.sort_values("sample_id").to_dict("records")))


def canonical_fingerprint(rollouts: pd.DataFrame, *, run: str | None = None) -> str:
    """Hash all canonical fields of one full run, independent of row order."""
    return _fingerprint(_select_run(validate_rollouts(rollouts), run))


def _content_ids(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            _sha256(_canonical_json([prompt, messages]))
            for prompt, messages in frame[["prompt", "prompt_messages"]].itertuples(
                index=False, name=None
            )
        ],
        index=frame.index,
        dtype=object,
    )


def _units_by_content(all_rows, full_run, content, mapping) -> pd.Series:
    resolved = resolve_sampling_units(all_rows, mapping)
    if mapping is None:
        names = pd.Index(sorted(content.unique()), name="prompt_id")
        return pd.Series(names, index=names, name="sampling_unit_id")
    selected = resolved.loc[resolved.run.eq(full_run.run.iloc[0])]
    aliases = pd.DataFrame(
        {
            "content": content,
            "unit": full_run.prompt_id.map(
                selected.set_index("prompt_id").sampling_unit_id
            ),
        }
    ).drop_duplicates()
    if aliases.groupby("content").unit.nunique().gt(1).any():
        raise ValueError("sampling units conflict for identical prompt content")
    return aliases.set_index("content").unit.sort_index()


def _compatible_units(a: pd.Series, b: pd.Series) -> bool:
    shared = a.index.intersection(b.index)
    pairs = pd.DataFrame({"a": a.loc[shared], "b": b.loc[shared]}).drop_duplicates()
    return not pairs.a.duplicated().any() and not pairs.b.duplicated().any()


def _select_window(frame, content, units, window, side):
    window = (
        (int(frame.step.min()), int(frame.step.max()))
        if window is None
        else _window(window)
    )
    mask = frame.step.between(*window)
    if not mask.any():
        raise ValueError("comparison window is empty")
    original = frame.loc[mask].reset_index(drop=True)
    identifiers = content.loc[mask].reset_index(drop=True)
    internal = original.assign(run=side, prompt_id=identifiers)
    selected_units = units.loc[sorted(identifiers.unique())]
    summary = {
        "side": side,
        "run": frame.run.iloc[0],
        "window": list(window),
        "observed_steps": sorted(int(step) for step in original.step.unique()),
        "n_rollouts": len(original),
        "n_prompts": len(selected_units),
        "n_sampling_units": int(selected_units.nunique()),
        "source": {
            "canonical_sha256": _fingerprint(frame),
            "format": "canonical",
            "files": [],
            "reward_column": None,
        },
    }
    return original, internal, selected_units, summary


def _mean(values) -> float | None:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        return None
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        ordinary = float(values.mean())
        result = (
            ordinary
            if math.isfinite(ordinary)
            else float((values / scale).mean() * scale)
        )
    return result if math.isfinite(result) else None


def _interval(estimate: float | None, se: float | None, status: str) -> dict:
    result = {
        "estimate": estimate,
        "se": se,
        "ci_low": None,
        "ci_high": None,
        "status": status,
    }
    if estimate is None:
        result.update(se=None, status="numeric_unavailable")
    elif se is not None:
        low, high = estimate - _Z95 * se, estimate + _Z95 * se
        if math.isfinite(low) and math.isfinite(high):
            result.update(ci_low=low, ci_high=high)
        else:
            result["status"] = "numeric_unavailable"
    return result


def _statistics(values, unit_ids) -> dict:
    estimate = _mean(values)
    if estimate is None:
        return _interval(None, None, "numeric_unavailable")
    try:
        se = cluster_standard_error(values, unit_ids)
    except ValueError:
        return _interval(estimate, None, "numeric_unavailable")
    return _interval(
        estimate, se, "available" if se is not None else "insufficient_units"
    )


def _side_estimate(
    means: pd.Series, units: pd.Series, n_rollouts: int, kind: str
) -> dict:
    result = _statistics(means.to_numpy(), units.to_numpy())
    result["mean"] = result.pop("estimate")
    result.update(
        n_rollouts=n_rollouts,
        n_prompts=len(means),
        n_sampling_units=int(units.nunique()),
        positive_prompts=int(means.gt(0).sum()) if kind == "binary" else None,
    )
    return result


def _paired(a: pd.Series, b: pd.Series, units_a: pd.Series, units_b: pd.Series) -> dict:
    shared = a.index.intersection(b.index)
    if not len(shared):
        result = _interval(0.0, None, "no_shared_prompts")
        result.update(estimate=None, n_prompts=0, n_sampling_units=0)
        return result
    with np.errstate(over="ignore", invalid="ignore"):
        changes = b.loc[shared].to_numpy() - a.loc[shared].to_numpy()
    if not _compatible_units(units_a, units_b):
        result = _interval(_mean(changes), None, "incompatible_units")
        result.update(n_prompts=len(shared), n_sampling_units=None)
        return result
    shared_units = units_a.loc[shared]
    result = _statistics(changes, shared_units.to_numpy())
    result.update(n_prompts=len(shared), n_sampling_units=int(shared_units.nunique()))
    return result


def _contrast(a, b, paired, units_a, units_b, *, same_run):
    estimate = None if a["mean"] is None or b["mean"] is None else b["mean"] - a["mean"]
    if estimate is None or not math.isfinite(estimate):
        return _interval(None, None, "numeric_unavailable") | {"method": "unavailable"}
    shared = paired["n_prompts"]
    if shared == len(units_a) == len(units_b):
        if paired["estimate"] is None or not math.isclose(
            estimate, paired["estimate"], rel_tol=1e-10, abs_tol=1e-12
        ):
            # Cancellation or per-prompt overflow prevents transferring paired
            # uncertainty to the separately computed marginal difference.
            return _interval(estimate, None, "numeric_unavailable") | {
                "method": "unavailable"
            }
        result = {key: paired[key] for key in ("se", "ci_low", "ci_high", "status")}
        # Center on B - A so the interval agrees with the displayed side means.
        result = _interval(estimate, result["se"], result["status"])
        method = (
            "paired"
            if result["status"] in {"available", "insufficient_units"}
            else "unavailable"
        )
        return result | {"method": method}
    if shared:
        return _interval(estimate, None, "partial_overlap") | {"method": "unavailable"}
    if same_run and set(units_a) & set(units_b):
        return _interval(estimate, None, "shared_units") | {"method": "unavailable"}
    if a["status"] == "numeric_unavailable" or b["status"] == "numeric_unavailable":
        return _interval(estimate, None, "numeric_unavailable") | {
            "method": "unavailable"
        }
    se = None if a["se"] is None or b["se"] is None else math.hypot(a["se"], b["se"])
    if se is not None and not math.isfinite(se):
        return _interval(estimate, None, "numeric_unavailable") | {
            "method": "unavailable"
        }
    result = _interval(
        estimate, se, "available" if se is not None else "insufficient_units"
    )
    return result | {
        "method": "independent"
        if result["status"] != "numeric_unavailable"
        else "unavailable"
    }


def _citations(original, aligned_values, kind):
    if kind != "binary":
        return {"status": "unsupported_kind", "items": []}
    candidates = original.assign(
        _value=aligned_values._value.to_numpy(),
        _content=aligned_values.prompt_id.to_numpy(),
    )
    candidates = candidates.loc[candidates._value.gt(0)].sort_values(
        ["step", "sample_id"]
    )
    if candidates.empty:
        return {"status": "no_positive_rollouts", "items": []}
    selected = [(candidates.index[0], "earliest_positive")]
    contents = {candidates.iloc[0]._content}
    for index, row in candidates.iloc[1:].iterrows():
        if row._content not in contents:
            selected.append((index, "prompt_diversity"))
            contents.add(row._content)
            if len(selected) == 3:
                break
    if len(selected) < 3:
        selected_indices = {index for index, _ in selected}
        for index in candidates.index:
            if index not in selected_indices:
                selected.append((index, "additional_positive"))
                if len(selected) == 3:
                    break
    items = []
    for index, reason in selected:
        row = candidates.loc[index]
        items.append(
            {
                "run": row.run,
                "sample_id": row.sample_id,
                "step": int(row.step),
                "prompt_id": row.prompt_id,
                "feature_value": float(row._value),
                "reward": float(row.reward),
                "response_sha256": _sha256(row.response),
                "selection_reason": reason,
            }
        )
    return {"status": "selected", "items": items}


def _prompt_comparability(a, b, keywords):
    grouped = []
    for side, frame in (("before", a), ("after", b)):
        grouped.append(frame.groupby("prompt_id").size().rename(f"{side}_n"))
    detail = pd.concat(grouped, axis=1).fillna(0)
    prompts = (
        pd.concat([a, b]).drop_duplicates("prompt_id").set_index("prompt_id").prompt
    )
    detail["prompt_chars"] = prompts.str.len()
    for word in keywords:
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")
        detail[f"keyword:{word}"] = prompts.str.casefold().str.contains(pattern)
    shared = int((detail.before_n.gt(0) & detail.after_n.gt(0)).sum())
    shift = _prompt_shift(detail, keywords, False)
    return {
        "status": "shared_prompts" if shared else "no_shared_prompts",
        "shared_prompts": shared,
        "only_a_prompts": int(detail.after_n.eq(0).sum()),
        "only_b_prompts": int(detail.before_n.eq(0).sum()),
        "prompt_shift": shift.to_dict("records"),
    }


def compare_rollouts(
    rollouts_a: pd.DataFrame,
    rollouts_b: pd.DataFrame | None = None,
    *,
    a: tuple[int, int] | None = None,
    b: tuple[int, int] | None = None,
    run_a: str | None = None,
    run_b: str | None = None,
    sampling_units_a: pd.DataFrame | None = None,
    sampling_units_b: pd.DataFrame | None = None,
    keywords=(),
) -> dict:
    """Compare selected ranges with equal weighting of exact prompts.

    Pool each prompt's completions within each range and select a shared vocabulary.
    One input requires ordered, disjoint inclusive ranges. Two inputs default to
    their full observed ranges. Paired estimates cover shared prompts. Partial
    overlap leaves the overall interval unavailable.
    """
    words = _keywords(keywords)
    same_run = rollouts_b is None
    all_a = validate_rollouts(rollouts_a)
    full_a = _select_run(all_a, run_a)
    if same_run:
        a, b = _window(a), _window(b)
        if a[1] >= b[0]:
            raise ValueError("windows must be ordered and must not overlap")
        if run_b is not None and run_b != full_a.run.iloc[0]:
            raise ValueError("one-input windows must select the same run")
        all_b, full_b = all_a, full_a
        if sampling_units_a is None:
            sampling_units_a = sampling_units_b
        if sampling_units_b is None:
            sampling_units_b = sampling_units_a
    else:
        all_b = validate_rollouts(rollouts_b)
        full_b = _select_run(all_b, run_b)
    content_a, content_b = _content_ids(full_a), _content_ids(full_b)
    units_a = _units_by_content(all_a, full_a, content_a, sampling_units_a)
    units_b = _units_by_content(all_b, full_b, content_b, sampling_units_b)
    if same_run:
        if not _compatible_units(units_a, units_b):
            raise ValueError("sampling unit maps contradict within the same run")
        units_b = units_a.copy()
    sides = [
        _select_window(full_a, content_a, units_a, a, "a"),
        _select_window(full_b, content_b, units_b, b, "b"),
    ]
    internal = pd.concat([side[1] for side in sides], ignore_index=True)
    features = featurize(internal)
    _validate_features(internal, features)
    index = _FeatureValueIndex(internal, features)
    catalog = [
        *features.catalog.to_dict("records"),
        {"feature": "scalar:reward", "kind": "continuous", "tier": "structural"},
    ]
    comparisons = []
    for metadata in catalog:
        name, kind = metadata["feature"], metadata["kind"]
        values = [index.values(side, name)[0] for side in ("a", "b")]
        means = [
            frame.groupby("prompt_id", sort=True)._value.mean().reindex(side[2].index)
            for frame, side in zip(values, sides, strict=True)
        ]
        estimates = [
            _side_estimate(mean, side[2], len(side[0]), kind)
            for mean, side in zip(means, sides, strict=True)
        ]
        paired = _paired(means[0], means[1], sides[0][2], sides[1][2])
        difference = _contrast(
            *estimates, paired, sides[0][2], sides[1][2], same_run=same_run
        )
        change = difference["estimate"]
        direction = (
            "unavailable"
            if change is None
            else "increase"
            if change > 0
            else "decrease"
            if change < 0
            else "unchanged"
        )
        comparisons.append(
            {
                "order": 0,
                **metadata,
                "direction": direction,
                "a": estimates[0],
                "b": estimates[1],
                "difference": difference,
                "paired": paired,
                "citations": {
                    label: _citations(side[0], frame, kind)
                    for label, side, frame in zip(
                        ("a", "b"), sides, values, strict=True
                    )
                },
            }
        )
    comparisons.sort(
        key=lambda row: (
            row["kind"] != "binary",
            -abs(row["difference"]["estimate"])
            if row["kind"] == "binary" and row["difference"]["estimate"] is not None
            else 0,
            row["feature"],
        )
    )
    for order, comparison in enumerate(comparisons, start=1):
        comparison["order"] = order
    report = {
        "report_type": "comparison",
        "schema_version": 1,
        "mode": "windows" if same_run else "dumps",
        "sides": [side[3] for side in sides],
        "prompt_mix": _prompt_comparability(sides[0][1], sides[1][1], words),
        "features": comparisons,
    }
    from .comparison_report import validate_comparison_report

    return validate_comparison_report(report)


def resolve_comparison_citations(report, rollouts, *, side: str, feature: str) -> dict:
    """Verify full-run provenance and every selected citation before returning text."""
    from .comparison_report import validate_comparison_report

    report = validate_comparison_report(report)
    if not isinstance(side, str) or side not in {"a", "b"}:
        raise ValueError("comparison side must be a or b")
    matches = [item for item in report["features"] if item["feature"] == feature]
    if not matches:
        raise ValueError("comparison feature is unavailable")
    source = next(item for item in report["sides"] if item["side"] == side)
    full_run = _select_run(validate_rollouts(rollouts), source["run"])
    if _fingerprint(full_run) != source["source"]["canonical_sha256"]:
        raise ValueError("comparison source fingerprint differs")
    citations = matches[0]["citations"][side]
    by_identity = full_run.set_index("sample_id")
    items = []
    for citation in citations["items"]:
        if citation["sample_id"] not in by_identity.index:
            raise ValueError("comparison citation identity is unavailable")
        row = by_identity.loc[citation["sample_id"]]
        if any(
            citation[field] != row[field]
            for field in ("run", "step", "prompt_id", "reward")
        ):
            raise ValueError("comparison citation metadata differs")
        if _sha256(row.response) != citation["response_sha256"]:
            raise ValueError("comparison citation response differs")
        extracted = extract(row.response)
        value = extracted.dense.get(feature, float(feature in extracted.sparse))
        if value != citation["feature_value"] or value != 1.0:
            raise ValueError("comparison citation feature differs")
        items.append({**citation, "response": row.response})
    return {
        "side": side,
        "feature": feature,
        "status": citations["status"],
        "items": items,
    }
