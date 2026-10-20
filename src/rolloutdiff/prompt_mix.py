"""Prompt-mix evidence for caller-selected windows of one run."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._feature_values import feature_values
from ._sampling import (
    cluster_standard_error,
    resolve_sampling_units,
    subset_sampling_units,
)
from .features import FeatureTables


@dataclass(frozen=True)
class PromptMixEvidence:
    run: str
    feature: str
    kind: str
    before: tuple[int, int]
    after: tuple[int, int]
    positive_prompts: int | None
    total_prompts: int
    shared_prompts: int
    shared_sampling_units: int
    shared_change: float | None
    shared_se: float | None
    per_prompt: pd.DataFrame
    prompt_shift: pd.DataFrame


def _window(value: tuple[int, int]) -> tuple[int, int]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(bound) is not int or bound < 0 for bound in value)
        or value[0] > value[1]
    ):
        raise ValueError(
            "window must be an inclusive (start, end) pair of nonnegative integers"
        )
    return value


def _keywords(values: Sequence[str]) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError("keywords must be a sequence of literal strings")
    result = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("keyword must be a nonempty string")
        word = value.strip().casefold()
        if word in result:
            raise ValueError("duplicate keyword after case normalization")
        result.append(word)
    return result


def _ks(before: np.ndarray, after: np.ndarray) -> float:
    points = np.union1d(before, after)
    a = np.searchsorted(np.sort(before), points, side="right") / len(before)
    b = np.searchsorted(np.sort(after), points, side="right") / len(after)
    return float(np.max(np.abs(a - b)))


def _prompt_shift(
    detail: pd.DataFrame, keywords: list[str], has_difficulty: bool
) -> pd.DataFrame:
    before = detail.loc[detail.before_n.gt(0)]
    after = detail.loc[detail.after_n.gt(0)]
    rows = []
    for descriptor in ["prompt_chars", *(f"keyword:{word}" for word in keywords)]:
        a, b = (
            before[descriptor].to_numpy(dtype=float),
            after[descriptor].to_numpy(dtype=float),
        )
        delta = float(b.mean() - a.mean())
        rows.append(
            {
                "descriptor": descriptor,
                "statistic": "ks"
                if descriptor == "prompt_chars"
                else "absolute_prevalence",
                "before": float(a.mean()),
                "after": float(b.mean()),
                "delta": delta,
                "distance": _ks(a, b) if descriptor == "prompt_chars" else abs(delta),
                "n_before": len(a),
                "n_after": len(b),
                "missing_before": 0,
                "missing_after": 0,
            }
        )
    if has_difficulty:
        a, b = before.difficulty.dropna(), after.difficulty.dropna()
        distance = float("nan")
        if len(a) and len(b):
            pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
            distance = float(pa.sub(pb, fill_value=0).abs().sum() / 2)
        rows.append(
            {
                "descriptor": "difficulty",
                "statistic": "total_variation",
                "before": float("nan"),
                "after": float("nan"),
                "delta": float("nan"),
                "distance": distance,
                "n_before": len(a),
                "n_after": len(b),
                "missing_before": len(before) - len(a),
                "missing_after": len(after) - len(b),
            }
        )
    return pd.DataFrame(rows)


def prompt_mix(
    rollouts: pd.DataFrame,
    features: FeatureTables,
    *,
    run: str,
    feature: str,
    before: tuple[int, int],
    after: tuple[int, int],
    keywords: Sequence[str] = (),
    difficulty: Mapping[str, str] | None = None,
    sampling_units: pd.DataFrame | None = None,
) -> PromptMixEvidence:
    """Measure breadth, paired prompt change, and prompt descriptor shift.

    Windows are inclusive, ordered, disjoint, and nonempty. Pool completions per
    prompt/window, then weight shared prompts equally. Default SE is paired-change
    SD(ddof=1) / sqrt(shared prompts); explicit units cluster those residuals.
    Fewer than two shared units makes SE unavailable; no shared prompts also
    makes change unavailable.

    Binary breadth counts after-window prompts with any positive value; continuous
    breadth is unavailable. Descriptors count distinct prompts once per window:
    length uses empirical KS distance. Literal, case-insensitive keywords bounded
    by non-word characters use absolute prevalence change. Difficulty total variation uses
    known tags only and is unavailable if either window has none.
    """
    before, after = _window(before), _window(after)
    if before[1] >= after[0]:
        raise ValueError("windows must not overlap and must be in before/after order")
    words = _keywords(keywords)
    units = resolve_sampling_units(rollouts, sampling_units)
    frame, kind = feature_values(rollouts, features, run=run, feature=feature)
    return _prompt_mix_prepared(
        frame,
        kind,
        run=run,
        feature=feature,
        before=before,
        after=after,
        keywords=words,
        difficulty=difficulty,
        sampling_units=subset_sampling_units(units, frame),
    )


def _prompt_mix_prepared(
    run_rows: pd.DataFrame,
    kind: str,
    *,
    run: str,
    feature: str,
    before: tuple[int, int],
    after: tuple[int, int],
    keywords: Sequence[str] = (),
    difficulty: Mapping[str, str] | None = None,
    sampling_units: pd.DataFrame | None = None,
) -> PromptMixEvidence:
    """Compute prompt-mix evidence from identity-aligned, validated values."""
    before, after = _window(before), _window(after)
    if before[1] >= after[0]:
        raise ValueError("windows must not overlap and must be in before/after order")
    words = _keywords(keywords)
    if difficulty is not None and (
        not isinstance(difficulty, Mapping)
        or any(
            not isinstance(k, str) or not isinstance(v, str) or not v.strip()
            for k, v in difficulty.items()
        )
        or set(difficulty) - set(run_rows.prompt_id)
    ):
        raise ValueError(
            "difficulty must map prompt IDs in this run to nonempty string tags"
        )
    masks = [run_rows.step.between(*window) for window in (before, after)]
    if any(not mask.any() for mask in masks):
        raise ValueError("comparison window is empty")
    selected = run_rows.loc[masks[0] | masks[1]]
    frame = selected[["run", "sample_id", "step", "prompt_id", "_value"]].rename(
        columns={"_value": "value"}
    )

    grouped = []
    for side, window in [("before", before), ("after", after)]:
        group = (
            frame.loc[frame.step.between(*window)]
            .groupby("prompt_id", sort=True)
            .value.agg(["mean", "size"])
        )
        grouped.append(
            group.rename(columns={"mean": f"{side}_mean", "size": f"{side}_n"})
        )
    detail = grouped[0].join(grouped[1], how="outer").sort_index()
    for side in ("before", "after"):
        detail[f"{side}_n"] = detail[f"{side}_n"].fillna(0).astype("int64")
    detail["change"] = detail.after_mean - detail.before_mean
    if sampling_units is None:
        detail["sampling_unit_id"] = detail.index
    else:
        detail["sampling_unit_id"] = sampling_units.set_index(
            "prompt_id"
        ).sampling_unit_id.reindex(detail.index)
    prompts = selected.drop_duplicates("prompt_id").set_index("prompt_id").prompt
    detail["prompt_chars"] = prompts.str.len()
    for word in words:
        pattern = re.compile(r"(?<!\w)" + re.escape(word) + r"(?!\w)")
        detail[f"keyword:{word}"] = prompts.str.casefold().str.contains(pattern)
    if difficulty is not None:
        # Ignore mapping defaults: only explicitly supplied tags are evidence.
        detail["difficulty"] = detail.index.to_series().map(dict(difficulty))

    changes = detail.change.dropna()
    shared_count = len(changes)
    shared_change = float(changes.mean()) if shared_count else None
    shared_units = detail.loc[changes.index, "sampling_unit_id"]
    if shared_count > 1 and shared_units.is_unique:
        # Preserve independent-prompt arithmetic when units are unique.
        shared_se = float(changes.std(ddof=1) / np.sqrt(shared_count))
    else:
        shared_se = cluster_standard_error(changes.to_numpy(), shared_units.to_numpy())
    return PromptMixEvidence(
        run=run,
        feature=feature,
        kind=kind,
        before=before,
        after=after,
        positive_prompts=int(detail.after_mean.gt(0).sum())
        if kind == "binary"
        else None,
        total_prompts=int(detail.after_n.gt(0).sum()),
        shared_prompts=shared_count,
        shared_sampling_units=int(shared_units.nunique()),
        shared_change=shared_change,
        shared_se=shared_se,
        per_prompt=detail.reset_index(),
        prompt_shift=_prompt_shift(detail, words, difficulty is not None),
    )
