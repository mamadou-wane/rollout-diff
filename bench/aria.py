"""Read the supplied, checksum-pinned local Aria file without evaluation labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from rolloutdiff.schema import validate_rollouts

ARIA_SHA256 = "59ffb1aa73d341f6bd070f639028da8ee041176c8c8e0e18678a4d6751a11556"


def load_aria_rollouts(
    path: str | Path, windows: Sequence[tuple[int, int]] = ((1, 200),)
) -> pd.DataFrame:
    """Preserve whole source groups and exact prompt/sample identities."""
    if not windows or any(
        not isinstance(w, tuple)
        or len(w) != 2
        or any(type(value) is not int for value in w)
        or not 1 <= w[0] <= w[1] <= 200
        for w in windows
    ):
        raise ValueError("Aria windows must be inclusive integer pairs within 1..200")
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARIA_SHA256:
        raise ValueError("local Aria fixture checksum changed")
    raw = pd.read_parquet(
        path,
        columns=["sample_id", "messages", "step", "problem_id", "score"],
        filters=[
            [("step", ">=", start), ("step", "<=", end)] for start, end in windows
        ],
    )
    expected_steps = {step for start, end in windows for step in range(start, end + 1)}
    if (
        set(raw.step) != expected_steps
        or len(raw) != 256 * len(expected_steps)
        or not raw.sample_id.is_unique
        or raw.isna().any().any()
        or not raw.groupby("step").size().eq(256).all()
        or not raw.groupby("step").problem_id.nunique().eq(16).all()
        or not raw.groupby(["step", "problem_id"]).size().eq(16).all()
    ):
        raise ValueError("unexpected Aria source groups or identities")
    if (
        not raw.sample_id.str.fullmatch(r"step\d{3}_row\d{3}").all()
        or not raw.sample_id.str.extract(r"^step(\d+)")[0]
        .astype(int)
        .eq(raw.step)
        .all()
    ):
        raise ValueError("Aria sample identity does not agree with step")
    messages = raw.messages.map(json.loads)
    if not messages.map(
        lambda rows: [row["role"] for row in rows] == ["system", "user", "assistant"]
    ).all():
        raise ValueError("unexpected Aria message layout")
    return validate_rollouts(
        pd.DataFrame(
            {
                "run": "aria",
                "step": raw.step,
                "sample_id": raw.sample_id,
                "prompt_id": raw.problem_id.astype(str),
                "prompt": messages.map(lambda rows: rows[1]["content"]),
                "response": messages.map(lambda rows: rows[2]["content"]),
                "reward": raw.score,
                "prompt_messages": messages.map(lambda rows: rows[:2]),
                "response_messages": messages.map(lambda rows: rows[2:]),
            }
        )
    )
