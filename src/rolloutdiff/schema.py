"""Canonical rollout records, excluding benchmark labels."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class Rollout(BaseModel):
    """One completion keyed by (run, sample_id), with unchanged text.

    Prompt IDs identify exact content within runs; messages retain roles/metadata.
    """

    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, allow_inf_nan=False
    )

    run: str = Field(min_length=1)
    step: int = Field(ge=0)
    sample_id: str = Field(min_length=1)
    prompt_id: str = Field(min_length=1)
    prompt: str
    response: str
    reward: float
    prompt_messages: list[dict[str, Any]] | None = None
    response_messages: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    @field_validator("run", "sample_id", "prompt_id")
    @classmethod
    def nonblank_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value


CANONICAL_COLUMNS = list(Rollout.model_fields)
REQUIRED_COLUMNS = [
    name for name, field in Rollout.model_fields.items() if field.is_required()
]


def validate_rollouts(table: pd.DataFrame) -> pd.DataFrame:
    """Validate required fields and step types; reset the positional index."""
    if not table.columns.is_unique:
        raise ValueError("duplicate rollout columns")
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    extra = set(table.columns) - set(CANONICAL_COLUMNS)
    if missing or extra:
        raise ValueError(
            f"invalid rollout columns: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    rows = []
    for position, row in enumerate(table.to_dict("records")):
        # pandas materializes absent optional values as NaN in mixed tables.
        for name in ("prompt_messages", "response_messages", "tool_calls"):
            value = row.get(name)
            if value is None or (isinstance(value, float) and pd.isna(value)):
                row[name] = None
        try:
            rows.append(Rollout.model_validate(row).model_dump())
        except ValidationError as exc:
            fields = ", ".join(
                ".".join(map(str, e["loc"])) for e in exc.errors(include_input=False)
            )
            raise ValueError(f"invalid rollout at row {position}: {fields}") from None
    out = pd.DataFrame(rows, columns=CANONICAL_COLUMNS)
    if out.duplicated(["run", "sample_id"]).any():
        raise ValueError("duplicate (run, sample_id) rollout identity")
    if not out.empty:
        prompt_keys = out[["run", "prompt_id", "prompt"]].copy()
        prompt_keys["messages"] = out.prompt_messages.map(
            lambda value: json.dumps(value, sort_keys=True, ensure_ascii=False)
        )
        distinct = prompt_keys.drop_duplicates().groupby(["run", "prompt_id"]).size()
        if (distinct > 1).any():
            raise ValueError(
                "prompt_id refers to different prompt content within a run"
            )
        out["step"] = out.step.astype("int64")
        out["reward"] = out.reward.astype("float64")
    return out
