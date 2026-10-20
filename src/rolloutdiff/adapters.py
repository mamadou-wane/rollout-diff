"""Text-only TRL/verl adapters and an explicit mapping for tabular input.

Layout references:
https://github.com/huggingface/trl/blob/main/trl/trainer/grpo_trainer.py (log)
https://github.com/verl-project/verl/blob/v0.9.1/verl/trainer/ppo/ray_trainer.py (_write_generations)
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .schema import CANONICAL_COLUMNS, validate_rollouts

_TRL_FILE = re.compile(r"completions_(\d+)\.parquet\Z")
_VERL_FILE = re.compile(r"(?:step_)?(\d+)\.jsonl(?:\.gz)?\Z")


def _plain(value: Any) -> Any:
    """Convert Arrow's nested numpy arrays to ordinary JSON-shaped containers."""
    if isinstance(value, np.ndarray):
        return [_plain(x) for x in value.tolist()]
    if isinstance(value, list):
        return [_plain(x) for x in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(value: Any) -> tuple[str, list[dict] | None]:
    value = _plain(value)
    if isinstance(value, str):
        return value, None
    if not isinstance(value, list):
        raise ValueError("content must be text or a list of chat messages")
    parts = []
    for message in value:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("chat content requires messages with a role")
        content = message.get("content")
        if content is None and message.get("tool_calls"):
            content = ""
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if (
                    not isinstance(block, dict)
                    or block.get("type") != "text"
                    or not isinstance(block.get("text"), str)
                ):
                    raise ValueError("only text content blocks are supported")
                parts.append(block["text"])
        else:
            raise ValueError("chat content must be text")
    return "\n".join(parts), value


def _identity(value: Any) -> str:
    value = _plain(value)
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    raise ValueError("identity column must contain nonempty strings or integers")


def from_table(
    table: pd.DataFrame,
    *,
    run: str,
    columns: Mapping[str, str] | None = None,
    source: str = "table",
) -> pd.DataFrame:
    """Map canonical names to input columns; discard other input columns.

    `step`, `prompt`, `response`, `reward` default to their canonical names.
    Optional `sample_id`, `prompt_id`, and `tool_calls` columns are used when
    present or explicitly mapped. Generated sample IDs are source:row-position.
    """
    if not table.columns.is_unique:
        raise ValueError("duplicate input column names")
    allowed = {
        "step",
        "prompt",
        "response",
        "reward",
        "sample_id",
        "prompt_id",
        "tool_calls",
    }
    mapping = dict(columns or {})
    if set(mapping) - allowed:
        raise ValueError("unsupported canonical column mapping")
    for name in ("step", "prompt", "response", "reward"):
        mapping.setdefault(name, name)
    for name in ("sample_id", "prompt_id", "tool_calls"):
        if name in table:
            mapping.setdefault(name, name)
    missing = set(mapping.values()) - set(table.columns)
    if missing:
        raise ValueError(f"missing input columns: {sorted(missing)}")
    rows = []
    for position, raw in enumerate(table.to_dict("records")):
        try:
            prompt, prompt_messages = _text(raw[mapping["prompt"]])
            response, response_messages = _text(raw[mapping["response"]])
            identity_source = (
                ["chat", prompt_messages]
                if prompt_messages is not None
                else ["text", prompt]
            )
            digest = hashlib.sha256(
                json.dumps(
                    identity_source,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            tool_calls = (
                _plain(raw[mapping["tool_calls"]]) if "tool_calls" in mapping else None
            )
            if not isinstance(tool_calls, (list, dict)) and pd.isna(tool_calls):
                tool_calls = None
            if tool_calls is None and response_messages is not None:
                tool_calls = [
                    call
                    for message in response_messages
                    for call in (message.get("tool_calls") or [])
                ] or None
            rows.append(
                {
                    "run": run,
                    "step": _plain(raw[mapping["step"]]),
                    "sample_id": _identity(raw[mapping["sample_id"]])
                    if "sample_id" in mapping
                    else f"{source}:{position}",
                    "prompt_id": _identity(raw[mapping["prompt_id"]])
                    if "prompt_id" in mapping
                    else digest,
                    "prompt": prompt,
                    "response": response,
                    "reward": _plain(raw[mapping["reward"]]),
                    "prompt_messages": prompt_messages,
                    "response_messages": response_messages,
                    "tool_calls": tool_calls,
                }
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{source}:{position + 1}: {exc}") from None
    try:
        return validate_rollouts(pd.DataFrame(rows, columns=CANONICAL_COLUMNS))
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from None


def _files(path: str | Path, pattern: re.Pattern, glob: str) -> list[Path]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    candidates = [path] if path.is_file() else list(path.glob(glob))
    files = [p for p in candidates if p.is_file() and pattern.fullmatch(p.name)]
    if not files:
        raise ValueError(f"No matching rollout files in {path}")
    return sorted(
        files, key=lambda p: (int(pattern.fullmatch(p.name).group(1)), p.name)
    )


def _check_steps(table: pd.DataFrame, file: Path, pattern: re.Pattern) -> None:
    expected = int(pattern.fullmatch(file.name).group(1))
    if "step" not in table or not table.step.eq(expected).all():
        raise ValueError(
            f"{file.name}: step column must agree with filename step {expected}"
        )


def load_trl(
    path: str | Path, *, run: str, reward_column: str | None = None
) -> pd.DataFrame:
    """Read completions_<step>.parquet, or a trainer output's completions/.

    Only `reward` is selected automatically; specify reward_column for other columns.
    """
    path = Path(path)
    if path.is_dir() and (path / "completions").is_dir():
        path = path / "completions"
    parts = []
    for file in _files(path, _TRL_FILE, "completions_*.parquet"):
        table = pd.read_parquet(file)
        selected_reward = reward_column or "reward"
        if selected_reward not in table:
            raise ValueError(
                f"{file.name}: select an available reward_column explicitly"
            )
        _check_steps(table, file, _TRL_FILE)
        parts.append(
            from_table(
                table,
                run=run,
                columns={"response": "completion", "reward": selected_reward},
                source=file.name,
            )
        )
    return (
        validate_rollouts(pd.concat(parts, ignore_index=True))
        .sort_values(["step", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )


def load_verl(path: str | Path, *, run: str) -> pd.DataFrame:
    """Read verl rollout_data_dir/<step>.jsonl with input/output/score/step.

    Also reads `step_<step>` names and gzip `.jsonl.gz` shards; steps may be
    zero-padded.
    """
    parts = []
    for file in _files(path, _VERL_FILE, "*.jsonl*"):
        rows = []
        open_text = gzip.open if file.suffix == ".gz" else open
        try:
            with open_text(file, "rt", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    try:
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            raise ValueError
                    except (ValueError, TypeError):
                        raise ValueError(
                            f"{file.name}:{line_number}: invalid JSON object"
                        ) from None
                    rows.append(row)
        except (EOFError, gzip.BadGzipFile, zlib.error):
            raise ValueError(f"{file.name}: invalid gzip data") from None
        table = pd.DataFrame(
            rows, columns=None if rows else ["input", "output", "score", "step"]
        )
        _check_steps(table, file, _VERL_FILE)
        parts.append(
            from_table(
                table,
                run=run,
                columns={"prompt": "input", "response": "output", "reward": "score"},
                source=file.name,
            )
        )
    return (
        validate_rollouts(pd.concat(parts, ignore_index=True))
        .sort_values(["step", "sample_id"], kind="stable")
        .reset_index(drop=True)
    )
