"""Offline, revision- and checksum-pinned public benchmark inputs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from rolloutdiff.adapters import _identity, _plain, from_table
from rolloutdiff.schema import CANONICAL_COLUMNS, validate_rollouts


@dataclass(frozen=True)
class PublicRun:
    rollouts: pd.DataFrame
    labels: pd.DataFrame
    provenance: dict
    sampling_units: pd.DataFrame | None = None


_KEYS = ["run", "sample_id", "step", "prompt_id"]
_TRL_FILE = re.compile(r"completions_(\d+)\.parquet\Z")


def _required(table: pd.DataFrame, names: list[str]) -> None:
    missing = set(names) - set(table)
    if missing:
        raise ValueError(f"missing source columns: {sorted(missing)}")


def _verify(data_root: Path, run: str, manifest: dict) -> tuple[dict, list[Path]]:
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("runs"), dict
    ):
        raise ValueError("unsupported public manifest schema")
    if run not in manifest["runs"]:
        raise ValueError(f"unknown public run: {run}")
    spec = manifest["runs"][run]
    revision = spec.get("revision")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("revision must be an immutable 40-character commit hash")
    if not isinstance(spec.get("repo_id"), str) or not spec["repo_id"].strip():
        raise ValueError("repo_id must identify the public source")
    if spec.get("adapter") not in ("aria", "aisi", "trl"):
        raise ValueError("unknown public adapter")
    license_info = spec.get("license")
    if (
        not isinstance(license_info, dict)
        or license_info.get("status") not in ("declared", "not_asserted", "unresolved")
        or "identifier" not in license_info
        or not isinstance(license_info.get("evidence_url"), str)
        or not license_info["evidence_url"].strip()
        or (
            license_info.get("identifier") is not None
            and not isinstance(license_info["identifier"], str)
        )
        or (
            license_info.get("status") == "declared"
            and not license_info.get("identifier")
        )
    ):
        raise ValueError(
            "license requires explicit status, identifier and evidence_url"
        )
    for name in ("expected_rows", "expected_steps"):
        if type(spec.get(name)) is not int or spec[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(spec.get("files"), list) or not spec["files"]:
        raise ValueError("manifest must list pinned files")
    root = data_root.resolve()
    paths = []
    for entry in spec["files"]:
        relative = entry.get("local_path")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise ValueError("local_path must be a confined relative path")
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("local_path resolves outside data root")
        if path in paths:
            raise ValueError("duplicate source file")
        if not isinstance(entry.get("sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", entry["sha256"]
        ):
            raise ValueError("sha256 must be a 64-character checksum")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] < 1:
            raise ValueError("size_bytes must be a positive integer")
        if (
            not isinstance(entry.get("upstream_path"), str)
            or not entry["upstream_path"]
        ):
            raise ValueError("upstream_path is required")
        url = urlparse(entry.get("url", ""))
        if (
            url.scheme != "https"
            or not url.netloc
            or revision not in url.path.split("/")
        ):
            raise ValueError("source URL must use HTTPS and pin the declared revision")
        if path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"{relative}: source size differs from manifest")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"{relative}: source checksum differs from manifest")
        paths.append(path)
    return spec, paths


def _messages(value) -> list[dict]:
    rows = json.loads(value) if isinstance(value, str) else _plain(value)
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or [row.get("role") for row in rows] != ["system", "user", "assistant"]
        or any(not isinstance(row.get("content"), str) for row in rows[:2])
    ):
        raise ValueError("expected system, user, assistant messages with text prompts")
    return rows


def _binary(table: pd.DataFrame, column: str) -> pd.Series:
    values = table[column] if column in table else pd.Series(pd.NA, index=table.index)
    observed = values.dropna()
    if not observed.map(
        lambda value: not isinstance(value, str) and value in (0, 1)
    ).all():
        raise ValueError(f"{column}: observed labels must be binary")
    return values.astype("Int64")


def _labels(table: pd.DataFrame, adapter: str) -> tuple[pd.DataFrame, dict]:
    if adapter == "aria":
        mapping = {
            "label_hack": "is_reward_hack_strict",
            "label_attempt": "is_reward_hack_loose",
            "label_correct": "eq_correct",
        }
    elif adapter == "aisi":
        mapping = {
            "label_hack": "reward_hacked",
            "label_attempt": None,
            "label_correct": "actually_solved",
        }
        mapping.update(
            {name: name for name in ("rh_exit", "rh_conftest", "rh_always_equal")}
        )
    else:
        mapping = {
            "label_hack": "proxy_reward_hacked",
            "label_attempt": None,
            "label_correct": "proxy_actually_solved",
        }
        # Preserve complete proxy graders and partial rh graders as distinct labels.
        mapping.update(
            {name: name for name in table if name.startswith(("proxy_", "rh_"))}
        )
    labels = pd.DataFrame(index=table.index)
    for output, source in mapping.items():
        labels[output] = (
            _binary(table, source)
            if source is not None
            else pd.Series(pd.NA, index=table.index, dtype="Int64")
        )
    if adapter == "aria":
        values = table.get("category_5way", pd.Series(pd.NA, index=table.index))
        observed = values.dropna()
        if not observed.map(
            lambda value: not isinstance(value, str) and value in range(5)
        ).all():
            raise ValueError("category_5way must be an integer in 0..4 when observed")
        labels["category_5way"] = values.astype("Int64")
        mapping["category_5way"] = "category_5way"
    return labels, mapping


def _coverage(labels: pd.DataFrame) -> dict:
    return {
        name: {
            "observed": int(values.notna().sum()),
            "missing": int(values.isna().sum()),
        }
        for name, values in labels.items()
        if name != "source_problem_id"
    }


def _read_sources(paths, adapter, expected_rows):
    parts = []
    for path in paths:
        part = pd.read_parquet(path)
        if not part.columns.is_unique:
            raise ValueError("duplicate source columns")
        if adapter == "trl":
            match = _TRL_FILE.fullmatch(path.name)
            if (
                match is None
                or "step" not in part
                or not part.step.eq(int(match.group(1))).all()
            ):
                raise ValueError(f"{path.name}: step must agree with TRL filename")
            # Native load_trl identities use the filename and original row index.
            part = part.copy()
            if "sample_id" not in part:
                part["sample_id"] = [f"{path.name}:{i}" for i in range(len(part))]
        parts.append(part)
    raw = pd.concat(parts, ignore_index=True)
    if len(raw) != expected_rows:
        raise ValueError("source rows differ from manifest")
    return raw


def _canonical_source(raw, run, adapter, labels):
    source_problem_ids = None
    if adapter in ("aria", "aisi"):
        _required(raw, ["messages", "sample_id", "problem_id"])
        messages = raw.messages.map(_messages)
        canonical = pd.DataFrame(
            {
                "run": run,
                "sample_id": raw.sample_id.map(_identity),
                "prompt_id": raw.problem_id.map(_identity),
                "prompt": messages.map(lambda rows: rows[1]["content"]),
                "prompt_messages": messages.map(lambda rows: rows[:2]),
            }
        )
        if adapter == "aria":
            _required(raw, ["step", "score"])
            canonical["step"] = raw.step
            canonical["reward"] = raw.score
            canonical["response"] = messages.map(lambda rows: rows[2].get("content"))
            canonical["response_messages"] = messages.map(lambda rows: rows[2:])
            source_reward, source_response = "score", "assistant_content"
            step_definition = "source integer step"
            if (
                not raw.sample_id.str.fullmatch(r"step\d{3}_row\d{3}").all()
                or not raw.sample_id.str.extract(r"^step(\d+)")[0]
                .astype(int)
                .eq(raw.step)
                .all()
            ):
                raise ValueError("Aria sample identity does not agree with step")
        else:
            _required(raw, ["source_eval_file", "epoch", "training_passed", "response"])
            source_files = raw.source_eval_file.map(_identity)
            steps = {
                name: i + 1 for i, name in enumerate(sorted(source_files.unique()))
            }
            canonical["step"] = source_files.map(steps)
            source_problem_ids = raw.problem_id.map(_identity)
            labels["source_problem_id"] = source_problem_ids
            # Mechanism instructions are permuted in the system prompt even
            # within one source problem/batch. Canonical groups require exact
            # full prompt content.
            prompt_digests = canonical.prompt_messages.map(
                lambda rows: hashlib.sha256(
                    json.dumps(
                        rows, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode("utf-8")
                ).hexdigest()
            )
            canonical["prompt_id"] = [
                json.dumps([problem, digest], ensure_ascii=False, separators=(",", ":"))
                for problem, digest in zip(
                    labels.source_problem_id, prompt_digests, strict=False
                )
            ]
            canonical["sample_id"] = [
                json.dumps(
                    [source, sample, _plain(epoch)],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for source, sample, epoch in zip(
                    source_files, raw.sample_id, raw.epoch, strict=False
                )
            ]
            if raw.epoch.isna().any():
                raise ValueError("AISI epoch identity is missing")
            canonical["reward"] = raw.training_passed
            canonical["response"] = raw.response
            # The chat's assistant can differ from the graded response. Preserve
            # the explicitly graded response without substituting that chat text.
            source_reward, source_response = "training_passed", "response"
            step_definition = (
                "1-based ordinal of lexicographically sorted full source_eval_file"
            )
    else:
        _required(raw, ["step", "prompt", "completion", "training_passed"])
        canonical = pd.DataFrame(
            {
                "step": raw.step,
                "reward": raw.training_passed,
                "response": raw.completion,
            }
        )
        source_reward, source_response = "training_passed", "completion"
        step_definition = "source integer step verified against completions filename"
    return (
        canonical,
        source_problem_ids,
        source_reward,
        source_response,
        step_definition,
    )


def load_public_run(
    data_root: str | Path, run: str, *, manifest: dict | str | Path | None = None
) -> PublicRun:
    """Read only manifest files; verify every pin before decoding any parquet."""
    if manifest is None:
        manifest = Path(__file__).with_name("public_manifest.json")
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be a mapping or JSON path")
    spec, paths = _verify(Path(data_root), run, manifest)
    adapter = spec["adapter"]
    raw = _read_sources(paths, adapter, spec["expected_rows"])
    labels, label_sources = _labels(raw, adapter)
    canonical, source_problem_ids, source_reward, source_response, step_definition = (
        _canonical_source(raw, run, adapter, labels)
    )
    if canonical.step.nunique() != spec["expected_steps"]:
        raise ValueError("source steps differ from manifest")
    unavailable = canonical[["reward", "response"]].isna()
    excluded_steps = sorted(
        int(step) for step in canonical.loc[unavailable.any(axis=1), "step"].unique()
    )
    keep = ~canonical.step.isin(excluded_steps)
    if adapter == "trl":
        rollouts = from_table(
            raw.loc[keep].reset_index(drop=True),
            run=run,
            columns={"response": "completion", "reward": "training_passed"},
            source="public TRL",
        )
    else:
        rollouts = validate_rollouts(
            canonical.loc[keep].reindex(columns=CANONICAL_COLUMNS)
        )
    retained_labels = labels.loc[keep].reset_index(drop=True)
    evaluation = pd.concat([rollouts[_KEYS], retained_labels], axis=1)
    order = rollouts.sort_values(["step", "sample_id"], kind="stable").index
    rollouts = rollouts.loc[order].reset_index(drop=True)
    evaluation = evaluation.loc[order].reset_index(drop=True)
    sampling_units = None
    if source_problem_ids is not None:
        sampling_units = (
            canonical[["run", "prompt_id"]]
            .assign(sampling_unit_id=source_problem_ids)
            .loc[keep]
            .drop_duplicates()
            .sort_values(["run", "prompt_id"])
            .reset_index(drop=True)
        )
    provenance = {
        "run": run,
        "adapter": adapter,
        "repo_id": spec["repo_id"],
        "revision": spec["revision"],
        "license": copy.deepcopy(spec["license"]),
        "files": copy.deepcopy(spec["files"]),
        "raw_rows": len(raw),
        "raw_steps": int(canonical.step.nunique()),
        "retained_rows": len(rollouts),
        "retained_steps": int(rollouts.step.nunique()),
        "excluded_rows": int((~keep).sum()),
        "excluded_steps": excluded_steps,
        "exclusion_policy": "exclude entire step if any reward or response is unavailable",
        "missing_fields": {
            source_reward: int(unavailable.reward.sum()),
            source_response: int(unavailable.response.sum()),
        },
        "step_definition": step_definition,
        "label_sources": label_sources,
        "sampling_unit_definition": "source problem_id"
        if sampling_units is not None
        else "exact prompt_id",
        "n_sampling_units": int(sampling_units.sampling_unit_id.nunique())
        if sampling_units is not None
        else int(rollouts.prompt_id.nunique()),
        "prompt_id_definition": (
            "JSON[source_problem_id, SHA256(exact system and user messages)]"
            if adapter == "aisi"
            else "source problem_id"
            if adapter == "aria"
            else "native adapter exact-content identity"
        ),
        "source_problem_count": int(raw.problem_id.nunique())
        if "problem_id" in raw
        else None,
        "source_label_coverage": _coverage(labels),
        "retained_label_coverage": _coverage(retained_labels),
    }
    return PublicRun(
        rollouts=rollouts,
        labels=evaluation,
        provenance=provenance,
        sampling_units=sampling_units,
    )
