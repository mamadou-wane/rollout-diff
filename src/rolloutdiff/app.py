"""Authenticated loopback investigation over an explicitly selected local root."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import math
import os
import re
import stat
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, asynccontextmanager, contextmanager
from itertools import islice
from numbers import Integral, Real
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import adapters
from .comparison import (
    canonical_fingerprint,
    compare_rollouts,
    resolve_comparison_citations,
)
from .comparison_report import (
    comparison_json,
    render_comparison_report,
    validate_comparison_report,
)
from .html_report import render_report
from .inspection import resolve_exemplars
from .pipeline import analyze_rollouts
from .report import _identifier, build_report, report_json

_SAFE_INTEGER = 2**53 - 1
_MAX_FILES = 10000
_MAX_LISTING = 1000
_MAX_JOBS = 32
_CSP = (
    "default-src 'none'; script-src 'self'; style-src 'self'; "
    "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; object-src 'none'"
)


def _require_local_platform():
    directory_functions = {
        function.__name__ for function in getattr(os, "supports_dir_fd", ())
    }
    descriptor_functions = {
        function.__name__ for function in getattr(os, "supports_fd", ())
    }
    if (
        os.name != "posix"
        or not all(
            hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
        )
        or not {"open", "stat"}.issubset(directory_functions)
        or "scandir" not in descriptor_functions
    ):
        raise ValueError(
            "Local investigation requires macOS or Linux with descriptor-relative "
            "and no-follow file access."
        )


def ui_value(value):
    """Preserve missing data and integers that JavaScript cannot represent exactly."""
    if isinstance(value, dict):
        return {key: ui_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [ui_value(item) for item in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        number = int(value)
        return str(number) if abs(number) > _SAFE_INTEGER else number
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        return value
    raise TypeError("unsupported view value")


def _view(value, *, status_code=200):
    return JSONResponse(ui_value(value), status_code=status_code)


def _rooted(root: Path, value: str) -> Path:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or "\\" in value
        or any(part == ".." or part.startswith(".") for part in path.parts)
    ):
        raise HTTPException(400, "Select a visible path within the selected root.")
    try:
        selected = (root / path).resolve(strict=True)
        relative = selected.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            raise ValueError
    except (OSError, RuntimeError, ValueError):
        raise HTTPException(
            400, "Path is unavailable within the selected root."
        ) from None
    return selected


@contextmanager
def _open_source(root: Path, value: str):
    """Open without following links, including links introduced during traversal."""
    relative = Path(value)
    if (
        not value
        or relative.is_absolute()
        or "\\" in value
        or any(part == ".." or part.startswith(".") for part in relative.parts)
    ):
        raise HTTPException(400, "Select a visible path within the selected root.")
    descriptors = []
    try:
        directory = os.open(root.anchor, os.O_RDONLY | os.O_DIRECTORY)
        descriptors.append(directory)
        parts = (*root.parts[1:], *relative.parts)
        for part in parts[:-1]:
            directory = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            descriptors.append(directory)
        descriptor = (
            os.open(
                parts[-1],
                os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                dir_fd=directory,
            )
            if parts
            else os.dup(directory)
        )
        descriptors.append(descriptor)
        mode = os.fstat(descriptor).st_mode
        if not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
            raise HTTPException(400, "Select a regular file or directory.")
        yield descriptor
    except (OSError, ValueError):
        raise HTTPException(
            400, "Source is unavailable or contains a symbolic link."
        ) from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _files(root: Path, path: str, format: str) -> list[Path]:
    relative = Path(path)
    pattern = adapters._TRL_FILE if format == "trl" else adapters._VERL_FILE
    with ExitStack() as opened:
        descriptor = opened.enter_context(_open_source(root, path))
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            names = [relative.name] if pattern.fullmatch(relative.name) else []
            parent = relative.parent
        else:
            if format == "trl":
                try:
                    child = os.stat(
                        "completions", dir_fd=descriptor, follow_symlinks=False
                    )
                except FileNotFoundError:
                    child = None
                if child is not None and stat.S_ISLNK(child.st_mode):
                    raise HTTPException(400, "Source contains a symbolic link.")
                if child is not None and stat.S_ISDIR(child.st_mode):
                    relative = relative / "completions"
                    descriptor = opened.enter_context(
                        _open_source(root, relative.as_posix())
                    )
            with os.scandir(descriptor) as entries:
                names = [entry.name for entry in islice(entries, _MAX_FILES + 1)]
            if len(names) > _MAX_FILES:
                raise HTTPException(400, "Too many files in this selection.")
            names = [name for name in names if pattern.fullmatch(name)]
            for name in names:
                mode = os.stat(name, dir_fd=descriptor, follow_symlinks=False).st_mode
                if not stat.S_ISREG(mode):
                    raise HTTPException(
                        400, "Source must contain regular files without symbolic links."
                    )
            parent = relative
    if not names:
        raise HTTPException(400, "No matching rollout files are available.")
    names.sort(key=lambda name: (int(pattern.fullmatch(name).group(1)), name))
    return [root / parent / name for name in names]


def _hash_file(root: Path, path: Path, *, destination=None) -> dict:
    """Hash the exact bytes copied, refusing nonregular and concurrently edited files."""
    relative = path.relative_to(root).as_posix()
    digest, size = hashlib.sha256(), 0
    with _open_source(root, relative) as descriptor:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise HTTPException(400, "Selected source must be a regular file.")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            if destination is not None:
                destination.write(chunk)
        after = os.fstat(descriptor)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or size != after.st_size:
            raise HTTPException(409, "Source changed while it was being read.")
    return {"path": relative, "sha256": digest.hexdigest(), "size_bytes": size}


def _snapshot(root: Path, path: str, format: str) -> list[dict]:
    return [_hash_file(root, file) for file in _files(root, path, format)]


@contextmanager
def _captured_source(root: Path, path: str, format: str):
    """Adapters read private copies, never mutable paths in the selected root."""
    originals = _files(root, path, format)
    with TemporaryDirectory(prefix="rollout-investigation-") as temporary:
        directory = Path(temporary)
        files = []
        for original in originals:
            descriptor = os.open(
                directory / original.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
            with os.fdopen(descriptor, "wb") as destination:
                files.append(_hash_file(root, original, destination=destination))
        if files != _snapshot(root, path, format):
            raise HTTPException(409, "Source changed while it was being copied.")
        yield directory, files


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RunInput(_Input):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    format: Literal["trl", "verl"]
    run: Annotated[str, Field(min_length=1, max_length=512)]
    reward_column: Annotated[str, Field(min_length=1, max_length=512)] | None = None

    @field_validator("run", "reward_column")
    @classmethod
    def identifier(cls, value):
        return _identifier(value) if value is not None else None


class InspectInput(_Input):
    event_rank: Annotated[int, Field(ge=1)]
    feature: Annotated[str, Field(min_length=1, max_length=4096)]
    sample_id: Annotated[str, Field(min_length=1, max_length=4096)]
    with_absent: bool = False


class ComparisonInput(_Input):
    run_a: str
    run_b: str | None = None
    a: Annotated[list[str], Field(min_length=2, max_length=2)]
    b: Annotated[list[str], Field(min_length=2, max_length=2)]

    @field_validator("a", "b")
    @classmethod
    def window(cls, value):
        if any(not re.fullmatch(r"[0-9]{1,20}", item) for item in value):
            raise ValueError("ranges require nonnegative decimal integers")
        if int(value[0]) > int(value[1]):
            raise ValueError("range endpoints must be ordered")
        return value


class ComparisonInspectInput(_Input):
    side: Literal["a", "b"]
    feature: Annotated[str, Field(min_length=1, max_length=4096)]
    sample_id: Annotated[str, Field(min_length=1, max_length=4096)]


class EvidenceInput(_Input):
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    run_id: str | None = None


class _Store:
    def __init__(self, root):
        self.root = root
        self.runs = {}
        self.comparisons = {}
        self.lock = threading.RLock()
        self.executor = None

    def run(self, identity, *, ready=True):
        with self.lock:
            item = self.runs.get(identity)
            if item is None:
                raise HTTPException(404, "Run was not found.")
            if ready and item["view"]["status"] != "ready":
                raise HTTPException(409, "Run analysis is not ready.")
            return item

    def comparison(self, identity, *, ready=True):
        with self.lock:
            item = self.comparisons.get(identity)
            if item is None:
                raise HTTPException(404, "Comparison was not found.")
            if ready and item["view"]["status"] != "ready":
                raise HTTPException(409, "Comparison is not ready.")
            return item

    def verify(self, item):
        try:
            source = item["view"]["source"]
            if (
                _snapshot(self.root, source["path"], source["format"])
                != source["files"]
            ):
                raise ValueError
            if canonical_fingerprint(item["rollouts"]) != source["canonical_sha256"]:
                raise ValueError
            for row in item["rollouts"][["sample_id", "response"]].itertuples(
                index=False
            ):
                if (
                    hashlib.sha256(row.response.encode()).hexdigest()
                    != item["responses"][row.sample_id]
                ):
                    raise ValueError
        except (HTTPException, OSError, ValueError, KeyError):
            raise HTTPException(
                409, "Source changed or is unavailable. Reload the run."
            ) from None

    def verify_comparison(self, item):
        for identity in dict.fromkeys(item["run_ids"]):
            self.verify(self.run(identity))

    def load(self, identity, request):
        item = self.runs[identity]
        started = time.monotonic()
        try:
            with self.lock:
                item["view"]["status"] = "loading"
            with _captured_source(self.root, request.path, request.format) as (
                path,
                before,
            ):
                rollouts = (
                    adapters.load_trl(
                        path, run=request.run, reward_column=request.reward_column
                    )
                    if request.format == "trl"
                    else adapters.load_verl(path, run=request.run)
                )
            if before != _snapshot(self.root, request.path, request.format):
                raise ValueError("source changed")
            source = {
                "path": request.path,
                "format": request.format,
                "canonical_sha256": canonical_fingerprint(rollouts),
                "files": before,
            }
            responses = {
                row.sample_id: hashlib.sha256(row.response.encode()).hexdigest()
                for row in rollouts[["sample_id", "response"]].itertuples(index=False)
            }
            with self.lock:
                item["view"]["status"] = "analyzing"
            analysis = analyze_rollouts(rollouts)
            report = build_report(analysis, run=request.run)
            with self.lock:
                item.update(
                    rollouts=rollouts,
                    analysis=analysis,
                    report=report,
                    responses=responses,
                )
                item["view"].update(source=source)
            self.verify(item)
            with self.lock:
                item["view"].update(status="ready", counts=report["runs"][0]["counts"])
        except Exception:  # noqa: BLE001 -- terminate jobs without exposing source text
            with self.lock:
                item["view"].update(
                    status="failed",
                    error="Could not analyze this source. Check its format, reward column, and unchanged files.",
                )
        finally:
            with self.lock:
                item["view"]["elapsed_seconds"] = time.monotonic() - started

    def compare(self, identity, request):
        item = self.comparisons[identity]
        try:
            with self.lock:
                item["view"]["status"] = "comparing"
            left, right = (self.run(run_id) for run_id in item["run_ids"])
            self.verify_comparison(item)
            same = request.run_b is None or request.run_a == request.run_b
            report = compare_rollouts(
                left["rollouts"],
                None if same else right["rollouts"],
                a=tuple(map(int, request.a)),
                b=tuple(map(int, request.b)),
            )
            for side, run in zip(report["sides"], (left, right), strict=True):
                source = run["view"]["source"]
                side["source"].update(
                    format=source["format"],
                    files=[
                        {key: file[key] for key in ("sha256", "size_bytes")}
                        for file in source["files"]
                    ],
                    reward_column=run["request"].reward_column,
                )
            report = validate_comparison_report(report)
            self.verify_comparison(item)
            with self.lock:
                item["view"].update(status="ready", report=report)
        except Exception:  # noqa: BLE001 -- terminate jobs without exposing source text
            with self.lock:
                item["view"].update(
                    status="failed",
                    error="Could not compare these ranges. Check coverage, range order, and unchanged sources.",
                )


def _rows(frame, features):
    if "feature" not in frame:
        return []
    return frame.loc[frame.feature.isin(features)].to_dict("records")


def _headlines(analysis, features):
    """Equal-weight prompt means whose difference is `prompt_shared_change`.

    Features without an exact prompt in both windows are omitted.
    """
    values = analysis.prompt_values
    if "change" not in values:
        return {}
    shared = values.loc[values.feature.isin(features) & values.change.notna()]
    return {
        feature: {
            "before": rows.before_mean.mean(),
            "after": rows.after_mean.mean(),
            "n_prompts": len(rows),
        }
        for feature, rows in shared.groupby("feature", sort=False)
    }


def _citation(item, citation):
    response = citation["response"]
    digest = hashlib.sha256(response.encode()).hexdigest()
    if digest != item["responses"].get(citation["sample_id"]):
        raise HTTPException(409, "Citation response changed. Reload the run.")
    counterpart = (citation.get("comparison") or {}).get("counterpart")
    if counterpart is not None:
        counterpart_digest = hashlib.sha256(
            counterpart["response"].encode()
        ).hexdigest()
        if counterpart_digest != item["responses"].get(counterpart["sample_id"]):
            raise HTTPException(409, "Comparison response changed. Reload the run.")
    return _view(
        {
            "citation": citation,
            "source": item["view"]["source"],
            "response_sha256": digest,
        }
    )


def _download(report, format, *, comparison=False):
    if format == "json":
        content = comparison_json(report) if comparison else report_json(report)
        media_type = "application/json"
    else:
        content = (
            render_comparison_report(report) if comparison else render_report(report)
        )
        media_type = "text/html"
    return Response(
        content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="rollout-{"comparison" if comparison else "analysis"}.{format}"'
        },
    )


def create_app(root, *, token: str, origin="http://127.0.0.1:8765") -> FastAPI:
    """Create a read-only local investigation session with serialized analysis jobs."""
    _require_local_platform()
    root = Path(root).resolve(strict=True)
    if not root.is_dir() or not isinstance(token, str) or not token:
        raise ValueError("a directory root and nonempty session token are required")
    parsed = urlsplit(origin)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.path
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        raise ValueError("origin must be a loopback HTTP origin")
    store = _Store(root)

    @asynccontextmanager
    async def lifespan(app):
        store.executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="rollout-investigation"
        )
        try:
            yield
        finally:
            store.executor.shutdown(wait=True, cancel_futures=True)
            store.executor = None

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store

    @app.middleware("http")
    async def local_session(request: Request, call_next):
        try:
            peer = request.client.host if request.client else ""
            local = ipaddress.ip_address(peer).is_loopback
        except ValueError:
            local = False
        hosts = request.headers.getlist("host")
        origins = request.headers.getlist("origin")
        accepted = (
            local and hosts == [parsed.netloc] and (not origins or origins == [origin])
        )
        if request.headers.get("sec-fetch-site") == "cross-site":
            accepted = False
        if request.url.path.startswith("/api/"):
            authorization = request.headers.getlist("authorization")
            accepted = (
                accepted
                and len(authorization) == 1
                and hmac.compare_digest(
                    authorization[0].encode(), f"Bearer {token}".encode()
                )
            )
        if accepted:
            try:
                response = await call_next(request)
            except Exception:  # noqa: BLE001 -- API errors must not expose local data
                response = JSONResponse(
                    {"detail": "Local request could not be completed."}, status_code=500
                )
        else:
            response = JSONResponse(
                {"detail": "Local session authorization required."}, status_code=403
            )
        response.headers.update(
            {
                "Content-Security-Policy": _CSP,
                "X-Frame-Options": "DENY",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
            }
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        return JSONResponse({"detail": "Invalid request fields."}, status_code=422)

    @app.get("/api/session")
    def session():
        return {
            "root_label": root.name,
            "formats": ["trl", "verl"],
            "read_only_execution": True,
        }

    @app.get("/api/files")
    def files(path: Annotated[str, Query(max_length=4096)] = "."):
        selected = _rooted(root, path)
        if not selected.is_dir():
            raise HTTPException(400, "Select a directory within the root.")
        entries = []
        try:
            children = list(islice(selected.iterdir(), _MAX_FILES + 1))
        except OSError:
            raise HTTPException(400, "Directory is unavailable.") from None
        if len(children) > _MAX_FILES:
            raise HTTPException(
                400,
                "Directory has too many entries; select a smaller source directory.",
            )
        for child in sorted(children, key=lambda item: item.name):
            if child.name.startswith("."):
                continue
            relative = child.relative_to(root).as_posix()
            try:
                resolved = _rooted(root, relative)
            except HTTPException:
                continue
            if resolved.is_dir():
                kind = "directory"
            elif adapters._TRL_FILE.fullmatch(child.name):
                kind = "trl"
            elif adapters._VERL_FILE.fullmatch(child.name):
                kind = "verl"
            elif child.suffix == ".json" and resolved.is_file():
                kind = "evidence"
            else:
                continue
            entries.append({"name": child.name, "path": relative, "kind": kind})
            if len(entries) == _MAX_LISTING:
                break
        return {"path": selected.relative_to(root).as_posix(), "entries": entries}

    @app.get("/api/runs")
    def runs():
        with store.lock:
            return _view({"runs": [item["view"] for item in store.runs.values()]})

    @app.post("/api/runs", status_code=202)
    def add_run(request: RunInput):
        _files(root, request.path, request.format)
        if request.format == "verl" and request.reward_column is not None:
            raise HTTPException(
                422, "Reward column selection is only available for TRL."
            )
        with store.lock:
            if len(store.runs) >= _MAX_JOBS:
                raise HTTPException(
                    409, "Remove a finished run before loading another."
                )
            identity = uuid4().hex
            view = {
                "id": identity,
                "run": request.run,
                "status": "queued",
                "error": None,
                "counts": None,
                "source": None,
                "elapsed_seconds": None,
            }
            store.runs[identity] = {"view": view, "request": request}
            if store.executor is None:
                raise HTTPException(503, "Investigation session is not running.")
            store.executor.submit(store.load, identity, request)
            return _view(view, status_code=202)

    @app.get("/api/runs/{identity}")
    def run(identity: str):
        with store.lock:
            return _view(store.run(identity, ready=False)["view"])

    @app.delete("/api/runs/{identity}", status_code=204)
    def delete_run(identity: str):
        with store.lock:
            item = store.run(identity, ready=False)
            dependent = [
                key
                for key, value in store.comparisons.items()
                if identity in value["run_ids"]
            ]
            if item["view"]["status"] not in {"ready", "failed"} or any(
                store.comparisons[key]["view"]["status"] not in {"ready", "failed"}
                for key in dependent
            ):
                raise HTTPException(
                    409, "Wait for active jobs before removing this run."
                )
            for key in dependent:
                del store.comparisons[key]
            del store.runs[identity]
        return Response(status_code=204)

    @app.get("/api/runs/{identity}/events")
    def events(
        identity: str,
        q: Annotated[str, Query(max_length=4096)] = "",
        kind: Literal["all", "behavioral", "contextual", "unavailable"] = "all",
        tier: Literal["all", "structural", "prose"] = "all",
        floor: Literal["all", "yes", "no"] = "all",
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ):
        item = store.run(identity)
        selected = []
        for event in item["report"]["runs"][0]["events"]:
            if kind != "all" and event["ranking_status"] != kind:
                continue
            if q and not any(
                q.casefold() in member["feature"].casefold()
                for member in event["members"]
            ):
                continue
            if tier != "all" and not any(
                member["evidence"]["tier"] == tier for member in event["members"]
            ):
                continue
            if floor != "all" and bool(event["ranking"]["numerical_floor_members"]) != (
                floor == "yes"
            ):
                continue
            selected.append(event)
        page = selected[offset : offset + limit]
        return _view(
            {
                "total": len(selected),
                "events": page,
                "headlines": _headlines(
                    item["analysis"],
                    [event["representative_feature"] for event in page],
                ),
            }
        )

    @app.get("/api/runs/{identity}/events/{rank}")
    def event(identity: str, rank: int):
        item = store.run(identity)
        selected = next(
            (row for row in item["report"]["runs"][0]["events"] if row["rank"] == rank),
            None,
        )
        if selected is None:
            raise HTTPException(404, "Event was not found.")
        names = [member["feature"] for member in selected["members"]]
        return _view(
            {
                "event": selected,
                "evidence": _rows(item["analysis"].evidence, names),
                "prompt_shifts": _rows(item["analysis"].prompt_shifts, names),
                "headlines": _headlines(item["analysis"], names),
            }
        )

    @app.get("/api/runs/{identity}/steps")
    def steps(identity: str):
        rollouts = store.run(identity)["rollouts"]
        return _view({"steps": sorted(rollouts.step.unique().tolist())})

    @app.get("/api/runs/{identity}/features")
    def features(
        identity: str,
        q: Annotated[str, Query(max_length=4096)] = "",
        limit: Annotated[int, Query(ge=1, le=5000)] = 100,
    ):
        analysis = store.run(identity)["analysis"]
        catalog = analysis.evidence[["feature", "tier", "kind"]]
        catalog = catalog.loc[
            catalog.feature.str.contains(q, case=False, regex=False)
        ].sort_values("feature")
        return _view(
            {"total": len(catalog), "features": catalog.head(limit).to_dict("records")}
        )

    @app.get("/api/runs/{identity}/series")
    def series(
        identity: str, feature: Annotated[str, Query(min_length=1, max_length=4096)]
    ):
        item = store.run(identity)
        analysis = item["analysis"]
        evidence = _rows(analysis.evidence, [feature])
        if not evidence:
            raise HTTPException(404, "Feature was not found.")
        return _view(
            {
                "feature": feature,
                "kind": evidence[0]["kind"],
                "tier": evidence[0]["tier"],
                "direction": next(
                    (
                        member["direction"]
                        for event in item["report"]["runs"][0]["events"]
                        for member in event["members"]
                        if member["feature"] == feature
                    ),
                    None,
                ),
                "points": _rows(analysis.aggregation.series, [feature]),
                "evidence": evidence[0],
                "persistence": _rows(analysis.persistence_steps, [feature]),
                "prompt_values": _rows(analysis.prompt_values, [feature]),
                "prompt_shifts": _rows(analysis.prompt_shifts, [feature]),
                "headline": _headlines(analysis, [feature]).get(feature),
            }
        )

    @app.post("/api/runs/{identity}/inspect")
    def inspect(identity: str, request: InspectInput):
        item = store.run(identity)
        store.verify(item)
        try:
            event = resolve_exemplars(
                item["report"],
                item["rollouts"],
                run=item["view"]["run"],
                event_rank=request.event_rank,
                with_absent=request.with_absent,
            )
            citation = next(
                example
                for member in event["members"]
                if member["feature"] == request.feature
                for example in member["exemplars"]
                if example["sample_id"] == request.sample_id
                and example["status"] == "selected"
            )
        except (ValueError, StopIteration):
            raise HTTPException(
                400, "Requested citation is unavailable or does not match the source."
            ) from None
        store.verify(item)
        return _citation(item, citation)

    @app.get("/api/runs/{identity}/export")
    def export(identity: str, format: Literal["json", "html"] = "json"):
        item = store.run(identity)
        store.verify(item)
        return _download(item["report"], format)

    @app.post("/api/comparisons", status_code=202)
    def add_comparison(request: ComparisonInput):
        identities = [request.run_a, request.run_b or request.run_a]
        for identity in dict.fromkeys(identities):
            store.verify(store.run(identity))
        with store.lock:
            if len(store.comparisons) >= _MAX_JOBS:
                raise HTTPException(409, "Remove finished runs to free comparisons.")
            identity = uuid4().hex
            view = {"id": identity, "status": "queued", "error": None, "report": None}
            store.comparisons[identity] = {"view": view, "run_ids": identities}
            if store.executor is None:
                raise HTTPException(503, "Investigation session is not running.")
            store.executor.submit(store.compare, identity, request)
            return _view(view, status_code=202)

    @app.get("/api/comparisons/{identity}")
    def comparison(identity: str):
        with store.lock:
            return _view(store.comparison(identity, ready=False)["view"])

    @app.post("/api/comparisons/{identity}/inspect")
    def inspect_comparison(identity: str, request: ComparisonInspectInput):
        item = store.comparison(identity)
        store.verify_comparison(item)
        run = store.run(item["run_ids"][0 if request.side == "a" else 1])
        try:
            resolved = resolve_comparison_citations(
                item["view"]["report"],
                run["rollouts"],
                side=request.side,
                feature=request.feature,
            )
            citation = next(
                row
                for row in resolved["items"]
                if row["sample_id"] == request.sample_id
            )
        except (ValueError, StopIteration):
            raise HTTPException(
                400,
                "Requested comparison citation is unavailable or does not match the source.",
            ) from None
        store.verify_comparison(item)
        return _citation(
            run, {**citation, "side": resolved["side"], "feature": resolved["feature"]}
        )

    @app.get("/api/comparisons/{identity}/export")
    def export_comparison(identity: str, format: Literal["json", "html"] = "json"):
        item = store.comparison(identity)
        store.verify_comparison(item)
        return _download(item["view"]["report"], format, comparison=True)

    @app.post("/api/evidence")
    def evidence(request: EvidenceInput):
        from .app_evidence import load_evidence

        path = _rooted(root, request.path)
        run = store.run(request.run_id) if request.run_id else None
        if run is not None:
            store.verify(run)
        try:
            result = load_evidence(
                path,
                root=root,
                rollouts=run["rollouts"] if run else None,
                source=run["view"]["source"] if run else None,
            )
        except (OSError, ValueError):
            raise HTTPException(
                400, "Saved evidence is unavailable, unsupported, or invalid."
            ) from None
        if run is not None:
            store.verify(run)
        return _view(result)

    web = Path(__file__).parent / "_web"

    @app.get("/")
    def index():
        path = web / "index.html"
        if not path.is_file():
            raise HTTPException(503, "Packaged investigation assets are unavailable.")
        return FileResponse(path)

    @app.get("/assets/{asset_path:path}")
    def asset(asset_path: str):
        path = _rooted(web, "assets/" + asset_path)
        if not path.is_file() or path.suffix not in {".js", ".css", ".svg", ".woff2"}:
            raise HTTPException(404, "Asset was not found.")
        return FileResponse(path)

    return app
