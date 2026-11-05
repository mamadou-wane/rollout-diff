"""Validated comparison JSON and self-contained HTML reports."""

from __future__ import annotations

import hashlib
import json
import math
from base64 import b64encode
from html import escape
from pathlib import Path
from statistics import NormalDist
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, ValidationError

from ._report_validation import _constant, _object
from .features import DENSE_BINARY, _tier, extract
from .html_report import _STYLE
from .report import _identifier

Identifier = Annotated[str, AfterValidator(_identifier)]
Count = Annotated[int, Field(ge=0)]
PositiveCount = Annotated[int, Field(ge=1)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Nonnegative = Annotated[float, Field(ge=0)]
_Z = NormalDist().inv_cdf(0.975)
_CONTINUOUS = (extract("").dense.keys() - DENSE_BINARY) | {"scalar:reward"}
_SPARSE_PREFIXES = (
    "call:",
    "def:",
    "class:",
    "import:",
    "setattr:",
    "strlit:",
    "file:",
    "w:",
    "b:",
)


class _Model(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class _File(_Model):
    sha256: Digest
    size_bytes: Count


class _Source(_Model):
    canonical_sha256: Digest
    format: Literal["canonical", "trl", "verl"]
    files: list[_File]
    reward_column: Identifier | None


class _Side(_Model):
    side: Literal["a", "b"]
    run: Identifier
    window: Annotated[list[Count], Field(min_length=2, max_length=2)]
    observed_steps: Annotated[list[Count], Field(min_length=1)]
    n_rollouts: PositiveCount
    n_prompts: PositiveCount
    n_sampling_units: PositiveCount
    source: _Source


class _Shift(_Model):
    descriptor: Identifier
    statistic: Identifier
    before: float | None
    after: float | None
    delta: float | None
    distance: Annotated[float, Field(ge=0, le=1)] | None
    n_before: Count
    n_after: Count
    missing_before: Count
    missing_after: Count


class _PromptMix(_Model):
    status: Literal["shared_prompts", "no_shared_prompts"]
    shared_prompts: Count
    only_a_prompts: Count
    only_b_prompts: Count
    prompt_shift: list[_Shift]


class _Interval(_Model):
    se: Nonnegative | None
    ci_low: float | None
    ci_high: float | None


class _Estimate(_Interval):
    mean: float | None
    n_rollouts: PositiveCount
    n_prompts: PositiveCount
    n_sampling_units: PositiveCount
    positive_prompts: Count | None
    status: Literal["available", "insufficient_units", "numeric_unavailable"]


class _Contrast(_Interval):
    estimate: float | None
    status: Literal[
        "available",
        "insufficient_units",
        "partial_overlap",
        "shared_units",
        "incompatible_units",
        "numeric_unavailable",
    ]
    method: Literal["independent", "paired", "unavailable"]


class _Paired(_Interval):
    estimate: float | None
    n_prompts: Count
    n_sampling_units: Count | None
    status: Literal[
        "available",
        "insufficient_units",
        "no_shared_prompts",
        "incompatible_units",
        "numeric_unavailable",
    ]


class _Citation(_Model):
    run: Identifier
    sample_id: Identifier
    step: Count
    prompt_id: Identifier
    feature_value: float
    reward: float
    response_sha256: Digest
    selection_reason: Literal[
        "earliest_positive", "prompt_diversity", "additional_positive"
    ]


class _Citations(_Model):
    status: Literal["selected", "no_positive_rollouts", "unsupported_kind"]
    items: Annotated[list[_Citation], Field(max_length=3)]


class _CitationSides(_Model):
    a: _Citations
    b: _Citations


class _Feature(_Model):
    order: PositiveCount
    feature: Identifier
    tier: Literal["structural", "prose"]
    kind: Literal["binary", "continuous"]
    direction: Literal["increase", "decrease", "unchanged", "unavailable"]
    a: _Estimate
    b: _Estimate
    difference: _Contrast
    paired: _Paired
    citations: _CitationSides


class _Report(_Model):
    report_type: Literal["comparison"]
    schema_version: Literal[1]
    mode: Literal["windows", "dumps"]
    sides: Annotated[list[_Side], Field(min_length=2, max_length=2)]
    prompt_mix: _PromptMix
    features: list[_Feature]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid comparison report: {message}")


def _same(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-12)


def _check_interval(value: _Interval, center: float | None, status: str) -> None:
    bounds = value.ci_low, value.ci_high
    if status == "available":
        _require(
            center is not None
            and value.se is not None
            and all(v is not None for v in bounds),
            "available interval requires finite measurements",
        )
        _require(
            _same(value.ci_low, center - _Z * value.se)
            and _same(value.ci_high, center + _Z * value.se),
            "interval does not match its estimate and standard error",
        )
    else:
        _require(bounds == (None, None), "unavailable interval requires null bounds")
        if status != "numeric_unavailable" or center is None:
            _require(
                value.se is None, "unavailable uncertainty requires null standard error"
            )
        if status not in {"numeric_unavailable", "no_shared_prompts"}:
            _require(center is not None, "observed estimate is missing")


def _check_side(side: _Side) -> None:
    _require(side.window[0] <= side.window[1], "window bounds are reversed")
    _require(
        side.observed_steps == sorted(set(side.observed_steps))
        and all(
            side.window[0] <= step <= side.window[1] for step in side.observed_steps
        ),
        "observed steps must be ordered, unique and within the window",
    )
    _require(
        side.n_sampling_units <= side.n_prompts <= side.n_rollouts
        and len(side.observed_steps) <= side.n_rollouts,
        "side counts are inconsistent",
    )


def _check_estimate(value: _Estimate, side: _Side, kind: str) -> None:
    _require(
        all(
            getattr(value, key) == getattr(side, key)
            for key in ("n_rollouts", "n_prompts", "n_sampling_units")
        ),
        "feature counts differ from side counts",
    )
    _check_interval(value, value.mean, value.status)
    if value.status == "available":
        _require(
            value.n_sampling_units >= 2,
            "available uncertainty requires at least two units",
        )
    if value.status == "insufficient_units":
        _require(
            value.n_sampling_units < 2, "insufficient uncertainty has enough units"
        )
    if kind == "continuous":
        _require(
            value.positive_prompts is None,
            "continuous features have no positive-prompt count",
        )
    else:
        _require(
            value.positive_prompts is not None
            and value.positive_prompts <= value.n_prompts,
            "binary positive-prompt count is invalid",
        )
        _require(
            value.mean is not None and 0 <= value.mean <= 1,
            "binary mean must be a prevalence",
        )
        _require(
            (value.mean == 0) == (value.positive_prompts == 0)
            and value.mean <= value.positive_prompts / value.n_prompts + 1e-12,
            "binary mean and positive-prompt count disagree",
        )


def _check_citations(value: _Citations, feature: _Feature, side: _Side) -> None:
    estimate = getattr(feature, side.side)
    if feature.kind == "continuous":
        _require(
            value.status == "unsupported_kind" and not value.items,
            "continuous citations must be unsupported",
        )
        return
    if not estimate.positive_prompts:
        _require(
            value.status == "no_positive_rollouts" and not value.items,
            "absent binary feature must have no citations",
        )
        return
    _require(
        value.status == "selected" and bool(value.items),
        "positive binary feature requires citations",
    )
    _require(len(value.items) <= side.n_rollouts, "too many citations for side")
    seen_samples, seen_prompts = set(), set()
    remaining = False
    first = value.items[0]
    for index, item in enumerate(value.items):
        _require(
            item.run == side.run
            and item.step in side.observed_steps
            and item.feature_value == 1
            and item.sample_id not in seen_samples,
            "citation identity or feature value is inconsistent",
        )
        _require(
            (item.step, item.sample_id) >= (first.step, first.sample_id),
            "first citation must be earliest",
        )
        if index == 0:
            _require(
                item.selection_reason == "earliest_positive",
                "first citation requires earliest-positive reason",
            )
        elif item.selection_reason == "prompt_diversity":
            _require(
                not remaining and item.prompt_id not in seen_prompts,
                "diversity citation repeats a prompt or follows additional citations",
            )
        else:
            _require(
                item.selection_reason == "additional_positive",
                "later citation has invalid selection reason",
            )
            remaining = True
        seen_samples.add(item.sample_id)
        seen_prompts.add(item.prompt_id)
    diversity = sum(item.selection_reason == "prompt_diversity" for item in value.items)
    _require(
        diversity < estimate.positive_prompts,
        "diversity citations exceed positive prompt count",
    )
    _require(
        len(value.items) >= min(3, estimate.positive_prompts),
        "citations omit known positive prompts",
    )


def _check_feature(feature: _Feature, report: _Report) -> None:
    name = feature.feature
    known = (
        name in DENSE_BINARY
        or name in _CONTINUOUS
        or any(
            name.startswith(prefix) and len(name) > len(prefix)
            for prefix in _SPARSE_PREFIXES
        )
    )
    _require(
        known and feature.tier == _tier(name), "feature identity or tier is invalid"
    )
    _require(
        feature.kind == ("continuous" if name in _CONTINUOUS else "binary"),
        "feature kind is inconsistent",
    )
    for side in report.sides:
        _check_estimate(getattr(feature, side.side), side, feature.kind)
        _check_citations(getattr(feature.citations, side.side), feature, side)
    difference, paired = feature.difference, feature.paired
    expected = (
        None
        if feature.a.mean is None or feature.b.mean is None
        else feature.b.mean - feature.a.mean
    )
    if expected is not None and not math.isfinite(expected):
        expected = None
    _require(_same(difference.estimate, expected), "difference must equal B minus A")
    direction = (
        "unavailable"
        if expected is None
        else "increase"
        if expected > 0
        else "decrease"
        if expected < 0
        else "unchanged"
    )
    _require(feature.direction == direction, "direction differs from observed effect")
    _check_interval(difference, difference.estimate, difference.status)
    _check_interval(paired, paired.estimate, paired.status)
    if feature.kind == "binary":
        _require(
            paired.estimate is None or -1 <= paired.estimate <= 1,
            "paired binary effect must be a prevalence difference",
        )
    mix = report.prompt_mix
    _require(
        paired.n_prompts == mix.shared_prompts,
        "paired count differs from shared-prompt count",
    )
    if not mix.shared_prompts:
        _require(
            paired.status == "no_shared_prompts"
            and paired.n_sampling_units == 0
            and paired.estimate is None,
            "unmatched prompts require unavailable paired effect",
        )
    elif paired.status == "incompatible_units" or (
        paired.status == "numeric_unavailable" and paired.n_sampling_units is None
    ):
        _require(
            paired.n_sampling_units is None,
            "incompatible paired units must have unavailable count",
        )
    else:
        _require(
            paired.status != "no_shared_prompts"
            and paired.n_sampling_units is not None
            and 1
            <= paired.n_sampling_units
            <= min(
                paired.n_prompts, feature.a.n_sampling_units, feature.b.n_sampling_units
            ),
            "paired unit counts are invalid",
        )
        if paired.status == "available":
            _require(
                paired.n_sampling_units >= 2,
                "available paired uncertainty requires two units",
            )
        if paired.status == "insufficient_units":
            _require(
                paired.n_sampling_units < 2,
                "insufficient paired uncertainty has enough units",
            )
    full_overlap = (
        mix.shared_prompts > 0 and mix.only_a_prompts == mix.only_b_prompts == 0
    )
    if full_overlap:
        if not _same(paired.estimate, difference.estimate):
            _require(
                difference.status == "numeric_unavailable"
                and difference.method == "unavailable"
                and all(
                    getattr(difference, key) is None
                    for key in ("se", "ci_low", "ci_high")
                ),
                "inconsistent numeric effects require unavailable marginal uncertainty",
            )
        elif paired.status != "incompatible_units":
            method = (
                "unavailable" if paired.status == "numeric_unavailable" else "paired"
            )
            _require(
                difference.method == method
                and difference.status == paired.status
                and all(
                    _same(getattr(difference, key), getattr(paired, key))
                    for key in ("se", "ci_low", "ci_high")
                ),
                "fully paired uncertainty must match paired measurements",
            )
        else:
            _require(
                difference.method == "unavailable"
                and difference.status == "incompatible_units",
                "incompatible paired uncertainty must remain unavailable",
            )
    elif mix.shared_prompts:
        _require(
            difference.method == "unavailable"
            and difference.status in {"partial_overlap", "numeric_unavailable"},
            "partial overlap requires unavailable marginal uncertainty",
        )
    elif difference.method == "independent":
        _require(
            difference.status
            in {"available", "insufficient_units", "numeric_unavailable"},
            "independent contrast has invalid status",
        )
        if difference.status == "available":
            _require(
                feature.a.status == feature.b.status == "available"
                and _same(difference.se, math.hypot(feature.a.se, feature.b.se)),
                "independent uncertainty differs from side uncertainty",
            )
        if difference.status == "insufficient_units":
            _require(
                min(feature.a.n_sampling_units, feature.b.n_sampling_units) < 2,
                "independent contrast has sufficient units",
            )
    else:
        _require(
            difference.method == "unavailable"
            and difference.status in {"shared_units", "numeric_unavailable"},
            "unmatched contrast requires independent or shared-unit uncertainty",
        )


def validate_comparison_report(value: object) -> dict:
    """Validate comparison fields and semantic links without changing the input."""
    _require(isinstance(value, dict), "expected a JSON object")
    _require(
        type(value.get("schema_version")) is int and value["schema_version"] == 1,
        "unsupported schema version",
    )
    try:
        report = _Report.model_validate(value)
    except (ValidationError, RecursionError, OverflowError):
        raise ValueError("invalid comparison report fields or types") from None
    _require(
        [side.side for side in report.sides] == ["a", "b"],
        "sides must be ordered A then B",
    )
    a, b = report.sides
    for side in report.sides:
        _check_side(side)
    if report.mode == "windows":
        _require(
            a.run == b.run and a.source == b.source and a.window[1] < b.window[0],
            "windows must be ordered and disjoint within the same source run",
        )
    mix = report.prompt_mix
    _require(
        mix.shared_prompts + mix.only_a_prompts == a.n_prompts
        and mix.shared_prompts + mix.only_b_prompts == b.n_prompts,
        "prompt overlap counts differ from side counts",
    )
    _require(
        mix.status == ("shared_prompts" if mix.shared_prompts else "no_shared_prompts"),
        "prompt overlap status differs from counts",
    )
    descriptors = set()
    for shift in mix.prompt_shift:
        _require(
            shift.descriptor not in descriptors, "prompt descriptors must be unique"
        )
        descriptors.add(shift.descriptor)
        _require(
            shift.n_before + shift.missing_before == a.n_prompts
            and shift.n_after + shift.missing_after == b.n_prompts,
            "prompt descriptor counts differ from side counts",
        )
        if shift.before is not None and shift.after is not None:
            _require(
                _same(shift.delta, shift.after - shift.before),
                "prompt shift differs from its means",
            )
        else:
            _require(shift.delta is None, "prompt delta requires both means")
    _require(
        len({feature.feature for feature in report.features}) == len(report.features),
        "features must be unique",
    )
    _require(
        [feature.order for feature in report.features]
        == list(range(1, len(report.features) + 1)),
        "feature order must be contiguous",
    )
    ordered = sorted(
        report.features,
        key=lambda f: (
            f.kind != "binary",
            f.difference.estimate is None if f.kind == "binary" else False,
            -abs(f.difference.estimate)
            if f.kind == "binary" and f.difference.estimate is not None
            else 0,
            f.feature,
        ),
    )
    _require(report.features == ordered, "feature order differs from effect ordering")
    citation_metadata = {}
    for feature in report.features:
        _check_feature(feature, report)
        for side in ("a", "b"):
            for item in getattr(feature.citations, side).items:
                key = side, item.run, item.sample_id
                metadata = item.step, item.prompt_id, item.reward, item.response_sha256
                _require(
                    key not in citation_metadata or citation_metadata[key] == metadata,
                    "canonical citation metadata differs across features",
                )
                citation_metadata[key] = metadata
    return report.model_dump()


def comparison_json(report: dict) -> str:
    """Encode a validated comparison as deterministic, strict UTF-8 JSON."""
    return (
        json.dumps(
            validate_comparison_report(report),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    )


def load_comparison_report(path: str | Path) -> dict:
    """Read strict JSON with duplicate-key checks and errors without payloads."""
    try:
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(
                stream, object_pairs_hook=_object, parse_constant=_constant
            )
    except (ValueError, RecursionError):
        raise ValueError("invalid comparison report JSON") from None
    return validate_comparison_report(value)


_SCRIPT = """
(() => {
  const cards = Array.from(document.querySelectorAll('.comparison-feature'));
  const text = cards.map(card => card.textContent.toLowerCase());
  const search = document.getElementById('feature-search');
  const kind = document.getElementById('feature-kind');
  const count = document.getElementById('result-count');
  function update() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach((card, index) => {
      card.hidden = !text[index].includes(query)
        || (kind.value !== 'all' && card.dataset.kind !== kind.value);
      if (!card.hidden) visible++;
    });
    count.textContent = `${visible} of ${cards.length} features shown`;
    document.getElementById('no-matches').hidden = visible !== 0 || cards.length === 0;
  }
  search.addEventListener('input', update);
  kind.addEventListener('change', update);
  document.getElementById('reset-filters').addEventListener('click', () => {
    search.value = ''; kind.value = 'all'; update(); search.focus();
  });
  document.getElementById('expand-features').addEventListener('click', () => {
    const visible = cards.filter(card => !card.hidden);
    const expand = visible.some(card => !card.open);
    visible.forEach(card => { card.open = expand; });
  });
  document.getElementById('controls').hidden = false;
  update();
})();
"""
_COMPARISON_STYLE = """
.table-wrap { overflow-x: auto; margin: 14px 0; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
th { font-weight: 650; } td { font-variant-numeric: tabular-nums; }
.delta { font-weight: 700; font-variant-numeric: tabular-nums; }
.comparison-feature .event-body > p { margin-bottom: 10px; }
.sources { margin: 14px 0 24px; }
.sources > summary { padding: 8px 0; }
"""


def _text(value: object) -> str:
    return escape(str(value), quote=True)


def _number(value: float | None, *, scale: float = 1, signed: bool = False) -> str:
    if value is None:
        return '<span class="unavailable">Unavailable</span>'
    number = value * scale
    spec = ".4g" if abs(number) >= 1e6 or 0 < abs(number) < 0.0001 else ".2f"
    return format(number, ("+" if signed else "") + spec)


def _measurement(row: dict, kind: str, *, difference: bool = False) -> str:
    scale = 100 if kind == "binary" else 1
    unit = (" pp" if difference else "%") if kind == "binary" else ""
    center = row["estimate" if difference else "mean"]
    mean = _number(center, scale=scale, signed=difference)
    if center is not None:
        mean += unit
    interval = "Unavailable"
    if row["ci_low"] is not None:
        interval = f"[{_number(row['ci_low'], scale=scale)}, {_number(row['ci_high'], scale=scale)}]{unit}"
    status = {
        "available": "Available",
        "insufficient_units": "Fewer than two sampling units",
        "partial_overlap": "Only some prompts are shared",
        "shared_units": "Shared sampling units without paired prompts",
        "incompatible_units": "Sampling-unit groupings differ",
        "numeric_unavailable": "Numerical limits",
        "no_shared_prompts": "No shared prompts",
    }[row["status"]]
    return f"<td>{mean}</td><td>{_number(row['se'], scale=scale)}{unit if row['se'] is not None else ''}</td><td>{interval}</td><td>{status}</td>"


def _citation_html(citations: dict, side: str) -> str:
    parts = [f'<section class="metric"><h4>Side {side.upper()} citations</h4>']
    if not citations["items"]:
        parts.append(
            '<p class="note">Continuous features have no selected citations.</p>'
            if citations["status"] == "unsupported_kind"
            else '<p class="note">No positive rollouts in this range.</p>'
        )
    for item in citations["items"]:
        parts.append(
            f'<details class="exemplar"><summary><code>{_text(item["sample_id"])}</code> · step {item["step"]}</summary><dl>'
        )
        for key, label in (
            ("run", "Run"),
            ("prompt_id", "Prompt ID"),
            ("feature_value", "Feature value"),
            ("reward", "Reward"),
            ("selection_reason", "Selection reason"),
            ("response_sha256", "Response SHA-256"),
        ):
            value = (
                item[key].replace("_", " ") if key == "selection_reason" else item[key]
            )
            parts.append(f"<dt>{label}</dt><dd><code>{_text(value)}</code></dd>")
        parts.append("</dl></details>")
    parts.append("</section>")
    return "".join(parts)


def _feature_html(feature: dict) -> str:
    kind, difference = feature["kind"], feature["difference"]
    effect = _number(
        difference["estimate"], scale=100 if kind == "binary" else 1, signed=True
    )
    if difference["estimate"] is not None:
        effect += " pp" if kind == "binary" else " native units"
    parts = [
        f'<details class="event-card comparison-feature" data-kind="{kind}" id="feature-{feature["order"]}">',
        f'<summary><span class="rank">{feature["order"]}</span><span><code class="representative">{_text(feature["feature"])}</code>',
        f'<span class="anchors">{feature["tier"]} · {kind} · {feature["direction"]}</span></span>',
        f'<span class="delta">{effect}</span></summary><div class="event-body">',
        f'<p class="note">B - A interval method: {_text(difference["method"])}. {"Binary prevalence and percentage-point differences." if kind == "binary" else "Continuous measurements retain native units."}</p>',
        '<div class="table-wrap"><table><thead><tr><th scope="col">Measurement</th><th scope="col">Estimate</th><th scope="col">Standard error</th><th scope="col">95% normal interval</th><th scope="col">Interval status</th></tr></thead><tbody>',
    ]
    for side in ("a", "b"):
        parts.append(
            f'<tr><th scope="row">Side {side.upper()}</th>{_measurement(feature[side], kind)}</tr>'
        )
    parts.append(
        f'<tr><th scope="row">B - A</th>{_measurement(difference, kind, difference=True)}</tr>'
    )
    parts.append(
        f'<tr><th scope="row">Shared-prompt change</th>{_measurement(feature["paired"], kind, difference=True)}</tr></tbody></table></div>'
    )
    paired = feature["paired"]
    parts.append(
        f'<p class="note">Shared prompts: {paired["n_prompts"]}. Sampling units: {"unavailable" if paired["n_sampling_units"] is None else paired["n_sampling_units"]}.</p>'
    )
    if kind == "binary":
        parts.append(
            f'<p class="note">Positive prompts: A {feature["a"]["positive_prompts"]} of {feature["a"]["n_prompts"]}; B {feature["b"]["positive_prompts"]} of {feature["b"]["n_prompts"]}.</p>'
        )
    parts.append('<div class="metrics">')
    parts.extend(
        _citation_html(feature["citations"][side], side) for side in ("a", "b")
    )
    parts.append("</div></div></details>")
    return "".join(parts)


def render_comparison_report(report: dict) -> str:
    """Render validated comparison evidence as escaped HTML."""
    report = validate_comparison_report(report)
    script_hash = b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode(
        "ascii"
    )
    policy = f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'"
    parts = [
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{escape(policy, quote=True)}">',
        "<title>Rollout Diff comparison report</title>",
        f"<style>{_STYLE}{_COMPARISON_STYLE}</style></head><body><main>",
        '<header class="intro"><p class="eyebrow">Rollout Diff · Comparison version 1</p><h1>Comparison report</h1>',
        "<p>Differences are B - A. Each exact prompt receives equal weight after pooling its completions within each selected range. Both ranges share one feature vocabulary. Binary features are ordered by absolute prevalence change, followed by continuous features in name order.</p>",
        '<p class="note">Pointwise 95% intervals use a normal approximation and measured standard errors. Supplied sampling units account for dependence among prompts while preserving equal prompt weights. Intervals are unadjusted for comparisons across features. No standard-error floor is applied. Causal and task-success claims require separate evidence.</p>',
        '<p class="note">Comparing rewards across dumps requires the same reward definition and scale.</p></header>',
        '<div class="table-wrap"><table><thead><tr><th>Side</th><th>Run</th><th>Inclusive steps</th><th>Rollouts</th><th>Exact prompts</th><th>Sampling units</th></tr></thead><tbody>',
    ]
    for side in report["sides"]:
        parts.append(
            f'<tr><th scope="row">{side["side"].upper()}</th><td><code>{_text(side["run"])}</code></td><td>{side["window"][0]} to {side["window"][1]} ({len(side["observed_steps"])} observed)</td><td>{side["n_rollouts"]}</td><td>{side["n_prompts"]}</td><td>{side["n_sampling_units"]}</td></tr>'
        )
    parts.append("</tbody></table></div>")
    mix = report["prompt_mix"]
    parts.append(
        f"<section><h2>Prompt overlap</h2><p>{mix['shared_prompts']} shared prompts, {mix['only_a_prompts']} only in A, {mix['only_b_prompts']} only in B.</p>"
    )
    if not mix["shared_prompts"]:
        parts.append(
            '<p class="warning-note">No exact prompts are shared, so a paired change is unavailable. Independent-side intervals require disjoint sampling units. Differences may reflect prompt composition.</p>'
        )
    elif mix["only_a_prompts"] or mix["only_b_prompts"]:
        parts.append(
            '<p class="warning-note">Partial overlap leaves the overall interval unavailable. The paired comparison covers only shared prompts. Differences may reflect prompt composition.</p>'
        )
    else:
        parts.append(
            '<p class="note">All exact prompts are shared. Paired uncertainty requires compatible sampling-unit groupings.</p>'
        )
    parts.append(
        '<p class="note">Intervals need at least two sampling units. Partial prompt overlap, shared units without paired prompts, incompatible groupings or numerical limits can leave an interval unavailable. Available estimates are still shown. A zero measured standard error does not establish zero uncertainty.</p>'
    )
    if mix["prompt_shift"]:
        parts.append(
            '<div class="table-wrap"><table><thead><tr><th>Prompt descriptor</th><th>Statistic</th><th>A mean</th><th>B mean</th><th>B - A</th><th>Distance</th><th>Observed A / B</th><th>Missing A / B</th></tr></thead><tbody>'
        )
        for shift in mix["prompt_shift"]:
            parts.append(
                f"<tr><th><code>{_text(shift['descriptor'])}</code></th><td>{_text(shift['statistic'])}</td><td>{_number(shift['before'])}</td><td>{_number(shift['after'])}</td><td>{_number(shift['delta'], signed=True)}</td><td>{_number(shift['distance'])}</td><td>{shift['n_before']} / {shift['n_after']}</td><td>{shift['missing_before']} / {shift['missing_after']}</td></tr>"
            )
        parts.append("</tbody></table></div>")
    parts.append(
        '</section><details class="sources"><summary>Source fingerprints</summary><div class="metrics">'
    )
    for side in report["sides"]:
        source = side["source"]
        parts.append(
            f'<section class="metric"><h3>Side {side["side"].upper()}</h3><dl><dt>Canonical SHA-256</dt><dd><code>{source["canonical_sha256"]}</code></dd><dt>Format</dt><dd>{source["format"]}</dd><dt>Reward column</dt><dd>{_text(source["reward_column"]) if source["reward_column"] is not None else "Unspecified"}</dd></dl>'
        )
        for file in source["files"]:
            parts.append(
                f'<p class="reference-note"><code>{file["sha256"]}</code> · {file["size_bytes"]} bytes</p>'
            )
        parts.append("</section>")
    parts.extend(
        [
            "</div></details><section><h2>Feature differences</h2>",
            '<p class="note">Citations are deterministic feature-positive examples selected without reward ranking. Their selection does not establish representativeness. Inspect responses in the original dump by side and feature.</p>',
            '<div class="controls" id="controls" hidden><label class="search-label">Search features<input id="feature-search" type="search" placeholder="Feature or sample ID" autocomplete="off"></label><label>Feature kind<select id="feature-kind"><option value="all">All features</option><option value="binary">Binary</option><option value="continuous">Continuous</option></select></label><button id="reset-filters" type="button">Reset filters</button><button id="expand-features" type="button">Expand / collapse</button><p class="result-count" id="result-count" role="status" aria-live="polite"></p></div><p class="empty" id="no-matches" hidden>No matching features.</p>',
        ]
    )
    if not report["features"]:
        parts.append('<p class="empty">No features in this comparison.</p>')
    parts.extend(_feature_html(feature) for feature in report["features"])
    parts.append(f"</section></main><script>{_SCRIPT}</script></body></html>\n")
    return "".join(parts)
