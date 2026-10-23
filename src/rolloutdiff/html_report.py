"""Static, self-contained presentation of version-1 investigation reports."""

from __future__ import annotations

import hashlib
from base64 import b64encode
from html import escape

from ._report_validation import load_report, validate_report

__all__ = ["load_report", "render_report"]

_STYLE = """
:root { color-scheme: light; --ink: #182d35; --muted: #52646c;
  --line: #d8e0e2; --accent: #17695f; --paper: #f5f7f7; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1200px; margin: auto; padding: 32px 28px 60px; }
h1, h2, h3, h4, p { margin: 0; }
h1 { font-size: 30px; letter-spacing: -.7px; line-height: 1.2; margin: 4px 0 12px; }
h2 { font-size: 22px; margin: 0 0 12px; }
h3 { font-size: 16px; margin-bottom: 10px; }
h4 { font-size: 14px; margin-bottom: 10px; }
code { font: .92em/1.5 ui-monospace, SFMono-Regular, Consolas, monospace;
  overflow-wrap: anywhere; }
.eyebrow { color: var(--accent); font-weight: 700; font-size: 12px;
  letter-spacing: 1.2px; text-transform: uppercase; }
.muted, .note, .unavailable { color: var(--muted); }
.note { font-size: 13px; }
.intro { max-width: 850px; margin-bottom: 22px; }
.controls { position: sticky; top: 0; z-index: 2; background: #f5f7f7;
  border-block: 1px solid var(--line); padding: 12px 0;
  display: grid; grid-template-columns: minmax(180px, 1fr) 180px auto auto;
  gap: 12px; align-items: end; }
label { display: block; font-size: 12px; font-weight: 650; }
input[type=search], select, button { font: inherit; color: var(--ink);
  background: white; border: 1px solid #9caaaf; border-radius: 5px; padding: 8px 10px; }
input[type=search], select { width: 100%; margin-top: 4px; }
input[type=checkbox] { accent-color: var(--accent); }
.checkbox { padding-bottom: 10px; white-space: nowrap; }
button, summary { cursor: pointer; }
button:hover { background: #e8f1ef; }
:focus-visible { outline: 3px solid #157b9e; outline-offset: 3px; }
.result-count { grid-column: 1 / -1; font-size: 12px; color: var(--muted); }
.run { margin-top: 30px; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
  margin: 14px 0 10px; }
.stat { background: white; border: 1px solid var(--line); border-radius: 6px;
  padding: 12px 16px; }
.stat strong { display: block; font-size: 24px; font-variant-numeric: tabular-nums; }
.stat span { font-size: 12px; color: var(--muted); }
.run-counts { margin-bottom: 18px; }
.event-card { background: white; border: 1px solid var(--line);
  border-radius: 7px; margin: 10px 0; overflow: clip; }
.event-card > summary { display: grid; grid-template-columns: 36px minmax(0, 1fr) auto 14px;
  gap: 12px; padding: 16px; align-items: start; list-style: none; }
summary::-webkit-details-marker { display: none; }
.event-card > summary::after { content: "+"; color: var(--muted); font-size: 18px; }
.event-card[open] > summary::after { content: "-"; }
.event-card[open] > summary { border-bottom: 1px solid var(--line); }
.rank { font-size: 13px; color: var(--muted); padding-top: 3px; }
.representative { display: block; font-weight: 650; font-size: 16px; }
.anchors { display: block; font-size: 12px; color: var(--muted); margin-top: 5px; }
.badges { display: flex; gap: 6px; flex-wrap: wrap; justify-content: end; max-width: 270px; }
.badge { font-size: 11px; border-radius: 4px; background: #e9f2ef;
  color: #1c584e; padding: 3px 7px; white-space: nowrap; }
.badge.contextual, .badge.other { background: #eef0f3; color: #4d5d70; }
.badge.warning { background: #fff1ce; color: #785015; }
.event-body { padding: 18px; }
.metrics { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px; margin: 14px 0; }
.metric { border: 1px solid var(--line); border-radius: 5px; padding: 14px; }
dl { display: grid; grid-template-columns: minmax(130px, 1fr) minmax(0, 1fr);
  gap: 6px 16px; font-size: 13px; margin: 0; }
dt { color: var(--muted); }
dd { margin: 0; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
.member { margin-top: 20px; padding-top: 20px; border-top: 1px solid var(--line); }
.member-title { display: flex; gap: 10px; flex-wrap: wrap; align-items: baseline; }
.member-title h3 { margin-bottom: 4px; }
.member-meta { margin: 10px 0; }
.warning-note { background: #fff7e2; border-left: 3px solid #c69936;
  padding: 8px 12px; font-size: 13px; margin: 10px 0; overflow-wrap: anywhere; }
.reference-note { font-size: 12px; color: var(--muted); margin: 8px 0; }
.exemplar { border: 1px solid var(--line); border-radius: 5px; margin-top: 7px; }
.exemplar > summary { padding: 10px 12px; font-size: 13px; }
.exemplar > dl { padding: 4px 14px 14px; }
.empty { padding: 24px 0; color: var(--muted); }
[hidden] { display: none !important; }
@media (max-width: 700px) {
  main { padding: 22px 16px 40px; } h1 { font-size: 26px; }
  .controls { position: static; grid-template-columns: 1fr 1fr; }
  .search-label { grid-column: 1 / -1; } .checkbox { white-space: normal; }
  .stats { grid-template-columns: 1fr 1fr; gap: 8px; }
  .event-card > summary { grid-template-columns: 26px minmax(0, 1fr) 12px; gap: 8px; padding: 12px; }
  .badges { grid-column: 2; justify-content: start; max-width: none; }
  .event-card > summary::after { grid-column: 3; grid-row: 1; }
  .event-body { padding: 12px; } .metrics { grid-template-columns: 1fr; }
  dl { grid-template-columns: minmax(110px, 1fr) minmax(0, 1fr); gap: 6px 10px; }
}
@media print {
  .controls { display: none; } main { max-width: none; padding: 0; }
  body { background: white; } .event-card { break-inside: avoid; }
}
"""

_SCRIPT = """
(() => {
  const cards = Array.from(document.querySelectorAll('.event-card'));
  const text = cards.map(card => card.textContent.toLowerCase());
  const search = document.getElementById('event-search');
  const status = document.getElementById('event-status');
  const floor = document.getElementById('floor-only');
  const count = document.getElementById('result-count');
  function update() {
    const query = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach((card, index) => {
      card.hidden = !text[index].includes(query)
        || (status.value !== 'all' && card.dataset.category !== status.value)
        || (floor.checked && card.dataset.floor !== 'true');
      if (!card.hidden) visible++;
    });
    count.textContent = `${visible} of ${cards.length} events shown`;
    document.getElementById('no-matches').hidden = visible !== 0 || cards.length === 0;
  }
  search.addEventListener('input', update);
  status.addEventListener('change', update);
  floor.addEventListener('change', update);
  document.getElementById('reset-filters').addEventListener('click', () => {
    search.value = '';
    status.value = 'all';
    floor.checked = false;
    update();
    search.focus();
  });
  document.getElementById('controls').hidden = false;
  update();
})();
"""

_CONTROLS = """
<div class="controls" id="controls" hidden>
  <label class="search-label">Search events
    <input id="event-search" type="search" placeholder="Feature, event or sample ID" autocomplete="off">
  </label>
  <label>Ranking status<select id="event-status">
    <option value="all">All events</option><option value="behavioral">Behavioral</option>
    <option value="contextual">Contextual</option><option value="other">Other / unavailable</option>
  </select></label>
  <label class="checkbox"><input type="checkbox" id="floor-only"> Numerical floor only</label>
  <button id="reset-filters" type="button">Reset filters</button>
  <p class="result-count" id="result-count" role="status" aria-live="polite"></p>
</div>
<p class="empty" id="no-matches" hidden>No matching events. Adjust or reset the filters.</p>
"""

_REPLICATION = (
    ("status", "Status"),
    ("replicates", "Replicates"),
    ("horizon_complete", "Horizon complete"),
    ("coverage_policy", "Coverage policy"),
    ("cutoff_step", "Cutoff step"),
)
_PERSISTENCE = (
    ("status", "Status"),
    ("fraction", "Retained fraction"),
    ("coverage", "Evaluable coverage"),
    ("n_observations", "Observed observations"),
    ("n_evaluable", "Evaluable observations"),
    ("n_retained", "Retained observations"),
    ("n_unavailable", "Unavailable observations"),
)
_REWARD = (
    ("status", "Status"),
    ("mean_difference", "Mean reward difference"),
    ("n_groups", "Groups"),
    ("n_mixed_groups", "Mixed groups"),
    ("n_rollouts", "Window rollouts"),
    ("n_included_rollouts", "Included rollouts"),
    ("window_start", "Window start"),
    ("window_end", "Window end"),
)
_BREADTH = (
    ("positive_prompts", "Positive prompts"),
    ("total_prompts", "Total prompts"),
    ("promptmix_status", "Shared-prompt status"),
)
_EXEMPLAR = (
    ("run", "Run"),
    ("sample_id", "Sample ID"),
    ("prompt_id", "Prompt ID"),
    ("step", "Original step"),
    ("status", "Status"),
    ("selection_order", "Selection order"),
    ("selection_reason", "Selection reason"),
    ("feature_value", "Feature value"),
    ("reward", "Reward"),
    ("reward_selection_status", "Reward selection status"),
    ("reward_group_size", "Reward group size"),
    ("reward_group_mean", "Reward group mean"),
    ("reward_deviation", "Reward deviation"),
    ("reward_deviation_status", "Reward deviation status"),
    ("n_window_rollouts", "Window rollouts"),
    ("n_candidates", "Candidates"),
)


def _value(value: object, *, identifier: bool = False) -> str:
    if value is None:
        return '<span class="unavailable">Unavailable</span>'
    if isinstance(value, bool):
        return "Yes" if value else "No"
    text = escape(str(value))
    return f"<code>{text}</code>" if identifier else text


def _fields(record: dict, fields: tuple) -> str:
    return (
        "<dl>"
        + "".join(
            f"<dt>{label}</dt><dd>{_value(record[key], identifier=key in {'run', 'sample_id', 'prompt_id'})}</dd>"
            for key, label in fields
        )
        + "</dl>"
    )


def _status(value: str) -> str:
    return escape(value.replace("_", " "))


def _references(references: list[dict]) -> str:
    return ", ".join(
        _value(item["feature"], identifier=True)
        for item in sorted(references, key=lambda item: (item["run"], item["feature"]))
    )


def _member(member: dict) -> str:
    evidence = member["evidence"]
    result = [
        '<article class="member"><div class="member-title">',
        f"<h3>{_value(member['feature'], identifier=True)}</h3>",
        f'<span class="note">{_status(member["kind"])} · {_status(member["direction"])}</span></div>',
        '<div class="member-meta">',
        _fields(
            member,
            (
                ("run", "Run"),
                ("onset_step", "Onset step"),
                ("alert_step", "Alert step"),
                ("status", "Membership status"),
            ),
        ),
        "</div>",
        _fields(
            evidence,
            (
                ("status", "Evidence record"),
                ("tier", "Feature tier"),
                ("scale_source", "Scale source"),
                ("numerical_floor_dominant", "Numerical floor dominant"),
            ),
        ),
    ]
    if evidence["numerical_floor_dominant"] is True:
        result.append(
            '<p class="warning-note">The numerical floor dominates this feature\'s baseline scale.</p>'
        )
    result.append('<div class="metrics">')
    for key, heading, fields in (
        ("replication", "Split-half replication", _REPLICATION),
        ("persistence", "Persistence", _PERSISTENCE),
        ("reward_association", "Signed reward association", _REWARD),
        ("prompt_breadth", "Prompt breadth", _BREADTH),
    ):
        result.append(f'<section class="metric"><h4>{heading}</h4>')
        result.append(_fields(evidence[key], fields))
        if key == "reward_association":
            result.append(
                '<p class="reference-note">Present-minus-absent reward in mixed groups.</p>'
            )
        result.append("</section>")
    result.append("</div><h4>Exemplar references</h4>")
    result.append(
        '<p class="reference-note">References only; raw rollout text is not included. Selected examples are descriptive, not representative estimates.</p>'
    )
    result.append(
        f'<p class="note">Record status: {_status(member["exemplar_status"])}</p>'
    )
    if not member["exemplars"]:
        result.append('<p class="note">No exemplar records in this report.</p>')
    for exemplar in sorted(
        member["exemplars"],
        key=lambda row: (
            row["selection_order"] is None,
            row["selection_order"] or 0,
            row["sample_id"] or "",
        ),
    ):
        result.append('<details class="exemplar"><summary>')
        if exemplar["sample_id"] is None:
            result.append(f"No selected reference · {_status(exemplar['status'])}")
        else:
            result.append(
                f"{_value(exemplar['sample_id'], identifier=True)} · step {_value(exemplar['step'])}"
            )
        result.extend(["</summary>", _fields(exemplar, _EXEMPLAR), "</details>"])
    result.append("</article>")
    return "".join(result)


def _event(event: dict, run_index: int) -> str:
    ranking = event["ranking"]
    category = event["ranking_status"]
    if category not in {"behavioral", "contextual"}:
        category = "other"
    floors = ranking["numerical_floor_members"]
    result = [
        f'<details class="event-card" id="event-{run_index}-{event["rank"]}" data-category="{category}" data-floor="{str(bool(floors)).lower()}">',
        f'<summary><span class="rank">#{event["rank"]}</span><span>',
        f'<code class="representative">{escape(event["representative_feature"])}</code>',
        f'<span class="anchors">{_value(event["event_id"], identifier=True)} · {event["n_members"]} members · onset {event["onset_step"]} · alert {event["alert_step"]}</span></span>',
        f'<span class="badges"><span class="badge {category}">{_status(event["ranking_status"])}</span>',
    ]
    if floors:
        result.append(
            f'<span class="badge warning">Numerical floor · {len(floors)}</span>'
        )
    if ranking["numerical_floor_unavailable_members"]:
        result.append('<span class="badge other">Floor evidence unavailable</span>')
    result.append('</span></summary><div class="event-body">')
    result.append(
        f'<p class="note">Event onset range: {event["onset_start"]} to {event["onset_end"]}. Representative evidence availability:</p>'
    )
    result.append(
        _fields(
            ranking,
            (
                ("n_prioritizable_members", "Prioritizable members"),
                ("replication_availability", "Replication"),
                ("persistence_availability", "Persistence"),
                ("reward_availability", "Reward association"),
                ("breadth_availability", "Prompt breadth"),
            ),
        )
    )
    for key, label in (
        ("numerical_floor_members", "Numerical floor"),
        ("numerical_floor_unavailable_members", "Floor evidence unavailable"),
        ("missing_evidence_members", "Missing evidence"),
    ):
        if ranking[key]:
            result.append(
                f'<p class="warning-note">{label}: {_references(ranking[key])}</p>'
            )
    for member in sorted(
        event["members"],
        key=lambda row: (row["onset_step"], row["alert_step"], row["feature"]),
    ):
        result.append(_member(member))
    result.append("</div></details>")
    return "".join(result)


def _run(run: dict, index: int) -> str:
    counts = run["counts"]
    result = [
        f'<section class="run" id="run-{index}"><h2>Run {_value(run["run"], identifier=True)}</h2>',
        f'<p class="note">Status: {_status(run["status"])}</p><div class="stats">',
    ]
    for key, label in (
        ("rollouts", "Rollouts"),
        ("screened_series", "Screened series"),
        ("sequential_alerts", "Sequential alerts"),
        ("events", "Ranked events"),
    ):
        result.append(
            f'<div class="stat"><strong>{counts[key]:,}</strong><span>{label}</span></div>'
        )
    result.append('</div><p class="note run-counts">')
    result.append(
        " · ".join(
            f"{counts[key]:,} {label}"
            for key, label in (
                ("series", "series"),
                ("unscorable_series", "unscorable"),
                ("observed_steps", "observed steps"),
                ("prompts", "prompts"),
                ("behavioral_events", "behavioral events"),
                ("contextual_events", "contextual events"),
                ("other_events", "other events"),
            )
        )
    )
    result.append("</p>")
    if not run["events"]:
        result.append('<p class="empty">No sequential events in this run.</p>')
    result.extend(
        _event(event, index)
        for event in sorted(run["events"], key=lambda row: row["rank"])
    )
    result.append("</section>")
    return "".join(result)


def render_report(report: dict) -> str:
    """Validate a version-1 report and render escaped, self-contained HTML.

    Existing within-run ranks are preserved. Unknown JSON fields are omitted.
    """
    report = validate_report(report)
    script_hash = b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode(
        "ascii"
    )
    policy = f"default-src 'none'; style-src 'unsafe-inline'; script-src 'sha256-{script_hash}'; base-uri 'none'; form-action 'none'"
    result = [
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f'<meta http-equiv="Content-Security-Policy" content="{escape(policy, quote=True)}">',
        "<title>Rollout Diff investigation report</title>",
        f"<style>{_STYLE}</style></head><body><main>",
        '<header class="intro"><p class="eyebrow">Rollout Diff · Report version 1</p>',
        "<h1>Investigation report</h1>",
        '<p class="note">Events are ranked using a heuristic assessment of supporting evidence. Grouped features share similar onset times and overlapping rollouts, providing leads for further investigation. Expand an event to examine its measurements, evidence availability, and exemplar references.</p></header>',
        _CONTROLS,
    ]
    if not report["runs"]:
        result.append('<p class="empty">No runs in this report.</p>')
    result.extend(
        _run(run, index)
        for index, run in enumerate(sorted(report["runs"], key=lambda row: row["run"]))
    )
    result.append(f"</main><script>{_SCRIPT}</script></body></html>\n")
    return "".join(result)
