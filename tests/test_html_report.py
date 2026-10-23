import copy
import hashlib
from base64 import b64encode
from html import escape
from html.parser import HTMLParser

import pytest
from test_full_pipeline import analyze, fixture

from rolloutdiff.report import build_report


@pytest.fixture(scope="module")
def report():
    return build_report(analyze(fixture()))


def render(report):
    from rolloutdiff import render_report

    return render_report(report)


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.tags = []
        self.text = []
        self.scripts = []
        self.in_script = False
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        if tag == "script":
            self.in_script = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        self.text.append(data)
        if self.in_script:
            self.scripts.append(data)


def replace(value, before, after):
    if isinstance(value, dict):
        return {key: replace(item, before, after) for key, item in value.items()}
    if isinstance(value, list):
        return [replace(item, before, after) for item in value]
    return after if value == before else value


def test_report_renders_all_events_members_and_exemplars_without_mutation(report):
    original = copy.deepcopy(report)
    html = render(report)
    document = Document(html)
    events = [
        attrs for tag, attrs in document.tags if attrs.get("class") == "event-card"
    ]
    expected = sum(len(run["events"]) for run in report["runs"])
    assert len(events) == expected
    for run in report["runs"]:
        representatives = [
            escape(event["representative_feature"])
            for event in sorted(run["events"], key=lambda event: event["rank"])
        ]
        summaries = html.split('class="event-card"')[1:]
        assert all(
            feature in card.split("</summary>")[0]
            for feature, card in zip(representatives, summaries, strict=True)
        )
        for event in run["events"]:
            for member in event["members"]:
                assert escape(member["feature"]) in html
                for example in member["exemplars"]:
                    if example["sample_id"] is not None:
                        assert escape(example["sample_id"]) in html
    assert "scalar:reward" in html
    assert report == original


def test_html_is_deterministic_under_shuffled_arrays(report):
    shuffled = copy.deepcopy(report)
    second_run = replace(shuffled["runs"][0], "r", "second")
    for event in second_run["events"]:
        event["event_id"] = "second:" + event["event_id"]
    shuffled["runs"].append(second_run)
    original = copy.deepcopy(shuffled)
    shuffled["runs"].reverse()
    for run in shuffled["runs"]:
        run["events"].reverse()
        for event in run["events"]:
            event["members"].reverse()
            for key in (
                "missing_evidence_members",
                "numerical_floor_members",
                "numerical_floor_unavailable_members",
            ):
                event["ranking"][key].reverse()
            for member in event["members"]:
                member["exemplars"].reverse()
    assert render(shuffled) == render(original)


@pytest.mark.parametrize("target", ["run", "feature", "sample_id", "prompt_id"])
def test_untrusted_identifiers_are_escaped_without_active_markup(report, target):
    payload = '</code><img src=x onerror="alert(1)"></script><script>alert(2)</script>'
    if target == "run":
        before = report["runs"][0]["run"]
    else:
        members = [
            member
            for event in report["runs"][0]["events"]
            for member in event["members"]
        ]
        if target == "feature":
            before = members[0]["feature"]
        else:
            before = next(
                example[target]
                for member in members
                for example in member["exemplars"]
                if example[target]
            )
    html = render(replace(report, before, payload))
    document = Document(html)
    assert escape(payload) in html
    assert payload in "".join(document.text)
    assert sum(tag == "script" for tag, attrs in document.tags) == 1
    assert not any(tag in {"img", "iframe", "object"} for tag, attrs in document.tags)
    assert not any(
        key.startswith("on") for tag, attrs in document.tags for key in attrs
    )
    assert payload not in document.scripts[0]


def test_missing_zero_false_and_signed_values_remain_distinct(report):
    modified = copy.deepcopy(report)
    member = modified["runs"][0]["events"][0]["members"][0]
    member["evidence"]["replication"]["replicates"] = False
    member["evidence"]["persistence"]["fraction"] = None
    member["evidence"]["persistence"]["n_retained"] = 0
    member["evidence"]["reward_association"]["mean_difference"] = -0.25
    html = render(modified)
    assert "<dt>Replicates</dt><dd>No</dd>" in html
    assert (
        '<dt>Retained fraction</dt><dd><span class="unavailable">Unavailable</span></dd>'
        in html
    )
    assert "<dt>Retained observations</dt><dd>0</dd>" in html
    assert "<dt>Mean reward difference</dt><dd>-0.25</dd>" in html


def test_floor_warnings_and_unavailable_statuses_are_visible(report):
    html = render(report)
    assert "Numerical floor" in html
    assert "Unavailable" in html
    assert "References only" in html
    assert "raw rollout text is not included" in html
    assert "not_binary" in html


def test_empty_report_and_empty_run():
    assert "No runs in this report" in render({"schema_version": 1, "runs": []})
    empty = build_report(analyze(fixture().iloc[:0]), run="empty")
    html = render(empty)
    assert "No sequential events" in html
    assert "empty" in html


def test_exact_integer_steps_survive_rendering(report):
    modified = copy.deepcopy(report)
    offset = 2**60

    def shift(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in {
                    "onset_start",
                    "onset_end",
                    "onset_step",
                    "alert_step",
                    "step",
                }:
                    if item is not None:
                        value[key] += offset
                else:
                    shift(item)
        elif isinstance(value, list):
            for item in value:
                shift(item)

    shift(modified)
    event = modified["runs"][0]["events"][0]
    assert str(event["onset_step"]) in render(modified)


def test_self_contained_html_has_fixed_script_and_matching_csp(report):
    document = Document(render(report))
    assert len(document.scripts) == 1
    script = document.scripts[0]
    digest = b64encode(hashlib.sha256(script.encode()).digest()).decode()
    csp = next(
        attrs["content"]
        for tag, attrs in document.tags
        if attrs.get("http-equiv") == "Content-Security-Policy"
    )
    assert f"'sha256-{digest}'" in csp
    assert "default-src 'none'" in csp
    assert "innerHTML" not in script
    assert not any("src" in attrs or "href" in attrs for tag, attrs in document.tags)


def test_unknown_fields_are_not_exported(report):
    modified = copy.deepcopy(report)
    modified["raw_response"] = "private-extra-text"
    modified["runs"][0]["events"][0]["raw_prompt"] = "private-extra-text"
    assert "private-extra-text" not in render(modified)


def test_status_text_is_escaped_and_missing_evidence_remains_explicit(report):
    modified = copy.deepcopy(report)
    event = modified["runs"][0]["events"][0]
    member = event["members"][0]
    member["evidence"]["status"] = "missing"
    event["ranking"]["missing_evidence_members"] = [
        {"run": member["run"], "feature": member["feature"]}
    ]
    payload = '<svg onload="alert(1)">'
    member["evidence"]["replication"]["status"] = payload
    html = render(modified)
    assert escape(payload) in html
    assert not any(tag == "svg" for tag, attrs in Document(html).tags)
    assert "<dt>Evidence record</dt><dd>missing</dd>" in html
    assert "Missing evidence:" in html


def test_renderer_rejects_unsupported_version():
    with pytest.raises(ValueError, match="version"):
        render({"schema_version": 2, "runs": []})
