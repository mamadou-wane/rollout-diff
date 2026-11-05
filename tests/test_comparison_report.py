import copy
import json
from html.parser import HTMLParser

import pytest


def _api():
    from rolloutdiff import comparison_report

    return comparison_report


@pytest.fixture
def report():
    def estimate(mean, positive):
        return {
            "mean": mean,
            "se": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "n_rollouts": 2,
            "n_prompts": 2,
            "n_sampling_units": 2,
            "positive_prompts": positive,
            "status": "available",
        }

    contrast = {
        "estimate": 1.0,
        "se": 0.0,
        "ci_low": 1.0,
        "ci_high": 1.0,
        "status": "available",
    }
    source = {
        "canonical_sha256": "a" * 64,
        "format": "canonical",
        "files": [],
        "reward_column": None,
    }
    return {
        "report_type": "comparison",
        "schema_version": 1,
        "mode": "windows",
        "sides": [
            {
                "side": side,
                "run": "run",
                "window": [offset, offset + 1],
                "observed_steps": [offset, offset + 1],
                "n_rollouts": 2,
                "n_prompts": 2,
                "n_sampling_units": 2,
                "source": copy.deepcopy(source),
            }
            for side, offset in [("a", 0), ("b", 2)]
        ],
        "prompt_mix": {
            "status": "shared_prompts",
            "shared_prompts": 2,
            "only_a_prompts": 0,
            "only_b_prompts": 0,
            "prompt_shift": [
                {
                    "descriptor": "prompt_chars",
                    "statistic": "ks",
                    "before": 10.0,
                    "after": 10.0,
                    "delta": 0.0,
                    "distance": 0.0,
                    "n_before": 2,
                    "n_after": 2,
                    "missing_before": 0,
                    "missing_after": 0,
                }
            ],
        },
        "features": [
            {
                "order": 1,
                "feature": "has_code",
                "tier": "structural",
                "kind": "binary",
                "direction": "increase",
                "a": estimate(0.0, 0),
                "b": estimate(1.0, 2),
                "difference": {**contrast, "method": "paired"},
                "paired": {**contrast, "n_prompts": 2, "n_sampling_units": 2},
                "citations": {
                    "a": {"status": "no_positive_rollouts", "items": []},
                    "b": {
                        "status": "selected",
                        "items": [
                            {
                                "run": "run",
                                "sample_id": "s3",
                                "step": 2,
                                "prompt_id": "p1",
                                "feature_value": 1.0,
                                "reward": 0.0,
                                "response_sha256": "b" * 64,
                                "selection_reason": "earliest_positive",
                            },
                            {
                                "run": "run",
                                "sample_id": "s4",
                                "step": 3,
                                "prompt_id": "p2",
                                "feature_value": 1.0,
                                "reward": 0.0,
                                "response_sha256": "c" * 64,
                                "selection_reason": "prompt_diversity",
                            },
                        ],
                    },
                },
            }
        ],
    }


def test_round_trip_is_deterministic_and_does_not_mutate(report, tmp_path):
    original = copy.deepcopy(report)
    api = _api()
    validated = api.validate_comparison_report(report)
    assert validated == original == report
    assert validated is not report
    encoded = api.comparison_json(report)
    assert encoded.endswith("\n")
    assert json.loads(encoded) == report
    assert encoded == api.comparison_json(dict(reversed(list(report.items()))))
    path = tmp_path / "comparison.json"
    path.write_text(encoded, encoding="utf-8")
    assert api.load_comparison_report(path) == report


@pytest.mark.parametrize(
    ("location", "value"),
    [
        (("schema_version",), True),
        (("schema_version",), 1.0),
        (("sides", 0, "n_rollouts"), True),
        (("sides", 0, "window", 0), 0.0),
        (("sides", 0, "source", "canonical_sha256"), "PRIVATE_VALUE"),
        (("sides", 0, "run"), "/private/PRIVATE_VALUE"),
        (("sides", 0, "source", "reward_column"), "/private/PRIVATE_VALUE"),
        (("features", 0, "a", "mean"), float("inf")),
        (("features", 0, "a", "mean"), True),
        (("features", 0, "a", "se"), -1.0),
        (("features", 0, "difference", "method"), "PRIVATE_VALUE"),
        (("features", 0, "citations", "b", "items", 0, "sample_id"), " "),
    ],
)
def test_invalid_types_and_private_identifiers_fail_without_echo(
    report, location, value
):
    target = report
    for key in location[:-1]:
        target = target[key]
    target[location[-1]] = value
    with pytest.raises(ValueError, match="comparison report") as exc:
        _api().validate_comparison_report(report)
    assert "PRIVATE_VALUE" not in str(exc.value)
    assert "input_value" not in str(exc.value)


@pytest.mark.parametrize(
    "location",
    [(), ("sides", 0), ("features", 0), ("features", 0, "a")],
)
def test_unknown_fields_are_rejected(report, location):
    target = report
    for key in location:
        target = target[key]
    target["response"] = "PRIVATE_VALUE"
    with pytest.raises(ValueError, match="comparison report") as exc:
        _api().comparison_json(report)
    assert "PRIVATE_VALUE" not in str(exc.value)


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["sides"][1].update(side="a"),
        lambda r: r["sides"][1].update(run="other"),
        lambda r: r["sides"][1].update(window=[1, 3]),
        lambda r: r["sides"][0].update(observed_steps=[1, 0]),
        lambda r: r["sides"][0].update(n_prompts=3),
        lambda r: r["sides"][1]["source"].update(canonical_sha256="d" * 64),
        lambda r: r["prompt_mix"].update(shared_prompts=1),
        lambda r: r["prompt_mix"].update(status="no_shared_prompts"),
        lambda r: r["prompt_mix"]["prompt_shift"][0].update(n_before=1),
        lambda r: r["features"][0].update(order=2),
        lambda r: r["features"][0].update(tier="prose"),
        lambda r: r["features"][0].update(kind="continuous"),
        lambda r: r["features"][0].update(direction="decrease"),
        lambda r: r["features"][0]["a"].update(n_rollouts=1),
        lambda r: r["features"][0]["a"].update(mean=1.1),
        lambda r: r["features"][0]["a"].update(positive_prompts=1),
        lambda r: r["features"][0]["a"].update(se=None),
        lambda r: r["features"][0]["a"].update(ci_high=None),
        lambda r: r["features"][0]["a"].update(ci_high=1.0),
        lambda r: r["features"][0]["a"].update(status="insufficient_units"),
        lambda r: r["features"][0]["difference"].update(estimate=0.5),
        lambda r: r["features"][0]["difference"].update(method="independent"),
        lambda r: r["features"][0]["paired"].update(n_prompts=1),
        lambda r: r["features"][0]["paired"].update(estimate=0.5),
        lambda r: r["features"][0]["citations"]["b"]["items"][0].update(run="other"),
        lambda r: r["features"][0]["citations"]["b"]["items"][0].update(step=0),
        lambda r: r["features"][0]["citations"]["b"]["items"][0].update(
            feature_value=0.0
        ),
        lambda r: r["features"][0]["citations"]["b"]["items"][0].update(
            selection_reason="additional_positive"
        ),
        lambda r: r["features"][0]["citations"]["b"]["items"][1].update(sample_id="s3"),
        lambda r: r["features"][0]["citations"]["b"].update(
            status="no_positive_rollouts"
        ),
    ],
)
def test_inconsistent_semantics_are_rejected(report, change):
    change(report)
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)


def test_duplicate_features_and_noncanonical_order_are_rejected(report):
    report["features"].append(copy.deepcopy(report["features"][0]))
    report["features"][1]["order"] = 2
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)
    report["features"][0]["feature"] = "raises"
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)


@pytest.mark.parametrize(
    "payload",
    [
        '{"report_type":"comparison","schema_version":1,"schema_version":1}',
        '{"report_type":"comparison","PRIVATE_VALUE":NaN}',
        '{"report_type":"comparison","PRIVATE_VALUE":Infinity}',
        '{"PRIVATE_VALUE":',
        "[" * 2000,
    ],
)
def test_loader_rejects_unsafe_json_without_echo(tmp_path, payload):
    path = tmp_path / "comparison.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="JSON") as exc:
        _api().load_comparison_report(path)
    assert "PRIVATE_VALUE" not in str(exc.value)


def test_loader_rejects_invalid_utf8(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="JSON"):
        _api().load_comparison_report(path)


class _Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []
        self.scripts = []
        self.in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))
        self.in_script = tag == "script" or self.in_script

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_script = False

    def handle_data(self, data):
        (self.scripts if self.in_script else self.text).append(data)


def test_html_preserves_signs_measurements_and_escapes_identifiers(report):
    report["features"][0]["citations"]["b"]["items"][0]["sample_id"] = (
        '<img src=x onerror="alert(1)">'
    )
    html = _api().render_comparison_report(report)
    document = _Document()
    document.feed(html)
    text = " ".join(document.text)
    assert not any(tag == "img" for tag, _ in document.tags)
    assert '<img src=x onerror="alert(1)">' in text
    assert "B - A" in text
    assert "B - A interval method: paired" in text
    assert "95% normal interval" in text
    assert "Interval status" in text
    assert "+100" in text and "pp" in text
    assert "95%" in text and "paired" in text
    assert "0.00" in text
    assert "sampling units" in text
    assert "A zero measured standard error does not establish zero uncertainty." in text
    assert "equal" in text.lower() and "prompt" in text.lower()
    assert "causal" in text.lower()
    assert all(
        "s3" not in script and "has_code" not in script for script in document.scripts
    )
    assert any(attrs.get("type") == "search" for _, attrs in document.tags)


def test_html_shows_partial_overlap_and_unavailable_intervals(report):
    report["mode"] = "dumps"
    report["prompt_mix"].update(shared_prompts=1, only_a_prompts=1, only_b_prompts=1)
    feature = report["features"][0]
    feature["difference"].update(
        status="partial_overlap",
        method="unavailable",
        se=None,
        ci_low=None,
        ci_high=None,
    )
    feature["paired"].update(
        status="insufficient_units",
        n_prompts=1,
        n_sampling_units=1,
        se=None,
        ci_low=None,
        ci_high=None,
    )
    html = _api().render_comparison_report(report)
    document = _Document()
    document.feed(html)
    text = " ".join(document.text).lower()
    assert "partial overlap" in text
    assert "shared prompts" in text
    assert "fewer than two sampling units" in text
    assert "b - a interval method: unavailable" in text
    assert "+100.00 pp" in text
    assert "difference method:" not in text
    assert "unavailable" in text


def test_continuous_values_use_native_units(report):
    feature = report["features"][0]
    feature.update(feature="resp_chars", kind="continuous")
    feature["a"].update(mean=100.0, ci_low=100.0, ci_high=100.0, positive_prompts=None)
    feature["b"].update(mean=101.0, ci_low=101.0, ci_high=101.0, positive_prompts=None)
    feature["citations"] = {
        side: {"status": "unsupported_kind", "items": []} for side in ("a", "b")
    }
    html = _api().render_comparison_report(report)
    document = _Document()
    document.feed(html)
    text = " ".join(document.text)
    assert "native units" in text
    assert "100.00" in text and "+1.00" in text
    assert "Continuous features have no selected citations" in text


@pytest.mark.parametrize(
    "name",
    [
        "n_assert",
        "n_calls",
        "n_classdefs",
        "n_funcdefs",
        "n_imports",
        "n_print",
        "n_return",
        "n_try",
    ],
)
def test_existing_continuous_ast_features_are_valid(report, name):
    feature = report["features"][0]
    feature.update(feature=name, kind="continuous")
    for side in ("a", "b"):
        feature[side]["positive_prompts"] = None
        feature["citations"][side] = {"status": "unsupported_kind", "items": []}
    assert _api().validate_comparison_report(report) == report


def test_original_prompt_aliases_can_share_one_positive_exact_prompt(report):
    feature = report["features"][0]
    report["sides"][1]["n_rollouts"] = 3
    feature["b"].update(
        n_rollouts=3, mean=0.5, ci_low=0.5, ci_high=0.5, positive_prompts=1
    )
    for key in ("difference", "paired"):
        feature[key].update(estimate=0.5, ci_low=0.5, ci_high=0.5)
    feature["citations"]["b"]["items"][1].update(
        prompt_id="p1-alias", selection_reason="additional_positive"
    )
    assert _api().validate_comparison_report(report) == report


def test_numeric_overflow_preserves_unavailable_effect_and_finite_sides(report):
    feature = report["features"][0]
    feature.update(feature="scalar:reward", kind="continuous", direction="unavailable")
    for side, mean in (("a", -1e308), ("b", 1e308)):
        feature[side].update(
            mean=mean, ci_low=mean, ci_high=mean, positive_prompts=None
        )
        feature["citations"][side] = {"status": "unsupported_kind", "items": []}
    for key in ("difference", "paired"):
        feature[key].update(
            estimate=None,
            se=None,
            ci_low=None,
            ci_high=None,
            status="numeric_unavailable",
        )
    feature["difference"]["method"] = "unavailable"
    assert json.loads(_api().comparison_json(report)) == report


def test_numeric_interval_overflow_retains_finite_measurements(report):
    feature = report["features"][0]
    feature.update(feature="scalar:reward", kind="continuous")
    for side in ("a", "b"):
        feature[side].update(
            se=1e308,
            ci_low=None,
            ci_high=None,
            positive_prompts=None,
            status="numeric_unavailable",
        )
        feature["citations"][side] = {"status": "unsupported_kind", "items": []}
    for key in ("difference", "paired"):
        feature[key].update(
            se=1e308, ci_low=None, ci_high=None, status="numeric_unavailable"
        )
    feature["difference"]["method"] = "unavailable"
    assert _api().validate_comparison_report(report) == report


def test_paired_binary_effect_cannot_exceed_prevalence_range(report):
    report["prompt_mix"].update(shared_prompts=1, only_a_prompts=1, only_b_prompts=1)
    feature = report["features"][0]
    feature["difference"].update(
        method="unavailable",
        status="partial_overlap",
        se=None,
        ci_low=None,
        ci_high=None,
    )
    feature["paired"].update(
        estimate=2.0,
        n_prompts=1,
        n_sampling_units=1,
        status="insufficient_units",
        se=None,
        ci_low=None,
        ci_high=None,
    )
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)


def test_paired_unit_count_cannot_exceed_either_side(report):
    feature = report["features"][0]
    for side in report["sides"]:
        side["n_sampling_units"] = 1
        feature[side["side"]].update(
            n_sampling_units=1,
            status="insufficient_units",
            se=None,
            ci_low=None,
            ci_high=None,
        )
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)


def test_canonical_citation_metadata_agrees_across_features(report):
    other = copy.deepcopy(report["features"][0])
    other.update(feature="raises", order=2)
    other["citations"]["b"]["items"][0]["reward"] = 12.0
    report["features"].append(other)
    with pytest.raises(ValueError, match="comparison report"):
        _api().validate_comparison_report(report)
