"""User-selected comparisons preserve prompt weights and resolvable evidence."""

import copy
import hashlib
import json

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def rows(values, *, run="r"):
    """Rows are (step, prompt_id, prompt, positive[, reward])."""
    return pd.DataFrame(
        [
            {
                "run": run,
                "sample_id": str(index),
                "step": value[0],
                "prompt_id": value[1],
                "prompt": value[2],
                "response": "def run_tests():\n    pass" if value[3] else "plain text",
                "reward": float(value[4] if len(value) > 4 else value[3]),
            }
            for index, value in enumerate(values)
        ]
    )


def compare(rollouts_a, rollouts_b=None, **kwargs):
    from rolloutdiff.comparison import compare_rollouts

    if rollouts_b is None:
        kwargs = {"a": (0, 0), "b": (1, 1)} | kwargs
    return compare_rollouts(rollouts_a, rollouts_b, **kwargs)


def feature(report, name="def:run_tests"):
    return next(row for row in report["features"] if row["feature"] == name)


def units(frame, names):
    return frame[["run", "prompt_id"]].drop_duplicates().assign(sampling_unit_id=names)


def test_equal_prompt_weights_pool_completions_across_steps_and_pair_changes():
    frame = rows(
        [(0, "p", "alpha", 1)]
        + [(1, "p", "alpha", 0)] * 9
        + [(0, "q", "beta", 1)]
        + [(2, "p", "alpha", 1), (2, "q", "beta", 0)]
    )
    report = compare(frame, a=(0, 1), b=(2, 2))
    value = feature(report)
    assert value["a"]["mean"] == pytest.approx(0.55)
    assert value["a"]["se"] == pytest.approx(0.45)
    assert value["a"]["n_rollouts"] == 11
    assert value["b"]["mean"] == 0.5
    assert value["difference"]["estimate"] == pytest.approx(-0.05)
    assert value["difference"]["method"] == "paired"
    assert value["paired"]["estimate"] == pytest.approx(-0.05)
    assert value["paired"]["se"] == pytest.approx(0.95)
    assert value["paired"]["ci_low"] == pytest.approx(-1.9119657853130514)
    assert value["a"]["positive_prompts"] == 2


def test_prompt_schedule_change_keeps_shared_subset_effect_separate():
    frame = rows(
        [
            (0, "a", "a", 0),
            (0, "b", "b", 1),
            (1, "a", "a", 0),
            (1, "b", "b", 1),
            (1, "c", "c", 1),
        ]
    )
    report = compare(frame)
    value = feature(report)
    assert value["difference"]["estimate"] == pytest.approx(1 / 6)
    assert value["difference"]["status"] == "partial_overlap"
    assert value["difference"]["se"] is None
    assert value["paired"]["estimate"] == value["paired"]["se"] == 0
    assert report["prompt_mix"]["shared_prompts"] == 2
    assert report["prompt_mix"]["only_b_prompts"] == 1


def test_zero_shared_prompts_uses_independent_contrast_and_explicit_unavailable_pair():
    frame = rows(
        [(0, "a", "a", 0), (0, "b", "b", 1), (1, "c", "c", 1), (1, "d", "d", 1)]
    )
    report = compare(frame)
    value = feature(report)
    assert value["difference"]["estimate"] == 0.5
    assert value["difference"]["se"] == 0.5
    assert value["difference"]["method"] == "independent"
    assert value["paired"]["status"] == "no_shared_prompts"
    assert value["paired"]["estimate"] is None
    assert value["paired"]["n_prompts"] == 0


def test_one_shared_prompt_has_effect_but_unavailable_uncertainty():
    value = feature(compare(rows([(0, "a", "a", 0), (1, "a", "a", 1)])))
    assert value["a"]["status"] == "insufficient_units"
    assert value["paired"]["estimate"] == 1
    assert value["paired"]["se"] is None
    assert value["difference"]["status"] == "insufficient_units"


def test_unequal_clusters_preserve_prompt_mean_and_cluster_residual_se():
    frame = rows(
        [(step, f"p{i}", f"text{i}", i == 3) for step in (0, 1) for i in range(4)]
    )
    report = compare(frame, sampling_units_a=units(frame, ["x", "x", "x", "y"]))
    value = feature(report)
    assert value["a"]["mean"] == 0.25
    assert value["a"]["se"] == 0.375
    assert value["a"]["n_sampling_units"] == 2
    assert value["b"]["se"] == 0.375
    assert value["paired"]["n_sampling_units"] == 2
    assert value["paired"]["se"] == 0


def test_same_run_shared_units_preclude_independent_interval_without_shared_prompts():
    frame = rows(
        [(0, "a", "a", 0), (0, "b", "b", 1), (1, "c", "c", 1), (1, "d", "d", 1)]
    )
    report = compare(frame, sampling_units_a=units(frame, ["x", "y", "x", "y"]))
    value = feature(report)
    assert value["difference"]["status"] == "shared_units"
    assert value["difference"]["se"] is None


def test_two_dumps_pair_content_with_different_ids_and_preserve_source_identities():
    a = rows([(0, "a", "alpha", 0), (0, "b", "beta", 0)])
    b = rows([(7, "x", "alpha", 1), (7, "y", "beta", 0)])
    report = compare(a, b)
    value = feature(report)
    assert report["mode"] == "dumps"
    assert report["prompt_mix"]["shared_prompts"] == 2
    assert value["paired"]["estimate"] == 0.5
    assert value["paired"]["se"] == 0.5
    citation = value["citations"]["b"]["items"][0]
    assert (citation["run"], citation["sample_id"], citation["prompt_id"]) == (
        "r",
        "0",
        "x",
    )
    assert (
        citation["response_sha256"]
        == hashlib.sha256(b.iloc[0].response.encode()).hexdigest()
    )


def test_reused_prompt_id_different_text_or_whitespace_does_not_pair():
    a = rows([(0, "a", "alpha", 0), (0, "b", "beta", 0)])
    b = rows([(0, "a", "alpha ", 1), (0, "b", "other", 1)])
    assert compare(a, b)["prompt_mix"]["shared_prompts"] == 0


def test_identical_rendered_chat_text_with_different_roles_does_not_pair():
    a = rows([(0, "a", "alpha", 0)])
    b = rows([(0, "a", "alpha", 1)])
    a["prompt_messages"] = [[{"role": "user", "content": "alpha"}]]
    b["prompt_messages"] = [[{"role": "system", "content": "alpha"}]]
    assert compare(a, b)["prompt_mix"]["shared_prompts"] == 0


def test_exact_content_aliases_are_pooled_before_prompt_weighting():
    a = rows([(0, "a", "alpha", 0), (0, "alias", "alpha", 1), (0, "b", "beta", 0)])
    b = rows([(0, "x", "alpha", 1), (0, "y", "beta", 0)])
    report = compare(a, b)
    value = feature(report)
    assert report["sides"][0]["n_prompts"] == 2
    assert value["a"]["mean"] == 0.25
    assert value["paired"]["estimate"] == 0.25


def test_aliases_with_conflicting_explicit_units_are_rejected():
    a = rows([(0, "a", "alpha", 0), (0, "alias", "alpha", 1)])
    b = rows([(0, "b", "alpha", 1)])
    with pytest.raises(ValueError, match="sampling"):
        compare(a, b, sampling_units_a=units(a, ["x", "y"]))


def test_cross_dump_unit_labels_alone_do_not_prove_dependence():
    a = rows([(0, "a", "alpha", 0), (0, "b", "beta", 1)])
    b = rows([(0, "c", "gamma", 1), (0, "d", "delta", 1)])
    value = feature(
        compare(
            a,
            b,
            sampling_units_a=units(a, ["x", "y"]),
            sampling_units_b=units(b, ["x", "y"]),
        )
    )
    assert value["difference"]["method"] == "independent"


def test_conflicting_paired_partitions_preserve_effect_without_uncertainty():
    a = rows([(0, str(i), str(i), 0) for i in range(3)])
    b = rows([(0, str(i), str(i), 1) for i in range(3)])
    value = feature(
        compare(
            a,
            b,
            sampling_units_a=units(a, ["x", "x", "y"]),
            sampling_units_b=units(b, ["a", "b", "b"]),
        )
    )
    assert value["paired"]["estimate"] == 1
    assert value["paired"]["status"] == "incompatible_units"
    assert value["paired"]["n_sampling_units"] is None
    assert value["difference"]["status"] == "incompatible_units"


def test_contradictory_same_run_sampling_maps_are_rejected():
    frame = rows([(step, str(i), str(i), i == 1) for step in (0, 1) for i in range(2)])
    with pytest.raises(ValueError, match="sampling"):
        compare(
            frame,
            sampling_units_a=units(frame, ["x", "x"]),
            sampling_units_b=units(frame, ["x", "y"]),
        )


def test_joint_catalog_observes_feature_only_present_on_one_side():
    a = rows([(0, "a", "alpha", 0), (0, "b", "beta", 0)])
    b = rows([(0, "a", "alpha", 1), (0, "b", "beta", 1)])
    value = feature(compare(a, b))
    assert value["a"]["mean"] == value["a"]["se"] == 0
    assert value["b"]["mean"] == 1
    assert value["citations"]["a"] == {"status": "no_positive_rollouts", "items": []}
    dense = feature(compare(a, b), "has_code")
    for key in ("a", "b", "difference", "paired"):
        assert dense[key] == value[key]


def test_order_puts_binary_effects_before_continuous_native_units():
    frame = rows(
        [
            (0, "a", "alpha", 0, 0),
            (0, "b", "beta", 0, 0),
            (1, "a", "alpha", 1, 1000),
            (1, "b", "beta", 0, 1000),
        ]
    )
    report = compare(frame)
    binary = [item for item in report["features"] if item["kind"] == "binary"]
    continuous = [item for item in report["features"] if item["kind"] == "continuous"]
    assert report["features"] == binary + continuous
    assert [
        (abs(item["difference"]["estimate"]), item["feature"]) for item in binary
    ] == sorted(
        [(abs(item["difference"]["estimate"]), item["feature"]) for item in binary],
        key=lambda item: (-item[0], item[1]),
    )
    assert [item["feature"] for item in continuous] == sorted(
        item["feature"] for item in continuous
    )
    reward = feature(report, "scalar:reward")
    assert reward["a"]["positive_prompts"] is None
    assert reward["citations"]["a"]["status"] == "unsupported_kind"


def test_citations_use_earliest_positive_then_prompt_diversity_and_ignore_reward():
    frame = rows(
        [
            (0, "p", "p", 0),
            (1, "p", "p", 1, -4),
            (1, "p", "p", 1, 100),
            (1, "q", "q", 1, -9),
            (1, "r", "r", 1, 0),
        ]
    )
    value = feature(compare(frame))
    citations = value["citations"]["b"]["items"]
    assert [item["sample_id"] for item in citations] == ["1", "3", "4"]
    assert [item["selection_reason"] for item in citations] == [
        "earliest_positive",
        "prompt_diversity",
        "prompt_diversity",
    ]


def test_remaining_positive_citations_fill_when_fewer_than_three_prompts():
    frame = rows(
        [(0, "p", "p", 0), (1, "p", "p", 1), (1, "p", "p", 1), (1, "p", "p", 1)]
    )
    citations = feature(compare(frame))["citations"]["b"]["items"]
    assert [item["selection_reason"] for item in citations] == [
        "earliest_positive",
        "additional_positive",
        "additional_positive",
    ]


def test_reordering_and_index_changes_preserve_report_and_inputs():
    frame = rows(
        [(step, str(i), str(i), step == i) for step in (0, 1) for i in range(3)]
    )
    original = frame.copy(deep=True)
    first = compare(frame, keywords=("0",))
    second = compare(
        frame.sample(frac=1, random_state=71).set_axis(range(40, 46)), keywords=("0",)
    )
    assert first == second
    assert_frame_equal(frame, original)
    encoded = json.dumps(first, allow_nan=False)
    assert "def run_tests():" not in encoded


def test_full_run_fingerprint_detects_changes_outside_selected_windows():
    from rolloutdiff.comparison import (
        canonical_fingerprint,
        resolve_comparison_citations,
    )

    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1), (9, "b", "b", 0)])
    report = compare(frame)
    assert report["sides"][0]["source"]["canonical_sha256"] == canonical_fingerprint(
        frame
    )
    altered = frame.copy()
    altered.loc[2, "response"] = "changed outside comparison"
    with pytest.raises(ValueError, match="fingerprint"):
        resolve_comparison_citations(report, altered, side="b", feature="def:run_tests")


def test_citation_resolution_returns_original_responses_without_mutating_report():
    from rolloutdiff.comparison import resolve_comparison_citations

    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1)])
    report = compare(frame)
    original = copy.deepcopy(report)
    resolved = resolve_comparison_citations(
        report, frame.iloc[::-1], side="b", feature="def:run_tests"
    )
    assert resolved["side"] == "b"
    assert resolved["status"] == "selected"
    assert resolved["items"][0]["response"] == frame.iloc[1].response
    assert report == original
    absent = resolve_comparison_citations(
        report, frame, side="a", feature="def:run_tests"
    )
    assert absent["status"] == "no_positive_rollouts" and not absent["items"]


@pytest.mark.parametrize(
    "field,value",
    [("reward", 77.0), ("prompt_id", "different"), ("response_sha256", "0" * 64)],
)
def test_citation_resolution_rejects_tampered_metadata(field, value):
    from rolloutdiff.comparison import resolve_comparison_citations

    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1)])
    report = compare(frame)
    feature(report)["citations"]["b"]["items"][0][field] = value
    with pytest.raises(ValueError):
        resolve_comparison_citations(report, frame, side="b", feature="def:run_tests")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": None},
        {"a": (0, 1)},
        {"a": (2, 2)},
        {"a": (-1, 0)},
        {"a": (True, 0)},
        {"b": (9, 10)},
        {"run_a": "missing"},
        {"keywords": "a"},
    ],
)
def test_invalid_selection_fails_explicitly(kwargs):
    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1)])
    with pytest.raises(ValueError):
        compare(frame, **kwargs)


def test_multirun_inputs_require_selection_and_fingerprint_only_selected_run():
    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1)])
    both = pd.concat([frame, frame.assign(run="other")], ignore_index=True)
    with pytest.raises(ValueError, match="run"):
        compare(both)
    assert compare(both, run_a="r") == compare(frame)


def test_extreme_finite_reward_overflow_never_produces_nonfinite_json():
    a = rows([(0, "a", "a", 0, -1e308), (0, "b", "b", 1, -1e308)])
    b = rows([(0, "a", "a", 1, 1e308), (0, "b", "b", 1, 1e308)])
    report = compare(a, b)
    value = feature(report, "scalar:reward")
    assert value["difference"]["status"] == "numeric_unavailable"
    assert value["difference"]["estimate"] is None
    assert value["direction"] == "unavailable"
    json.dumps(report, allow_nan=False)


def test_resolver_rejects_invalid_side_type_with_value_error():
    from rolloutdiff.comparison import resolve_comparison_citations

    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1)])
    report = compare(frame)
    with pytest.raises(ValueError, match="side"):
        resolve_comparison_citations(report, frame, side=[], feature="def:run_tests")


def test_resolver_rejects_a_valid_identity_that_does_not_express_the_feature():
    from rolloutdiff.comparison import resolve_comparison_citations

    frame = rows([(0, "a", "a", 0), (1, "a", "a", 1), (1, "a", "a", 0)])
    report = compare(frame)
    citation = feature(report)["citations"]["b"]["items"][0]
    citation.update(
        sample_id="2",
        reward=0.0,
        response_sha256=hashlib.sha256(frame.iloc[2].response.encode()).hexdigest(),
    )
    with pytest.raises(ValueError, match="feature"):
        resolve_comparison_citations(report, frame, side="b", feature="def:run_tests")


def test_alias_citations_preserve_ids_without_inventing_prompt_diversity():
    a = rows([(0, "a", "alpha", 0)])
    b = rows([(0, "x", "alpha", 1), (0, "y", "alpha", 1)])
    report = compare(a, b)
    value = feature(report)
    assert value["b"]["positive_prompts"] == 1
    citations = value["citations"]["b"]["items"]
    assert [item["prompt_id"] for item in citations] == ["x", "y"]
    assert [item["selection_reason"] for item in citations] == [
        "earliest_positive",
        "additional_positive",
    ]


def test_overflowing_interval_preserves_finite_estimate_and_se():
    a = rows([(0, "a", "a", 0, -1e308), (0, "b", "b", 1, 1e308)])
    b = rows([(0, "a", "a", 1, 0), (0, "b", "b", 1, 0)])
    report = compare(a, b)
    value = feature(report, "scalar:reward")
    assert value["a"]["status"] == "numeric_unavailable"
    assert value["a"]["mean"] == 0
    assert value["a"]["se"] == 1e308
    assert value["a"]["ci_low"] is value["a"]["ci_high"] is None
    json.dumps(report, allow_nan=False)


def test_full_overlap_cancellation_preserves_paired_change_and_marks_marginal_numeric():
    a = rows([(0, "a", "a", 0, 1e16), (0, "b", "b", 0, 1e16)])
    b = rows([(0, "a", "a", 1, 1e16 + 2), (0, "b", "b", 0, 1e16)])
    report = compare(a, b)
    value = feature(report, "scalar:reward")
    assert value["a"]["mean"] == value["b"]["mean"] == 1e16
    assert value["paired"]["estimate"] == 1
    assert value["paired"]["se"] == 1
    assert value["paired"]["status"] == "available"
    assert value["difference"] == {
        "estimate": 0.0,
        "se": None,
        "ci_low": None,
        "ci_high": None,
        "status": "numeric_unavailable",
        "method": "unavailable",
    }
    assert feature(report)["difference"]["estimate"] == 0.5
    json.dumps(report, allow_nan=False)


def test_full_overlap_paired_overflow_preserves_finite_marginal_without_uncertainty():
    a = rows([(0, "a", "a", 0, -1e308), (0, "b", "b", 0, 1e308)])
    b = rows([(0, "a", "a", 1, 1e308), (0, "b", "b", 0, -1e308)])
    report = compare(a, b)
    value = feature(report, "scalar:reward")
    assert value["a"]["mean"] == value["b"]["mean"] == 0
    assert value["paired"]["estimate"] is None
    assert value["paired"]["se"] is None
    assert value["paired"]["status"] == "numeric_unavailable"
    assert value["difference"] == {
        "estimate": 0.0,
        "se": None,
        "ci_low": None,
        "ci_high": None,
        "status": "numeric_unavailable",
        "method": "unavailable",
    }
    assert feature(report)["difference"]["estimate"] == 0.5
    json.dumps(report, allow_nan=False)


def test_incompatible_paired_units_and_overflow_preserve_unavailable_counts():
    a = rows([(0, str(i), str(i), 0, -1e308) for i in range(3)])
    b = rows([(0, str(i), str(i), 1, 1e308) for i in range(3)])
    report = compare(
        a,
        b,
        sampling_units_a=units(a, ["x", "x", "y"]),
        sampling_units_b=units(b, ["x", "y", "y"]),
    )
    value = feature(report, "scalar:reward")
    assert value["paired"]["status"] == "numeric_unavailable"
    assert value["paired"]["estimate"] is None
    assert value["paired"]["n_sampling_units"] is None
    assert value["paired"]["n_prompts"] == 3
    assert value["difference"]["estimate"] is None
    assert value["difference"]["status"] == "numeric_unavailable"
    assert feature(report)["paired"]["status"] == "incompatible_units"
    json.dumps(report, allow_nan=False)
