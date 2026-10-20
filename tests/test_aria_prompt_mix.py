"""Opt-in named-feature prompt-mix checks."""

import pytest
from _aria import load_aria

from rolloutdiff.prompt_mix import prompt_mix

pytestmark = pytest.mark.local_data
_KEYWORDS = ("array", "graph")

# Fixed comparison windows: the report places bisect at 81 without an archived
# detector record; archived run_tests onset is 110 and alert is 112 (three-step
# sustain).
_CASES = [
    {
        "feature": "call:bisect.bisect_left",
        "before": (78, 80),
        "after": (81, 83),
        "prompts": 48,
        "positive_before": 4,
        "positive_after": 3,
        "positive_rows_before": 40,
        "positive_rows_after": 18,
        "shared": 0,
        "shared_change": None,
        "shared_se": None,
        "chars_before": 1421.5,
        "chars_after": 1463.9791666666667,
        "length_distance": 0.14583333333333337,
        "array_counts": (31, 32),
        "graph_counts": (0, 1),
    },
    {
        "feature": "def:run_tests",
        "before": (107, 109),
        "after": (110, 112),
        "prompts": 48,
        "positive_before": 14,
        "positive_after": 29,
        "positive_rows_before": 32,
        "positive_rows_after": 107,
        "shared": 0,
        "shared_change": None,
        "shared_se": None,
        "chars_before": 1520.25,
        "chars_after": 1381.5416666666667,
        "length_distance": 0.20833333333333331,
        "array_counts": (27, 24),
        "graph_counts": (0, 3),
    },
    {
        "feature": "def:run_tests",
        "before": (80, 100),
        "after": (140, 160),
        "prompts": 336,
        "positive_before": 42,
        "positive_after": 336,
        "positive_rows_before": 75,
        "positive_rows_after": 5321,
        "shared": 119,
        "shared_change": 0.9732142857142857,
        "shared_se": 0.007843357149100373,
        "chars_before": 1547.044642857143,
        "chars_after": 1500.3720238095239,
        "length_distance": 0.07440476190476192,
        "array_counts": (196, 188),
        "graph_counts": (11, 9),
    },
]


@pytest.fixture(scope="module")
def aria():
    rollouts, features = load_aria([(78, 100), (107, 112), (140, 160)])
    assert len(rollouts) == 12800
    assert rollouts.step.nunique() == 50
    return rollouts, features


@pytest.mark.parametrize(
    "case",
    _CASES,
    ids=["bisect_report_anchor", "run_tests_local_onset", "run_tests_plan_windows"],
)
def test_local_aria_prompt_mix(aria, case):
    rollouts, features = aria
    result = prompt_mix(
        rollouts,
        features,
        run="aria",
        feature=case["feature"],
        before=case["before"],
        after=case["after"],
        keywords=_KEYWORDS,
    )
    assert result.positive_prompts == case["positive_after"]
    assert result.total_prompts == case["prompts"]
    assert result.shared_prompts == case["shared"]
    if case["shared"] == 0:
        assert result.shared_change is None
        assert result.shared_se is None
    else:
        assert result.shared_change == pytest.approx(case["shared_change"])
        assert result.shared_se == pytest.approx(case["shared_se"])
    detail = result.per_prompt
    assert detail.before_n.gt(0).sum() == case["prompts"]
    assert detail.before_mean.gt(0).sum() == case["positive_before"]
    assert detail.before_n.sum() == detail.after_n.sum() == case["prompts"] * 16

    # Literal expected counts were computed independently from source responses
    # with extract() and direct problem_id grouping, without prompt_mix().
    presence = features.sparse.loc[features.sparse.feature.eq(case["feature"])].merge(
        rollouts[["run", "sample_id", "step"]],
        on=["run", "sample_id"],
        validate="one_to_one",
    )
    assert presence.step.between(*case["before"]).sum() == case["positive_rows_before"]
    assert presence.step.between(*case["after"]).sum() == case["positive_rows_after"]

    shift = result.prompt_shift.set_index("descriptor")
    length = shift.loc["prompt_chars"]
    assert length.statistic == "ks"
    assert length["before"] == pytest.approx(case["chars_before"])
    assert length["after"] == pytest.approx(case["chars_after"])
    assert length.delta == pytest.approx(case["chars_after"] - case["chars_before"])
    assert length.distance == pytest.approx(case["length_distance"])
    for keyword in _KEYWORDS:
        row = shift.loc[f"keyword:{keyword}"]
        a, b = case[f"{keyword}_counts"]
        assert row.statistic == "absolute_prevalence"
        assert row["before"] == pytest.approx(a / case["prompts"])
        assert row["after"] == pytest.approx(b / case["prompts"])
        assert row.delta == pytest.approx((b - a) / case["prompts"])
        assert row.distance == pytest.approx(abs(b - a) / case["prompts"])
    assert shift.n_before.eq(case["prompts"]).all()
    assert shift.n_after.eq(case["prompts"]).all()
    assert shift.missing_before.eq(0).all()
    assert shift.missing_after.eq(0).all()
