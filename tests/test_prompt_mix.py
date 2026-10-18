from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


def make(rows):
    records = []
    for i, (run, step, prompt_id, text, value) in enumerate(rows):
        records.append(
            {
                "run": run,
                "step": step,
                "sample_id": str(i),
                "prompt_id": prompt_id,
                "prompt": text,
                "response": "",
                "reward": float(value),
            }
        )
    rollouts = pd.DataFrame(records)
    dense = rollouts[["run", "sample_id"]].copy()
    dense["flag"] = [r[-1] for r in rows]
    dense["size"] = dense.flag * 10
    sparse = dense.loc[dense.flag.eq(1), ["run", "sample_id"]].assign(feature="s:flag")
    catalog = pd.DataFrame(
        [
            ("flag", "structural", "binary"),
            ("s:flag", "structural", "binary"),
            ("s:absent", "structural", "binary"),
            ("size", "structural", "continuous"),
        ],
        columns=["feature", "tier", "kind"],
    )
    return rollouts, SimpleNamespace(dense=dense, sparse=sparse, catalog=catalog)


def compare(rollouts, features, **kwargs):
    from rolloutdiff.prompt_mix import prompt_mix

    return prompt_mix(
        rollouts,
        features,
        **({"run": "r", "feature": "flag", "before": (1, 1), "after": (2, 2)} | kwargs),
    )


def example():
    return make(
        [("r", 1, "a", "alpha", 0)] * 3
        + [("r", 1, "b", "beta", 1)]
        + [("r", 2, "a", "alpha", 1)]
        + [("r", 2, "b", "beta", 0)] * 3
        + [("r", 2, "c", "gamma", 0)]
    )


def test_breadth_counts_distinct_after_prompts_and_pairs_receive_equal_weight():
    result = compare(*example())
    assert (result.positive_prompts, result.total_prompts) == (1, 3)
    assert result.shared_prompts == 2
    assert result.shared_change == 0
    assert result.shared_se == pytest.approx(1)
    detail = result.per_prompt.set_index("prompt_id")
    assert detail.before_n.tolist() == [3, 1, 0]
    assert detail.after_n.tolist() == [1, 3, 1]
    assert detail.loc["a", "change"] == 1
    assert detail.loc["b", "change"] == -1
    assert pd.isna(detail.loc["c", "change"])
    assert pd.isna(detail.loc["c", "before_mean"])
    assert detail.loc["c", "after_mean"] == 0


def test_window_means_pool_rollouts_within_each_prompt_not_steps():
    rows = [("r", 1, "a", "alpha", 1)] + [("r", 2, "a", "alpha", 0)] * 9
    rows += [("r", 3, "a", "alpha", 1), ("r", 3, "b", "beta", 0)]
    result = compare(*make(rows), before=(1, 2), after=(3, 3))
    assert result.per_prompt.set_index("prompt_id").loc[
        "a", "before_mean"
    ] == pytest.approx(0.1)
    assert result.shared_change == pytest.approx(0.9)
    assert result.shared_se is None
    assert result.positive_prompts == 1


def test_zero_shared_prompts_is_unavailable_not_zero_change():
    result = compare(*make([("r", 1, "a", "alpha", 1), ("r", 2, "b", "beta", 0)]))
    assert result.shared_prompts == 0
    assert result.shared_change is None
    assert result.shared_se is None
    assert result.positive_prompts == 0


def test_one_shared_prompt_has_change_but_no_uncertainty():
    result = compare(*make([("r", 1, "a", "alpha", 0), ("r", 2, "a", "alpha", 0)]))
    assert result.shared_prompts == 1
    assert result.shared_change == 0
    assert result.shared_se is None


def test_dense_sparse_equivalence_and_absent_features():
    rollouts, features = example()
    dense = compare(rollouts, features)
    sparse = compare(rollouts, features, feature="s:flag")
    assert_frame_equal(dense.per_prompt, sparse.per_prompt)
    assert (dense.positive_prompts, dense.shared_change, dense.shared_se) == (
        sparse.positive_prompts,
        sparse.shared_change,
        sparse.shared_se,
    )
    absent = compare(rollouts, features, feature="s:absent")
    assert absent.positive_prompts == 0
    assert absent.shared_change == 0
    assert absent.shared_se == 0


def test_join_identity_and_run_isolation():
    rollouts, features = example()
    expected = compare(rollouts, features)
    extra = rollouts.assign(run="other", reward=0)
    rollouts = pd.concat([rollouts, extra]).sample(frac=1, random_state=1)
    features.dense = pd.concat(
        [features.dense, features.dense.assign(run="other", flag=0)]
    ).sample(frac=1, random_state=2)
    features.sparse = features.sparse.sample(frac=1, random_state=3)
    result = compare(rollouts, features)
    assert_frame_equal(result.per_prompt, expected.per_prompt)
    assert_frame_equal(result.prompt_shift, expected.prompt_shift)
    assert result.shared_change == expected.shared_change


@pytest.mark.parametrize("feature", ["size", "scalar:reward"])
def test_continuous_features_do_not_invent_a_presence_threshold(feature):
    result = compare(*example(), feature=feature)
    assert result.kind == "continuous"
    assert result.positive_prompts is None
    assert result.total_prompts == 3
    assert result.shared_change == 0


def test_prompt_length_distribution_shift_can_have_zero_mean_change():
    rows = [
        ("r", 1, "a", "aa", 0),
        ("r", 1, "b", "bbbbbb", 0),
        ("r", 2, "c", "cccc", 0),
        ("r", 2, "d", "dddd", 0),
    ]
    result = compare(*make(rows))
    length = result.prompt_shift.set_index("descriptor").loc["prompt_chars"]
    assert length["before"] == length["after"] == 4
    assert length.delta == 0
    assert length.distance == 0.5
    assert length.statistic == "ks"


def test_prompt_descriptors_count_prompts_once_and_keywords_are_literal():
    rows = [("r", 1, "a", "C++ graph", 0)] * 20 + [("r", 1, "b", "graphs", 0)]
    rows += [("r", 2, "a", "C++ graph", 0), ("r", 2, "b", "graphs", 0)] * 3
    result = compare(*make(rows), keywords=("c++", "graph"))
    shifts = result.prompt_shift.set_index("descriptor")
    for descriptor in ("keyword:c++", "keyword:graph"):
        assert (
            shifts.loc[descriptor, "before"] == shifts.loc[descriptor, "after"] == 0.5
        )
        assert shifts.loc[descriptor, "distance"] == 0
        assert shifts.loc[descriptor, "n_before"] == 2
    assert shifts.loc["prompt_chars", "distance"] == 0


def test_keyword_shift_reports_observed_prevalence():
    result = compare(
        *make([("r", 1, "a", "graph", 0), ("r", 2, "b", "number", 0)]),
        keywords=["graph"],
    )
    keyword = result.prompt_shift.set_index("descriptor").loc["keyword:graph"]
    assert (keyword["before"], keyword["after"], keyword.delta, keyword.distance) == (
        1,
        0,
        -1,
        1,
    )


def test_difficulty_uses_only_known_tags_and_reports_missing_coverage():
    rows = [
        ("r", 1, "a", "a", 0),
        ("r", 1, "b", "b", 0),
        ("r", 2, "c", "c", 0),
        ("r", 2, "d", "d", 0),
    ]
    result = compare(*make(rows), difficulty={"a": "easy", "b": "hard", "c": "hard"})
    shift = result.prompt_shift.set_index("descriptor").loc["difficulty"]
    assert shift.distance == 0.5
    assert shift.statistic == "total_variation"
    assert (
        shift.n_before,
        shift.n_after,
        shift.missing_before,
        shift.missing_after,
    ) == (2, 1, 0, 1)
    assert pd.isna(shift["before"]) and pd.isna(shift["after"])
    assert result.per_prompt.set_index("prompt_id").loc["c", "difficulty"] == "hard"


def test_difficulty_without_known_values_is_unavailable():
    result = compare(*example(), difficulty={})
    shift = result.prompt_shift.set_index("descriptor").loc["difficulty"]
    assert pd.isna(shift.distance)
    assert shift.missing_before == 2 and shift.missing_after == 3


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"before": (2, 1)}, "window"),
        ({"before": (True, 1)}, "window"),
        ({"before": (-1, 1)}, "window"),
        ({"after": (1, 2)}, "overlap"),
        ({"after": (0, 0)}, "order"),
        ({"after": (3, 4)}, "empty"),
        ({"run": "missing"}, "run"),
        ({"feature": "missing"}, "feature"),
        ({"keywords": "graph"}, "keywords"),
        ({"keywords": [""]}, "keyword"),
        ({"keywords": ["Graph", "graph"]}, "keyword"),
        ({"difficulty": {"missing": "easy"}}, "difficulty"),
        ({"difficulty": {"a": 1}}, "difficulty"),
    ],
)
def test_invalid_requests_fail_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        compare(*example(), **kwargs)


def test_invalid_feature_tables_cannot_silently_change_the_evidence():
    rollouts, features = example()
    features.sparse = pd.concat([features.sparse, features.sparse])
    with pytest.raises(ValueError, match="duplicate"):
        compare(rollouts, features)


def test_public_api_uses_extracted_features_end_to_end():
    from rolloutdiff import featurize, from_table, prompt_mix

    raw = pd.DataFrame(
        {
            "step": [1, 1, 2, 2],
            "prompt": ["graph problem", "array problem"] * 2,
            "response": [
                "plain prose.",
                "plain prose.",
                "def run_tests():\n    pass",
                "plain prose.",
            ],
            "reward": [0, 0, 1, 0],
        }
    )
    rollouts = from_table(raw, run="r")
    result = prompt_mix(
        rollouts,
        featurize(rollouts),
        run="r",
        feature="def:run_tests",
        before=(1, 1),
        after=(2, 2),
        keywords=("graph",),
    )
    assert result.positive_prompts == 1
    assert result.shared_change == 0.5
    assert result.shared_se == 0.5
    assert result.prompt_shift.distance.eq(0).all()


def test_difficulty_distance_unavailable_when_only_one_window_has_known_tags():
    data = make([("r", 1, "a", "alpha", 0), ("r", 2, "b", "beta", 0)])
    result = compare(*data, difficulty={"a": "easy"})
    shift = result.prompt_shift.set_index("descriptor").loc["difficulty"]
    assert pd.isna(shift.distance)
    assert (
        shift.n_before,
        shift.n_after,
        shift.missing_before,
        shift.missing_after,
    ) == (1, 0, 0, 1)


def test_zero_difficulty_distance_does_not_hide_changed_coverage():
    data = make(
        [("r", 1, "a", "alpha", 0), ("r", 2, "b", "beta", 0), ("r", 2, "c", "gamma", 0)]
    )
    result = compare(*data, difficulty={"a": "easy", "b": "easy"})
    shift = result.prompt_shift.set_index("descriptor").loc["difficulty"]
    assert shift.distance == 0
    assert (
        shift.n_before,
        shift.n_after,
        shift.missing_before,
        shift.missing_after,
    ) == (1, 1, 0, 1)


def test_difficulty_mapping_defaults_do_not_invent_tags_or_mutate_input():
    from collections import defaultdict

    tags = defaultdict(lambda: "easy", {"a": "hard"})
    data = make([("r", 1, "a", "alpha", 0), ("r", 2, "b", "beta", 0)])
    result = compare(*data, difficulty=tags)
    shift = result.prompt_shift.set_index("descriptor").loc["difficulty"]
    assert shift.n_after == 0
    assert shift.missing_after == 1
    assert pd.isna(shift.distance)
    assert dict(tags) == {"a": "hard"}
