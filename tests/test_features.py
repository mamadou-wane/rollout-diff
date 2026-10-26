import pandas as pd
import pytest

from rolloutdiff import features


def rows(*entries):
    return pd.DataFrame(
        [
            {
                "run": run,
                "step": step,
                "sample_id": sample,
                "prompt_id": prompt,
                "prompt": prompt,
                "response": response,
                "reward": reward,
            }
            for run, step, sample, prompt, response, reward in entries
        ],
        columns=[
            "run",
            "step",
            "sample_id",
            "prompt_id",
            "prompt",
            "response",
            "reward",
        ],
    )


def test_plain_prose_survives_without_code():
    result = features.extract("Careful reasoning reveals useful patterns.")
    assert result.dense["has_code"] == 0
    assert result.dense["prose_words"] == 5
    assert {"w:careful", "b:useful patterns"} <= result.sparse


def test_valid_block_survives_an_independent_invalid_block():
    result = features.extract(
        "```python\ndef run_tests():\n    return 1\n```\n```python\ndef broken(\n```"
    )
    assert result.dense["code_parse_fail"] == 1
    assert result.dense["n_code_blocks"] == 2
    assert result.dense["n_funcdefs"] == 1
    assert "def:run_tests" in result.sparse
    assert result.dense.get("defines_test_token") == 1


@pytest.mark.parametrize(
    "name, expected",
    [
        ("test_case", 1),
        ("test_solution", 1),
        ("run_tests", 1),
        ("testCase", 1),
        ("HTTPTests", 1),
        ("run2Tests", 1),
        ("TESTCase", 1),
        ("TEST_CASE", 1),
        ("test_2", 1),
        ("shortestPath", 0),
        ("latest", 0),
        ("contest", 0),
        ("testify", 0),
        ("test2", 0),
        ("test2Case", 0),
        ("testcase", 0),
        ("TESTcase", 0),
        ("βTest", 0),
        ("findRotateSteps", 0),
        ("deleteString", 0),
        ("findMaxConsecutiveOnesInfiniteStream", 0),
        ("solve", 0),
    ],
)
def test_test_indicator_requires_a_complete_name_token_and_preserves_sparse_names(
    name, expected
):
    result = features.extract(f"def {name}():\n    pass\n")
    assert result.dense.get("defines_test_token") == expected
    assert result.dense["n_funcdefs"] == 1
    assert result.sparse == {f"def:{name}"}


@pytest.mark.parametrize(
    "response, count",
    [
        ("def outer():\n    def test_case():\n        pass", 2),
        ("class Suite:\n    def test_case(self):\n        pass", 1),
        ("async def test_case():\n    pass", 1),
        ("async def outer():\n    async def test_case():\n        pass", 2),
        ("if False:\n    def test_case():\n        return 42", 1),
    ],
)
def test_test_token_counts_parsed_definitions_in_all_scopes(response, count):
    result = features.extract(response)
    assert result.dense.get("defines_test_token") == 1
    assert result.dense["n_funcdefs"] == count
    assert "def:test_case" in result.sparse


@pytest.mark.parametrize(
    "response",
    [
        "class TestCase:\n    pass",
        "run_tests()",
        "# def test_case(): pass",
        'text = "def test_case(): pass"',
        "```javascript\ndef test_case(): pass\n```",
        "<think>def test_case(): pass</think>",
        "<thinking>def test_case(): pass</thinking>",
        "def helper():\n    assert True",
        "```python\ndef test_case(\n```",
    ],
)
def test_test_token_ignores_non_definitions_and_unparsed_regions(response):
    result = features.extract(response)
    assert result.dense.get("defines_test_token") == 0
    assert "def:test_case" not in result.sparse


def test_test_token_is_the_only_test_indicator_in_dense_output_and_catalog():
    result = features.featurize(
        rows(
            ("run", 0, "a", "p", "def run_tests():\n    pass", 0),
            ("run", 0, "b", "p", "def shortestPath():\n    pass", 0),
        )
    )
    assert result.dense.filter(like="defines_test").to_dict("list") == {
        "defines_test_token": [1, 0]
    }
    catalog = result.catalog.loc[result.catalog.feature.str.startswith("defines_test")]
    assert catalog.to_dict("records") == [
        {"feature": "defines_test_token", "tier": "structural", "kind": "binary"}
    ]


def test_file_wrappers_and_headers_share_a_vocabulary_without_double_counting():
    result = features.extract("""<file name='pkg/solver.py'>
```python
# filename: pkg/solver.py
def solve():
    return 1
```
</file>
```python
# filename: tests/test_solver.py
def test_solve():
    assert solve() == 1
```""")
    assert {
        "file:solver.py",
        "file:test_solver.py",
        "def:solve",
        "def:test_solve",
    } <= result.sparse
    assert result.dense["n_files_declared"] == 2
    assert result.dense["n_funcdefs"] == 2
    assert result.dense["n_code_blocks"] == 2
    assert result.dense["code_parse_fail"] == 0


@pytest.mark.parametrize(
    "condition, expected",
    [
        ('__name__ == "__main__"', 1),
        ('"__main__" == __name__', 1),
        ('__name__ != "__main__"', 0),
        ('__name__ == "different"', 0),
        ('__name__ == "__main__" == other', 0),
    ],
)
def test_main_guard_requires_equality_to_main(condition, expected):
    result = features.extract(f"if {condition}:\n    main()")
    assert result.dense["has_main_guard"] == expected


def test_ast_vocabulary_is_lexical_and_never_executes_response(tmp_path):
    marker = tmp_path / "must-not-exist"
    result = features.extract(f"""import os
import sys
class Solution:
    def __eq__(self, other):
        return True
sys.exit = lambda: None
open({str(marker)!r}, "w").write("unsafe")
try:
    sys.exit()
except Exception:
    raise SystemExit()
""")
    assert not marker.exists()
    assert {
        "class:Solution",
        "def:__eq__",
        "import:sys",
        "call:sys.exit",
        "setattr:sys.exit",
    } <= result.sparse
    assert result.dense["defines_cmp_dunder"] == 1
    assert result.dense["calls_exit"] == 1
    assert result.dense["file_write"] == 1
    assert result.dense["module_attr_assign"] == 1
    assert result.dense["catches_all"] == 1


def test_complete_thinking_blocks_are_excluded_but_response_length_is_raw():
    text = "<think>secret analysis</think><thinking>hidden notes</thinking>Visible explanation."
    result = features.extract(text)
    assert result.dense["resp_chars"] == len(text)
    assert "w:visible" in result.sparse
    assert "w:secret" not in result.sparse
    assert "w:hidden" not in result.sparse
    assert "w:unfinished" in features.extract("<think>unfinished reasoning").sparse


def test_non_python_fences_do_not_become_prose_or_python():
    result = features.extract(
        "Explanation here.\n```javascript\nconst hidden = true;\n```"
    )
    assert result.dense["has_code"] == 0
    assert "w:explanation" in result.sparse
    assert "w:hidden" not in result.sparse


def test_featurize_preserves_keys_is_label_blind_and_declares_tiers():
    rollouts = rows(
        (
            "run-a",
            0,
            "same",
            "p",
            "Helpful explanation.\n```py\ndef solve():\n    pass\n```",
            1,
        ),
        ("run-b", 0, "same", "p", "print(1)", 0),
    )
    original = rollouts.copy(deep=True)
    result = features.featurize(rollouts, min_prevalence=0)
    pd.testing.assert_frame_equal(rollouts, original)
    assert list(
        result.dense[["run", "sample_id"]].itertuples(index=False, name=None)
    ) == [("run-a", "same"), ("run-b", "same")]
    assert not result.sparse.duplicated(["run", "sample_id", "feature"]).any()
    catalog = result.catalog.set_index("feature")
    assert catalog.loc["def:solve", ["tier", "kind"]].tolist() == [
        "structural",
        "binary",
    ]
    assert catalog.loc["w:helpful", ["tier", "kind"]].tolist() == ["prose", "binary"]
    assert catalog.loc["prose_chars", ["tier", "kind"]].tolist() == [
        "prose",
        "continuous",
    ]
    assert catalog.loc["has_code", "kind"] == "binary"
    rollouts["reward"] = [999, -10]
    changed = features.featurize(rollouts, min_prevalence=0)
    pd.testing.assert_frame_equal(result.dense, changed.dense)
    pd.testing.assert_frame_equal(result.sparse, changed.sparse)
    pd.testing.assert_frame_equal(result.catalog, changed.catalog)


def test_sparse_threshold_uses_prompt_balance_and_peak_step():
    entries = [
        ("run", 0, f"a{i}", "prompt-a", "def frequent():\n    pass", 0)
        for i in range(9)
    ]
    entries += [
        ("run", 0, "b", "prompt-b", "def rare():\n    pass", 0),
        ("run", 1, "c", "prompt-c", "def late():\n    pass", 0),
    ]
    result = features.featurize(rows(*entries), min_prevalence=0.5)
    assert {"def:frequent", "def:rare", "def:late"} <= set(result.sparse.feature)
    stricter = features.featurize(rows(*entries), min_prevalence=0.6)
    assert set(stricter.sparse.feature) == {"def:late"}


def test_sparse_threshold_does_not_pool_runs():
    entries = [("large", 0, f"n{i}", f"p{i}", "pass", 0) for i in range(9)]
    entries += [("small", 0, "yes", "p", "def isolated():\n    pass", 0)]
    result = features.featurize(rows(*entries), min_prevalence=0.9)
    assert "def:isolated" in set(result.sparse.feature)


def test_sparse_caps_are_per_tier_and_ties_are_lexical():
    rollouts = rows(
        (
            "run",
            0,
            "a",
            "p",
            "Zebra alpha.\n```py\ndef zebra():\n    pass\ndef alpha():\n    pass\n```",
            0,
        )
    )
    result = features.featurize(rollouts, max_sparse_per_tier=1)
    assert set(result.sparse.feature) == {"def:alpha", "b:zebra alpha"}
    reversed_result = features.featurize(rollouts.iloc[::-1], max_sparse_per_tier=1)
    pd.testing.assert_frame_equal(result.catalog, reversed_result.catalog)


def test_empty_tables_keep_their_schema():
    result = features.featurize(rows())
    assert list(result.sparse.columns) == ["run", "sample_id", "feature"]
    assert {"run", "sample_id", "n_calls", "has_code"} <= set(result.dense.columns)
    assert not result.catalog.empty
    assert result.catalog.feature.is_unique


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_sparse_per_tier": -1},
        {"max_sparse_per_tier": 1.5},
        {"min_prevalence": -0.1},
        {"min_prevalence": 1.1},
    ],
)
def test_invalid_selection_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        features.featurize(rows(), **kwargs)


def test_duplicate_rollout_keys_are_rejected():
    rollouts = rows(("run", 0, "a", "p", "pass", 0), ("run", 1, "a", "p", "pass", 0))
    with pytest.raises(ValueError, match="duplicate"):
        features.featurize(rollouts)


@pytest.mark.parametrize(
    "call, read, write",
    [
        ('Path("x.txt").open("w")', 0, 1),
        ('pathlib.Path("x.txt").open("a")', 0, 1),
        ('Path("x.txt").open()', 1, 0),
        ('Path("x.txt").open(mode="w")', 0, 1),
        ('open("x.txt", "w")', 0, 1),
    ],
)
def test_literal_path_open_uses_its_first_argument_as_mode(call, read, write):
    result = features.extract(call)
    assert result.dense["file_read"] == read
    assert result.dense["file_write"] == write


def test_empty_dense_columns_remain_numeric_for_aggregation():
    result = features.featurize(rows())
    for column in result.dense.columns.difference(["run", "sample_id"]):
        assert pd.api.types.is_numeric_dtype(result.dense[column])


def test_rollout_syntax_warnings_do_not_escape_extraction():
    import warnings

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        result = features.extract("pattern = '\\(' \n")
    assert result.dense["has_code"] == 1
    assert not emitted


@pytest.mark.parametrize("closing", ["````", ""])
def test_longer_or_missing_fence_closer_preserves_python(closing):
    result = features.extract("```python\ndef solve():\n    pass\n" + closing)
    assert result.dense["has_code"] == 1
    assert result.dense["n_funcdefs"] == 1
    assert "def:solve" in result.sparse
    assert "w:python" not in result.sparse


def test_truncated_invalid_python_fence_reports_parse_failure():
    result = features.extract("```python\ndef solve(")
    assert result.dense["has_code"] == 1
    assert result.dense["code_parse_fail"] == 1
    assert "w:def" not in result.sparse


def test_different_fence_character_does_not_close_a_block():
    result = features.extract("```python\ndef solve():\n    pass\n~~~\n")
    assert result.dense["has_code"] == 1
    assert result.dense["code_parse_fail"] == 1
