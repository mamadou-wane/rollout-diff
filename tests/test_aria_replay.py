"""Static replay preparation must never execute fixture or response code."""

import hashlib
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pytest

from rolloutdiff import _aria_replay as replay


def test_static_parse_does_not_echo_response_text_in_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        replay.historical_extract("```python\nvalue = '\\qPRIVATE_RESPONSE'\n```")
    assert not caught


def test_deep_lookup_abstains_without_guessing_an_invocation():
    response = "```python\ndef run_tests():\n return " + "+".join(["1"] * 600) + "\n```"
    result = replay.historical_extract(response)
    assert result["parse_status"] == "lookup_error"
    assert result["requested_call"] is None
    assert result["program"] is not None


def task(tests=None):
    return {
        "id": 7,
        "func_name": "Solution().answer",
        "gt_answer": tests or ["assert Solution().answer(value=[1, 2]) == 3"],
        "setup_code": "from math import inf\n",
        "canonical_solution": "class Solution: pass",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task and should pass all tests"},
        ],
    }


def test_parser_preserves_thinking_and_unclosed_fences():
    result = replay.historical_extract(
        "<think>```python\nx = 1\n```</think>\n```PYTHON\ndef run_tests():\n    return False"
    )
    assert result["program"] == "x = 1\n\ndef run_tests():\n    return False"
    assert result["parse_status"] == "parsed"
    assert result["requested_call"] == "run_tests()"
    assert result["response_has_test_func"] is True


@pytest.mark.parametrize(
    "response",
    ["def run_tests(): pass", "```python\n \n```", "```python\r\nx=1\r\n```"],
)
def test_missing_code_still_records_historical_call(response):
    result = replay.historical_extract(response)
    assert result["program"] is None
    assert result["parse_status"] == "missing_code"
    assert result["requested_call"] == "run_tests()"


def test_failed_syntax_keeps_program_and_historical_fallback():
    result = replay.historical_extract("```python\ndef run_tests(:\n```")
    assert result["program"] == "def run_tests(:"
    assert result["parse_status"] == "syntax_error"
    assert result["parent_class"] is None
    assert result["selected_function_sha256"] is None
    assert result["requested_call"] == "run_tests()"


def test_function_body_and_parent_follow_different_historical_traversals():
    source = "def run_tests():\n    return 1\nclass Last:\n    def wrapper(self):\n        def run_tests():\n            return 2\n"
    result = replay.historical_extract(f"```python\n{source}```")
    assert (
        result["selected_function_sha256"]
        == hashlib.sha256(b"def run_tests():\n    return 1").hexdigest()
    )
    assert result["parent_class"] == "Last"
    assert result["requested_call"] == "Last().run_tests()"
    assert len(result["definitions"]) == 2
    assert result["definitions"][1]["function_depth"] == 1


def test_async_definition_does_not_disable_requested_call():
    result = replay.historical_extract(
        "```python\nasync def run_tests(): return True\n```"
    )
    assert result["response_has_test_func"] is False
    assert result["requested_call"] == "run_tests()"
    assert result["definitions"][0]["kind"] == "async"


def test_main_guard_only_disables_function_flag():
    result = replay.historical_extract(
        "```python\nclass Solution:\n    def run_tests(self):\n        if __name__ == '__main__':\n            pass\n```"
    )
    assert result["response_has_test_func"] is False
    assert result["response_test_func_compile_error"] is True
    assert result["requested_call"] == "Solution().run_tests()"


def test_protected_cases_preserve_keyword_names_values_and_infinity():
    cases = replay.protected_cases(
        task(
            [
                "assert Solution().answer(value=[1, [2.5, 'x']], other=-3) == [True, None, inf, -inf]"
            ]
        )
    )
    assert cases == [
        {
            "method": "answer",
            "kwargs": {"value": [1, [2.5, "x"]], "other": -3},
            "expected": [True, None, float("inf"), -float("inf")],
        }
    ]


@pytest.mark.parametrize(
    "statement",
    [
        "assert Solution().other(value=1) == 1",
        "assert Solution().answer(value=evil()) == 1",
        "assert Solution().answer(value=1) == evil()",
        "assert Solution().answer(value=1) != 1",
        "assert Solution().answer(value=1) == 1 == True",
        "assert Solution().answer(1) == 1",
        "assert Solution().answer(**values) == 1",
        "assert Solution().answer(value={1: 2}) == 1",
        "assert Solution().answer(value=1) == 1; print('side effect')",
        "assert Solution().answer(value=1) == 1, 'message'",
    ],
)
def test_protected_cases_reject_unsupported_assertions(statement):
    with pytest.raises(ValueError, match="unsupported"):
        replay.protected_cases(task([statement]))


def test_artifact_hash_failure_prevents_loading(tmp_path):
    module = replay
    for filename in module.ARTIFACT_FILES.values():
        (tmp_path / filename).write_text("raise RuntimeError('must not execute')")
    with pytest.raises(ValueError, match="checksum"):
        module.verify_artifacts(tmp_path)


def test_fixture_loader_checks_hash_id_and_exact_prompt(tmp_path, monkeypatch):
    module = replay
    path = tmp_path / "tasks.jsonl"
    content = (json.dumps(task()) + "\n").encode()
    path.write_bytes(content)
    monkeypatch.setitem(
        module.ARTIFACT_SHA256, "fixture", hashlib.sha256(content).hexdigest()
    )
    prompt = (
        "\nuser\ntask and will be evaluated by calling a function called run_tests()"
    )
    assert module.load_task(path, "7", prompt)["id"] == 7
    with pytest.raises(ValueError, match="prompt"):
        module.load_task(path, "7", prompt + " ")
    with pytest.raises(ValueError, match="problem"):
        module.load_task(path, "8", prompt)
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="checksum"):
        module.load_task(path, "7", prompt)


def test_template_rendering_is_literal_and_never_imports_source(tmp_path, monkeypatch):
    module = replay
    source = '''raise RuntimeError("must not execute")
_SUBPROCESS_CODE = textwrap.dedent(f"""\n    # outer\n    namespace = {{}}\n""").strip()
def create_test_runner_code(setup_code, program, test_list, max_failures):
    test_list_repr = repr(test_list)
    max_failures_repr = repr(max_failures)
    test_runner = f"""{setup_code}\n{program}\nchecks = {test_list_repr}\nlimit = {max_failures_repr}\n"""
    return test_runner
def _get_python_executable():
    return sys.executable
def _execute_in_subprocess(code, timeout, memory_limit, raise_exceptions=False):
    return CodeRunResult(success=False, timeout=True)
'''
    path = tmp_path / "helpers.py"
    path.write_text(source)
    monkeypatch.setitem(
        module.ARTIFACT_SHA256, "helpers", hashlib.sha256(source.encode()).hexdigest()
    )
    outer, stdin = module.historical_script(
        {"helpers": path}, "setup", "program", "run_tests()"
    )
    assert "# outer\\nnamespace = {}" in outer
    assert stdin == b"setup\nprogram\nchecks = ['run_tests()']\nlimit = 1\n"


@pytest.mark.parametrize(
    ("reported", "expected"),
    [
        (
            "success=False, timeout=True",
            {
                "success": False,
                "compiled": True,
                "timeout": True,
                "oom": False,
                "stdout": {},
            },
        ),
        (
            "success=False, compiled=False, stdout={'raw': 'invalid JSON'}",
            {
                "success": False,
                "compiled": False,
                "timeout": False,
                "oom": False,
                "stdout": {"raw": "invalid JSON"},
            },
        ),
    ],
)
def test_supervisor_preserves_source_parent_results(
    tmp_path, monkeypatch, reported, expected
):
    # This executes only hand-written fixture functions, never the upstream parent or child.
    source = '''_SUBPROCESS_CODE = textwrap.dedent(f"""pass""").strip()
def create_test_runner_code(setup_code, program, test_list, max_failures):
    test_runner = f"{program}"
    return test_runner
def _get_python_executable():
    return sys.executable
def _execute_in_subprocess(code, timeout, memory_limit, raise_exceptions=False):
    assert code == 'fixture payload'
    assert timeout == 3 and memory_limit == 1024
    assert os.environ['TOKENIZERS_PARALLELISM'] == 'false'
    return CodeRunResult(REPORTED)
'''.replace("REPORTED", reported)
    path = tmp_path / "helpers.py"
    path.write_text(source)
    monkeypatch.setitem(
        replay.ARTIFACT_SHA256, "helpers", hashlib.sha256(source.encode()).hexdigest()
    )
    outer, payload = replay.historical_script(
        {"helpers": path}, "", "fixture payload", "run_tests()"
    )
    process = subprocess.run(
        [sys.executable, "-c", outer],
        input=payload,
        capture_output=True,
        timeout=2,
        check=False,
    )
    assert process.returncode == 0
    assert json.loads(process.stdout) == expected


def test_supervisor_marks_pydantic_coercions_unavailable(tmp_path, monkeypatch):
    source = '''_SUBPROCESS_CODE = textwrap.dedent(f"""pass""").strip()
def create_test_runner_code(setup_code, program, test_list, max_failures):
    test_runner = f"{program}"
    return test_runner
def _get_python_executable():
    return sys.executable
def _execute_in_subprocess(code, timeout, memory_limit, raise_exceptions=False):
    return CodeRunResult(success='true')
'''
    path = tmp_path / "helpers.py"
    path.write_text(source)
    monkeypatch.setitem(
        replay.ARTIFACT_SHA256, "helpers", hashlib.sha256(source.encode()).hexdigest()
    )
    outer, payload = replay.historical_script(
        {"helpers": path}, "", "fixture payload", "run_tests()"
    )
    process = subprocess.run(
        [sys.executable, "-c", outer],
        input=payload,
        capture_output=True,
        timeout=2,
        check=False,
    )
    assert process.returncode == 0
    assert json.loads(process.stdout) == {
        "historical_unavailable": "unsupported_result_validation_or_supervisor_error",
        "exception_type": "ValueError",
    }


def test_template_rejects_unexpected_interpolation(tmp_path, monkeypatch):
    module = replay
    source = """_SUBPROCESS_CODE = textwrap.dedent(f"{danger()}").strip()
def create_test_runner_code(setup_code, program, test_list, max_failures):
    test_runner = f"{program}"
    return test_runner
"""
    path = tmp_path / "helpers.py"
    path.write_text(source)
    monkeypatch.setitem(
        module.ARTIFACT_SHA256, "helpers", hashlib.sha256(source.encode()).hexdigest()
    )
    with pytest.raises(ValueError, match="template"):
        module.historical_script({"helpers": path}, "", "pass", "run_tests()")


@pytest.mark.local_data
def test_pinned_task_fixtures_are_all_supported():
    source = os.environ.get("ROLLOUTDIFF_ARIA_TASKS_PATH")
    if not source:
        pytest.skip("set ROLLOUTDIFF_ARIA_TASKS_PATH to the pinned task JSONL")
    path = Path(source)
    module = replay
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == module.ARTIFACT_SHA256["fixture"]
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 992
    assert sum(len(module.protected_cases(record)) for record in records) == 100794
