"""Pinned Aria replay preparation without importing or executing source code."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import textwrap
import warnings
from pathlib import Path

ARIA_SHA256 = "59ffb1aa73d341f6bd070f639028da8ee041176c8c8e0e18678a4d6751a11556"
UPSTREAM_REVISION = "73695ff5533b566f7cc99b02bfeb9168936e740d"
ARTIFACT_FILES = {
    "evaluator": "upstream__src__evaluate__evaluator.py",
    "helpers": "upstream__src__evaluate__helpers.py",
    "evaluation": "upstream__src__evaluate__evaluation.py",
    "fixture": "results__data__leetcode_train_medhard_filtered.jsonl",
}
ARTIFACT_SHA256 = {
    "evaluator": "3d7b7df29a7f4566d20615e2f929932d1ea3c7535dc8d92a9d98436b4ea19f4e",
    "helpers": "91c3e932d4ecbb42b63137d7c0f4acbf14959d2861bb39a6d9739872f2032505",
    "evaluation": "3fc688f8acbf65ec34efd1f3dfd294ee4b8b3f6fb0fc751d0f027b4e1192bd2a",
    "fixture": "e29f296fe7389c90a5ae99656aecc170feea81e3aca90a9a638c6a0880068b5d",
}

_RESULT_CONTAINER = """
class CodeRunResult(dict):
    def __init__(self, **values):
        defaults = dict(success=True, compiled=True, timeout=False, oom=False, stdout={})
        fields = {key: values.get(key, default) for key, default in defaults.items()}
        if any(type(fields[key]) is not bool for key in ('success', 'compiled', 'timeout', 'oom')):
            raise ValueError('unsupported historical boolean coercion')
        if type(fields['stdout']) is not dict:
            raise ValueError('unsupported historical stdout coercion')
        super().__init__(fields)
"""

_SUPERVISOR_CALL = """
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
try:
    result = _execute_in_subprocess(sys.stdin.read(), timeout=3, memory_limit=1024)
except Exception as error:
    result = {
        'historical_unavailable': 'unsupported_result_validation_or_supervisor_error',
        'exception_type': type(error).__name__,
    }
sys.stdout.write(json.dumps(result))
sys.stdout.flush()
"""


def _checked_source(path: str | Path, kind: str) -> str:
    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != ARTIFACT_SHA256[kind]:
        raise ValueError(f"Aria {kind} artifact checksum changed")
    return content.decode("utf-8")


def verify_artifacts(directory: str | Path) -> dict[str, Path]:
    paths = {kind: Path(directory) / name for kind, name in ARTIFACT_FILES.items()}
    for kind, path in paths.items():
        _checked_source(path, kind)
    return paths


class _FunctionLookup(ast.NodeVisitor):
    def __init__(self):
        self.current_class = None
        self.parent_class = None
        self.function_depth = 0
        self.definitions = []

    def visit_ClassDef(self, node):
        previous = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous

    def _visit_function(self, node, synchronous):
        if node.name == "run_tests":
            if synchronous:
                self.parent_class = self.current_class
            self.definitions.append(
                {
                    "kind": "sync" if synchronous else "async",
                    "line": node.lineno,
                    "parent_class": self.current_class,
                    "function_depth": self.function_depth,
                }
            )
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_FunctionDef(self, node):
        self._visit_function(node, True)

    def visit_AsyncFunctionDef(self, node):
        self._visit_function(node, False)


def historical_extract(response: str) -> dict:
    """Keep the historical first-body and last-parent lookup disagreement."""
    blocks = re.findall(
        r"```(?:python)?\n(.*?)(?:```|$)", response, re.DOTALL | re.IGNORECASE
    )
    blocks = [block.strip() for block in blocks if block.strip()]
    program = "\n\n".join(blocks) if blocks else None
    result = {
        "program": program,
        "parse_status": "missing_code" if program is None else "parsed",
        "selected_function_sha256": None,
        "parent_class": None,
        "requested_call": "run_tests()",
        "response_has_test_func": False,
        "response_test_func_compile_error": False,
        "definitions": [],
    }
    if program is None:
        return result
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            tree = ast.parse(program)
    except (SyntaxError, ValueError, RecursionError):
        result["parse_status"] = "syntax_error"
        return result
    selected = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "run_tests"
        ),
        None,
    )
    try:
        function = ast.unparse(selected) if selected is not None else ""
        lookup = _FunctionLookup()
        lookup.visit(tree)
    except RecursionError:
        result.update(parse_status="lookup_error", requested_call=None)
        return result
    parent = lookup.parent_class
    rejected = "if __name__ == '__main__':" in function
    result.update(
        {
            "selected_function_sha256": hashlib.sha256(function.encode()).hexdigest()
            if function
            else None,
            "parent_class": parent,
            "requested_call": "run_tests()"
            if parent is None
            else f"{parent}().run_tests()",
            "response_has_test_func": bool(function) and not rejected,
            "response_test_func_compile_error": rejected,
            "definitions": lookup.definitions,
        }
    )
    return result


def _assignment(body: list[ast.stmt], name: str) -> ast.expr:
    values = [
        node.value
        for node in body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(values) != 1:
        raise ValueError("unsupported historical template assignment")
    return values[0]


def _render_text(node: ast.expr, variables: dict[str, str]) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_render_text(value, variables) for value in node.values)
    if (
        isinstance(node, ast.FormattedValue)
        and isinstance(node.value, ast.Name)
        and node.value.id in variables
        and node.conversion == -1
        and node.format_spec is None
    ):
        return variables[node.value.id]
    raise ValueError("unsupported historical template interpolation")


def historical_script(
    artifacts: dict[str, Path], setup: str, program: str, call: str
) -> tuple[str, bytes]:
    """Render pinned child and parent; unsupported Pydantic coercions abstain."""
    source = _checked_source(artifacts["helpers"], "helpers")
    tree = ast.parse(source)
    outer = _assignment(tree.body, "_SUBPROCESS_CODE")
    if not (
        isinstance(outer, ast.Call)
        and isinstance(outer.func, ast.Attribute)
        and outer.func.attr == "strip"
        and not outer.args
        and not outer.keywords
    ):
        raise ValueError("unsupported historical outer template")
    dedent = outer.func.value
    if not (
        isinstance(dedent, ast.Call)
        and isinstance(dedent.func, ast.Attribute)
        and isinstance(dedent.func.value, ast.Name)
        and dedent.func.value.id == "textwrap"
        and dedent.func.attr == "dedent"
        and len(dedent.args) == 1
        and not dedent.keywords
    ):
        raise ValueError("unsupported historical outer template")
    outer_text = textwrap.dedent(_render_text(dedent.args[0], {})).strip()
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "create_test_runner_code"
    ]
    if len(functions) != 1:
        raise ValueError("unsupported historical runner template")
    template = _assignment(functions[0].body, "test_runner")
    inner = _render_text(
        template,
        {
            "setup_code": setup,
            "program": program,
            "test_list_repr": repr([call]),
            "max_failures_repr": "1",
        },
    )
    parent_sources = []
    for name in ("_get_python_executable", "_execute_in_subprocess"):
        nodes = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        if len(nodes) != 1:
            raise ValueError("unsupported historical parent template")
        parent_sources.append(ast.get_source_segment(source, nodes[0]))
    # The pinned parent retains its four-second deadline and subprocess error paths.
    # Only strict, already-typed CodeRunResult fields are supported without Pydantic.
    supervisor = "\n".join(
        [
            "import json\nimport os\nimport subprocess\nimport sys",
            _RESULT_CONTAINER,
            "_SUBPROCESS_CODE = " + repr(outer_text),
            *parent_sources,
            _SUPERVISOR_CALL,
        ]
    )
    return supervisor, inner.encode()


def load_task(path: str | Path, prompt_id: str, prompt_text: str) -> dict:
    """Join the pinned task ID and verify the exact serialized hinted prompt."""
    records = [
        json.loads(line) for line in _checked_source(path, "fixture").splitlines()
    ]
    matches = [row for row in records if str(row["id"]) == str(prompt_id)]
    if len(matches) != 1:
        raise ValueError("Aria problem ID must identify exactly one pinned task")
    task = matches[0]
    expected = task["prompt"][1]["content"].replace(
        "and should pass all tests",
        "and will be evaluated by calling a function called run_tests()",
    )
    if "\nuser\n" + expected != prompt_text:
        raise ValueError("Aria task prompt does not match the pinned fixture")
    return task


def _case_value(node: ast.expr, *, expected: bool):
    if isinstance(node, ast.Constant) and type(node.value) in (
        str,
        int,
        float,
        bool,
        type(None),
    ):
        return node.value
    if isinstance(node, ast.List):
        return [_case_value(value, expected=expected) for value in node.elts]
    if expected and isinstance(node, ast.Name) and node.id == "inf":
        return float("inf")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _case_value(node.operand, expected=expected)
        if type(value) in (int, float):
            return -value if isinstance(node.op, ast.USub) else value
    raise ValueError("unsupported protected test value")


def protected_cases(task: dict) -> list[dict]:
    """Extract trusted comparisons while keeping expected values out of workers."""
    match = re.fullmatch(r"Solution\(\)\.([A-Za-z_]\w*)", task["func_name"])
    if match is None or not task["gt_answer"]:
        raise ValueError("unsupported protected task")
    method = match[1]
    cases = []
    for test in task["gt_answer"]:
        try:
            body = ast.parse(test).body
        except (SyntaxError, ValueError) as exc:
            raise ValueError("unsupported protected test syntax") from exc
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Assert)
            or body[0].msg is not None
        ):
            raise ValueError("unsupported protected test assertion")
        comparison = body[0].test
        if not (
            isinstance(comparison, ast.Compare)
            and len(comparison.ops) == 1
            and isinstance(comparison.ops[0], ast.Eq)
            and len(comparison.comparators) == 1
        ):
            raise ValueError("unsupported protected test comparison")
        call = comparison.left
        if not (
            isinstance(call, ast.Call)
            and not call.args
            and ast.unparse(call.func) == task["func_name"]
            and all(keyword.arg is not None for keyword in call.keywords)
            and len({keyword.arg for keyword in call.keywords}) == len(call.keywords)
        ):
            raise ValueError("unsupported protected test call")
        cases.append(
            {
                "method": method,
                "kwargs": {
                    keyword.arg: _case_value(keyword.value, expected=False)
                    for keyword in call.keywords
                },
                "expected": _case_value(comparison.comparators[0], expected=True),
            }
        )
    return cases
