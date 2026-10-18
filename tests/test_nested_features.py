"""Invented fixtures for nested, lexical file-declaration ownership."""

import sys

import pytest

from rolloutdiff.features import extract


def check(text, files, functions, blocks):
    result = extract(text)
    assert {f for f in result.sparse if f.startswith("file:")} == set(files)
    assert result.dense["n_files_declared"] == len(files)
    assert result.dense["n_funcdefs"] == functions
    assert result.dense["n_code_blocks"] == blocks
    return result


@pytest.mark.parametrize("language", ["", "python", "py", "python3"])
def test_file_inside_fence_is_parsed_once(language):
    result = check(
        f'```{language}\n<file path="solver.py">\ndef solve():\n    pass\n</file>\n```',
        {"file:solver.py"},
        1,
        1,
    )
    assert result.dense["code_parse_fail"] == 0


def test_balanced_nested_files_preserve_direct_parent_code_once():
    result = check(
        """<file path="outer.py">
import sys
<file name='inner.py'>
def inner(): pass
</file>
def outer(): pass
</file>""",
        {"file:outer.py", "file:inner.py"},
        2,
        2,
    )
    assert result.dense["n_imports"] == 1
    assert result.dense["code_parse_fail"] == 0
    assert "w:file" not in result.sparse


def test_file_fence_file_fence_nesting_does_not_duplicate_ast():
    result = check(
        """<file path="outer.py">
````python
<file path="inner.py">
```python
def inner(): pass
```
</file>
````
</file>""",
        {"file:outer.py", "file:inner.py"},
        1,
        1,
    )
    assert result.dense["code_parse_fail"] == 0


def test_complete_inner_file_survives_incomplete_parent():
    result = extract("""<file path="unfinished.py">
<file path="complete.py">def complete(): pass</file>""")
    assert "file:complete.py" in result.sparse
    assert "file:unfinished.py" not in result.sparse
    assert result.dense["n_funcdefs"] == 1


def test_child_close_is_not_borrowed_by_incomplete_parent():
    result = extract("""<file path="unfinished.py">
<file path="complete.py">pass</file>""")
    assert result.dense["n_files_declared"] == 1


@pytest.mark.parametrize(
    "text",
    [
        '```python\n<file path="ambiguous.py">\npass\n```\n</file>',
        '<file path="ambiguous.py">\n```python\npass\n</file>\n```',
    ],
)
def test_file_cannot_close_across_fence_scope(text):
    assert "file:ambiguous.py" not in extract(text).sparse


@pytest.mark.parametrize(
    "prefix",
    [
        "if True:\n pass\n  pass\n pass\n",
        "broken = 'unterminated\npass\n",
        'broken = "unterminated\npass\n',
    ],
)
def test_prior_invalid_python_does_not_hide_complete_file(prefix):
    result = extract(
        "```python\n"
        + prefix
        + '<file path="complete.py">def recovered(): pass</file>\n```'
    )
    assert "file:complete.py" in result.sparse
    assert "def:recovered" in result.sparse
    assert result.dense["code_parse_fail"] == 1


def test_declaration_does_not_require_valid_python_body():
    result = check(
        '<file path="broken.py">def unfinished(</file>', {"file:broken.py"}, 0, 1
    )
    assert result.dense["code_parse_fail"] == 1


@pytest.mark.parametrize("fenced", [False, True])
def test_string_comment_and_inline_prose_tags_are_not_declarations(fenced):
    code = '''example = """<file path="fake.py">
def fake(): pass
</file>"""
# <file path="comment.py">pass</file>
def real(): pass
'''
    result = check(("```python\n" + code + "```") if fenced else code, set(), 1, 1)
    assert "def:fake" not in result.sparse
    prose = extract(
        'For example, `<file path="example.py">pass</file>` describes the format.'
    )
    assert prose.dense["n_files_declared"] == 0
    assert prose.dense["has_code"] == 0


def test_literal_close_does_not_truncate_file_and_real_child_survives():
    result = check(
        '''<file path="outer.py">
example = """</file>
<file path="literal.py">pass</file>"""
<file path="inner.py">def real(): pass</file>
</file>''',
        {"file:outer.py", "file:inner.py"},
        1,
        2,
    )
    assert "file:literal.py" not in result.sparse
    assert result.dense["code_parse_fail"] == 0


def test_inline_file_body_and_repeated_path_header_deduplicate_declaration():
    result = check(
        """<file path="solver.py"><file path="solver.py">
# filename: solver.py
def solve(): pass
</file></file>""",
        {"file:solver.py"},
        1,
        1,
    )
    assert result.dense["code_parse_fail"] == 0


def test_identical_separate_snippets_are_not_deduplicated():
    text = '<file path="solver.py">def solve(): pass</file>\n' * 2
    check(text, {"file:solver.py"}, 2, 2)


def test_non_python_fence_remains_opaque():
    check(
        '```xml\n<file path="example.py">def example(): pass</file>\n```', set(), 0, 0
    )


def test_same_basename_keeps_distinct_full_path_count():
    result = extract(
        '<file path="a/solver.py">pass</file>\n<file path="b/solver.py">pass</file>'
    )
    assert result.dense["n_files_declared"] == 2
    assert {f for f in result.sparse if f.startswith("file:")} == {"file:solver.py"}


def test_python_fences_preserve_existing_prose_spacing():
    result = extract("Explanation.\n```python\npass\n```\nAfter.\n```py\npass\n```")
    assert result.dense["prose_chars"] == len("Explanation.\n After.\n ")


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="same-quote f-string grammar requires Python 3.12",
)
def test_parseable_python_fstring_nested_quotes_are_literal():
    code = 's = f"""{str("""\n<file path="fake.py">\nprint(1)\n</file>\n""")}"""'
    result = extract("```python\n" + code + "\n```")
    assert result.dense["n_files_declared"] == 0
    assert result.dense["n_calls"] == 1
    assert result.dense["code_parse_fail"] == 0


def test_filename_header_inside_python_string_is_not_declaration():
    result = extract('text = """\n# filename: fake.py\n"""\n# filename: real.py\npass')
    assert {f for f in result.sparse if f.startswith("file:")} == {"file:real.py"}


def test_deep_file_nesting_does_not_depend_on_python_recursion_limit():
    text = '<file path="same.py">' * 1100 + "print(1)" + "</file>" * 1100
    result = check(text, {"file:same.py"}, 0, 1)
    assert result.dense["n_calls"] == 0
    assert result.dense["code_parse_fail"] == 1


@pytest.mark.parametrize(
    "attributes",
    [
        'note=\'example path="fake.py"\' name="real.py"',
        'name="real.py" note=\'example path="fake.py"\'',
    ],
)
def test_path_text_in_another_attribute_is_not_a_declaration(attributes):
    check(f"<file {attributes}>pass</file>", {"file:real.py"}, 0, 1)


def test_inline_comment_does_not_become_filename_header():
    result = extract("# filename: real.py\npass # filename: fake.py\n")
    assert {f for f in result.sparse if f.startswith("file:")} == {"file:real.py"}


@pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="same-quote f-string grammar requires Python 3.12",
)
@pytest.mark.parametrize("prefix", ["", "if True:\n pass\n  pass\n pass\n"])
def test_complex_fstring_inside_file_cannot_close_its_wrapper(prefix):
    code = (
        's = f"""{str("""\n</file>\n<file path="fake.py">\nprint(1)\n</file>\n""")}"""'
    )
    result = extract('<file path="real.py">\n' + prefix + code + "\n</file>")
    assert {f for f in result.sparse if f.startswith("file:")} == {"file:real.py"}
    assert result.dense["n_calls"] == (0 if prefix else 1)
    assert result.dense["code_parse_fail"] == bool(prefix)


@pytest.mark.parametrize("prefix", ["f", "fr", "rf"])
def test_malformed_ordinary_fstring_does_not_hide_later_declaration(prefix):
    result = extract(
        "```python\ns = " + prefix + '"unfinished\npass\n'
        '<file path="real.py">def real(): pass</file>\n```'
    )
    assert {f for f in result.sparse if f.startswith("file:")} == {"file:real.py"}
    assert "def:real" in result.sparse
    assert result.dense["code_parse_fail"] == 1
