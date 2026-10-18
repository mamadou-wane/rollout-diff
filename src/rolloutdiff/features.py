"""Lexical and structural response features.

Complete think/thinking sections are excluded; resp_chars retains original length.
Python snippets are parsed independently without execution. AST indicators omit
alias resolution, dataflow, and outcomes. Non-Python fences are excluded from prose.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
import warnings
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

# File tags cannot pair across fence boundaries.
_FENCE_RE = re.compile(
    r"^[ \t]*(?P<fence>(?P<marker>`|~)(?P=marker){2,})(?P<language>[^\n]*)\n"
    r"(?P<fence_body>.*?)(?:^[ \t]*(?P=fence)(?P=marker)*[ \t]*(?:\n|$)|\Z)",
    re.DOTALL | re.MULTILINE | re.IGNORECASE,
)
_FILE_OPEN_RE = re.compile(
    r"<file\b(?P<attrs>(?:[^>'\"]|'[^']*'|\"[^\"]*\")*)>", re.IGNORECASE
)
_FILE_CLOSE_RE = re.compile(r"</file\s*>", re.IGNORECASE)
_FILE_ATTR_RE = re.compile(r"(?:^|\s)([\w:.-]+)\s*=\s*(['\"])(.*?)\2", re.DOTALL)
_FILE_HINT_RE = re.compile(r"</?file\b", re.IGNORECASE)
_MAX_MARKUP_DEPTH = 64
_FSTRING_PREFIX_RE = re.compile(r"(?:fr|rf|f)(?=['\"])", re.IGNORECASE)
_FILENAME_RE = re.compile(
    r"^[ \t]*#[ \t]*filename[ \t]*:[ \t]*(.+?)[ \t]*$", re.MULTILINE | re.IGNORECASE
)
_THINK_RE = re.compile(r"<(think|thinking)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE)
_WORD_RE = re.compile(r"\b[a-zA-Z]{3,}\b")
_STOP = frozenset(
    [
        "the",
        "and",
        "for",
        "that",
        "with",
        "you",
        "this",
        "are",
        "can",
        "not",
        "your",
        "from",
        "have",
        "but",
        "will",
        "what",
        "how",
        "all",
        "out",
        "use",
        "any",
        "has",
        "also",
        "more",
        "make",
        "which",
        "then",
        "there",
        "their",
        "about",
        "when",
        "some",
        "into",
        "other",
        "just",
        "its",
        "only",
        "these",
        "could",
        "would",
        "should",
        "been",
        "very",
        "who",
        "where",
        "does",
        "did",
        "they",
        "them",
        "than",
        "because",
        "while",
        "after",
        "such",
        "through",
        "between",
        "being",
        "most",
        "here",
        "much",
        "those",
        "well",
        "one",
    ]
)
_EXIT_CALLS = {"sys.exit", "os._exit", "exit", "quit", "os.abort"}
_EXEC_CALLS = {
    "exec",
    "eval",
    "compile",
    "subprocess.run",
    "subprocess.call",
    "subprocess.Popen",
    "os.system",
    "os.popen",
}
_CMP_DUNDERS = {
    "__eq__",
    "__ne__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__bool__",
    "__contains__",
}
_FILE_EXT_RE = re.compile(r"^[\w\-./]+\.(py|txt|json|cfg|ini|toml|yaml|yml|sh|csv)$")


_AST_FLAGS = (
    "calls_exit",
    "calls_exec_or_shell",
    "file_write",
    "file_read",
    "defines_cmp_dunder",
    "defines_any_dunder",
    "defines_test_named",
    "module_attr_assign",
    "has_main_guard",
    "top_level_call",
    "imports_sys",
    "imports_os",
    "raises",
    "catches_all",
    "defines_class",
    "overrides_builtin_name",
)
_BUILTINS_LIKE = {
    "print",
    "input",
    "open",
    "len",
    "range",
    "exit",
    "quit",
    "any",
    "all",
    "sorted",
    "map",
    "filter",
}


@dataclass
class RolloutFeatures:
    dense: dict[str, float] = field(default_factory=dict)
    sparse: set[str] = field(default_factory=set)


def _callee_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _callee_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def _parse(code: str) -> ast.Module | None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return None


@dataclass(frozen=True)
class _Region:
    start: int
    end: int
    body_start: int
    body_end: int
    kind: str
    value: str = ""


def _quoted_end(text: str, start: int) -> int:
    """Skip literals despite invalid Python; stop ordinary quotes at newlines.

    Unfinished triple quotes protect the remaining scope from markup scanning.
    """
    quote = text[start]
    delimiter = quote * (3 if text.startswith(quote * 3, start) else 1)
    position = start + len(delimiter)
    while position < len(text):
        if text[position] == "\\":
            position += 2
        elif text.startswith(delimiter, position):
            return position + len(delimiter)
        elif len(delimiter) == 1 and text[position] == "\n":
            return position
        else:
            position += 1
    return len(text)


def _fstring_end(text: str, start: int, quote_start: int) -> int:
    """Lex nested f-strings independently of earlier malformed indentation.

    Python 3.11 emits STRING; 3.12+ exposes nested FSTRING_START/END tokens.
    """
    depth = 0
    first = True
    try:
        for token in tokenize.generate_tokens(io.StringIO(text[start:]).readline):
            if first and token.type not in (
                tokenize.STRING,
                getattr(tokenize, "FSTRING_START", -1),
            ):
                break
            first = False
            if token.type == getattr(tokenize, "FSTRING_START", -1):
                depth += 1
            elif token.type == getattr(tokenize, "FSTRING_END", -1):
                depth -= 1
            elif token.type != tokenize.STRING or depth:
                continue
            if depth == 0:
                line_start = start
                for _ in range(token.end[0] - 1):
                    line_start = text.index("\n", line_start) + 1
                return line_start + token.end[1]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return _quoted_end(text, quote_start)


def _regions(text: str, *, fences: bool = True) -> list[_Region]:
    """Find outermost files/fences while ignoring literal markup.

    Open tags must follow whitespace from a line start or another structural tag.
    Retain completed children of unclosed parents without reusing child closers.
    """
    regions: list[_Region] = []
    stack: list[tuple[int, int, str]] = []
    position = 0
    structural = True
    while position < len(text):
        fence = _FENCE_RE.match(text, position) if fences or stack else None
        if fence:
            regions.append(
                _Region(
                    position,
                    fence.end(),
                    *fence.span("fence_body"),
                    "fence",
                    fence.group("language").strip().lower(),
                )
            )
            position = fence.end()
            structural = position == 0 or text[position - 1] == "\n"
            continue
        opening = _FILE_OPEN_RE.match(text, position) if structural else None
        if opening:
            path = next(
                (
                    value.strip()
                    for key, _, value in _FILE_ATTR_RE.findall(opening.group("attrs"))
                    if key.lower() in {"path", "name"}
                ),
                "",
            )
            stack.append((position, opening.end(), path))
            position = opening.end()
            continue
        closing = _FILE_CLOSE_RE.match(text, position)
        if closing:
            if stack:
                start, body_start, path = stack.pop()
                regions.append(
                    _Region(start, closing.end(), body_start, position, "file", path)
                )
            position = closing.end()
            structural = True
            continue
        char = text[position]
        fstring = (
            _FSTRING_PREFIX_RE.match(text, position)
            if char in "fFrR"
            and (
                position == 0
                or not (text[position - 1].isalnum() or text[position - 1] == "_")
            )
            else None
        )
        if fstring:
            position = _fstring_end(text, position, fstring.end())
            structural = False
        elif text.startswith("<!--", position):
            end = text.find("-->", position + 4)
            position = len(text) if end < 0 else end + 3
            structural = False
        elif char in "\"'":
            position = _quoted_end(text, position)
            structural = False
        elif char == "#":
            end = text.find("\n", position)
            position = len(text) if end < 0 else end
            structural = False
        elif char == "`":
            # Inline code is literal; unmatched ticks cannot hide later lines.
            end = position + 1
            while end < len(text) and text[end] == "`":
                end += 1
            delimiter = text[position:end]
            line_end = text.find("\n", end)
            close = text.find(delimiter, end, len(text) if line_end < 0 else line_end)
            position = end if close < 0 else close + len(delimiter)
            structural = False
        else:
            if char == "\n":
                structural = True
            elif not char.isspace():
                structural = False
            position += 1
    outermost: list[_Region] = []
    consumed = 0
    for region in sorted(regions, key=lambda r: (r.start, -r.end)):
        if region.start >= consumed:
            outermost.append(region)
            consumed = region.end
    return outermost


def _filename_paths(code: str) -> set[str]:
    matches = _FILENAME_RE.findall(code)
    if not matches:
        return set()
    try:
        comments = [
            token.string
            for token in tokenize.generate_tokens(io.StringIO(code).readline)
            if token.type == tokenize.COMMENT
            and not token.line[: token.start[1]].strip()
        ]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {path.strip() for path in matches}
    return {
        path.strip() for comment in comments for path in _FILENAME_RE.findall(comment)
    }


def _split_without_files(
    text: str, *, file_body: bool
) -> tuple[list[str], str, set[str]]:
    """Separate code fences, prose, and filename declarations; retain empty fences."""
    blocks: list[str] = []
    prose: list[str] = []
    position = 0
    matched = False
    for match in _FENCE_RE.finditer(text):
        matched = True
        prose.append(text[position : match.start()])
        position = match.end()
        if match.group("language").strip().lower() in ("", "python", "python3", "py"):
            blocks.append(match.group("fence_body"))
    prose.append(text[position:])
    if not matched and text.strip() and (file_body or _parse(text) is not None):
        blocks.append(text)
        prose.clear()
    declared = {path for block in blocks for path in _filename_paths(block)}
    return blocks, " ".join(prose), declared


def _split_response(
    text: str,
    *,
    file_body: bool = False,
    fence_body: bool = False,
    _depth: int = 0,
) -> tuple[list[str], str, set[str]]:
    # Bound adversarial nesting; remaining markup becomes a parse-failing leaf.
    if _depth >= _MAX_MARKUP_DEPTH:
        return [text], "", set()
    if not _FILE_HINT_RE.search(text):
        if fence_body:
            return [text], "", _filename_paths(text)
        return _split_without_files(text, file_body=file_body)
    if _parse(text) is not None:
        # Valid Python is atomic, protecting complex f-string literals.
        return [text], "", _filename_paths(text)
    blocks: list[str] = []
    direct: list[str] = []
    prose: list[str] = []
    declared: set[str] = set()
    position = 0
    regions = _regions(text, fences=not fence_body)
    for region in regions:
        gap = text[position : region.start]
        direct.append(gap)
        if not file_body:
            prose.append(gap)
        position = region.end
        if region.kind == "file" or region.value in ("", "python", "python3", "py"):
            if region.kind == "file" and region.value:
                declared.add(region.value)
            nested, nested_prose, nested_paths = _split_response(
                text[region.body_start : region.body_end],
                file_body=True,
                fence_body=region.kind == "fence",
                _depth=_depth + 1,
            )
            blocks.extend(nested)
            if region.kind == "file" or nested_prose:
                prose.append(nested_prose)
            declared.update(nested_paths)
    tail = text[position:]
    direct.append(tail)
    if file_body:
        # Reassemble parent fragments without counting child spans twice.
        code = "\n".join(direct)
        if code.strip():
            blocks.append(code)
    else:
        prose.append(tail)
    for block in blocks:
        declared.update(_filename_paths(block))
    return blocks, " ".join(prose), declared


def _is_main_guard(node: ast.AST) -> bool:
    if not (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and isinstance(node.ops[0], ast.Eq)
    ):
        return False
    left, right = node.left, node.comparators[0]
    return (
        isinstance(left, ast.Name)
        and left.id == "__name__"
        and isinstance(right, ast.Constant)
        and right.value == "__main__"
    ) or (
        isinstance(right, ast.Name)
        and right.id == "__name__"
        and isinstance(left, ast.Constant)
        and left.value == "__main__"
    )


def _call_features(
    node: ast.Call, counts: Counter, flags: dict, sparse: set[str]
) -> None:
    name = _callee_name(node.func)
    if not name:
        return
    sparse.add(f"call:{name}")
    if name in _EXIT_CALLS:
        flags["calls_exit"] = 1.0
    if name in _EXEC_CALLS:
        flags["calls_exec_or_shell"] = 1.0
    if name == "print":
        counts["n_print"] += 1
    if name in ("open", "io.open", "Path.open"):
        # Path("x").open takes mode first; builtin open takes it second.
        literal_path = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "open"
            and isinstance(node.func.value, ast.Call)
            and _callee_name(node.func.value.func) in ("Path", "pathlib.Path")
        )
        mode_index = 0 if literal_path else 1
        mode = None
        if len(node.args) > mode_index and isinstance(
            node.args[mode_index], ast.Constant
        ):
            mode = node.args[mode_index].value
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                mode = kw.value.value
        if isinstance(mode, str) and any(c in mode for c in "wax+"):
            flags["file_write"] = 1.0
        else:
            flags["file_read"] = 1.0
    if name.endswith(("write_text", "write_bytes", "to_csv", "dump")):
        flags["file_write"] = 1.0
    if name.endswith(("read_text", "read_bytes", "load")) and "json" not in name:
        flags["file_read"] = 1.0


def _ast_features(trees: list[ast.Module], sparse: set[str]) -> dict[str, float]:
    counts = Counter()
    flags = dict.fromkeys(_AST_FLAGS, 0.0)
    for tree in trees:
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                counts["n_funcdefs"] += 1
                name = node.name
                sparse.add(f"def:{name}")
                if name.startswith("__") and name.endswith("__"):
                    flags["defines_any_dunder"] = 1.0
                    if name in _CMP_DUNDERS:
                        flags["defines_cmp_dunder"] = 1.0
                if "test" in name.lower():
                    flags["defines_test_named"] = 1.0
                if name in _BUILTINS_LIKE:
                    flags["overrides_builtin_name"] = 1.0
            elif isinstance(node, ast.ClassDef):
                counts["n_classdefs"] += 1
                flags["defines_class"] = 1.0
                sparse.add(f"class:{node.name}")
            elif isinstance(node, ast.Import):
                for a in node.names:
                    mod = a.name.split(".")[0]
                    imported.add(mod)
                    sparse.add(f"import:{mod}")
                counts["n_imports"] += 1
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                imported.add(mod)
                sparse.add(f"import:{mod}")
                counts["n_imports"] += 1
            elif isinstance(node, ast.Call):
                counts["n_calls"] += 1
                _call_features(node, counts, flags, sparse)
            elif isinstance(node, ast.Try):
                counts["n_try"] += 1
                for h in node.handlers:
                    if h.type is None or (
                        isinstance(h.type, ast.Name)
                        and h.type.id in ("Exception", "BaseException")
                    ):
                        flags["catches_all"] = 1.0
            elif isinstance(node, ast.Raise):
                flags["raises"] = 1.0
                if (
                    isinstance(node.exc, ast.Call)
                    and _callee_name(node.exc.func) == "SystemExit"
                ):
                    flags["calls_exit"] = 1.0
            elif isinstance(node, ast.Assert):
                counts["n_assert"] += 1
            elif isinstance(node, ast.Return):
                counts["n_return"] += 1
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Attribute):
                        base = _callee_name(t.value)
                        if base:
                            sparse.add(f"setattr:{base}.{t.attr}")
                            if base.split(".")[0] in imported:
                                flags["module_attr_assign"] = 1.0
                    elif isinstance(t, ast.Name) and t.id in _BUILTINS_LIKE:
                        flags["overrides_builtin_name"] = 1.0
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                v = node.value.strip()
                if 3 <= len(v) <= 40 and _FILE_EXT_RE.match(v):
                    sparse.add(f"strlit:{v}")
        for st in tree.body:
            if isinstance(st, ast.If):
                if _is_main_guard(st.test):
                    flags["has_main_guard"] = 1.0
            elif isinstance(st, ast.Expr) and isinstance(st.value, ast.Call):
                flags["top_level_call"] = 1.0
        flags["imports_sys"] = max(flags["imports_sys"], float("sys" in imported))
        flags["imports_os"] = max(flags["imports_os"], float("os" in imported))
    names = (
        "n_funcdefs",
        "n_classdefs",
        "n_imports",
        "n_calls",
        "n_print",
        "n_try",
        "n_assert",
        "n_return",
    )
    return {**{name: float(counts.get(name, 0)) for name in names}, **flags}


def extract(response: str) -> RolloutFeatures:
    """Extract syntax and prose features from response text."""
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    result = RolloutFeatures()
    dense, sparse = result.dense, result.sparse
    text = _THINK_RE.sub(" ", response)
    blocks, prose, declared_files = _split_response(text)
    for path in declared_files:
        sparse.add(f"file:{path.replace(chr(92), '/').split('/')[-1]}")
    dense["n_files_declared"] = float(len(declared_files))
    code = "\n".join(blocks)
    dense["resp_chars"] = float(len(response))
    dense["prose_chars"] = float(len(prose))
    dense["code_chars"] = float(len(code))
    dense["n_code_blocks"] = float(len(blocks))
    dense["code_lines"] = float(sum(len(block.splitlines()) for block in blocks))
    dense["has_code"] = float(any(block.strip() for block in blocks))
    trees = []
    dense["code_parse_fail"] = 0.0
    for block in blocks:
        tree = _parse(block)
        if tree is None:
            dense["code_parse_fail"] = 1.0
        else:
            trees.append(tree)

    dense.update(_ast_features(trees, sparse))

    words = [w for w in _WORD_RE.findall(prose.lower()) if w not in _STOP]
    dense["prose_words"] = float(len(words))
    for w in set(words):
        sparse.add(f"w:{w}")
    for i in range(len(words) - 1):
        sparse.add(f"b:{words[i]} {words[i + 1]}")
    return result


DENSE_BINARY = {"has_code", "code_parse_fail", *_AST_FLAGS}


@dataclass
class FeatureTables:
    """Features keyed by (run, sample_id), plus metadata for retained features."""

    dense: pd.DataFrame
    sparse: pd.DataFrame
    catalog: pd.DataFrame


def _tier(feature: str) -> str:
    return (
        "prose"
        if feature.startswith(("w:", "b:")) or feature in {"prose_chars", "prose_words"}
        else "structural"
    )


def _select_sparse(
    rollouts: pd.DataFrame,
    sparse: pd.DataFrame,
    max_sparse_per_tier: int,
    min_prevalence: float,
) -> list[str]:
    if sparse.empty or not max_sparse_per_tier:
        return []
    metadata = rollouts[["run", "sample_id", "step", "prompt_id"]]
    presence = sparse.merge(metadata, on=["run", "sample_id"], validate="many_to_one")
    prompt_keys = ["run", "step", "prompt_id"]
    denominators = (
        rollouts.groupby(prompt_keys, observed=True, sort=False)
        .size()
        .rename("n_rollouts")
    )
    positive = (
        presence.groupby([*prompt_keys, "feature"], observed=True, sort=False)
        .size()
        .rename("n_positive")
        .reset_index()
    )
    positive = positive.merge(
        denominators.reset_index(), on=prompt_keys, validate="many_to_one"
    )
    positive["within_prompt"] = positive.n_positive / positive.n_rollouts
    n_prompts = (
        denominators.groupby(level=["run", "step"], sort=False)
        .size()
        .rename("n_prompts")
    )
    prevalence = (
        positive.groupby(["run", "step", "feature"], observed=True, sort=False)
        .within_prompt.sum()
        .reset_index()
    )
    prevalence = prevalence.merge(
        n_prompts.reset_index(), on=["run", "step"], validate="many_to_one"
    )
    prevalence["prevalence"] = prevalence.within_prompt / prevalence.n_prompts
    peaks = (
        prevalence.groupby("feature", sort=False)
        .prevalence.max()
        .rename("peak")
        .reset_index()
    )
    peaks = peaks.loc[peaks.peak >= min_prevalence].sort_values(
        ["peak", "feature"], ascending=[False, True]
    )
    peaks["tier"] = peaks.feature.map(_tier)
    return peaks.groupby("tier", sort=False).head(max_sparse_per_tier).feature.tolist()


def featurize(
    rollouts: pd.DataFrame,
    *,
    max_sparse_per_tier: int = 2000,
    min_prevalence: float = 0.05,
) -> FeatureTables:
    """Retain sparse features by peak prompt-balanced prevalence per tier.

    Average within prompts, then across all prompts at each run/step, including
    zeros for absent features. Ties break lexically. Selection is label-free and
    retrospective over all supplied data.
    """
    if (
        isinstance(max_sparse_per_tier, bool)
        or not isinstance(max_sparse_per_tier, int)
        or max_sparse_per_tier < 0
    ):
        raise ValueError("max_sparse_per_tier must be a nonnegative integer")
    if not 0 <= min_prevalence <= 1:
        raise ValueError("min_prevalence must be between zero and one")
    required = {"run", "step", "sample_id", "prompt_id", "response"}
    missing = required - set(rollouts.columns)
    if missing:
        raise ValueError(f"missing rollout columns: {sorted(missing)}")
    if rollouts[list(required)].isna().any().any():
        raise ValueError("feature input keys and responses cannot be null")
    if rollouts.duplicated(["run", "sample_id"]).any():
        raise ValueError("duplicate rollout keys (run, sample_id)")

    dense_names = list(extract("").dense)
    dense_rows = []
    sparse_rows = []
    columns = ["run", "sample_id", "step", "prompt_id", "response"]
    for run, sample, _step, _prompt, response in rollouts[columns].itertuples(
        index=False, name=None
    ):
        result = extract(response)
        dense_rows.append({"run": run, "sample_id": sample, **result.dense})
        sparse_rows.extend((run, sample, token) for token in sorted(result.sparse))
    dense = pd.DataFrame(dense_rows, columns=["run", "sample_id", *dense_names])
    if dense.empty:
        dense = dense.astype({name: "float64" for name in dense_names})
    sparse = pd.DataFrame(sparse_rows, columns=["run", "sample_id", "feature"])
    kept = _select_sparse(rollouts, sparse, max_sparse_per_tier, min_prevalence)
    sparse = sparse.loc[sparse.feature.isin(kept)].reset_index(drop=True)
    catalog = pd.DataFrame(
        [
            (name, _tier(name), "binary" if name in DENSE_BINARY else "continuous")
            for name in dense_names
        ]
        + [(name, _tier(name), "binary") for name in kept],
        columns=["feature", "tier", "kind"],
    ).sort_values("feature", ignore_index=True)
    return FeatureTables(dense=dense, sparse=sparse, catalog=catalog)
