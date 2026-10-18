"""Inspect staged blobs and enforce repository exclusions."""

import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

MAX_BYTES = 5 * 1024 * 1024
PRIVATE_DOCUMENTS = {"AGENTS.md", "Rollout Diff — Product & Engineering Plan.md"}
CREDENTIALS = re.compile(
    rb"-----BEGIN (?:(?:RSA|DSA|EC|OPENSSH|ENCRYPTED) )?PRIVATE KEY-----"
    rb"|\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{30,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{30,}|AKIA[A-Z0-9]{16}"
    rb"|xox[baprs]-[A-Za-z0-9-]{20,})\b"
    rb"|https?://[^\s/\"':]+:[^\s/@\"']+@"
)


def git(*arguments, input=None, allowed=(0,)):
    result = subprocess.run(
        ["git", *arguments], input=input, capture_output=True, check=False
    )
    if result.returncode not in allowed:
        raise RuntimeError(f"git {arguments[0]} failed (exit {result.returncode})")
    return result.stdout


def index_entries():
    entries = {}
    for record in git("ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode("ascii").split()
        if stage != "0":
            raise RuntimeError("resolve unmerged index entries before committing")
        entries[os.fsdecode(path)] = (mode, oid)
    return entries


def blob(oid):
    size = int(git("cat-file", "-s", oid))
    if size > MAX_BYTES:
        raise RuntimeError("staged file exceeds the 5 MiB limit")
    return git("cat-file", "blob", oid)


def check_ignore_policy(paths, entries):
    # Git evaluates working-tree ignore files. Require their staged equivalents.
    policies = {
        str(parent / ".gitignore")
        for path in paths
        for parent in PurePosixPath(path).parents
    }
    for name in sorted(policies):
        path = Path(name)
        entry = entries.get(name)
        if path.is_symlink() or (entry and entry[0] not in ("100644", "100755")):
            raise RuntimeError(f"ignore policy {name!r} must be a regular file")
        if path.is_file() and path.stat().st_size > MAX_BYTES:
            raise RuntimeError(f"ignore policy {name!r} exceeds the 5 MiB limit")
        working = path.read_bytes() if path.is_file() else None
        staged = blob(entry[1]) if entry else None
        if working != staged:
            raise RuntimeError(
                f"ignore policy {name!r} differs from the index; stage or restore it"
            )


def ignored_paths(paths):
    output = git(
        "check-ignore",
        "--no-index",
        "--stdin",
        "-v",
        "-z",
        input=b"\0".join(os.fsencode(path) for path in paths) + b"\0",
        allowed=(0, 1),
    )
    fields = output.split(b"\0")[:-1]
    if len(fields) % 4:
        raise RuntimeError("unexpected git check-ignore output")
    return {
        os.fsdecode(fields[index + 3])
        for index in range(0, len(fields), 4)
        if not fields[index + 2].startswith(b"!")
    }


def main():
    root = os.fsdecode(git("rev-parse", "--show-toplevel")).rstrip("\n")
    os.chdir(root)
    entries = index_entries()
    paths = [
        os.fsdecode(path)
        for path in git(
            "diff",
            "--cached",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACMT",
            "-z",
        ).split(b"\0")
        if path
    ]
    if not paths:
        return 0
    check_ignore_policy(paths, entries)
    ignored = ignored_paths(paths)
    failures = []
    for path in paths:
        reason = None
        mode, oid = entries[path]
        if PurePosixPath(path).name in PRIVATE_DOCUMENTS:
            reason = "private project document"
        elif path in ignored:
            reason = "excluded by Git ignore rules"
        elif mode not in ("100644", "100755"):
            reason = "staged entry must be a regular file"
        else:
            try:
                if CREDENTIALS.search(blob(oid)):
                    reason = "credential marker in staged content"
            except RuntimeError as error:
                reason = str(error)
        if reason:
            failures.append(f"staged safety: {path!r}: {reason}")
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError) as error:
        print(f"staged safety: {error}", file=sys.stderr)
        sys.exit(1)
