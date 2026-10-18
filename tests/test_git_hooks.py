"""Exercise hooks in disposable repositories without touching the project index."""

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[1]
HOOKS = PROJECT / ".githooks"


def test_hook_entrypoints_exist():
    for name in ("check_staged.py", "pre-commit", "pre-push"):
        assert (HOOKS / name).is_file(), f"missing hook: {name}"


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "repository with spaces"
    repo.mkdir()
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.hooksPath", ".githooks"],
        env=env,
        check=True,
    )
    assert (HOOKS / "check_staged.py").is_file(), "staged checker is missing"
    shutil.copytree(HOOKS, repo / ".githooks")
    (repo / ".github").mkdir()
    shutil.copyfile(
        PROJECT / ".github/ci-constraints.txt", repo / ".github/ci-constraints.txt"
    )
    (repo / ".gitignore").write_bytes((PROJECT / ".gitignore").read_bytes())
    subprocess.run(["git", "-C", str(repo), "add", ".gitignore"], env=env, check=True)
    return repo, env


def git(repository, *args):
    repo, env = repository
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=True
    )


def stage(repository, name, content=b"ordinary content\n"):
    repo, _ = repository
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    git(repository, "add", "-f", "--", name)
    return path


def check(repository):
    repo, env = repository
    return subprocess.run(
        [sys.executable, str(repo / ".githooks/check_staged.py")],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    "name",
    [
        "README.md",
        "docs/architecture.md",
        "docs/agents.md",
        "docs/product-plan.md",
        "bench/public_manifest.json",
        "bench/external_manifest.json",
        "tests/test_example.py",
        "tests/fixtures/example.json",
        "tests/data/small.csv",
        "tests/snapshots/expected.json",
        ".env.example",
        ".env.sample",
        ".env.template",
        "docs/reports/method.md",
        "docs/space and ünicode.md",
        "docs/line\nbreak.md",
        "-leading.md",
    ],
)
def test_legitimate_staged_files_are_allowed(repository, name):
    stage(repository, name)
    result = check(repository)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "name",
    [
        "AGENTS.md",
        "src/nested/AGENTS.md",
        "Rollout Diff — Product & Engineering Plan.md",
        "docs/Rollout Diff — Product & Engineering Plan.md",
        ".env",
        ".env.local",
        "secrets/key",
        "credentials.json",
        "batch.parquet",
        "batch.jsonl",
        "inputs.tgz",
        "data/raw.csv",
        "reports/metrics.json",
        "outputs/result.html",
        "bench/results/metrics.json",
        "snapshots/tool/source.py",
        "external_sources/tool/source.py",
        "build/module.py",
    ],
)
def test_private_or_generated_paths_are_rejected_even_when_force_staged(
    repository, name
):
    stage(repository, name)
    result = check(repository)
    assert result.returncode == 1
    assert "staged safety:" in result.stderr and repr(name) in result.stderr


def test_staged_secret_is_detected_after_worktree_is_cleaned(repository):
    token = b"ghp_" + b"A" * 36
    path = stage(repository, "config.txt", token)
    path.write_text("now clean\n")
    result = check(repository)
    assert result.returncode == 1 and "credential" in result.stderr
    assert token.decode() not in result.stdout + result.stderr


def test_unstaged_secret_does_not_change_staged_verdict(repository):
    path = stage(repository, "config.txt")
    path.write_bytes(b"ghp_" + b"A" * 36)
    assert check(repository).returncode == 0


def test_env_example_is_still_scanned_for_secrets(repository):
    stage(repository, ".env.example", b"TOKEN=ghp_" + b"A" * 36)
    assert check(repository).returncode == 1


def test_binary_content_is_scanned_and_large_files_fail_closed(repository):
    stage(repository, "small.bin", b"\0" + b"ghp_" + b"A" * 36)
    assert "credential" in check(repository).stderr
    stage(repository, "small.bin", b"\0safe")
    assert check(repository).returncode == 0
    stage(repository, "large.bin", b"\0" * (5 * 1024 * 1024 + 1))
    result = check(repository)
    assert result.returncode == 1 and "5 MiB" in result.stderr


def test_private_key_and_credential_url_detection(repository):
    stage(repository, "config.txt", b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----")
    assert "credential" in check(repository).stderr
    stage(repository, "config.txt", b"https://user:" + b"password@example.invalid/api")
    assert "credential" in check(repository).stderr


def test_ignore_exceptions_work_and_unstaged_policy_changes_are_rejected(repository):
    repo, _ = repository
    policy = repo / ".gitignore"
    policy.write_text(policy.read_text() + "\n*.fixture\n!allowed.fixture\n")
    git(repository, "add", ".gitignore")
    stage(repository, "allowed.fixture")
    assert check(repository).returncode == 0
    stage(repository, "blocked.fixture")
    assert check(repository).returncode == 1
    policy.write_text("")
    result = check(repository)
    assert result.returncode == 1 and "ignore policy" in result.stderr


def test_nested_unstaged_ignore_cannot_hide_a_policy_difference(repository):
    stage(repository, "docs/allowed.md")
    repo, _ = repository
    (repo / "docs/.gitignore").write_text("*.md\n")
    result = check(repository)
    assert result.returncode == 1 and "ignore policy" in result.stderr


def test_deleted_sensitive_file_is_allowed_and_renamed_destination_checked(repository):
    stage(repository, "old.txt")
    stage(repository, ".env")
    git(
        repository,
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "fixture",
    )
    git(repository, "rm", ".env")
    assert check(repository).returncode == 0
    git(repository, "mv", "old.txt", "AGENTS.md")
    assert check(repository).returncode == 1


def test_symlink_payload_is_not_followed(repository, tmp_path):
    repo, _ = repository
    secret = tmp_path / "private"
    secret.write_bytes(b"ghp_" + b"A" * 36)
    (repo / "link").symlink_to(secret)
    git(repository, "add", "link")
    result = check(repository)
    assert result.returncode == 1 and "regular file" in result.stderr
    assert "credential" not in result.stderr


@pytest.mark.parametrize(
    "lint_status,advisory_status,expected_status", [(0, 0, 0), (1, 0, 1), (0, 2, 1)]
)
def test_hook_runs_read_only_checks_and_reports_lint_failure(
    repository, lint_status, advisory_status, expected_status
):
    repo, _ = repository
    binary = repo / ".venv/bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    ruff = binary / "ruff"
    ruff.write_text(
        '#!/bin/sh\nif [ "$1" = --version ]; then echo "ruff 0.17.0"; exit 0; fi\nprintf "%s\\n" "$*" >> ruff-arguments\n'
        f'case "$*" in *--select*) exit {advisory_status} ;; esac\n'
        f'if [ "$1" = check ]; then exit {lint_status}; fi\n'
    )
    ruff.chmod(0o755)
    stage(repository, "README.md")
    index_before = (repo / ".git/index").read_bytes()
    result = subprocess.run(
        ["git", "hook", "run", "pre-commit"],
        cwd=repo,
        env=repository[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == expected_status
    assert ("Ruff lint failed" in result.stderr) == bool(lint_status)
    assert ("Ruff advisory check failed" in result.stderr) == bool(advisory_status)
    expected_commands = [
        "format --check --no-cache src bench tests",
        "check --no-cache src bench tests",
    ]
    if not lint_status:
        expected_commands.append(
            "check --no-cache --select TRY004 --exit-zero src bench tests"
        )
    assert (repo / "ruff-arguments").read_text().splitlines() == expected_commands
    assert (repo / ".git/index").read_bytes() == index_before


def test_precommit_stops_on_whitespace_without_modifying_files(repository):
    repo, _ = repository
    path = stage(repository, "README.md", b"trailing space \n")
    before = path.read_bytes()
    result = subprocess.run(
        ["git", "hook", "run", "pre-commit"],
        cwd=repo,
        env=repository[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1 and "whitespace" in result.stderr
    assert path.read_bytes() == before


def test_pre_push_runs_fast_tests_despite_external_input_environment(repository):
    repo, env = repository
    binary = repo / ".venv/bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\nmarkers = ["local_data: optional inputs"]\n'
    )
    (repo / "tests").mkdir()
    (repo / "tests/test_fast.py").write_text(
        'import pytest\n\ndef test_fast():\n    assert True\n\n@pytest.mark.local_data\ndef test_external():\n    pytest.fail("external test ran")\n'
    )
    env = env | {
        "ROLLOUTDIFF_EXTERNAL_SOURCES": "/nonexistent",
        "ROLLOUTDIFF_FULL_ARIA": "1",
    }
    result = subprocess.run(
        ["git", "hook", "run", "pre-push"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed, 1 deselected" in result.stdout + result.stderr


def test_precommit_rejects_secrets_before_whitespace_can_print_them(repository):
    repo, env = repository
    token = b"ghp_" + b"A" * 36
    stage(repository, "README.md", token + b" \n")
    result = subprocess.run(
        ["git", "hook", "run", "pre-commit"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "staged-file safety failed" in result.stderr
    assert token.decode() not in result.stdout + result.stderr


def test_pre_push_reports_failed_tests(repository):
    repo, env = repository
    binary = repo / ".venv/bin"
    binary.mkdir(parents=True)
    python = binary / "python"
    python.write_text(f'#!/bin/sh\nexec {shlex.quote(sys.executable)} "$@"\n')
    python.chmod(0o755)
    (repo / "tests").mkdir()
    (repo / "tests/test_failure.py").write_text(
        "def test_failure():\n    assert False\n"
    )
    result = subprocess.run(
        ["git", "hook", "run", "pre-push"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "fast tests failed" in result.stderr
    assert "1 failed" in result.stdout + result.stderr


def test_project_hook_and_ci_files_are_safe_to_stage(repository):
    for name in ("pre-commit", "pre-push", "check_staged.py"):
        stage(repository, f".githooks/{name}", (HOOKS / name).read_bytes())
    for name in (
        ".github/ci-constraints.txt",
        ".github/workflows/ci.yml",
        "tests/test_git_hooks.py",
    ):
        stage(repository, name, (PROJECT / name).read_bytes())
    result = check(repository)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture
def ruff_binary():
    binary = Path(sys.executable).with_name("ruff")
    if binary.is_file():
        return str(binary)
    binary = shutil.which("ruff")
    if binary is None:
        pytest.skip("install the pinned Ruff to exercise lint policy")
    return binary


def test_required_lint_preserves_value_errors_and_other_rules(tmp_path, ruff_binary):
    (tmp_path / "pyproject.toml").write_bytes((PROJECT / "pyproject.toml").read_bytes())
    source = tmp_path / "validation.py"
    value_error = (
        "def validate(value):\n"
        "    if not isinstance(value, str):\n"
        '        raise ValueError("value must be text")\n'
    )
    source.write_text(value_error)
    command = [ruff_binary, "check", "--no-cache", "--output-format", "json"]
    required = subprocess.run(
        [*command, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert required.returncode == 0, required.stdout + required.stderr
    assert json.loads(required.stdout) == []

    source.write_text("import os\n\n\n" + value_error)
    required = subprocess.run(
        [*command, str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert required.returncode == 1
    assert {row["code"] for row in json.loads(required.stdout)} == {"F401"}

    advisory = subprocess.run(
        [*command, "--select", "TRY004", "--exit-zero", str(source)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert advisory.returncode == 0
    assert [row["code"] for row in json.loads(advisory.stdout)] == ["TRY004"]


def test_precommit_reports_real_advisory_without_blocking(repository, ruff_binary):
    repo, env = repository
    binary = repo / ".venv/bin"
    binary.mkdir(parents=True)
    for name, executable in (("python", sys.executable), ("ruff", ruff_binary)):
        wrapper = binary / name
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(executable)} "$@"\n')
        wrapper.chmod(0o755)
    for name in ("src", "bench", "tests"):
        (repo / name).mkdir()
    stage(repository, "pyproject.toml", (PROJECT / "pyproject.toml").read_bytes())
    stage(
        repository,
        "src/validation.py",
        (
            b"def validate(value):\n"
            b"    if not isinstance(value, str):\n"
            b'        raise ValueError("value must be text")\n'
        ),
    )
    result = subprocess.run(
        ["git", "hook", "run", "pre-commit"],
        cwd=repo,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "TRY004" in result.stdout + result.stderr
    assert "advisory" in result.stdout + result.stderr
