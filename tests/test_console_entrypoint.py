import subprocess
import sysconfig
from importlib.metadata import distributions
from pathlib import Path

import pytest


def console():
    # pytest adds src to sys.path, where stale editable-build metadata may remain.
    locations = sorted({sysconfig.get_path("purelib"), sysconfig.get_path("platlib")})
    installed = [
        package
        for package in distributions(path=locations)
        if package.metadata["Name"] == "rolloutdiff"
    ]
    assert len(installed) == 1
    scripts = {
        point.name: point.value
        for point in installed[0].entry_points
        if point.group == "console_scripts"
    }
    assert scripts == {"rollout": "rolloutdiff.cli:main"}
    executable = Path(sysconfig.get_path("scripts")) / "rollout"
    assert executable.is_file(), "reinstall the project to refresh its entry point"
    return executable


def test_installed_metadata_is_not_shadowed_by_source_build_metadata(
    tmp_path, monkeypatch
):
    stale = tmp_path / "rolloutdiff-0.0.1.dist-info"
    stale.mkdir()
    (stale / "METADATA").write_text("Name: rolloutdiff\nVersion: 0.0.1\n")
    (stale / "entry_points.txt").write_text(
        "[console_scripts]\nrolloutdiff = rolloutdiff.cli:main\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    assert console().name == "rollout"


@pytest.mark.parametrize(
    "subcommand",
    [
        [],
        ["analyze"],
        ["diff"],
        ["render"],
        ["inspect"],
        ["replay-aria"],
        ["investigate-aria"],
        ["investigate-aria", "prepare"],
        ["investigate-aria", "approve"],
        ["investigate-aria", "run"],
        ["investigate-aria", "compare"],
    ],
)
def test_installed_console_help_uses_rollout(subcommand):
    completed = subprocess.run(
        [str(console()), *subcommand, "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage: rollout " in completed.stdout
    if subcommand == ["render"]:
        assert "analysis or comparison JSON" in completed.stdout
    assert "rolloutdiff" not in completed.stdout
    assert not completed.stderr


def test_installed_console_errors_use_rollout():
    completed = subprocess.run(
        [str(console()), "unknown-command"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "rollout: error:" in completed.stderr
    assert "rolloutdiff" not in completed.stderr
    assert not completed.stdout


def test_installed_replay_refuses_execution_without_authorization(tmp_path):
    output = tmp_path / "replay.json"
    completed = subprocess.run(
        [
            str(console()),
            "replay-aria",
            str(tmp_path / "missing.parquet"),
            "--run",
            "aria",
            "--sample-id",
            "step001_row001",
            "--response-sha256",
            "0" * 64,
            "--artifacts",
            str(tmp_path / "absent"),
            "--docker-host",
            "unix:///unused-test-socket",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 1
    assert (
        "rollout: error: replay requires explicit execution authorization"
        in completed.stderr
    )
    assert "Traceback" not in completed.stderr
    assert not completed.stdout
    assert not output.exists()
