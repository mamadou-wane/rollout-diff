"""The local launcher binds its socket before exposing a session URL."""

from urllib.parse import parse_qs, urlsplit

import pytest
from test_cli import _subprocess


def test_installed_investigate_help_is_readonly():
    result = _subprocess(["investigate", "--help"])
    assert result.returncode == 0, result.stderr
    assert "--root" in result.stdout
    assert "--port" in result.stdout
    assert "--no-browser" in result.stdout
    assert "--host" not in result.stdout
    assert "--allow-execution" not in result.stdout


def test_launcher_uses_loopback_socket_without_proxy_headers(
    tmp_path, monkeypatch, capsys
):
    import uvicorn

    from rolloutdiff import cli

    calls = []

    def run(server, sockets):
        calls.append((server.config, sockets[0].getsockname()))

    monkeypatch.setattr(uvicorn.Server, "run", run)
    assert (
        cli.main(
            ["investigate", "--root", str(tmp_path), "--port", "0", "--no-browser"]
        )
        == 0
    )
    assert len(calls) == 1
    config, address = calls[0]
    assert address[0] == "127.0.0.1" and address[1] > 0
    assert config.proxy_headers is False
    assert config.access_log is False
    url = next(
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("http://")
    )
    parsed = urlsplit(url)
    assert parsed.netloc == f"127.0.0.1:{address[1]}"
    assert parsed.query == ""
    assert len(parse_qs(parsed.fragment)["token"][0]) >= 32


@pytest.mark.parametrize("port", ["-1", "65536", "x"])
def test_launcher_rejects_invalid_ports(port, tmp_path):
    from rolloutdiff.cli import main

    with pytest.raises(SystemExit):
        main(["investigate", "--root", str(tmp_path), "--port", port, "--no-browser"])


def test_launcher_missing_root_fails_before_server_or_browser(
    tmp_path, monkeypatch, capsys
):
    import webbrowser

    import uvicorn

    from rolloutdiff.cli import main

    monkeypatch.setattr(
        uvicorn.Server, "run", lambda *a, **k: pytest.fail("unexpected server")
    )
    monkeypatch.setattr(
        webbrowser, "open", lambda *a, **k: pytest.fail("unexpected browser")
    )
    assert main(["investigate", "--root", str(tmp_path / "missing")]) == 1
    assert "error:" in capsys.readouterr().err


def test_launcher_rejects_unsupported_file_boundary_before_binding(
    tmp_path, monkeypatch, capsys
):
    from rolloutdiff import app, cli

    def unavailable():
        raise ValueError("Local investigation requires macOS or Linux.")

    monkeypatch.setattr(app, "_require_local_platform", unavailable)
    monkeypatch.setattr(
        cli.socket, "socket", lambda *a, **k: pytest.fail("unexpected socket")
    )
    assert cli.main(["investigate", "--root", str(tmp_path), "--no-browser"]) == 1
    assert "macOS or Linux" in capsys.readouterr().err


def test_launcher_interrupt_closes_without_a_traceback(tmp_path, monkeypatch, capsys):
    import uvicorn

    from rolloutdiff.cli import main

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(uvicorn.Server, "run", interrupted)
    try:
        result = main(
            ["investigate", "--root", str(tmp_path), "--port", "0", "--no-browser"]
        )
    except KeyboardInterrupt:
        pytest.fail("launcher leaked the interruption")
    assert result == 130
    assert not capsys.readouterr().err
