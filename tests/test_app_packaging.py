"""Bundled font assets retain their redistribution notice."""

from importlib.resources import files
from pathlib import Path


def test_mona_sans_assets_include_copyright_and_license():
    assets = files("rolloutdiff").joinpath("_web", "assets")
    notice = assets.joinpath("mona-sans-OFL.txt").read_bytes()
    source = (
        Path(__file__).resolve().parents[1] / "frontend/public/assets/mona-sans-OFL.txt"
    ).read_bytes()
    assert notice == source
    assert b"Copyright 2022 The Mona Sans Project Authors" in notice
    assert b"SIL OPEN FONT LICENSE Version 1.1" in notice
    fonts = [entry for entry in assets.iterdir() if entry.name.endswith(".woff2")]
    assert len(fonts) == 3
    assert all(entry.name.startswith("mona-sans-") for entry in fonts)
    assert all(entry.read_bytes().startswith(b"wOF2") for entry in fonts)
