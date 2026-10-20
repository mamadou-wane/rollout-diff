"""Data-free checks of the shipped public manifest."""

import json
from pathlib import Path

from test_pipeline import _LOCAL_FILES

MANIFEST = Path(__file__).resolve().parents[1] / "bench" / "public_manifest.json"
EXPECTED_FILES = {"aria": 1, "aisi0": 1, "aisi2": 1, "trl": 115}


def shipped():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_shipped_manifest_pins_every_file_to_its_declared_revision():
    manifest = shipped()
    assert manifest["schema_version"] == 1
    assert set(manifest["runs"]) == set(EXPECTED_FILES)
    for run, spec in manifest["runs"].items():
        assert len(spec["files"]) == EXPECTED_FILES[run]
        prefix = (
            f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/"
            f"{spec['revision']}/"
        )
        for entry in spec["files"]:
            assert entry["url"] == prefix + entry["upstream_path"]
            assert len(entry["sha256"]) == 64 and entry["size_bytes"] > 0
    assert sum(len(spec["files"]) for spec in manifest["runs"].values()) == 118


def test_shipped_manifest_license_status_matches_reuse_class_and_attribution():
    manifest = shipped()
    assert manifest["all_four_licensed"] is False
    for spec in manifest["runs"].values():
        declared = spec["license"]["status"] == "declared"
        caveated = "declared_apache_2_0_with_underlying_source_caveats"
        assert (spec["reuse_class"] == caveated) == declared
        if declared:
            assert spec["license"]["identifier"] == "apache-2.0"
        else:
            assert spec["reuse_class"] == "optional_external_unresolved_license"
        assert spec["attribution"].strip()


def test_shipped_trl_pins_match_the_local_fixture_hashes():
    trl = {
        Path(entry["local_path"]).name: entry["sha256"]
        for entry in shipped()["runs"]["trl"]["files"]
    }
    for name, digest in _LOCAL_FILES.items():
        assert trl[name] == digest
