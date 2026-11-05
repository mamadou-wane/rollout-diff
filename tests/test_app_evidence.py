"""Saved evidence uses synthetic records and never invokes an execution transport."""

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest
import test_comparison_report
import test_investigation
from test_investigation import api, prepare, write_json

from rolloutdiff import _replay_container as boundary
from rolloutdiff import replay

comparison_report = test_comparison_report.report
fixture = test_investigation.fixture


def load(path, root, **kwargs):
    module = importlib.import_module("rolloutdiff.app_evidence")
    return module.load_evidence(path, root=root, **kwargs)


def prohibit_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("saved evidence attempted execution or preparation")

    for module, names in [
        (replay, ["replay_aria", "run_container"]),
        (boundary, ["run_container"]),
        (
            api(),
            [
                "prepare_aria_intervention",
                "approve_aria_intervention",
                "execute_aria_intervention",
                "compare_aria_intervention",
                "_inputs",
                "_case",
            ],
        ),
    ]:
        for name in names:
            monkeypatch.setattr(module, name, forbidden)


def comparison(data):
    manifest = prepare(data)
    manifest.pop("case_sha256")
    value = api()._comparison(
        manifest, data["record"], data["record"], status="completed", origin="saved"
    )
    path = data["directory"] / "comparison.json"
    write_json(path, value)
    return path, value


def test_replay_view_separates_untrusted_claim_and_protected_failure(
    fixture, monkeypatch, tmp_path
):
    prohibit_execution(monkeypatch)
    value = load(fixture["control"], tmp_path)
    assert value["kind"] == "replay"
    assert value["identity"] == {
        "run": "aria",
        "sample_id": "step001_row000",
        "step": 1,
        "prompt_id": "7",
    }
    condition = value["conditions"][0]
    assert condition["historical_passed"] == 1
    assert condition["historical_evaluated"] == 1
    assert condition["task_determinate"] is True
    assert condition["task_passed"] == 1
    assert condition["task_failed"] == 1
    assert condition["task_completed"] == condition["task_applicable"] == 2
    assert condition["discrepancy"] is True
    assert value["mechanism_status"] == "unresolved"
    assert value["source_link"]["status"] == "not_loaded"
    assert "recorded" in value["verification"]


def test_prepared_case_verifies_files_and_only_returns_review_fields(
    fixture, monkeypatch, tmp_path
):
    fixture["hypothesis"] = '<script>alert("untrusted")</script>'
    prepare(fixture)
    prohibit_execution(monkeypatch)
    value = load(fixture["directory"] / "case.json", tmp_path)
    assert value["kind"] == "case"
    assert value["hypothesis"] == fixture["hypothesis"]
    assert value["edit"] == {**fixture["edit"], "occurrence": 1}
    assert value["source_diff"].startswith("--- canonical-response")
    assert value["predictions"][0]["status"] == "unavailable"
    text = json.dumps(value)
    for private in [
        "docker_host",
        "canonical_solution",
        "worker_inputs",
        str(tmp_path),
    ]:
        assert private not in text
    assert fixture["response"] not in text


def test_intervention_preserves_both_conditions_and_literal_predictions(
    fixture, monkeypatch, tmp_path
):
    path, record = comparison(fixture)
    prohibit_execution(monkeypatch)
    value = load(path, tmp_path)
    assert value["kind"] == "intervention"
    assert [row["name"] for row in value["conditions"]] == ["control", "variant"]
    assert value["predictions"] == record["predictions"]
    assert value["hypothesis"] == fixture["hypothesis"]
    assert value["edit"] == {**fixture["edit"], "occurrence": 1}
    assert value["summary"]["answer_stream_equal"] is True


def test_partial_and_missing_counters_stay_unavailable(fixture, tmp_path):
    record = fixture["record"]
    record["historical"] = {"status": "unavailable", "reason": "not_attempted"}
    record["task"] = {
        "status": "unavailable",
        "reason": "not_attempted",
        "determinate": False,
    }
    write_json(fixture["control"], record)
    condition = load(fixture["control"], tmp_path)["conditions"][0]
    assert condition["historical_passed"] is None
    assert condition["task_passed"] is None
    assert condition["task_determinate"] is False
    assert condition["discrepancy"] is None


def test_legacy_zero_claim_is_not_a_negative_result(fixture, tmp_path):
    claim = fixture["record"]["historical"]["reported_result"]
    claim.update(tests_passed=0, tests_evaluated=0)
    fixture["record"]["implementation_sha256"] = {"replay.py": "0" * 64}
    write_json(fixture["control"], fixture["record"])
    value = load(fixture["control"], tmp_path)
    condition = value["conditions"][0]
    assert condition["historical_passed"] is None
    assert condition["historical_evaluated"] is None
    assert condition["historical_status"] == "unavailable"
    assert condition["discrepancy"] is None
    assert "legacy" in condition["historical_reason"]


@pytest.mark.parametrize(
    "change",
    [
        {"tests_passed": 2},
        {"tests_applicable": 3},
        {"test_coverage_complete": False},
        {"tests_completed": True},
    ],
)
def test_contradictory_task_counts_are_rejected(fixture, tmp_path, change):
    fixture["record"]["task"].update(change)
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError, match=r"task|determinate"):
        load(fixture["control"], tmp_path)


@pytest.mark.parametrize("field", ["control.json", "variant.txt", "source.diff"])
def test_case_rejects_changed_sibling(fixture, tmp_path, field):
    prepare(fixture)
    (fixture["directory"] / field).write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        load(fixture["directory"] / "case.json", tmp_path)


def test_case_rejects_self_hash_and_intervention_provenance(fixture, tmp_path):
    path, record = comparison(fixture)
    record["provenance"]["parent_response_sha256"] = "0" * 64
    write_json(path, record)
    with pytest.raises(ValueError, match=r"provenance|metadata"):
        load(path, tmp_path)
    manifest_path = fixture["directory"] / "case.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["hypothesis"] = "changed"
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="identity hash"):
        load(manifest_path, tmp_path)


@pytest.mark.parametrize("sibling", [False, True])
def test_symlinks_cannot_read_outside_root(fixture, tmp_path, sibling):
    root = tmp_path / "root"
    root.mkdir()
    if sibling:
        fixture["directory"] = root / "case"
        prepare(fixture)
        diff = fixture["directory"] / "source.diff"
        outside = tmp_path / "private.diff"
        outside.write_bytes(diff.read_bytes())
        diff.unlink()
        diff.symlink_to(outside)
        path = fixture["directory"] / "case.json"
    else:
        path = root / "record.json"
        path.symlink_to(fixture["control"])
    with pytest.raises(ValueError, match=r"root|regular|symlink") as caught:
        load(path, root)
    assert str(tmp_path) not in str(caught.value)


@pytest.mark.parametrize(
    "content",
    ['{"schema_version":1,"schema_version":1}', '{"x":NaN}', '{"x":1e999}'],
)
def test_json_is_strict(tmp_path, content):
    path = tmp_path / "record.json"
    path.write_text(content)
    with pytest.raises(ValueError, match="JSON"):
        load(path, tmp_path)


def test_source_binding_distinguishes_export_and_original(fixture, tmp_path):
    identity = fixture["record"]["identity"]
    rows = pd.DataFrame([{**identity, "response": fixture["response"]}])
    source = {"files": [{"sha256": fixture["record"]["source"]["sha256"]}]}
    assert (
        load(fixture["control"], tmp_path, rollouts=rows, source=source)["source_link"][
            "status"
        ]
        == "matched"
    )
    source["files"][0]["sha256"] = "0" * 64
    assert (
        load(fixture["control"], tmp_path, rollouts=rows, source=source)["source_link"][
            "status"
        ]
        == "response_only"
    )
    rows.loc[0, "prompt_id"] = "other"
    assert (
        load(fixture["control"], tmp_path, rollouts=rows, source=source)["source_link"][
            "status"
        ]
        == "mismatch"
    )


def test_saved_reports_are_validated_without_source_or_payloads(
    tmp_path, comparison_report, monkeypatch
):
    prohibit_execution(monkeypatch)
    for name, report in [
        ("analysis", {"schema_version": 1, "runs": [], "response": "PRIVATE"}),
        ("comparison", comparison_report),
    ]:
        path = tmp_path / (name + ".json")
        write_json(path, report)
        view = load(path, tmp_path)
        assert view["kind"] == name
        assert view["source_link"]["status"] == "not_loaded"
        assert "PRIVATE" not in json.dumps(view)
        assert view["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unknown_runtime_is_not_reinterpreted(fixture, tmp_path):
    record = copy.deepcopy(fixture["record"])
    record["task"]["runtime"]["metadata"]["python_version"] = "unknown"
    write_json(fixture["control"], record)
    with pytest.raises(ValueError, match="runtime"):
        load(fixture["control"], tmp_path)


@pytest.mark.parametrize("mutation", ["control", "prediction", "expectation"])
def test_intervention_rejects_contradictions_with_frozen_case(
    fixture, tmp_path, mutation
):
    path, record = comparison(fixture)
    if mutation == "control":
        record["control"]["historical"]["reported_result"]["tests_passed"] = 0
    elif mutation == "prediction":
        record["predictions"][0]["status"] = "matched"
    else:
        record["predictions"][0]["expected"] = ["TypeError"]
    write_json(path, record)
    with pytest.raises(ValueError, match=r"control|prediction"):
        load(path, tmp_path)


def test_report_only_intervention_remains_readable(fixture, tmp_path):
    _, record = comparison(fixture)
    archive = tmp_path / "archive"
    archive.mkdir()
    path = archive / "comparison.json"
    write_json(path, record)
    view = load(path, tmp_path)
    assert view["kind"] == "intervention"
    assert view["source_diff"] is None
    assert view["edit"] is None
    assert view["conditions"][0]["task_determinate"] is True


def test_missing_legacy_historical_counters_are_unknown(fixture, tmp_path):
    del fixture["record"]["historical"]["reported_result"]["tests_evaluated"]
    write_json(fixture["control"], fixture["record"])
    condition = load(fixture["control"], tmp_path)["conditions"][0]
    assert condition["historical_passed"] is None
    assert condition["historical_evaluated"] is None


def test_legacy_claim_cannot_hide_contradictory_counts(fixture, tmp_path):
    fixture["record"]["historical"]["reported_result"].update(
        tests_passed=1, tests_evaluated=0
    )
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError, match="historical counts"):
        load(fixture["control"], tmp_path)


def test_unverified_cleanup_preserves_counts_without_determinate(fixture, tmp_path):
    task = fixture["record"]["task"]
    task.update(
        status="unresolved",
        determinate=False,
        test_coverage_complete=False,
        reason="task_container_infrastructure_failure",
    )
    task["runtime"]["cleanup_verified"] = False
    write_json(fixture["control"], fixture["record"])
    condition = load(fixture["control"], tmp_path)["conditions"][0]
    assert condition["task_completed"] == 2
    assert condition["task_determinate"] is False
    assert condition["cleanup_verified"] is False
    assert condition["discrepancy"] is None


def test_large_file_and_directory_are_rejected(tmp_path):
    path = tmp_path / "large.json"
    with path.open("wb") as stream:
        stream.truncate(16 * 1024 * 1024 + 1)
    for candidate in [path, tmp_path / "directory"]:
        if candidate != path:
            candidate.mkdir()
        with pytest.raises(ValueError, match="regular bounded"):
            load(candidate, tmp_path)


def test_recorded_discrepancy_requires_usable_historical_runtime(fixture, tmp_path):
    fixture["record"]["historical"]["runtime"]["cleanup_verified"] = False
    write_json(fixture["control"], fixture["record"])
    condition = load(fixture["control"], tmp_path)["conditions"][0]
    assert condition["historical_passed"] == 1
    assert condition["discrepancy"] is None


def test_missing_evaluator_contract_cannot_support_determinate(fixture, tmp_path):
    fixture["record"].pop("evaluator")
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError, match="evaluator"):
        load(fixture["control"], tmp_path)


def test_unknown_payloads_and_private_metadata_never_enter_view(fixture, tmp_path):
    record = fixture["record"]
    for target in [
        record,
        record["source"],
        record["historical"],
        record["task"],
        record["task"]["runtime"],
    ]:
        target.update(
            stdout="PRIVATE_STDOUT",
            answers=["PRIVATE_ANSWER"],
            program="PRIVATE_PROGRAM",
            path="/private/SECRET",
        )
    write_json(fixture["control"], record)
    serialized = json.dumps(load(fixture["control"], tmp_path))
    assert "PRIVATE_" not in serialized
    assert "/private/SECRET" not in serialized


@pytest.mark.parametrize("swap_ancestor", [False, True])
def test_root_and_ancestor_swaps_cannot_redirect_evidence_read(
    tmp_path, monkeypatch, swap_ancestor
):
    parent = tmp_path / "granted-parent"
    root = parent / "root"
    root.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    replacement = outside / "root" if swap_ancestor else outside
    replacement.mkdir(exist_ok=True)
    for directory in [root, replacement]:
        write_json(directory / "saved.json", {"schema_version": 1, "runs": []})
    validated_path = (root / "saved.json").resolve(strict=True)
    swapped = parent if swap_ancestor else root
    retained = tmp_path / "retained"
    original_open = os.open
    did_swap = False

    def racing_open(path, flags, *args, **kwargs):
        nonlocal did_swap
        if not did_swap and Path(path) in {root, Path(swapped.name)}:
            did_swap = True
            swapped.rename(retained)
            swapped.symlink_to(outside, target_is_directory=True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", racing_open)
    try:
        with pytest.raises(ValueError, match=r"root|symlink") as caught:
            load(validated_path, root)
        assert str(tmp_path) not in str(caught.value)
        assert did_swap
    finally:
        if did_swap:
            swapped.unlink()
            retained.rename(swapped)
