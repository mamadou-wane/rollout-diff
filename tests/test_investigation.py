"""Investigation orchestration uses synthetic data and mocked transports only."""

import copy
import hashlib
import importlib
import json

import pandas as pd
import pytest

from rolloutdiff import _aria_replay as aria
from rolloutdiff import _replay_container as boundary
from rolloutdiff import replay


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True))


def raw_result(output, index=0, **kwargs):
    return boundary.ContainerResult(
        "completed",
        exit_code=0,
        stdout=json.dumps(output).encode(),
        config_verified=True,
        cleanup_verified=True,
        metadata={
            "image_id": boundary.DEFAULT_IMAGE,
            "architecture": "arm64",
            "python_version": "3.12.13",
            "runtime": "runc",
            "cgroup_version": "2",
            "container_id": str(index + 1) * 64,
            "limits": {
                "memory_bytes": 1073741824,
                "cpus": 1,
                "pids": 16,
                "wall_seconds": 6,
                "output_bytes": 1048576,
                "stdin_bytes": 2097152,
                "tmp_bytes": 16777216,
            },
        },
        **kwargs,
    )


def historical(passed=1):
    return {
        "success": True,
        "compiled": True,
        "timeout": False,
        "oom": False,
        "stdout": {
            "tests_evaluated": 1,
            "tests_total": 1,
            "tests_passed": passed,
            "test_errors": [] if passed else ["AssertionError: fixture"],
        },
    }


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    response = "```python\nclass Solution:\n def answer(self, x): return x + 1\n def run_tests(self):\n  print(self.answer(0))\n```"
    prompt = (
        "\nuser\ntask and will be evaluated by calling a function called run_tests()"
    )
    source = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "sample_id": "step001_row000",
                "step": 1,
                "problem_id": 7,
                "messages": json.dumps(
                    [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                ),
            }
        ]
    ).to_parquet(source)
    monkeypatch.setattr(replay, "ARIA_SHA256", digest(source.read_bytes()))
    monkeypatch.setattr(aria, "ARIA_SHA256", digest(source.read_bytes()))
    task = {
        "id": 7,
        "func_name": "Solution().answer",
        "gt_answer": [
            "assert Solution().answer(x=0) == 1",
            "assert Solution().answer(x=1) == 2",
        ],
        "setup_code": "",
        "canonical_solution": "private task answer",
        "prompt": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task and should pass all tests"},
        ],
    }
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    for kind, name in aria.ARTIFACT_FILES.items():
        content = (
            json.dumps(task) + "\n" if kind == "fixture" else "# pinned fixture " + kind
        )
        (artifacts / name).write_text(content)
        monkeypatch.setitem(aria.ARTIFACT_SHA256, kind, digest(content.encode()))
    monkeypatch.setattr(
        replay,
        "historical_script",
        lambda paths, setup, program, call: (
            "synthetic runner",
            (program + "\n" + call).encode(),
        ),
    )
    monkeypatch.setattr(aria, "historical_script", replay.historical_script)
    outputs = iter(
        [
            raw_result(historical()),
            raw_result({"results": [{"value": ["int", 1]}, {"value": ["int", 3]}]}, 1),
        ]
    )
    monkeypatch.setattr(replay, "run_container", lambda *a, **kw: next(outputs))
    record = replay.replay_aria(
        source,
        run="aria",
        sample_id="step001_row000",
        response_sha256=digest(response.encode()),
        artifacts=artifacts,
        execution_authorized=True,
    )
    control = tmp_path / "control.json"
    write_json(control, record)
    monkeypatch.setattr(
        boundary,
        "run_container",
        lambda *a, **kw: pytest.fail("unmocked Docker transport"),
    )
    return {
        "source": source,
        "artifacts": artifacts,
        "control": control,
        "record": record,
        "response": response,
        "directory": tmp_path / "case",
        "edit": {"old": "print(self.answer(0))", "new": "assert self.answer(0) == 2"},
        "hypothesis": "An executable assertion changes the reported pass.",
        "predictions": {
            "historical_passed": 0,
            "historical_evaluated": 1,
            "exception_types": ["AssertionError"],
            "task_matched": 1,
            "task_mismatched": 1,
            "task_stdout_sha256": record["task"]["runtime"]["stdout_sha256"],
        },
    }


def api():
    return importlib.import_module("rolloutdiff.investigation")


def prepare(data):
    return api().prepare_aria_intervention(
        data["control"],
        data["source"],
        **{
            key: data[key]
            for key in ("artifacts", "edit", "hypothesis", "predictions", "directory")
        },
    )


def approve(data, prepared):
    return api().approve_aria_intervention(
        data["directory"],
        case_sha256=prepared["case_sha256"],
        variant_sha256=prepared["variant_response_sha256"],
        max_containers=2,
        execution_authorized=True,
    )


def test_prepare_freezes_exact_control_and_one_edit(fixture):
    module = api()
    original = fixture["control"].read_bytes(), fixture["source"].read_bytes()
    before = copy.deepcopy(fixture["edit"])
    prepared = prepare(fixture)
    directory = fixture["directory"]
    assert (directory / "control.json").read_bytes() == original[0]
    assert (directory / "variant.txt").read_text() == fixture["response"].replace(
        fixture["edit"]["old"], fixture["edit"]["new"]
    )
    assert prepared["parent_identity"] == fixture["record"]["identity"]
    assert prepared["variant_response_sha256"] != prepared["parent_response_sha256"]
    assert prepared["case_sha256"] == digest((directory / "case.json").read_bytes())
    assert fixture["edit"] == before
    assert (fixture["control"].read_bytes(), fixture["source"].read_bytes()) == original
    assert not (directory / "approval.json").exists()
    assert callable(module.compare_aria_intervention)


@pytest.mark.parametrize(
    "change", ["source", "identity", "counts", "cleanup", "version", "implementation"]
)
def test_incompatible_control_is_rejected(fixture, change):
    record = fixture["record"]
    if change == "source":
        fixture["source"].write_bytes(b"changed")
    elif change == "identity":
        record["identity"]["step"] = 2
    elif change == "counts":
        record["task"]["tests_applicable"] = 1
    elif change == "cleanup":
        record["task"]["runtime"]["cleanup_verified"] = False
    elif change == "version":
        record["replay_schema_version"] = 2
    else:
        record["implementation_sha256"]["replay.py"] = "0" * 64
    write_json(fixture["control"], record)
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture["directory"].exists()


@pytest.mark.parametrize(
    "edit",
    [
        {"old": "missing", "new": "x"},
        {"old": "def", "new": "x"},
        {"old": "", "new": "x"},
        {"old": "print", "new": "print"},
    ],
)
def test_edit_requires_one_nonempty_changed_occurrence(fixture, edit):
    fixture["edit"] = edit
    with pytest.raises(ValueError):
        prepare(fixture)


def test_approval_is_explicit_bound_and_single_use(fixture):
    prepared = prepare(fixture)
    module = api()
    with pytest.raises(ValueError, match="authorization"):
        module.approve_aria_intervention(
            fixture["directory"],
            case_sha256=prepared["case_sha256"],
            variant_sha256=prepared["variant_response_sha256"],
            max_containers=2,
        )
    with pytest.raises(ValueError, match="hash"):
        module.approve_aria_intervention(
            fixture["directory"],
            case_sha256="0" * 64,
            variant_sha256=prepared["variant_response_sha256"],
            max_containers=2,
            execution_authorized=True,
        )
    approve(fixture, prepared)
    with pytest.raises((ValueError, FileExistsError)):
        approve(fixture, prepared)
    with pytest.raises(ValueError, match="authorization"):
        module.execute_aria_intervention(
            fixture["directory"], fixture["source"], artifacts=fixture["artifacts"]
        )


def test_run_preserves_answers_and_compares_predictions(fixture, monkeypatch):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    module = api()
    calls = []

    def transport(script, payload, config, *, execution_authorized):
        assert execution_authorized is True
        assert (fixture["directory"] / "consumed.json").is_file()
        calls.append(payload)
        return raw_result(
            historical(0)
            if len(calls) == 1
            else {"results": [{"value": ["int", 1]}, {"value": ["int", 3]}]},
            len(calls) + 1,
        )

    monkeypatch.setattr(boundary, "run_container", transport)
    report = module.execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert len(calls) == 2
    assert all(row["status"] == "matched" for row in report["predictions"])
    assert report["answer_stream_equal"] is True
    assert report["mechanism_status"] == "unresolved"
    assert report["variant"]["task"]["tests_failed"] == 1
    assert (fixture["directory"] / "comparison.html").is_file()
    assert b"private task answer" not in b"".join(calls)
    with pytest.raises(ValueError, match="spent"):
        module.execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert len(calls) == 2


@pytest.mark.parametrize("cancel", [False, True])
def test_cleanup_failure_preserves_partial_evidence_and_stops(
    fixture, monkeypatch, cancel
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    module = api()
    calls = []
    failure = boundary.ContainerResult(
        "infrastructure_error",
        cleanup_verified=False,
        metadata={"container_id": "c" * 64},
    )

    def transport(*args, **kwargs):
        calls.append(True)
        if cancel:
            error = KeyboardInterrupt()
            error.replay_result = failure
            raise error
        return failure

    monkeypatch.setattr(boundary, "run_container", transport)
    if cancel:
        with pytest.raises(KeyboardInterrupt):
            module.execute_aria_intervention(
                fixture["directory"],
                fixture["source"],
                artifacts=fixture["artifacts"],
                execution_authorized=True,
            )
    else:
        result = module.execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
        assert result["status"] == "stopped"
    assert len(calls) == 1
    saved = json.loads((fixture["directory"] / "result.json").read_text())
    assert saved["historical"]["runtime"]["metadata"]["container_id"] == "c" * 64
    assert saved["task"]["status"] == "unavailable"
    assert (fixture["directory"] / "consumed.json").is_file()


def test_spent_approval_cannot_be_reused_after_copy(fixture, tmp_path):
    import shutil

    prepared = prepare(fixture)
    approve(fixture, prepared)
    copied = tmp_path / "copied"
    shutil.copytree(fixture["directory"], copied)
    with pytest.raises(ValueError, match="directory"):
        api().execute_aria_intervention(
            copied,
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )


def test_modified_variant_is_rejected_before_execution(fixture):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    (fixture["directory"] / "variant.txt").write_text("modified")
    with pytest.raises(ValueError, match="hash"):
        api().execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert not (fixture["directory"] / "consumed.json").exists()


def test_missing_approval_and_invalid_budget_never_execute(fixture):
    prepared = prepare(fixture)
    with pytest.raises(FileNotFoundError):
        api().execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    for budget in (0, 1, 3, True):
        with pytest.raises(ValueError, match="budget"):
            api().approve_aria_intervention(
                fixture["directory"],
                case_sha256=prepared["case_sha256"],
                variant_sha256=prepared["variant_response_sha256"],
                max_containers=budget,
                execution_authorized=True,
            )
    assert not (fixture["directory"] / "consumed.json").exists()


def test_changed_source_after_approval_rejected(fixture):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    fixture["source"].write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        api().execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert not (fixture["directory"] / "consumed.json").exists()


def test_valid_unresolved_control_remains_unresolved(fixture):
    task = fixture["record"]["task"]
    task.update(
        status="unresolved",
        reason="incomplete_or_unsupported_answers",
        determinate=False,
        test_coverage_complete=False,
        tests_completed=1,
        tests_passed=0,
        tests_failed=1,
        reported_exceptions=["ValueError"],
    )
    write_json(fixture["control"], fixture["record"])
    prepare(fixture)
    assert (
        json.loads((fixture["directory"] / "control.json").read_text())["task"] == task
    )


def archived_result(data, prepared, tmp_path):
    protocol = {
        key: prepared[key]
        for key in (
            "control_result_sha256",
            "parent_identity",
            "parent_response_sha256",
            "source_sha256",
            "variant_response_sha256",
            "variant_program_sha256",
            "implementation_sha256",
            "evaluator",
            "task_source",
            "configuration",
            "worker_inputs",
        )
    }
    protocol["variant_id"] = "archived-variant"
    provenance = tmp_path / "protocol.json"
    write_json(provenance, protocol)
    result = {
        key: copy.deepcopy(data["record"][key])
        for key in ("historical", "task", "mechanism_status")
    }
    result.update(
        derived_replay_schema_version=1,
        parent_identity=prepared["parent_identity"],
        variant_id=protocol["variant_id"],
        variant_response_sha256=prepared["variant_response_sha256"],
        extraction=prepared["extraction"],
        protocol_sha256=digest(provenance.read_bytes()),
    )
    result["historical"]["reported_result"].update(
        tests_passed=0, exception_types=["AssertionError"]
    )
    path = tmp_path / "derived.json"
    write_json(path, result)
    return path, provenance, result


def test_saved_comparison_import_never_executes_and_preserves_records(
    fixture, tmp_path
):
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    before = path.read_bytes(), provenance.read_bytes(), fixture["control"].read_bytes()
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    assert all(row["status"] == "matched" for row in report["predictions"])
    assert report["origin"] == "saved_evidence"
    assert report["answer_stream_equal"] is True
    assert (
        path.read_bytes(),
        provenance.read_bytes(),
        fixture["control"].read_bytes(),
    ) == before
    assert not (fixture["directory"] / "approval.json").exists()


@pytest.mark.parametrize(
    "mutation", ["task_count", "protocol", "extraction", "variant"]
)
def test_saved_comparison_rejects_mismatched_evidence(fixture, tmp_path, mutation):
    prepared = prepare(fixture)
    path, provenance, result = archived_result(fixture, prepared, tmp_path)
    if mutation == "task_count":
        result["task"]["tests_applicable"] = 1
    elif mutation == "protocol":
        provenance.write_text("{}")
    elif mutation == "extraction":
        result["extraction"]["requested_call"] = "other()"
    else:
        result["variant_response_sha256"] = "0" * 64
    write_json(path, result)
    with pytest.raises(ValueError):
        api().compare_aria_intervention(
            fixture["directory"], path, provenance=provenance
        )


def test_prediction_violations_and_html_escaping(fixture, tmp_path):
    fixture["hypothesis"] = '<script>alert("untrusted")</script>'
    fixture["predictions"]["historical_passed"] = 1
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    assert (
        next(
            row for row in report["predictions"] if row["field"] == "historical_passed"
        )["status"]
        == "violated"
    )
    rendered = api().render_aria_comparison(report)
    assert "<script>" not in rendered and "&lt;script&gt;" in rendered
    assert fixture["response"] not in rendered
    assert "private task answer" not in rendered
    assert "AssertionError" in rendered
    assert report["variant"]["task"]["runtime"]["stdout_sha256"] in rendered


def test_invalid_historical_protocol_stops_before_second_mode(fixture, monkeypatch):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    calls = []

    def run(*a, **kw):
        calls.append(True)
        return raw_result({"arbitrary": "claim"})

    monkeypatch.setattr(boundary, "run_container", run)
    report = api().execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert len(calls) == 1 and report["status"] == "stopped"
    assert report["variant"]["task"]["status"] == "unavailable"


def test_fsync_failure_consumes_without_execution(fixture, monkeypatch):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    original = api().os.fsync
    import stat

    def fail_directory(fd):
        if stat.S_ISDIR(api().os.fstat(fd).st_mode):
            raise OSError("directory fsync failed")
        return original(fd)

    monkeypatch.setattr(api().os, "fsync", fail_directory)
    with pytest.raises(OSError, match="fsync"):
        api().execute_aria_intervention(
            fixture["directory"],
            fixture["source"],
            artifacts=fixture["artifacts"],
            execution_authorized=True,
        )
    assert (fixture["directory"] / "consumed.json").exists()


def test_duplicate_control_json_is_rejected(fixture):
    content = fixture["control"].read_text()
    fixture["control"].write_text(content[:-1] + ',"replay_schema_version":1}')
    with pytest.raises(ValueError, match="JSON"):
        prepare(fixture)


def test_report_always_includes_both_answer_hashes_and_identity(fixture, tmp_path):
    fixture["predictions"] = {"historical_passed": 0}
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    rendered = api().render_aria_comparison(report)
    assert "step001_row000" in rendered
    assert fixture["record"]["task"]["runtime"]["stdout_sha256"] in rendered
    assert fixture["record"]["historical"]["runtime"]["stdout_sha256"] in rendered


def test_non_answer_streams_do_not_establish_equal_answers(fixture, tmp_path):
    empty = digest(b"")
    task = fixture["record"]["task"]
    task.update(
        status="unresolved",
        reason="nonzero_exit",
        determinate=False,
        test_coverage_complete=False,
        tests_completed=0,
        tests_passed=0,
        tests_failed=0,
    )
    task["runtime"].update(
        status="nonzero_exit", exit_code=137, stdout_sha256=empty, stdout_bytes=0
    )
    write_json(fixture["control"], fixture["record"])
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    assert report["answer_stream_equal"] is None
    assert (
        next(
            row for row in report["predictions"] if row["field"] == "task_stdout_sha256"
        )["status"]
        == "unavailable"
    )


def test_prediction_text_does_not_assert_reporting_rule_for_any_hypothesis(
    fixture, tmp_path
):
    fixture["hypothesis"] = "The solution edit changes task answers."
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    assert "reporting-rule explanation" not in report["interpretation"]


def test_stopped_infrastructure_record_can_be_compared_without_execution(
    fixture, monkeypatch
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    monkeypatch.setattr(
        boundary,
        "run_container",
        lambda *a, **k: boundary.ContainerResult(
            "infrastructure_error", cleanup_verified=True, error="Docker unavailable"
        ),
    )
    report = api().execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert report["status"] == "stopped"
    restored = api().compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    assert restored["variant"] == report["variant"]
    assert all(row["status"] == "unavailable" for row in restored["predictions"])


def test_task_cleanup_failure_retains_counts_without_determinate_verdict(
    fixture, monkeypatch
):
    prepared = prepare(fixture)
    approve(fixture, prepared)
    calls = []

    def run(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            return raw_result(historical(0))
        from dataclasses import replace

        return replace(
            raw_result({"results": [{"value": ["int", 1]}, {"value": ["int", 3]}]}, 1),
            cleanup_verified=False,
        )

    monkeypatch.setattr(boundary, "run_container", run)
    report = api().execute_aria_intervention(
        fixture["directory"],
        fixture["source"],
        artifacts=fixture["artifacts"],
        execution_authorized=True,
    )
    assert report["status"] == "stopped" and len(calls) == 2
    task = report["variant"]["task"]
    assert task["tests_passed"] == task["tests_failed"] == 1
    assert task["determinate"] is False and task["status"] == "unresolved"
    restored = api().compare_aria_intervention(
        fixture["directory"], fixture["directory"] / "result.json"
    )
    assert restored["variant"]["task"] == task


def test_control_requires_distinct_mode_containers(fixture):
    fixture["record"]["task"]["runtime"]["metadata"]["container_id"] = fixture[
        "record"
    ]["historical"]["runtime"]["metadata"]["container_id"]
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError, match="distinct"):
        prepare(fixture)


@pytest.mark.parametrize("field,value", [("step", True), ("task_checks", 2.0)])
def test_provenance_requires_exact_integer_types(fixture, field, value):
    if field == "step":
        fixture["record"]["identity"]["step"] = value
    else:
        fixture["record"]["task_source"]["checks"] = value
    write_json(fixture["control"], fixture["record"])
    with pytest.raises(ValueError):
        prepare(fixture)


def test_renderer_distinguishes_absent_reason_from_unavailable_outcome(
    fixture, tmp_path
):
    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    rendered = api().render_aria_comparison(report)
    assert (
        "<th>Protected unavailable reason</th><td>Not applicable</td><td>Not applicable</td>"
        in rendered
    )
    report["variant"]["task"].update(
        determinate=False, status="unresolved", reason=None
    )
    rendered = api().render_aria_comparison(report)
    assert (
        "<th>Protected unavailable reason</th><td>Not applicable</td><td>Unavailable</td>"
        in rendered
    )
    report["variant"]["task"]["reason"] = "incomplete_or_unsupported_answers"
    assert "incomplete_or_unsupported_answers" in api().render_aria_comparison(report)


def test_renderer_contains_tables_in_keyboard_scroll_regions(fixture, tmp_path):
    from html.parser import HTMLParser

    class Tables(HTMLParser):
        def __init__(self):
            super().__init__()
            self.regions = []
            self.tables = []
            self.current = None

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "div":
                self.current = attrs
                self.regions.append(attrs)
            elif tag == "table":
                self.tables.append(self.current)

        def handle_endtag(self, tag):
            if tag == "div":
                self.current = None

    prepared = prepare(fixture)
    path, provenance, _ = archived_result(fixture, prepared, tmp_path)
    report = api().compare_aria_intervention(
        fixture["directory"], path, provenance=provenance
    )
    parser = Tables()
    parser.feed(api().render_aria_comparison(report))
    assert len(parser.tables) == 2
    assert all(
        region
        and region.get("class") == "table-scroll"
        and region.get("tabindex") == "0"
        for region in parser.tables
    )
    assert {region["aria-label"] for region in parser.tables} == {
        "Measurements",
        "Predictions",
    }
