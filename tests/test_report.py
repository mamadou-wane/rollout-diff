import json
from dataclasses import fields, is_dataclass, replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from test_full_pipeline import analyze, fixture

import rolloutdiff


def build(analysis, **kwargs):
    assert hasattr(rolloutdiff, "build_report"), "report builder API is missing"
    return rolloutdiff.build_report(analysis, **kwargs)


def tables(result, prefix=""):
    found = {}
    for field in fields(result):
        value = getattr(result, field.name)
        name = f"{prefix}{field.name}"
        if isinstance(value, pd.DataFrame):
            found[name] = value
        elif is_dataclass(value):
            found.update(tables(value, name + "."))
    return found


def test_report_links_ranked_events_members_evidence_and_canonical_exemplars():
    data = fixture()
    result = analyze(data)
    report = build(result)
    assert set(report) == {"schema_version", "runs"}
    assert report["schema_version"] == 1
    run = report["runs"][0]
    assert run["run"] == "r"
    screening = result.sequential.screening.iloc[0]
    assert run["counts"]["rollouts"] == len(data)
    assert run["counts"]["observed_steps"] == 14
    assert run["counts"]["prompts"] == 6
    assert run["counts"]["screened_series"] == screening.n_screened
    assert run["counts"]["sequential_alerts"] == len(result.sequential.alerts)
    assert [
        event["event_id"] for event in run["events"]
    ] == result.event_ranking.event_id.tolist()
    source = data.set_index(["run", "sample_id"])
    evidence = result.evidence.set_index(["run", "feature"])
    seen = set()
    for event, ranking in zip(
        run["events"], result.event_ranking.to_dict("records"), strict=True
    ):
        assert event["rank"] == ranking["rank"]
        assert event["representative_feature"] == ranking["representative_feature"]
        assert event["n_members"] == len(event["members"])
        assert event["ranking"]["numerical_floor_members"] == [
            {"run": r, "feature": f} for r, f in ranking["numerical_floor_members"]
        ]
        for member in event["members"]:
            key = member["run"], member["feature"]
            assert key not in seen
            seen.add(key)
            row = evidence.loc[key]
            supporting = member["evidence"]
            assert supporting["status"] == "recorded"
            assert (
                supporting["numerical_floor_dominant"] == row.numerical_floor_dominant
            )
            assert supporting["replication"]["status"] == row.replication_status
            assert (
                supporting["persistence"]["n_evaluable"] == row.persistence_n_evaluable
            )
            assert supporting["reward_association"]["status"] == row.reward_status
            assert supporting["prompt_breadth"]["positive_prompts"] == (
                None
                if pd.isna(row.prompt_positive_prompts)
                else row.prompt_positive_prompts
            )
            assert member["exemplar_status"] == "recorded"
            for exemplar in member["exemplars"]:
                assert (exemplar["run"], exemplar["feature"]) == key
                if exemplar["status"] == "selected":
                    canonical = source.loc[(exemplar["run"], exemplar["sample_id"])]
                    assert exemplar["step"] == canonical.step
                    assert exemplar["prompt_id"] == canonical.prompt_id
                    assert exemplar["reward"] == canonical.reward
                    assert exemplar["feature_value"] == 1
                else:
                    assert exemplar["sample_id"] is None
    assert seen == set(
        result.event_members[["run", "feature"]].itertuples(index=False, name=None)
    )
    contextual = next(
        e for e in run["events"] if e["representative_feature"] == "scalar:reward"
    )
    assert contextual["ranking_status"] == "contextual"
    assert contextual["members"][0]["exemplars"][0]["status"] == "unsupported_kind"


def shuffle(value):
    if isinstance(value, pd.DataFrame):
        value = value.sample(frac=1, random_state=7)
        value.index = [9] * len(value)
        return value
    if is_dataclass(value):
        return replace(
            value, **{f.name: shuffle(getattr(value, f.name)) for f in fields(value)}
        )
    return value


def test_json_and_schema_are_stable_under_every_table_order_without_mutation():
    data = fixture()
    data = pd.concat([data, data.assign(run="another")], ignore_index=True)
    result = analyze(data)
    before = {name: frame.copy(deep=True) for name, frame in tables(result).items()}
    expected = build(result)
    from rolloutdiff.report import report_json

    assert [run["run"] for run in expected["runs"]] == ["another", "r"]
    assert report_json(build(shuffle(result))) == report_json(expected)
    assert json.loads(report_json(expected)) == expected
    assert len(before) == 24
    for name, frame in tables(result).items():
        assert_frame_equal(frame, before[name], check_exact=True)


def test_only_allowlisted_evidence_and_metadata_are_exported():
    result = analyze(fixture())
    secret = "/Users/private/DO_NOT_EXPORT prompt response tool payload"
    for frame in tables(result).values():
        for name in ["prompt", "response", "tool_calls", "label_hack", "source_path"]:
            if name not in frame:
                frame[name] = secret
    text = json.dumps(build(result))
    assert secret not in text
    for name in ["prompt", "response", "tool_calls", "label_hack", "source_path"]:
        assert f'"{name}":' not in text


def test_missing_evidence_and_exemplars_remain_explicit():
    result = analyze(fixture())
    feature = result.event_members.feature.iloc[0]
    evidence = result.evidence.loc[~result.evidence.feature.eq(feature)]
    exemplars = result.exemplars.loc[~result.exemplars.feature.eq(feature)]
    report = build(replace(result, evidence=evidence, exemplars=exemplars))
    member = next(
        m
        for e in report["runs"][0]["events"]
        for m in e["members"]
        if m["feature"] == feature
    )
    assert member["evidence"]["status"] == "missing"
    assert member["evidence"]["persistence"]["fraction"] is None
    assert member["evidence"]["replication"]["replicates"] is None
    assert member["exemplar_status"] == "missing"
    assert member["exemplars"] == []


def test_unavailable_measurements_are_strict_json_nulls_with_statuses():
    result = analyze(fixture(last=6, one_half=True))
    report = build(result)
    from rolloutdiff.report import report_json

    text = report_json(report)
    assert "NaN" not in text and "Infinity" not in text
    for event in report["runs"][0]["events"]:
        for member in event["members"]:
            persistence = member["evidence"]["persistence"]
            assert persistence["status"] == "no_future_observations"
            assert persistence["fraction"] is None and persistence["coverage"] is None
    assert json.loads(text) == report


def test_empty_alerts_and_empty_input_keep_run_identity_and_schema():
    result = analyze(fixture().assign(response="constant", reward=0.0))
    report = build(result, run="r")
    assert report["runs"][0]["events"] == []
    assert report["runs"][0]["counts"]["sequential_alerts"] == 0
    empty = analyze(fixture().iloc[:0])
    assert build(empty)["runs"] == []
    report = build(empty, run="empty")
    assert report["runs"][0]["run"] == "empty"
    assert report["runs"][0]["status"] == "empty"
    assert all(count == 0 for count in report["runs"][0]["counts"].values())
    with pytest.raises(ValueError, match="run"):
        build(result, run="different")


def test_integer_steps_above_float_precision_survive_serialization():
    data = fixture()
    offset = 2**53 + 1
    data["step"] += offset
    report = build(analyze(data))
    from rolloutdiff.report import report_json

    event = next(
        e
        for e in json.loads(report_json(report))["runs"][0]["events"]
        if e["representative_feature"] == "def:solve"
    )
    assert event["onset_step"] == offset + 4
    assert type(event["onset_step"]) is int
    assert event["alert_step"] == offset + 6
    for exemplar in event["members"][0]["exemplars"]:
        if exemplar["status"] == "selected":
            assert type(exemplar["step"]) is int
            assert offset + 4 <= exemplar["step"] <= offset + 6


def test_absolute_path_identifiers_are_rejected_without_rewriting_namespaced_ids():
    result = analyze(fixture().assign(run="experiment/seed-1"))
    assert build(result)["runs"][0]["run"] == "experiment/seed-1"
    for run in ["/Users/private/run", r"C:\private\run", "~/private/run"]:
        with pytest.raises(ValueError, match="path"):
            build(analyze(fixture().assign(run=run)))


def test_namespaced_canonical_ids_are_not_rewritten_as_windows_drive_paths():
    data = fixture().assign(run="r:one")
    data.loc[data.response.str.startswith("def"), "response"] = (
        "```python\ndef solve():\n    pass\n```\nfresh prose"
    )
    result = analyze(data)
    assert {"w:fresh", "b:fresh prose"} <= set(result.event_members.feature)
    assert build(result)["runs"][0]["run"] == "r:one"


@pytest.mark.parametrize("path", ["/tmp/private.py", "/home/person/secret.json"])
def test_path_bearing_literal_features_cannot_leak_private_paths(path):
    data = fixture()
    changed = data.response.str.startswith("def")
    data.loc[changed, "response"] += f"\nopen({path!r})"
    result = analyze(data)
    assert f"strlit:{path}" in set(result.event_members.feature)
    with pytest.raises(ValueError, match="path"):
        build(result)


@pytest.mark.parametrize(
    "identifier",
    [
        "16:p0:0:/Users/private/project/raw",
        "source=/var/private/raw.json",
        "sample [/home/private/raw]",
        "sample:~/private/raw",
        r"sample:C:\Users\private\raw",
        "sample:C:/Users/private/raw",
        r"sampleC:\private\raw",
        r"sample:\\server\share\private\raw",
        "sample=//server/share/private/raw",
        r"sample=//server\share/private/raw",
        "sample:file:///Users/private/raw",
        "sample-file://server/share/private/raw",
        "sample:FILE:///C:/private/raw",
        "sample:file://localhost/etc/private",
    ],
)
def test_embedded_filesystem_structures_are_rejected_without_echo(identifier):
    from rolloutdiff.report import _identifier

    with pytest.raises(ValueError, match="filesystem paths") as exc:
        _identifier(identifier)
    assert (
        str(exc.value)
        == "report identifiers must not contain absolute filesystem paths"
    )


@pytest.mark.parametrize(
    "field",
    ["run", "event_id", "feature", "representative_feature", "prompt_id", "sample_id"],
)
def test_every_report_identifier_uses_the_embedded_path_check(field):
    from rolloutdiff.report import _value

    with pytest.raises(ValueError, match="filesystem paths"):
        _value(field, "namespace:/Users/private/raw")


@pytest.mark.parametrize(
    "identifier",
    [
        "experiment/seed-1",
        "r:one",
        "completions_16.parquet:0",
        "16:p0:0",
        "namespace:relative/path",
        "folder-/file.txt",
        "./relative/file.txt",
        "../relative/file.txt",
        r"relative\file.txt",
        "w:word",
        "b:two words",
        "strlit:src/solution.py",
        "https://example.test/items/1",
        "profile://settings",
        "b:ratio / denominator",
    ],
)
def test_legitimate_identifier_structures_are_preserved(identifier):
    from rolloutdiff.report import _identifier

    assert _identifier(identifier) == identifier


@pytest.mark.parametrize("field", ["run", "prompt_id", "sample_id"])
def test_builder_rejects_embedded_paths_in_canonical_ids(field):
    data = fixture()
    data[field] = data[field].map(lambda value: f"namespace:{value}:/Users/private/raw")
    with pytest.raises(ValueError, match="filesystem paths"):
        build(analyze(data))


@pytest.mark.parametrize("value", [True, np.bool_(False)])
def test_boolean_screening_counts_are_not_serialized_as_integers(value):
    result = analyze(fixture())
    result.sequential.screening["n_screened"] = value
    with pytest.raises(ValueError, match="integer"):
        build(result)


def test_version_one_schema_keeps_member_evidence_and_rank_metadata_separate():
    report = build(analyze(fixture()))
    run = report["runs"][0]
    assert set(run) == {"run", "status", "counts", "events"}
    assert set(run["counts"]) == {
        "rollouts",
        "observed_steps",
        "prompts",
        "series",
        "screened_series",
        "sequential_alerts",
        "unscorable_series",
        "events",
        "behavioral_events",
        "contextual_events",
        "other_events",
    }
    event = run["events"][0]
    assert set(event) == {
        "event_id",
        "run",
        "rank",
        "representative_feature",
        "ranking_status",
        "onset_start",
        "onset_end",
        "onset_step",
        "alert_step",
        "n_members",
        "ranking",
        "members",
    }
    assert set(event["ranking"]) == {
        "n_prioritizable_members",
        "replication_availability",
        "persistence_availability",
        "reward_availability",
        "breadth_availability",
        "missing_evidence_members",
        "numerical_floor_members",
        "numerical_floor_unavailable_members",
    }
    member = event["members"][0]
    assert set(member) == {
        "run",
        "feature",
        "kind",
        "direction",
        "onset_step",
        "alert_step",
        "status",
        "evidence",
        "exemplar_status",
        "exemplars",
    }
    assert set(member["evidence"]) == {
        "status",
        "tier",
        "scale_source",
        "numerical_floor_dominant",
        "replication",
        "persistence",
        "reward_association",
        "prompt_breadth",
    }


def test_lossy_float_steps_are_rejected_instead_of_inventing_integer_identities():
    result = analyze(fixture())
    result.event_ranking["onset_step"] = float(2**53)
    with pytest.raises(ValueError, match="exact integers"):
        build(result)


@pytest.mark.parametrize(
    "field", ["event_ranking", "events", "event_members", "evidence"]
)
def test_duplicate_join_keys_cannot_silently_replace_records(field):
    result = analyze(fixture())
    frame = getattr(result, field)
    with pytest.raises(ValueError, match="duplicate"):
        build(replace(result, **{field: pd.concat([frame, frame.iloc[:1]])}))


def test_missing_ranking_or_orphan_exemplar_is_rejected():
    result = analyze(fixture())
    with pytest.raises(ValueError, match="event"):
        build(replace(result, event_ranking=result.event_ranking.iloc[:0]))
    result.exemplars.loc[result.exemplars.status.eq("selected"), "sample_id"] = (
        "not-canonical"
    )
    with pytest.raises(ValueError, match="rollout"):
        build(result)


def test_writer_publishes_complete_json_and_refuses_overwrite_or_symlinks(tmp_path):
    from rolloutdiff.report import report_json, write_report

    report = build(analyze(fixture()))
    path = tmp_path / "report.json"
    write_report(report, path)
    assert path.read_text() == report_json(report)
    assert json.loads(path.read_text()) == report
    with pytest.raises(FileExistsError):
        write_report(report, path)
    assert path.read_text() == report_json(report)
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        write_report(report, link)
    assert not (tmp_path / "absent").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "dangling.json",
        "report.json",
    ]


def test_writer_cleans_temporary_file_and_preserves_racing_destination(
    tmp_path, monkeypatch
):
    from rolloutdiff import report as module

    path = tmp_path / "report.json"

    def competing_writer(source, destination):
        destination.write_text("another writer")
        raise FileExistsError("destination appeared")

    monkeypatch.setattr(module.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        module.write_report({"schema_version": 1}, path)
    assert path.read_text() == "another writer"
    assert list(tmp_path.iterdir()) == [path]


def test_invalid_json_leaves_no_output_or_temporary_file(tmp_path):
    from rolloutdiff.report import write_report

    with pytest.raises(ValueError):
        write_report({"bad": np.nan}, tmp_path / "report.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operation", ["fsync", "link"])
def test_writer_io_failure_leaves_no_partial_report(tmp_path, monkeypatch, operation):
    from rolloutdiff import report as module

    def fail(*args):
        raise OSError("simulated output failure")

    monkeypatch.setattr(module.os, operation, fail)
    with pytest.raises(OSError, match="output failure"):
        module.write_report({"schema_version": 1}, tmp_path / "report.json")
    assert list(tmp_path.iterdir()) == []
