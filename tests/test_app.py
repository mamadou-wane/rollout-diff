"""Local HTTP boundary, analytical parity, and source integrity checks."""

import hashlib
import importlib.util
import json
import time

import pytest

TOKEN = "test-session-token-0123456789"
ORIGIN = "http://127.0.0.1:8765"


def api(root, **kwargs):
    assert importlib.util.find_spec("rolloutdiff.app") is not None, (
        "the local investigation API must exist"
    )
    from fastapi.testclient import TestClient

    from rolloutdiff.app import create_app

    return TestClient(
        create_app(root, token=TOKEN),
        base_url=ORIGIN,
        client=kwargs.pop("client", ("127.0.0.1", 50000)),
        headers={"Authorization": f"Bearer {TOKEN}", "Origin": ORIGIN},
        **kwargs,
    )


def write_run(root, *, large=False):
    directory = root / "rollouts"
    directory.mkdir(exist_ok=True)
    offset = 2**53 + 1 if large else 0
    for step in range(1, 24):
        rows = [
            {
                "step": step + offset,
                "input": f"private prompt {prompt}",
                "output": "def solve():\n    pass"
                if step > 15 and j == 0
                else "plain words.",
                "score": 2 if step > 15 and j == 0 else 0,
                "sample_id": f"s-{step}-{prompt}-{j}",
            }
            for prompt in range(6)
            for j in range(2)
        ]
        (directory / f"{step + offset}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    return directory


def wait_ready(client, location):
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        response = client.get(location)
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] in {"ready", "failed"}:
            assert value["status"] == "ready", value
            return value
        time.sleep(0.01)
    pytest.fail("local job did not finish")


def load_run(client, **kwargs):
    response = client.post(
        "/api/runs", json={"path": "rollouts", "format": "verl", "run": "r", **kwargs}
    )
    assert response.status_code == 202, response.text
    return wait_ready(client, f"/api/runs/{response.json()['id']}")


def test_session_requires_token_host_origin_and_loopback_peer(tmp_path):
    with api(tmp_path) as client:
        response = client.get("/api/session")
        assert response.status_code == 200
        assert response.json()["read_only_execution"] is True
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-frame-options"] == "DENY"
        for headers in (
            {"Authorization": ""},
            {"Authorization": "Bearer wrong"},
            {"Host": "evil.example"},
            {"Origin": "https://evil.example"},
            {"Origin": "null"},
        ):
            assert client.get("/api/session", headers=headers).status_code == 403
        assert (
            client.get(
                f"/api/session?token={TOKEN}", headers={"Authorization": ""}
            ).status_code
            == 403
        )
        assert client.post("/api/replay", json={}).status_code == 404
        assert client.get("/docs").status_code == 404
    with api(tmp_path, client=("192.0.2.10", 80)) as client:
        assert client.get("/api/session").status_code == 403


def test_rooted_files_reject_escape_and_omit_hidden_files(tmp_path):
    write_run(tmp_path)
    (tmp_path / ".private.json").write_text("secret")
    (tmp_path / "report.json").write_text("{}")
    (tmp_path / "outside").symlink_to(tmp_path.parent, target_is_directory=True)
    with api(tmp_path) as client:
        listing = client.get("/api/files").json()
        assert [entry["name"] for entry in listing["entries"]] == [
            "report.json",
            "rollouts",
        ]
        for path in ("../", str(tmp_path.parent), "outside", ".private.json"):
            response = client.get("/api/files", params={"path": path})
            assert response.status_code == 400
            assert str(tmp_path.parent) not in response.text
        response = client.post(
            "/api/runs", json={"path": "outside", "format": "verl", "run": "r"}
        )
        assert response.status_code == 400


def test_analysis_events_series_inspection_and_exports_match_existing_pipeline(
    tmp_path,
):
    from rolloutdiff import analyze_rollouts, build_report, load_verl
    from rolloutdiff.report import report_json

    directory = write_run(tmp_path)
    expected = build_report(analyze_rollouts(load_verl(directory, run="r")), run="r")
    with api(tmp_path) as client:
        run = load_run(client)
        prefix = f"/api/runs/{run['id']}"
        assert run["counts"] == expected["runs"][0]["counts"]
        assert run["source"]["files"][0]["path"].startswith("rollouts/")
        response = client.get(prefix + "/events")
        assert response.json()["events"] == expected["runs"][0]["events"]
        events = response.json()["events"]
        selected = next(
            event for event in events if event["ranking_status"] == "behavioral"
        )
        detail = client.get(prefix + f"/events/{selected['rank']}").json()
        assert detail["event"] == selected
        assert detail["evidence"]
        feature = selected["representative_feature"]
        series = client.get(prefix + "/series", params={"feature": feature}).json()
        assert len(series["points"]) == 23
        assert "private prompt" not in json.dumps(series)
        assert "scalar:reward" in [
            row["feature"]
            for row in client.get(prefix + "/features").json()["features"]
        ]
        member = next(
            member for member in selected["members"] if member["feature"] == feature
        )
        example = next(
            item for item in member["exemplars"] if item["status"] == "selected"
        )
        citation = client.post(
            prefix + "/inspect",
            json={
                "event_rank": selected["rank"],
                "feature": feature,
                "sample_id": example["sample_id"],
                "with_absent": True,
            },
        )
        assert citation.status_code == 200, citation.text
        value = citation.json()
        assert value["citation"]["response"] == "def solve():\n    pass"
        assert (
            value["response_sha256"]
            == hashlib.sha256(value["citation"]["response"].encode()).hexdigest()
        )
        assert (
            value["citation"]["comparison"]["counterpart"]["response"] == "plain words."
        )
        assert client.get(prefix + "/export?format=json").text == report_json(expected)
        assert (
            "attachment"
            in client.get(prefix + "/export?format=html").headers["content-disposition"]
        )
        assert (
            client.get(prefix + "/events", params={"q": feature, "limit": 1}).json()[
                "events"
            ][0]["rank"]
            == selected["rank"]
        )
        assert client.get(prefix + "/events", params={"offset": -1}).status_code == 422
        assert (
            client.get(prefix + "/series", params={"feature": "unknown"}).status_code
            == 404
        )


def test_changed_sources_block_inspection_comparison_and_export(tmp_path):
    directory = write_run(tmp_path)
    with api(tmp_path) as client:
        run = load_run(client)
        prefix = f"/api/runs/{run['id']}"
        event = client.get(prefix + "/events").json()["events"][0]
        member = next(member for member in event["members"] if member["exemplars"])
        sample = member["exemplars"][0]["sample_id"]
        original = directory / "1.jsonl"
        original.write_text(
            original.read_text().replace("plain words.", "changed response")
        )
        assert (
            client.post(
                prefix + "/inspect",
                json={
                    "event_rank": event["rank"],
                    "feature": member["feature"],
                    "sample_id": sample,
                },
            ).status_code
            == 409
        )
        assert client.get(prefix + "/export?format=json").status_code == 409
        response = client.post(
            "/api/comparisons",
            json={
                "run_a": run["id"],
                "run_b": None,
                "a": ["1", "15"],
                "b": ["16", "23"],
            },
        )
        assert response.status_code == 409


def test_comparison_citations_and_dependent_deletion(tmp_path):
    write_run(tmp_path)
    with api(tmp_path) as client:
        run = load_run(client)
        response = client.post(
            "/api/comparisons",
            json={
                "run_a": run["id"],
                "run_b": None,
                "a": ["1", "15"],
                "b": ["16", "23"],
            },
        )
        assert response.status_code == 202, response.text
        prefix = f"/api/comparisons/{response.json()['id']}"
        comparison = wait_ready(client, prefix)
        feature = next(
            row
            for row in comparison["report"]["features"]
            if row["feature"] == "def:solve"
        )
        sample = feature["citations"]["b"]["items"][0]["sample_id"]
        response = client.post(
            prefix + "/inspect",
            json={"side": "b", "feature": "def:solve", "sample_id": sample},
        )
        assert response.status_code == 200, response.text
        assert response.json()["citation"]["response"].startswith("def solve")
        assert response.json()["citation"]["feature"] == "def:solve"
        assert response.json()["citation"]["side"] == "b"
        assert client.get(prefix + "/export?format=json").json()["mode"] == "windows"
        assert client.delete(f"/api/runs/{run['id']}").status_code == 204
        assert client.get(prefix).status_code == 404


def test_large_steps_are_exact_in_ui_and_unchanged_in_exports(tmp_path):
    write_run(tmp_path, large=True)
    with api(tmp_path) as client:
        run = load_run(client)
        prefix = f"/api/runs/{run['id']}"
        points = client.get(
            prefix + "/series", params={"feature": "scalar:reward"}
        ).json()["points"]
        assert points[0]["step"] == str(2**53 + 2)
        exported = client.get(prefix + "/export?format=json").json()
        assert type(exported["runs"][0]["events"][0]["onset_step"]) is int


def test_duplicate_headers_and_nonascii_credentials_are_denied(tmp_path):
    with api(tmp_path) as client:
        for headers in (
            [("Host", "127.0.0.1:8765"), ("Host", "evil.example")],
            [("Origin", ORIGIN), ("Origin", ORIGIN)],
            [
                ("Authorization", f"Bearer {TOKEN}"),
                ("Authorization", f"Bearer {TOKEN}"),
            ],
            [(b"Authorization", b"Bearer \xff")],
        ):
            assert client.get("/api/session", headers=headers).status_code == 403


def test_invalid_inputs_never_echo_private_text_or_outside_paths(tmp_path):
    write_run(tmp_path)
    secret = "/outside/private-secret-value"
    with api(tmp_path) as client:
        for body in (
            {"path": "rollouts", "format": "verl", "run": secret},
            {"path": secret, "format": "verl", "run": "r"},
            {"path": "rollouts", "format": "bad", "run": "r", "private": secret},
            {
                "path": "rollouts",
                "format": "verl",
                "run": "r",
                "reward_column": "score",
            },
        ):
            response = client.post("/api/runs", json=body)
            assert response.status_code in {400, 422}
            assert secret not in response.text
        assert client.get("/api/runs").json()["runs"] == []


def test_nested_source_symlink_is_rejected_before_loading(tmp_path):
    directory = tmp_path / "rollouts"
    directory.mkdir()
    target = tmp_path.parent / f"{tmp_path.name}-private.jsonl"
    target.write_text('{"private": "outside data"}')
    (directory / "1.jsonl").symlink_to(target)
    with api(tmp_path) as client:
        response = client.post(
            "/api/runs", json={"path": "rollouts", "format": "verl", "run": "r"}
        )
        assert response.status_code == 400
        assert "outside data" not in response.text
        assert client.get("/api/runs").json()["runs"] == []


def test_trl_custom_reward_and_failed_jobs_have_terminal_states(tmp_path):
    import pandas as pd

    directory = tmp_path / "rollouts"
    directory.mkdir()
    pd.DataFrame(
        [{"step": 1, "prompt": "p", "completion": "plain", "task_reward": 3}]
    ).to_parquet(directory / "completions_1.parquet")
    with api(tmp_path) as client:
        ready = load_run(client, format="trl", reward_column="task_reward")
        assert ready["counts"]["rollouts"] == 1
        response = client.post(
            "/api/runs", json={"path": "rollouts", "format": "trl", "run": "r"}
        )
        identity = response.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            failed = client.get(f"/api/runs/{identity}").json()
            if failed["status"] == "failed":
                break
            time.sleep(0.01)
        assert failed["status"] == "failed"
        assert failed["error"]
        assert failed["counts"] is None
        assert "task_reward" not in failed["error"]
        assert client.get(f"/api/runs/{identity}/events").status_code == 409
        assert client.delete(f"/api/runs/{identity}").status_code == 204


def test_analysis_jobs_are_sequential_and_running_runs_cannot_be_deleted(
    tmp_path, monkeypatch
):
    import threading

    import rolloutdiff.app as module

    write_run(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = module.analyze_rollouts

    def blocked_analysis(rollouts):
        entered.set()
        assert release.wait(10)
        return original(rollouts)

    monkeypatch.setattr(module, "analyze_rollouts", blocked_analysis)
    with api(tmp_path) as client:
        first = client.post(
            "/api/runs", json={"path": "rollouts", "format": "verl", "run": "a"}
        ).json()
        try:
            assert entered.wait(10)
            second = client.post(
                "/api/runs", json={"path": "rollouts", "format": "verl", "run": "b"}
            ).json()
            assert (
                client.get(f"/api/runs/{first['id']}").json()["status"] == "analyzing"
            )
            assert client.get(f"/api/runs/{second['id']}").json()["status"] == "queued"
            assert client.delete(f"/api/runs/{first['id']}").status_code == 409
        finally:
            release.set()
        assert wait_ready(client, f"/api/runs/{first['id']}")["run"] == "a"
        assert wait_ready(client, f"/api/runs/{second['id']}")["run"] == "b"


def test_added_source_file_invalidates_existing_loaded_run(tmp_path):
    directory = write_run(tmp_path)
    with api(tmp_path) as client:
        run = load_run(client)
        (directory / "24.jsonl").write_text(
            '{"step":24,"input":"p","output":"plain","score":0}'
        )
        assert client.get(f"/api/runs/{run['id']}/export").status_code == 409


def test_cross_run_comparison_uses_each_source_and_preserves_report_hashes(tmp_path):
    from rolloutdiff import compare_rollouts, load_verl

    directory = write_run(tmp_path)
    expected = compare_rollouts(
        load_verl(directory, run="a"),
        load_verl(directory, run="b"),
        a=(1, 15),
        b=(16, 23),
    )
    with api(tmp_path) as client:
        left = load_run(client, run="a")
        right = load_run(client, run="b")
        response = client.post(
            "/api/comparisons",
            json={
                "run_a": left["id"],
                "run_b": right["id"],
                "a": ["1", "15"],
                "b": ["16", "23"],
            },
        )
        prefix = f"/api/comparisons/{response.json()['id']}"
        comparison = wait_ready(client, prefix)
        assert comparison["report"]["mode"] == "dumps"
        assert comparison["report"]["features"] == expected["features"]
        assert comparison["report"]["prompt_mix"] == expected["prompt_mix"]
        for side, source in zip(
            comparison["report"]["sides"], (left, right), strict=True
        ):
            assert (
                side["source"]["canonical_sha256"]
                == source["source"]["canonical_sha256"]
            )
        feature = next(
            row
            for row in comparison["report"]["features"]
            if row["feature"] == "def:solve"
        )
        sample = feature["citations"]["b"]["items"][0]["sample_id"]
        response = client.post(
            prefix + "/inspect",
            json={"side": "b", "feature": "def:solve", "sample_id": sample},
        )
        assert response.json()["citation"]["run"] == "b"
        assert client.get(prefix + "/export?format=html").status_code == 200
        original = directory / "1.jsonl"
        original.write_text(
            original.read_text().replace("plain words.", "changed response")
        )
        assert (
            client.post(
                prefix + "/inspect",
                json={"side": "b", "feature": "def:solve", "sample_id": sample},
            ).status_code
            == 409
        )
        assert client.get(prefix + "/export").status_code == 409


def test_symlink_to_hidden_directory_cannot_reveal_dotfiles(tmp_path):
    hidden = tmp_path / ".private"
    hidden.mkdir()
    (hidden / "evidence.json").write_text("{}")
    (tmp_path / "visible").symlink_to(hidden, target_is_directory=True)
    with api(tmp_path) as client:
        assert client.get("/api/files", params={"path": "visible"}).status_code == 400
        assert client.get("/api/files").json()["entries"] == []


def test_unexpected_request_errors_are_sanitized_and_keep_security_headers(
    tmp_path, monkeypatch
):
    import rolloutdiff.app as module

    def unavailable(*args, **kwargs):
        raise RuntimeError("private response /outside/secret")

    monkeypatch.setattr(module, "_rooted", unavailable)
    with api(tmp_path, raise_server_exceptions=False) as client:
        response = client.get("/api/files")
        assert response.status_code == 500
        assert "private response" not in response.text
        assert "/outside/secret" not in response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.headers["x-content-type-options"] == "nosniff"


def test_source_swap_during_adapter_load_cannot_import_outside_root(
    tmp_path, monkeypatch
):
    from rolloutdiff import adapters

    directory = write_run(tmp_path)
    source = directory / "16.jsonl"
    original = source.read_bytes()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.jsonl"
    outside.write_bytes(original.replace(b"solve", b"outside_root_secret_marker"))
    loader = adapters.load_verl

    def swapped_load(path, *, run):
        source.unlink()
        source.symlink_to(outside)
        try:
            return loader(path, run=run)
        finally:
            source.unlink()
            source.write_bytes(original)

    monkeypatch.setattr(adapters, "load_verl", swapped_load)
    with api(tmp_path) as client:
        run = load_run(client)
        prefix = f"/api/runs/{run['id']}"
        assert (
            client.get(
                prefix + "/features", params={"q": "outside_root_secret_marker"}
            ).json()["features"]
            == []
        )
        assert "outside_root_secret_marker" not in client.get(prefix + "/export").text
        source_hash = next(
            file["sha256"]
            for file in run["source"]["files"]
            if file["path"] == "rollouts/16.jsonl"
        )
        assert source_hash == hashlib.sha256(original).hexdigest()


def test_private_snapshot_preserves_names_permissions_and_cleans_up_on_load_failure(
    tmp_path, monkeypatch
):
    import stat

    from rolloutdiff import adapters

    directory = tmp_path / "rollouts"
    directory.mkdir()
    content = b"x" * (17 * 1024 * 1024)
    (directory / "step_000001.jsonl").write_bytes(content)
    loaded = []

    def failed_load(path, *, run):
        from pathlib import Path

        path = Path(path)
        loaded.append(
            {"path": path, "directory_mode": stat.S_IMODE(path.stat().st_mode)}
        )
        copied = path / "step_000001.jsonl"
        loaded[0].update(
            mode=stat.S_IMODE(copied.stat().st_mode), content=copied.read_bytes()
        )
        raise ValueError("synthetic malformed source")

    monkeypatch.setattr(adapters, "load_verl", failed_load)
    with api(tmp_path) as client:
        response = client.post(
            "/api/runs", json={"path": "rollouts", "format": "verl", "run": "r"}
        )
        identity = response.json()["id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = client.get(f"/api/runs/{identity}").json()
            if run["status"] == "failed":
                break
            time.sleep(0.01)
        assert run["status"] == "failed"
    assert len(loaded) == 1
    assert loaded[0]["directory_mode"] == 0o700
    assert loaded[0]["mode"] == 0o600
    assert loaded[0]["content"] == content
    assert not loaded[0]["path"].exists()
    assert directory.is_dir()


@pytest.mark.parametrize("kind", ["file_link", "directory_link", "fifo"])
def test_source_files_refuse_symlinks_and_nonregular_nodes(tmp_path, kind):
    import os

    directory = tmp_path / "rollouts"
    directory.mkdir()
    source = directory / "1.jsonl"
    if kind == "fifo":
        os.mkfifo(source)
    else:
        valid = tmp_path / "valid"
        valid.mkdir()
        (valid / "1.jsonl").write_text(
            '{"step":1,"input":"p","output":"plain","score":0}'
        )
        if kind == "file_link":
            source.symlink_to(valid / "1.jsonl")
        else:
            directory.rmdir()
            directory.symlink_to(valid, target_is_directory=True)
    with api(tmp_path) as client:
        response = client.post(
            "/api/runs", json={"path": "rollouts", "format": "verl", "run": "r"}
        )
        assert response.status_code == 400
        assert client.get("/api/runs").json()["runs"] == []


@pytest.mark.parametrize("format", ["trl", "verl"])
def test_copied_sources_preserve_generated_identities_and_cleanup_after_success(
    tmp_path, monkeypatch, format
):
    import gzip

    import pandas as pd

    from rolloutdiff import adapters
    from rolloutdiff.comparison import canonical_fingerprint

    if format == "trl":
        directory = tmp_path / "trainer" / "completions"
        directory.mkdir(parents=True)
        pd.DataFrame(
            [{"step": 1, "prompt": "p", "completion": "plain", "reward": 0}]
        ).to_parquet(directory / "completions_1.parquet")
        requested = "trainer"
        loader = adapters.load_trl
    else:
        source = tmp_path / "step_000001.jsonl.gz"
        with gzip.open(source, "wt") as stream:
            stream.write('{"step":1,"input":"p","output":"plain","score":0}\n')
        requested = source.name
        loader = adapters.load_verl
    expected = loader(tmp_path / requested, run="r")
    copied_paths = []

    def copied_load(path, **kwargs):
        copied_paths.append(path)
        return loader(path, **kwargs)

    monkeypatch.setattr(adapters, f"load_{format}", copied_load)
    with api(tmp_path) as client:
        run = load_run(client, path=requested, format=format)
        assert run["source"]["canonical_sha256"] == canonical_fingerprint(expected)
        stored = client.app.state.store.runs[run["id"]]["rollouts"]
        assert stored.sample_id.tolist() == expected.sample_id.tolist()
        assert stored.response.tolist() == expected.response.tolist()
    assert len(copied_paths) == 1
    assert not copied_paths[0].exists()


def test_source_hashing_refuses_symlink_inserted_at_final_open(tmp_path, monkeypatch):
    from fastapi import HTTPException

    import rolloutdiff.app as module

    source = tmp_path / "1.jsonl"
    source.write_bytes(b"original")
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.write_bytes(b"outside secret")
    original_open = module.os.open

    def replaced_open(path, flags, *args, **kwargs):
        if path == "1.jsonl":
            source.unlink()
            source.symlink_to(outside)
            try:
                return original_open(path, flags, *args, **kwargs)
            finally:
                source.unlink()
                source.write_bytes(b"original")
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", replaced_open)
    with pytest.raises(HTTPException) as caught:
        module._hash_file(tmp_path, source)
    assert caught.value.status_code == 400
    assert source.read_bytes() == b"original"


def test_trl_selection_uses_only_the_adapter_supported_completions_level(tmp_path):
    import pandas as pd

    directory = tmp_path / "trainer" / "completions" / "completions"
    directory.mkdir(parents=True)
    pd.DataFrame(
        [{"step": 1, "prompt": "p", "completion": "plain", "reward": 0}]
    ).to_parquet(directory / "completions_1.parquet")
    with api(tmp_path) as client:
        response = client.post(
            "/api/runs", json={"path": "trainer", "format": "trl", "run": "r"}
        )
        assert response.status_code == 400


def test_app_refuses_missing_secure_descriptor_capability_before_startup(
    tmp_path, monkeypatch
):
    import rolloutdiff.app as module

    monkeypatch.delattr(module.os, "O_NOFOLLOW")
    with pytest.raises(ValueError, match=r"macOS or Linux.*descriptor.*no-follow"):
        module.create_app(tmp_path, token=TOKEN)


def test_headline_values_equal_shared_prompt_change(tmp_path):
    write_run(tmp_path)
    with api(tmp_path) as client:
        prefix = f"/api/runs/{load_run(client)['id']}"
        listing = client.get(prefix + "/events").json()
        event = listing["events"][0]
        detail = client.get(prefix + f"/events/{event['rank']}").json()
        for row in detail["evidence"]:
            feature = row["feature"]
            series = client.get(prefix + "/series", params={"feature": feature}).json()
            headline = series["headline"]
            assert detail["headlines"][feature] == headline
            assert headline["after"] - headline["before"] == pytest.approx(
                row["prompt_shared_change"]
            )
            assert headline["n_prompts"] == row["prompt_shared_prompts"]
            assert series["direction"] in {"up", "down"}
        assert (
            listing["headlines"][event["representative_feature"]]
            == detail["headlines"][event["representative_feature"]]
        )


def test_run_without_alerts_has_no_headlines(tmp_path):
    directory = tmp_path / "rollouts"
    directory.mkdir()
    for step in range(1, 24):
        rows = [
            {
                "step": step,
                "input": f"private prompt {prompt}",
                "output": "plain words.",
                "score": 0,
                "sample_id": f"s-{step}-{prompt}-{j}",
            }
            for prompt in range(6)
            for j in range(2)
        ]
        (directory / f"{step}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    with api(tmp_path) as client:
        prefix = f"/api/runs/{load_run(client)['id']}"
        listing = client.get(prefix + "/events")
        assert listing.status_code == 200
        assert listing.json()["events"] == []
        assert listing.json()["headlines"] == {}
        series = client.get(prefix + "/series", params={"feature": "scalar:reward"})
        assert series.status_code == 200
        assert series.json()["headline"] is None
        assert series.json()["direction"] is None


def test_features_report_the_total_before_the_limit(tmp_path):
    write_run(tmp_path)
    with api(tmp_path) as client:
        prefix = f"/api/runs/{load_run(client)['id']}"
        limited = client.get(prefix + "/features", params={"limit": 2}).json()
        everything = client.get(prefix + "/features", params={"limit": 5000}).json()
        assert len(limited["features"]) == 2
        assert limited["total"] == everything["total"] == len(everything["features"])


def test_steps_lists_observed_steps_in_order(tmp_path):
    write_run(tmp_path)
    with api(tmp_path) as client:
        prefix = f"/api/runs/{load_run(client)['id']}"
        steps = client.get(prefix + "/steps").json()["steps"]
        feature = client.get(prefix + "/events").json()["events"][0][
            "representative_feature"
        ]
        points = client.get(prefix + "/series", params={"feature": feature}).json()[
            "points"
        ]
        assert steps == sorted(steps)
        assert {point["step"] for point in points} <= set(steps)


def test_headline_is_absent_without_shared_prompts():
    import pandas as pd

    from rolloutdiff.app import _headlines

    class Analysis:
        prompt_values = pd.DataFrame(
            {
                "feature": ["kept", "kept", "kept", "split", "split"],
                "before_mean": [0.0, 0.5, None, 0.2, None],
                "after_mean": [1.0, 0.5, 0.9, None, 0.4],
                "change": [1.0, 0.0, None, None, None],
            }
        )

    assert _headlines(Analysis, ["kept", "split", "unknown"]) == {
        "kept": {"before": 0.25, "after": 0.75, "n_prompts": 2}
    }

    class Empty:
        prompt_values = pd.DataFrame(columns=["run", "feature"])

    assert _headlines(Empty, ["kept"]) == {}
