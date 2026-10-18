import copy
import hashlib
import json

import pandas as pd
import pytest

from rolloutdiff.schema import CANONICAL_COLUMNS


def manifest_for(tmp_path, tables, adapter="aria", license_status="declared"):
    files = []
    all_steps = set()
    n_rows = 0
    for name, table in tables.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        files.append(
            {
                "local_path": name,
                "upstream_path": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
                "url": f"https://example.org/resolve/{'a' * 40}/{name}",
            }
        )
        n_rows += len(table)
        all_steps.update(table.source_eval_file if adapter == "aisi" else table.step)
    return {
        "schema_version": 1,
        "runs": {
            "test": {
                "adapter": adapter,
                "revision": "a" * 40,
                "repo_id": "owner/data",
                "license": {
                    "status": license_status,
                    "identifier": "mit" if license_status == "declared" else None,
                    "evidence_url": "https://example.org/card",
                },
                "files": files,
                "expected_rows": n_rows,
                "expected_steps": len(all_steps),
            }
        },
    }


def aria_table():
    return pd.DataFrame(
        [
            {
                "sample_id": f"step{step:03d}_row{row:03d}",
                "step": step,
                "problem_id": 42,
                "score": float(row),
                "messages": json.dumps(
                    [
                        {"role": "system", "content": "system"},
                        {"role": "user", "content": " exact prompt "},
                        {
                            "role": "assistant",
                            "content": "assistant\n<think>reason</think>\ncode ",
                        },
                    ]
                ),
                "is_reward_hack_strict": row,
                "is_reward_hack_loose": 1,
                "eq_correct": 1 - row,
                "category_5way": row + 1,
            }
            for step in (1, 2)
            for row in (0, 1)
        ]
    )


def load(tmp_path, manifest):
    from bench.public_data import load_public_run

    return load_public_run(tmp_path, "test", manifest=manifest)


def test_aria_separates_labels_and_preserves_source_text(tmp_path):
    result = load(tmp_path, manifest_for(tmp_path, {"aria.parquet": aria_table()}))
    assert list(result.rollouts) == CANONICAL_COLUMNS
    assert result.rollouts.response.eq("assistant\n<think>reason</think>\ncode ").all()
    assert result.rollouts.prompt.eq(" exact prompt ").all()
    assert result.rollouts.prompt_id.eq("42").all()
    assert result.labels.label_hack.tolist() == [0, 1, 0, 1]
    assert result.labels.label_attempt.eq(1).all()
    assert result.labels.category_5way.tolist() == [1, 2, 1, 2]
    assert result.rollouts[["run", "sample_id", "step", "prompt_id"]].equals(
        result.labels[["run", "sample_id", "step", "prompt_id"]]
    )
    assert result.provenance["raw_rows"] == 4
    assert result.provenance["retained_rows"] == 4
    assert result.provenance["license"]["status"] == "declared"


@pytest.mark.parametrize("status", ["not_asserted", "unresolved"])
def test_unasserted_license_is_not_upgraded(tmp_path, status):
    result = load(
        tmp_path,
        manifest_for(tmp_path, {"aria.parquet": aria_table()}, license_status=status),
    )
    assert result.provenance["license"]["status"] == status
    assert result.provenance["license"]["identifier"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda m: m["runs"]["test"].update(revision="main"), "revision"),
        (lambda m: m["runs"]["test"]["license"].update(status="open"), "license"),
        (lambda m: m["runs"]["test"].pop("license"), "license"),
        (lambda m: m["runs"]["test"]["files"][0].update(sha256="bad"), "sha256"),
        (lambda m: m["runs"]["test"]["files"][0].update(size_bytes=1), "size"),
        (lambda m: m["runs"]["test"]["files"][0].update(sha256="b" * 64), "checksum"),
        (
            lambda m: m["runs"]["test"]["files"][0].update(
                url="https://example.org/main/file"
            ),
            "revision",
        ),
        (lambda m: m["runs"]["test"].update(expected_rows=3), "rows"),
        (lambda m: m["runs"]["test"].update(expected_steps=3), "steps"),
    ],
)
def test_invalid_manifest_or_source_fails(tmp_path, mutation, match):
    manifest = manifest_for(tmp_path, {"aria.parquet": aria_table()})
    mutation(manifest)
    with pytest.raises(ValueError, match=match):
        load(tmp_path, manifest)


@pytest.mark.parametrize("path", ["../aria.parquet", "/tmp/aria.parquet"])
def test_manifest_paths_are_confined(tmp_path, path):
    manifest = manifest_for(tmp_path, {"aria.parquet": aria_table()})
    manifest["runs"]["test"]["files"][0]["local_path"] = path
    with pytest.raises(ValueError, match=r"relative|outside"):
        load(tmp_path, manifest)


def test_symlink_cannot_escape_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    manifest = manifest_for(tmp_path, {"aria.parquet": aria_table()})
    (data_root / "aria.parquet").symlink_to(tmp_path / "aria.parquet")
    with pytest.raises(ValueError, match="outside"):
        load(data_root, manifest)


def test_checks_all_hashes_before_decoding(tmp_path, monkeypatch):
    table = aria_table()
    manifest = manifest_for(
        tmp_path, {"a.parquet": table.iloc[:2], "b.parquet": table.iloc[2:]}
    )
    manifest["runs"]["test"]["files"][1]["sha256"] = "b" * 64
    monkeypatch.setattr(
        pd,
        "read_parquet",
        lambda *a, **k: pytest.fail("decoded before pin verification"),
    )
    with pytest.raises(ValueError, match="checksum"):
        load(tmp_path, manifest)


def aisi_table():
    # Equal timestamp prefixes must still yield distinct sample identities.
    return pd.DataFrame(
        [
            {
                "source_eval_file": f"2026-05-07T00-00-{second}.eval",
                "sample_id": str(row),
                "epoch": 1,
                "problem_id": "p",
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "p"},
                    {"role": "assistant", "content": "different source chat answer"},
                ],
                "response": "<thinking>exact</thinking>\nanswer ",
                "training_passed": float(row),
                "reward_hacked": float(row),
                "actually_solved": float(1 - row),
                "rh_exit": float(row),
                "rh_conftest": 0.0,
                "rh_always_equal": 0.0,
            }
            for second in ("02", "01")
            for row in (0, 1)
        ]
    )


def test_aisi_identity_chronology_and_unknown_attempts(tmp_path):
    table = aisi_table()
    result = load(
        tmp_path, manifest_for(tmp_path, {"aisi.parquet": table}, adapter="aisi")
    )
    assert result.rollouts.step.tolist() == [1, 1, 2, 2]
    assert result.rollouts.sample_id.is_unique
    identities = result.rollouts.sample_id.map(json.loads)
    assert identities.iloc[0] == ["2026-05-07T00-00-01.eval", "0", 1]
    assert result.rollouts.response.eq("<thinking>exact</thinking>\nanswer ").all()
    assert result.labels.label_attempt.isna().all()
    assert result.labels.rh_exit.tolist() == [0, 1, 0, 1]
    assert (
        result.provenance["step_definition"]
        == "1-based ordinal of lexicographically sorted full source_eval_file"
    )


@pytest.mark.parametrize("missing_column", ["training_passed", "response"])
def test_missing_aisi_required_field_excludes_whole_step(tmp_path, missing_column):
    table = aisi_table()
    table.loc[0, missing_column] = None
    table.loc[0, "reward_hacked"] = None
    result = load(
        tmp_path, manifest_for(tmp_path, {"aisi.parquet": table}, adapter="aisi")
    )
    assert len(result.rollouts) == 2
    assert result.rollouts.step.eq(1).all()
    assert result.provenance["excluded_steps"] == [2]
    assert result.provenance["excluded_rows"] == 2
    assert result.provenance["missing_fields"][missing_column] == 1
    assert result.provenance["source_label_coverage"]["label_hack"]["observed"] == 3
    assert result.provenance["source_label_coverage"]["label_hack"]["missing"] == 1
    assert result.provenance["retained_label_coverage"]["label_hack"]["observed"] == 2


def test_missing_label_does_not_exclude_rollout_or_become_negative(tmp_path):
    table = aria_table()
    table.loc[0, "is_reward_hack_strict"] = None
    result = load(tmp_path, manifest_for(tmp_path, {"aria.parquet": table}))
    assert len(result.rollouts) == 4
    assert pd.isna(result.labels.label_hack.iloc[0])
    assert result.provenance["excluded_steps"] == []


@pytest.mark.parametrize("bad_label", [2, -1, 0.5, "yes"])
def test_invalid_observed_binary_label_rejected(tmp_path, bad_label):
    table = aria_table().astype({"is_reward_hack_strict": object})
    table.loc[0, "is_reward_hack_strict"] = bad_label
    if isinstance(bad_label, str):
        table.is_reward_hack_strict = table.is_reward_hack_strict.astype(str)
    with pytest.raises(ValueError, match="binary"):
        load(tmp_path, manifest_for(tmp_path, {"aria.parquet": table}))


def trl_table(step):
    return pd.DataFrame(
        {
            "step": [step, step],
            "prompt": ["p", "p"],
            "completion": ["<thinking>x</thinking>\na", "b"],
            "training_passed": [0.0, 1.0],
            "proxy_reward_hacked": [0.0, 1.0],
            "proxy_actually_solved": [1.0, 0.0],
            "proxy_rh_exit": [0.0, 1.0],
            "proxy_rh_conftest": [0.0, 0.0],
            "proxy_rh_always_equal": [0.0, 0.0],
            "rh_reward_hacked": [None, 0.0],
            "rh_actually_solved": [None, 1.0],
        }
    )


def test_trl_native_identity_and_proxy_label_preference(tmp_path):
    tables = {f"trl/completions_{step:05d}.parquet": trl_table(step) for step in (2, 1)}
    result = load(tmp_path, manifest_for(tmp_path, tables, adapter="trl"))
    assert result.rollouts.sample_id.tolist() == [
        f"completions_{s:05d}.parquet:{r}" for s in (1, 2) for r in (0, 1)
    ]
    assert result.rollouts.response.iloc[0] == "<thinking>x</thinking>\na"
    assert result.labels.label_hack.tolist() == [0, 1, 0, 1]
    assert result.labels.rh_reward_hacked.tolist()[1::2] == [0, 0]
    assert result.labels.rh_reward_hacked.isna().sum() == 2
    assert result.labels.label_attempt.isna().all()
    assert result.labels.proxy_rh_exit.tolist() == [0, 1, 0, 1]


def test_trl_missing_proxy_label_stays_unknown_despite_partial_grader(tmp_path):
    table = trl_table(1).drop(columns=["proxy_reward_hacked"])
    result = load(
        tmp_path,
        manifest_for(tmp_path, {"completions_00001.parquet": table}, adapter="trl"),
    )
    assert result.labels.label_hack.isna().all()
    assert result.labels.rh_reward_hacked.notna().sum() == 1


def test_trl_filename_step_and_explicit_reward_contract(tmp_path):
    table = trl_table(2)
    manifest = manifest_for(
        tmp_path, {"completions_00001.parquet": table}, adapter="trl"
    )
    with pytest.raises(ValueError, match="filename"):
        load(tmp_path, manifest)
    table = trl_table(1).drop(columns=["training_passed"])
    manifest = manifest_for(
        tmp_path, {"completions_00001.parquet": table}, adapter="trl"
    )
    with pytest.raises(ValueError, match="training_passed"):
        load(tmp_path, manifest)


def test_load_does_not_mutate_manifest(tmp_path):
    manifest = manifest_for(tmp_path, {"aria.parquet": aria_table()})
    original = copy.deepcopy(manifest)
    load(tmp_path, manifest)
    assert manifest == original


def test_aisi_system_prompt_variants_have_distinct_canonical_identity(tmp_path):
    table = aisi_table()
    # Same source problem, different ordering of system instructions within a batch.
    for index in (1, 3):
        messages = copy.deepcopy(table.at[index, "messages"])
        messages[0]["content"] = "different system instruction order"
        table.at[index, "messages"] = messages
    result = load(
        tmp_path, manifest_for(tmp_path, {"aisi.parquet": table}, adapter="aisi")
    )
    assert result.rollouts.prompt_id.nunique() == 2
    assert result.rollouts.groupby("step").prompt_id.nunique().eq(2).all()
    assert result.rollouts.prompt.nunique() == 1
    assert result.labels.source_problem_id.eq("p").all()
    assert result.provenance["source_problem_count"] == 1
    assert "source_problem_id" not in result.provenance["source_label_coverage"]


def test_evaluation_labels_cannot_change_canonical_rollouts(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    table = aisi_table()
    first = load(
        first_root, manifest_for(first_root, {"aisi.parquet": table}, adapter="aisi")
    )
    changed = table.copy()
    for column in [
        "reward_hacked",
        "actually_solved",
        "rh_exit",
        "rh_conftest",
        "rh_always_equal",
    ]:
        changed[column] = 1 - changed[column]
    second = load(
        second_root,
        manifest_for(second_root, {"aisi.parquet": changed}, adapter="aisi"),
    )
    pd.testing.assert_frame_equal(first.rollouts, second.rollouts)
    assert not first.labels.label_hack.equals(second.labels.label_hack)


def test_aisi_sampling_units_are_explicit_source_problems_not_prompt_replacements(
    tmp_path,
):
    table = aisi_table()
    for index in (1, 3):
        messages = copy.deepcopy(table.at[index, "messages"])
        messages[0]["content"] = "second exact variant"
        table.at[index, "messages"] = messages
    data = load(
        tmp_path, manifest_for(tmp_path, {"aisi.parquet": table}, adapter="aisi")
    )
    assert list(data.sampling_units) == ["run", "prompt_id", "sampling_unit_id"]
    assert len(data.sampling_units) == 2
    assert data.sampling_units.sampling_unit_id.tolist() == ["p", "p"]
    assert set(data.sampling_units.prompt_id) == set(data.rollouts.prompt_id)
    assert data.rollouts.prompt_messages.map(lambda m: m[0]["content"]).nunique() == 2
    assert set(data.rollouts) == set(CANONICAL_COLUMNS)
    assert data.provenance["sampling_unit_definition"] == "source problem_id"
    inverted = table.copy()
    inverted["reward_hacked"] = 1 - inverted.reward_hacked
    other_root = tmp_path / "inverted"
    other = load(
        other_root, manifest_for(other_root, {"aisi.parquet": inverted}, adapter="aisi")
    )
    pd.testing.assert_frame_equal(data.sampling_units, other.sampling_units)
    pd.testing.assert_frame_equal(data.rollouts, other.rollouts)
