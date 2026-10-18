import hashlib
import os
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from bench.external_baselines import (
    _prepare,
    _scalar_records,
    _source_context,
    _verify_source,
    run_external,
)


def rollouts():
    return pd.DataFrame(
        [
            {
                "run": "r",
                "step": 1,
                "sample_id": "a",
                "prompt_id": "p",
                "prompt": "p",
                "response": "a",
                "reward": 0.0,
            },
            {
                "run": "r",
                "step": 1,
                "sample_id": "b",
                "prompt_id": "p",
                "prompt": "p",
                "response": "b",
                "reward": 1.0,
            },
            {
                "run": "r",
                "step": 1,
                "sample_id": "c",
                "prompt_id": "q",
                "prompt": "q",
                "response": "c",
                "reward": 1.0,
            },
            {
                "run": "r",
                "step": 1,
                "sample_id": "d",
                "prompt_id": "q",
                "prompt": "q",
                "response": "d",
                "reward": 1.0,
            },
        ]
    )


def dense():
    return pd.DataFrame(
        {
            "run": ["r"] * 4,
            "sample_id": ["d", "b", "a", "c"],
            "resp_chars": [80.0, 40.0, 20.0, 60.0],
        }
    )


def fixture_source(tmp_path):
    folder = tmp_path / "fixture"
    package = folder / "src" / "fixture_tool"
    package.mkdir(parents=True)
    (folder / "LICENSE").write_text("test license")
    (package / "__init__.py").write_text("VALUE=12\n")
    files = {
        str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (folder / "LICENSE", package / "__init__.py")
    }
    return {
        "directory": "fixture",
        "import_path": "src",
        "module_prefix": "fixture_tool",
        "files": files,
    }


def test_records_align_dense_by_identity_and_keep_original_scalar_weighting():
    frame = _prepare(rollouts().sample(frac=1, random_state=2), dense())
    assert frame.sample_id.tolist() == ["a", "b", "c", "d"]
    assert frame.resp_chars.tolist() == [20.0, 40.0, 60.0, 80.0]
    record = _scalar_records(frame)[0]
    assert record["reward_mean"] == 0.75
    assert record["reward_std"] == np.std([0.0, 1.0, 1.0, 1.0])
    assert record["response_len_mean"] == 12.5
    assert record["response_len_max"] == 20.0
    assert record["zero_var_group_frac"] == 0.5
    assert record["group_reward_std_mean"] == pytest.approx(np.sqrt(0.5) / 2)
    assert record["advantage_std"] == pytest.approx(np.std([-0.5, 0.5, 0.0, 0.0]))
    assert not any("label" in key for key in record)


def test_dense_identity_mismatch_fails_instead_of_positional_pairing():
    with pytest.raises(ValueError, match="exactly match"):
        _prepare(rollouts(), dense().iloc[:-1])


def test_singleton_groups_have_unavailable_group_variance():
    frame = _prepare(
        rollouts().iloc[[0, 2]], dense().loc[dense().sample_id.isin(["a", "c"])]
    )
    record = _scalar_records(frame)[0]
    assert record["group_reward_std_mean"] is None
    assert record["zero_var_group_frac"] is None


def test_sources_checked_before_import_and_extra_code_rejected(tmp_path):
    metadata = fixture_source(tmp_path)
    _verify_source(tmp_path, metadata)
    package = tmp_path / "fixture" / "src" / "fixture_tool"
    (package / "surprise.py").write_text('raise RuntimeError("must never run")\n')
    with pytest.raises(ValueError, match="unmanifested"):
        _verify_source(tmp_path, metadata)
    (package / "surprise.py").unlink()
    (package / "__init__.py").write_text('raise RuntimeError("must never run")\n')
    with pytest.raises(ValueError, match="checksum"):
        _verify_source(tmp_path, metadata)


def test_bytecode_and_ambient_modules_cannot_shadow_pinned_sources(tmp_path):
    metadata = fixture_source(tmp_path)
    bytecode = tmp_path / "fixture" / "src" / "fixture_tool" / "shadow.pyc"
    bytecode.write_bytes(b"x")
    with pytest.raises(ValueError, match="unmanifested"):
        _verify_source(tmp_path, metadata)
    bytecode.unlink()
    sys.modules["fixture_tool"] = ModuleType("fixture_tool")
    try:
        with (
            pytest.raises(ValueError, match="already imported"),
            _source_context(tmp_path, metadata),
        ):
            pytest.fail("must reject ambient module before execution")
    finally:
        del sys.modules["fixture_tool"]


def test_missing_optional_sources_report_unavailable():
    result = run_external(rollouts(), dense(), None)
    assert set(result) == {"rlprobe", "rldoctor", "rhda_surface"}
    assert all(item["status"] == "unavailable" for item in result.values())
    assert all(item["provenance"]["revision"] for item in result.values())


def test_real_pinned_apis_preserve_finding_serialization_and_missingness():
    source = os.environ.get("ROLLOUTDIFF_EXTERNAL_SOURCES")
    if not source:
        pytest.skip(
            "set ROLLOUTDIFF_EXTERNAL_SOURCES for pinned external baseline API checks"
        )
    rows = []
    lengths = []
    for step in range(1, 41):
        for i in range(4):
            sample = f"{step}:{i}"
            rows.append(
                {
                    "run": "r",
                    "step": step,
                    "sample_id": sample,
                    "prompt_id": f"p{i // 2}",
                    "prompt": f"p{i // 2}",
                    "response": "print(1)",
                    "reward": 1.0,
                }
            )
            lengths.append({"run": "r", "sample_id": sample, "resp_chars": 8.0})
    result = run_external(pd.DataFrame(rows), pd.DataFrame(lengths), Path(source))
    assert all(item["status"] == "executed" for item in result.values())
    assert result["rlprobe"]["coverage"]["catalogue_size"] == 14
    assert result["rlprobe"]["findings"]
    assert all(
        "signature_id" in row and "explanation" in row
        for row in result["rlprobe"]["findings"]
    )
    doctor = result["rldoctor"]["findings"]
    assert any(row["severity"] == "SKIPPED" and row["missing_fields"] for row in doctor)
    assert result["rhda_surface"]["coverage"]["agent_executed"] is False
    assert len(result["rhda_surface"]["steps"]) == 40
    assert not any(
        name in ("rlprobe", "rldoctor", "detection")
        or name.startswith(("rlprobe.", "rldoctor.", "detection."))
        for name in sys.modules
    )


def test_mixed_singletons_make_whole_step_group_variance_unavailable():
    frame = _prepare(rollouts().iloc[:3], dense().loc[dense().sample_id.ne("d")])
    record = _scalar_records(frame)[0]
    assert record["group_reward_std_mean"] is None
    assert record["zero_var_group_frac"] is None


def test_all_available_sources_are_verified_before_any_external_execution(
    tmp_path, monkeypatch
):
    import json

    import bench.external_baselines as module

    first = fixture_source(tmp_path)
    second = fixture_source(tmp_path / "second")
    second["directory"] = "second/fixture"
    for metadata in (first, second):
        metadata.update(
            repository="example/fixture",
            revision="a" * 40,
            license="fixture",
            license_url="https://example.invalid",
            comparison="fixture",
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"tools": {"rhda_surface": first, "rlprobe": second}})
    )
    (tmp_path / "second" / "fixture" / "LICENSE").write_text("changed")
    monkeypatch.setattr(module, "_MANIFEST", manifest)

    def must_not_run(frame):
        pytest.fail("executed an external tool before checking every available source")

    monkeypatch.setattr(module, "_run_rhda", must_not_run)
    with pytest.raises(ValueError, match="checksum"):
        run_external(rollouts(), dense(), tmp_path)


def test_surface_output_exports_counts_without_response_ngrams(monkeypatch):
    import json
    from types import SimpleNamespace

    import bench.external_baselines as module

    sentinel = "quartzviper titaniumcloak"
    fake = ModuleType("detection.rhda.common.surface_stats")
    fake.compute_surface_stats = lambda entries, baseline, step: (
        f'"{sentinel}" 100% of responses',
        SimpleNamespace(step=step) if baseline is None else None,
    )
    monkeypatch.setitem(sys.modules, "detection.rhda.common.surface_stats", fake)
    result = module._run_rhda(_prepare(rollouts().assign(response=sentinel), dense()))
    assert result["steps"] == [{"step": 1, "n_drift_lines": 1}]
    assert sentinel not in json.dumps(result)


def test_findings_export_templates_and_metrics_without_sample_text_or_ids():
    import json

    import bench.external_baselines as module

    sentinel = "quartzviper titaniumcloak"
    probe = module._rlprobe_finding(
        {
            "signature_id": "DEGENERATE_GROUPS",
            "severity": "warn",
            "first_step": 1,
            "confidence": 0.9,
            "explanation": "Fixed explanation with 90% groups",
            "evidence": [{"metric": "zero_variance_group_frac", "value": 0.9}],
            "suggested_actions": ["Fixed suggestion"],
            "links_to_samples": [sentinel],
            "completion_text": sentinel,
        }
    )
    assert probe["n_linked_samples"] == 1
    assert probe["evidence"][0]["value"] == 0.9
    doctor = module._rldoctor_finding(
        {
            "detector": "advantage_collapse",
            "severity": "WARNING",
            "title": "Fixed title",
            "summary": "Fixed scalar summary",
            "evidence": ["90% groups"],
            "metrics": {"zero_var_frac": 0.9},
            "missing_fields": [],
            "response": sentinel,
        }
    )
    assert doctor["summary"] == "Fixed scalar summary"
    assert sentinel not in json.dumps([probe, doctor])


def test_real_external_results_never_export_distinctive_input_text():
    import json

    source = os.environ.get("ROLLOUTDIFF_EXTERNAL_SOURCES")
    if not source:
        pytest.skip(
            "set ROLLOUTDIFF_EXTERNAL_SOURCES for pinned external baseline API checks"
        )
    sentinel = "quartzviper titaniumcloak"
    rows = []
    lengths = []
    for step in range(1, 41):
        response = "ordinary baseline phrases" if step == 1 else sentinel
        for i in range(32):
            sample = f"{step}:{i}"
            rows.append(
                {
                    "run": "r",
                    "step": step,
                    "sample_id": sample,
                    "prompt_id": f"p{i // 16}",
                    "prompt": f"p{i // 16}",
                    "response": response,
                    "reward": 1.0,
                }
            )
            lengths.append(
                {"run": "r", "sample_id": sample, "resp_chars": float(len(response))}
            )
    result = run_external(pd.DataFrame(rows), pd.DataFrame(lengths), Path(source))
    assert result["rhda_surface"]["coverage"]["n_steps_with_drift"] > 0
    assert sentinel not in json.dumps(result)
    assert "quartzviper" not in json.dumps(result)
    assert all(
        set(row) == {"step", "n_drift_lines"} for row in result["rhda_surface"]["steps"]
    )


def test_namespace_root_cannot_be_replaced_by_ambient_regular_package(
    tmp_path, monkeypatch
):
    import importlib

    metadata = fixture_source(tmp_path)
    initializer = tmp_path / "fixture" / "src" / "fixture_tool" / "__init__.py"
    initializer.unlink()
    metadata["files"].pop("src/fixture_tool/__init__.py")
    pinned = initializer.with_name("pinned.py")
    pinned.write_text("VALUE=12\n")
    metadata["files"]["src/fixture_tool/pinned.py"] = hashlib.sha256(
        pinned.read_bytes()
    ).hexdigest()
    ambient = tmp_path / "ambient"
    package = ambient / "fixture_tool"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise AssertionError("ambient package executed")\n'
    )
    monkeypatch.syspath_prepend(str(ambient))
    with _source_context(tmp_path, metadata):
        module = importlib.import_module("fixture_tool.pinned")
        assert module.VALUE == 12
        assert Path(module.__file__).resolve() == pinned.resolve()


def test_real_cherrl_namespace_ignores_ambient_detection_package(tmp_path, monkeypatch):
    import importlib
    import json

    import bench.external_baselines as module

    source = os.environ.get("ROLLOUTDIFF_EXTERNAL_SOURCES")
    if not source:
        pytest.skip(
            "set ROLLOUTDIFF_EXTERNAL_SOURCES for pinned external baseline API checks"
        )
    package = tmp_path / "detection"
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise AssertionError("ambient detection executed")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    metadata = json.loads(module._MANIFEST.read_text())["tools"]["rhda_surface"]
    with _source_context(Path(source), metadata):
        surface = importlib.import_module("detection.rhda.common.surface_stats")
        expected = (
            Path(source)
            / "CHERRL"
            / "detection"
            / "rhda"
            / "common"
            / "surface_stats.py"
        )
        assert Path(surface.__file__).resolve() == expected.resolve()
    assert "detection" not in sys.modules


@pytest.mark.parametrize("name", ["rlprobe", "rldoctor"])
def test_real_regular_packages_are_bound_to_verified_source(
    tmp_path, monkeypatch, name
):
    import importlib
    import json

    import bench.external_baselines as module

    source = os.environ.get("ROLLOUTDIFF_EXTERNAL_SOURCES")
    if not source:
        pytest.skip(
            "set ROLLOUTDIFF_EXTERNAL_SOURCES for pinned external baseline API checks"
        )
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text(
        'raise AssertionError("ambient package executed")\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    metadata = json.loads(module._MANIFEST.read_text())["tools"][name]
    with _source_context(Path(source), metadata):
        actual = importlib.import_module(name)
        expected = Path(source) / name / "src" / name / "__init__.py"
        assert Path(actual.__file__).resolve() == expected.resolve()
    assert name not in sys.modules


def test_explicit_reward_groups_group_variants_without_rewriting_canonical_prompts():
    source = rollouts()
    before = source.copy(deep=True)
    groups = pd.DataFrame(
        {"run": ["r", "r"], "prompt_id": ["p", "q"], "group_id": ["problem", "problem"]}
    )
    frame = _prepare(source, dense(), reward_groups=groups)
    pd.testing.assert_frame_equal(source, before)
    assert frame.prompt_id.tolist() == ["p", "p", "q", "q"]
    assert frame.prompt.tolist() == ["p", "p", "q", "q"]
    assert frame.group_id.tolist() == ["problem"] * 4
    record = _scalar_records(frame)[0]
    assert record["reward_mean"] == 0.75
    assert record["group_reward_std_mean"] == pytest.approx(0.5)
    assert record["zero_var_group_frac"] == 0.0
    assert record["advantage_std"] == pytest.approx(np.std([-0.75, 0.25, 0.25, 0.25]))
    result = run_external(source, dense(), reward_groups=groups)
    for item in result.values():
        coverage = item["coverage"]["prompt_groups"][0]
        assert coverage["n_exact_prompts"] == 2
        assert coverage["n_reward_groups"] == coverage["n_groups"] == 1
        assert coverage["group_variance_available"] is True
        assert item["input_policy"]["grouping"] == "explicit descriptive reward groups"
        assert "not logged trainer advantages" in item["input_policy"]["advantage"]


def test_explicit_reward_groups_require_complete_mapping_and_keep_singletons_unavailable():
    groups = pd.DataFrame(
        {
            "run": ["r", "r"],
            "prompt_id": ["p", "q"],
            "group_id": ["problem-p", "problem-q"],
        }
    )
    with pytest.raises(ValueError, match="exactly match"):
        run_external(rollouts(), dense(), reward_groups=groups.iloc[:1])
    subset = rollouts().iloc[[0, 2]]
    features = dense().loc[dense().sample_id.isin(["a", "c"])]
    result = run_external(subset, features, reward_groups=groups)
    for item in result.values():
        coverage = item["coverage"]["prompt_groups"][0]
        assert coverage["n_exact_prompts"] == coverage["n_reward_groups"] == 2
        assert coverage["n_singleton_groups"] == 2
        assert coverage["group_variance_available"] is False
    record = _scalar_records(_prepare(subset, features, reward_groups=groups))[0]
    assert record["group_reward_std_mean"] is None
    assert record["zero_var_group_frac"] is None


def test_real_rlprobe_sample_records_keep_exact_prompt_ids_with_explicit_reward_groups(
    monkeypatch,
):
    import json

    import bench.external_baselines as module

    source = os.environ.get("ROLLOUTDIFF_EXTERNAL_SOURCES")
    if not source:
        pytest.skip(
            "set ROLLOUTDIFF_EXTERNAL_SOURCES for pinned external baseline API checks"
        )
    groups = pd.DataFrame(
        {"run": ["r", "r"], "prompt_id": ["p", "q"], "group_id": ["problem", "problem"]}
    )
    frame = _prepare(rollouts(), dense(), reward_groups=groups)
    metadata = json.loads(module._MANIFEST.read_text())["tools"]["rlprobe"]
    with _source_context(Path(source), metadata):
        from rlprobe.recorder import sampling

        captured = []

        def capture(candidates, config, rng):
            captured.extend(candidates)
            return candidates

        monkeypatch.setattr(sampling, "select_samples", capture)
        module._run_rlprobe(frame, _scalar_records(frame))
    assert [sample.prompt_id for sample in captured] == ["p", "p", "q", "q"]
    assert [sample.group_id for sample in captured] == ["problem"] * 4
    assert [sample.advantage for sample in captured] == [-0.75, 0.25, 0.25, 0.25]
