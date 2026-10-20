"""Opt-in ingestion check on one shard of a public verl 0.9.1 GRPO coding run."""

import hashlib
import os
from pathlib import Path

import pytest

from rolloutdiff import featurize, load_verl

pytestmark = pytest.mark.local_data

REPO_ID = (
    "hi-todayis-jh/"
    "grpo-qwen3-1.7b-nemotron-leetcode-clean-3.2k-bs32-n8-verl091-146102-rollouts"
)
REVISION = "1f4fdec9825f0ff963e9cc414b775eaa3cf69acc"
LICENSE_EVIDENCE = (
    f"https://huggingface.co/datasets/{REPO_ID}/resolve/{REVISION}/README.md"
)
SHARD = "step_000001.jsonl.gz"
SHARD_BYTES = 2962495
SHARD_SHA256 = "6107a19548fd5024e6e58de22396fbda0d3321a18a54e8ca009cfbd667b80625"


@pytest.fixture(scope="module")
def rollouts():
    source = os.environ.get("ROLLOUTDIFF_VERL_SHARD")
    if not source:
        pytest.skip(f"set ROLLOUTDIFF_VERL_SHARD to {SHARD} from {REPO_ID}@{REVISION}")
    path = Path(source)
    if path.is_dir():
        path = path / SHARD
    data = path.read_bytes()
    assert len(data) == SHARD_BYTES
    assert hashlib.sha256(data).hexdigest() == SHARD_SHA256
    return load_verl(path, run="verl_leetcode")


def test_local_verl_shard_maps_to_canonical_rows(rollouts):
    assert len(rollouts) == 256
    assert rollouts.step.eq(1).all()
    assert rollouts.sample_id.is_unique
    assert rollouts.prompt_id.nunique() == 32
    assert rollouts.groupby("prompt_id").size().eq(8).all()
    assert rollouts.reward.between(0, 1).all()
    assert rollouts.reward.value_counts().to_dict() == {0.0: 158, 1.0: 98}
    assert not {"gts", "uid", "input", "output", "score"} & set(rollouts)


def test_local_verl_shard_featurizes(rollouts):
    tables = featurize(rollouts)
    assert len(tables.dense) == 256
    assert not tables.catalog.empty
