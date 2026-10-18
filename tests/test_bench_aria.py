import pytest

from bench.aria import load_aria_rollouts


@pytest.mark.parametrize(
    "windows", [[], [(0, 15)], [(1, 201)], [(True, 15)], [(15, 1)]]
)
def test_local_aria_loader_rejects_invalid_windows_before_reading(windows):
    with pytest.raises(ValueError, match="windows"):
        load_aria_rollouts("/nonexistent", windows)


def test_local_aria_loader_rejects_unpinned_data(tmp_path):
    path = tmp_path / "aria.parquet"
    path.write_bytes(b"not the pinned source")
    with pytest.raises(ValueError, match="checksum"):
        load_aria_rollouts(path)
