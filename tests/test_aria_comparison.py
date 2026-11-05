"""Known pinned development case for general window comparison."""

import os

import pytest

from bench.aria import load_aria_rollouts
from rolloutdiff import compare_rollouts, resolve_comparison_citations

pytestmark = pytest.mark.local_data


def test_aria_roadmap_windows_report_run_tests_and_comparability():
    source = os.environ.get("ROLLOUTDIFF_ARIA_PATH")
    if not source:
        pytest.skip("set ROLLOUTDIFF_ARIA_PATH to the pinned local aria.parquet")
    rows = load_aria_rollouts(source, ((80, 100), (140, 160)))
    report = compare_rollouts(rows, a=(80, 100), b=(140, 160))
    feature = next(
        row for row in report["features"] if row["feature"] == "def:run_tests"
    )
    assert feature["a"]["n_rollouts"] == feature["b"]["n_rollouts"] == 5376
    assert feature["a"]["positive_prompts"] == 42
    assert feature["b"]["positive_prompts"] == 336
    assert feature["a"]["n_prompts"] == feature["b"]["n_prompts"] == 336
    assert feature["a"]["mean"] == pytest.approx(75 / 5376)
    assert feature["b"]["mean"] == pytest.approx(5321 / 5376)
    assert feature["paired"]["n_prompts"] == 119
    assert feature["paired"]["estimate"] == pytest.approx(0.9732142857142857)
    assert feature["paired"]["se"] == pytest.approx(0.007843357149100373)
    assert feature["difference"]["status"] == "partial_overlap"
    assert feature["difference"]["se"] is None
    shift = report["prompt_mix"]["prompt_shift"][0]
    assert shift["descriptor"] == "prompt_chars"
    assert shift["distance"] == pytest.approx(0.07440476190476192)
    assert shift["delta"] == pytest.approx(-46.67261904761904)
    for side in ("a", "b"):
        resolved = resolve_comparison_citations(
            report, rows, side=side, feature="def:run_tests"
        )
        assert resolved["status"] == "selected"
        assert len(resolved["items"]) == 3
