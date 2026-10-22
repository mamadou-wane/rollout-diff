import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import rolloutdiff
from rolloutdiff import FeatureTables


def inputs(positives, *, steps=range(1, 10), sparse=(), run="r"):
    rows = [
        {
            "run": run,
            "sample_id": f"{step}:{sample}",
            "step": step,
            "prompt_id": "p",
            "prompt": "prompt",
            "response": "private response",
            "reward": 0.0,
        }
        for step in steps
        for sample in range(4)
    ]
    data = pd.DataFrame(rows)
    dense = data[["run", "sample_id"]].copy()
    sparse_rows = []
    for name, present in positives.items():
        mask = data.sample_id.isin(present)
        if name in sparse:
            sparse_rows.extend(
                data.loc[mask, ["run", "sample_id"]]
                .assign(feature=name)
                .to_dict("records")
            )
        else:
            dense[name] = mask.astype(float)
    features = FeatureTables(
        dense,
        pd.DataFrame(sparse_rows, columns=["run", "sample_id", "feature"]),
        pd.DataFrame(
            [(name, "structural", "binary") for name in positives],
            columns=["feature", "tier", "kind"],
        ),
    )
    return data, features


def alerts(*specs, run="r"):
    return pd.DataFrame(
        [(run, name, "binary", "up", onset, end) for name, onset, end in specs],
        columns=["run", "feature", "kind", "direction", "onset_step", "alert_step"],
    )


def group(data, features, selected):
    assert hasattr(rolloutdiff, "group_events"), "event grouping API is missing"
    return rolloutdiff.group_events(data, features, selected)


def member_names(result):
    return [[feature for _, feature in refs] for refs in result.events.members]


@pytest.mark.parametrize(
    "distance,expected", [(2, [["a", "b"]]), (3, [["a", "b"]]), (4, [["a"], ["b"]])]
)
def test_numeric_onset_distance_boundary(distance, expected):
    present = {f"{step}:0" for step in range(1, 10)}
    out = group(
        *inputs({"a": present, "b": present}),
        alerts(("a", 1, 3), ("b", 1 + distance, 3 + distance)),
    )
    assert member_names(out) == expected
    assert (out.events.onset_end - out.events.onset_start).le(3).all()


@pytest.mark.parametrize(
    "b,expected", [({"1:0", "1:1"}, 1), ({"1:0", "1:1", "1:2"}, 2), ({"1:1"}, 2)]
)
def test_jaccard_half_is_inclusive(b, expected):
    out = group(*inputs({"a": {"1:0"}, "b": b}), alerts(("a", 1, 1), ("b", 1, 1)))
    assert len(out.events) == expected


def test_pair_window_includes_both_alert_windows_and_excludes_later_data():
    # Shared pair window 1..4: intersection=2, union=4. Restricting either
    # feature to its own window (1..2 or 2..4) would give intersection=1/4.
    data, features = inputs(
        {"a": {"1:0", "2:0", "3:0", "5:0", "5:1", "5:2"}, "b": {"2:0", "3:0", "4:0"}}
    )
    out = group(data, features, alerts(("a", 1, 2), ("b", 2, 4)))
    assert member_names(out) == [["a", "b"]]
    # The intersection window 2..2 has perfect overlap; the declared window
    # rejects this variant because intersection=2, union=5.
    features.dense.loc[data.sample_id.eq("4:1"), "b"] = 1
    out = group(data, features, alerts(("a", 1, 2), ("b", 2, 4)))
    assert member_names(out) == [["a"], ["b"]]


def test_temporal_chain_cannot_extend_event_span():
    present = {f"{step}:0" for step in range(1, 10)}
    out = group(
        *inputs(dict.fromkeys(["a", "b", "c"], present)),
        alerts(("a", 1, 3), ("b", 4, 6), ("c", 7, 9)),
    )
    assert member_names(out) == [["a", "b"], ["c"]]


def test_nontransitive_overlap_requires_every_member_pair():
    out = group(
        *inputs({"a": {"1:0"}, "b": {"1:0", "1:1"}, "c": {"1:1"}}),
        alerts(("a", 1, 1), ("b", 1, 1), ("c", 1, 1)),
    )
    assert member_names(out) == [["a", "b"], ["c"]]


@pytest.mark.parametrize(
    "ends,expected",
    [((1, 1, 2), [["a", "c"], ["b"]]), ((2, 1, 2), [["b", "c"], ["a"]])],
)
def test_first_created_compatible_event_wins_with_alert_and_feature_ties(
    ends, expected
):
    out = group(
        *inputs({"a": {"1:0"}, "b": {"1:1"}, "c": {"1:0", "1:1"}}),
        alerts(
            *[(name, 1, end) for name, end in zip(["a", "b", "c"], ends, strict=True)]
        ),
    )
    assert member_names(out) == expected


def test_sparse_absence_is_zero_and_empty_positive_sets_do_not_match():
    data, features = inputs(
        {"a": {"1:0"}, "b": {"1:0"}, "absent": set(), "empty": set()},
        sparse=["b", "absent"],
    )
    out = group(
        data, features, alerts(*[(name, 1, 2) for name in features.catalog.feature])
    )
    assert member_names(out) == [["a", "b"], ["absent"], ["empty"]]
    statuses = out.members.set_index("feature").status.to_dict()
    assert statuses == {
        "a": "grouped",
        "b": "grouped",
        "absent": "no_positive_rollouts",
        "empty": "no_positive_rollouts",
    }


def test_unsupported_and_unobserved_alerts_are_identifiable_singletons():
    data, features = inputs(
        {name: {"1:0"} for name in ["up", "down", "size", "missing"]}, steps=[1, 3]
    )
    features.catalog.loc[features.catalog.feature.eq("size"), "kind"] = "continuous"
    selected = alerts(
        ("up", 1, 3),
        ("down", 1, 3),
        ("size", 1, 3),
        ("missing", 2, 3),
        ("scalar:reward", 1, 3),
    )
    selected.loc[selected.feature.eq("down"), "direction"] = "down"
    selected.loc[selected.feature.isin(["size", "scalar:reward"]), "kind"] = (
        "continuous"
    )
    out = group(data, features, selected)
    assert len(out.events) == len(out.members) == 5
    assert out.events.n_members.eq(1).all()
    assert out.members.set_index("feature").status.to_dict() == {
        "down": "unsupported_direction",
        "size": "unsupported_kind",
        "scalar:reward": "unsupported_kind",
        "up": "singleton_no_match",
        "missing": "unavailable_window",
    }


def test_canonical_identity_alignment_cross_run_isolation_and_input_order():
    data, features = inputs({"a": {"1:0"}, "b": {"1:0"}}, sparse=["b"])
    other, other_features = inputs(
        {"a": {"1:0"}, "b": {"1:1"}}, sparse=["b"], run="other"
    )
    data = pd.concat([data, other])
    features = FeatureTables(
        pd.concat([features.dense, other_features.dense]),
        pd.concat([features.sparse, other_features.sparse]),
        features.catalog,
    )
    selected = pd.concat(
        [
            alerts(("a", 1, 3), ("b", 1, 3)),
            alerts(("a", 1, 3), ("b", 1, 3), run="other"),
        ]
    )
    original = [
        frame.copy(deep=True)
        for frame in [data, features.dense, features.sparse, features.catalog, selected]
    ]
    expected = group(data, features, selected)
    assert expected.events.run.tolist() == ["other", "other", "r"]
    assert member_names(expected) == [["a"], ["b"], ["a", "b"]]
    assert expected.events.event_id.tolist() == [
        "event-000001",
        "event-000002",
        "event-000003",
    ]
    for seed in [0, 1, 2]:
        shuffled = [frame.sample(frac=1, random_state=seed) for frame in original]
        for frame in shuffled:
            frame.index = [7] * len(frame)
        actual = group(shuffled[0], FeatureTables(*shuffled[1:4]), shuffled[4])
        assert_frame_equal(actual.events, expected.events)
        assert_frame_equal(actual.members, expected.members)
    for before, after in zip(
        original,
        [data, features.dense, features.sparse, features.catalog, selected],
        strict=True,
    ):
        assert_frame_equal(before, after)
    assert not {"response", "prompt", "sample_id", "reward"}.intersection(
        expected.events.columns
    )
    assert not {"response", "prompt", "sample_id", "reward"}.intersection(
        expected.members.columns
    )


def test_exact_large_steps_and_numeric_distance_on_gapped_schedule():
    step = 2**53 + 1
    present = {f"{value}:0" for value in [step, step + 3, step + 7]}
    out = group(
        *inputs(
            dict.fromkeys(["a", "b", "c"], present), steps=[step, step + 3, step + 7]
        ),
        alerts(
            ("a", step, step + 3), ("b", step + 3, step + 3), ("c", step + 7, step + 7)
        ),
    )
    assert member_names(out) == [["a", "b"], ["c"]]
    assert out.events.onset_start.tolist() == [step, step + 7]
    assert out.events.onset_end.tolist() == [step + 3, step + 7]
    assert out.members.onset_step.tolist() == [step, step + 3, step + 7]


def test_singleton_and_empty_results_have_stable_schema():
    data, features = inputs({"a": {"1:0"}})
    one = group(data, features, alerts(("a", 1, 3)))
    assert one.events.members.tolist() == [(("r", "a"),)]
    assert one.members.status.tolist() == ["singleton_no_match"]
    for frame, tables in [
        (data, features),
        (
            data.iloc[:0],
            FeatureTables(
                features.dense.iloc[:0], features.sparse.iloc[:0], features.catalog
            ),
        ),
    ]:
        out = group(frame, tables, alerts())
        assert_frame_equal(out.events, one.events.iloc[:0])
        assert_frame_equal(out.members, one.members.iloc[:0])


@pytest.mark.parametrize(
    "column,value",
    [
        ("run", "unknown"),
        ("feature", "unknown"),
        ("kind", "continuous"),
        ("direction", "left"),
        ("onset_step", -1),
        ("alert_step", 0),
        ("onset_step", 1.5),
        ("onset_step", True),
        ("kind", pd.NA),
    ],
)
def test_invalid_alerts_raise_value_error(column, value):
    selected = alerts(("a", 1, 3)).astype(object)
    selected.loc[0, column] = value
    with pytest.raises(ValueError):
        group(*inputs({"a": {"1:0"}}), selected)


def test_duplicate_alert_or_misaligned_feature_identity_is_rejected():
    data, features = inputs({"a": {"1:0"}})
    selected = alerts(("a", 1, 3))
    with pytest.raises(ValueError, match="duplicate"):
        group(data, features, pd.concat([selected, selected]))
    features.dense.loc[0, "sample_id"] = "unknown"
    with pytest.raises(ValueError, match="keys"):
        group(data, features, selected)
