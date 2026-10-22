import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import rolloutdiff
from rolloutdiff.pipeline import _EVIDENCE_COLUMNS


def inputs(*specs):
    events, members, evidence = [], [], []
    for spec in specs:
        feature = spec["feature"]
        run = spec.get("run", "r")
        event_id = spec.get("event_id", f"{run}:{feature}")
        onset, alert = spec.get("onset_step", 10), spec.get("alert_step", 12)
        kind, direction = spec.get("kind", "binary"), spec.get("direction", "up")
        members.append(
            {
                "event_id": event_id,
                "run": run,
                "feature": feature,
                "kind": kind,
                "direction": direction,
                "onset_step": onset,
                "alert_step": alert,
                "status": spec.get("member_status", "singleton_no_match"),
            }
        )
        row = dict.fromkeys(_EVIDENCE_COLUMNS)
        row.update(
            run=run,
            feature=feature,
            tier="structural",
            kind=kind,
            sequential_status="alerted",
            onset_step=onset,
            alert_step=alert,
            scale_source="baseline_sd",
            numerical_floor_dominant=False,
            replication_status="replicated",
            replicates=True,
            replication_coverage_policy="contiguous_prefix",
            replication_horizon_complete=True,
            persistence_status="partial",
            persistence_fraction=0.5,
            persistence_coverage=0.5,
            persistence_n_observations=20,
            persistence_n_evaluable=10,
            persistence_n_retained=5,
            persistence_n_unavailable=10,
            reward_status="available",
            reward_mean_difference=1.0,
            reward_n_groups=6,
            reward_n_mixed_groups=4,
            reward_n_rollouts=12,
            reward_n_included_rollouts=8,
            promptmix_status="available",
            prompt_positive_prompts=2,
            prompt_total_prompts=4,
        )
        row.update({k: v for k, v in spec.items() if k in row})
        if not spec.get("omit_evidence", False):
            evidence.append(row)
    member_table = pd.DataFrame(
        members,
        columns=[
            "event_id",
            "run",
            "feature",
            "kind",
            "direction",
            "onset_step",
            "alert_step",
            "status",
        ],
        dtype=object,
    )
    for event_id, group in member_table.groupby("event_id", sort=True):
        ordered = group.sort_values(["onset_step", "alert_step", "feature"])
        events.append(
            {
                "event_id": event_id,
                "run": ordered.run.iloc[0],
                "onset_start": ordered.onset_step.min(),
                "onset_end": ordered.onset_step.max(),
                "n_members": len(ordered),
                "members": tuple(
                    ordered[["run", "feature"]].itertuples(index=False, name=None)
                ),
            }
        )
    return (
        pd.DataFrame(
            events,
            columns=[
                "event_id",
                "run",
                "onset_start",
                "onset_end",
                "n_members",
                "members",
            ],
            dtype=object,
        ),
        member_table,
        pd.DataFrame(evidence, columns=_EVIDENCE_COLUMNS, dtype=object),
    )


def rank(*specs):
    assert hasattr(rolloutdiff, "rank_events"), "event ranking API is missing"
    return rolloutdiff.rank_events(*inputs(*specs))


def names(result):
    return result.representative_feature.tolist()


def test_tier_precedes_replication_without_treating_reward_as_behavioral():
    result = rank(
        {"feature": "prose", "tier": "prose"},
        {
            "feature": "structure",
            "replication_status": "not_replicated",
            "replicates": False,
        },
        {
            "feature": "scalar:reward",
            "kind": "continuous",
            "reward_mean_difference": 1e9,
        },
    )
    assert names(result) == ["structure", "prose", "scalar:reward"]
    assert result.ranking_status.tolist() == ["behavioral", "behavioral", "contextual"]
    assert result.iloc[-1].tier == "structural"


def test_replication_positive_negative_and_unknown_are_distinct():
    result = rank(
        {
            "feature": "a_unknown",
            "replication_status": "pending",
            "replicates": None,
            "persistence_fraction": 1,
            "persistence_n_retained": 10,
        },
        {
            "feature": "b_no",
            "replication_status": "not_replicated",
            "replicates": False,
            "persistence_fraction": 1,
            "persistence_n_retained": 10,
        },
        {"feature": "c_yes", "persistence_fraction": 0, "persistence_n_retained": 0},
    )
    assert names(result) == ["c_yes", "b_no", "a_unknown"]
    assert result.replication_availability.tolist() == [
        "available",
        "available",
        "unavailable",
    ]
    assert pd.isna(result.iloc[-1].replicates)
    assert result.iloc[-1].replication_status == "pending"


@pytest.mark.parametrize(
    "better,worse",
    [
        (
            {"persistence_fraction": 1, "persistence_n_retained": 10},
            {
                "persistence_fraction": 0.9,
                "persistence_n_retained": 9,
                "persistence_coverage": 1,
                "persistence_status": "available",
                "persistence_n_observations": 10,
                "persistence_n_unavailable": 0,
            },
        ),
        (
            {
                "persistence_coverage": 1,
                "persistence_status": "available",
                "persistence_n_observations": 10,
                "persistence_n_unavailable": 0,
            },
            {"persistence_coverage": 0.5},
        ),
        (
            {
                "persistence_n_evaluable": 20,
                "persistence_n_observations": 40,
                "persistence_n_retained": 10,
                "persistence_n_unavailable": 20,
            },
            {"persistence_n_evaluable": 10},
        ),
        (
            {"persistence_fraction": 0, "persistence_n_retained": 0},
            {
                "persistence_fraction": None,
                "persistence_status": "unavailable_uncertainty",
            },
        ),
    ],
)
def test_persistence_uses_fraction_then_coverage_then_evaluable_count(better, worse):
    result = rank(dict(feature="z", **better), dict(feature="a", **worse))
    assert names(result) == ["z", "a"]


def test_partial_persistence_and_unavailable_values_remain_visible():
    result = rank(
        {"feature": "partial"},
        {
            "feature": "missing",
            "persistence_status": "no_future_observations",
            "persistence_fraction": None,
            "persistence_coverage": None,
            "persistence_n_evaluable": 0,
            "persistence_n_observations": 0,
            "persistence_n_retained": 0,
            "persistence_n_unavailable": 0,
        },
    )
    assert names(result) == ["partial", "missing"]
    assert result.persistence_status.tolist() == ["partial", "no_future_observations"]
    assert result.persistence_availability.tolist() == ["available", "unavailable"]
    assert pd.isna(result.iloc[1].persistence_fraction)


def test_signed_reward_prioritizes_positive_zero_negative_then_missing():
    result = rank(
        *[
            {"feature": name, "reward_mean_difference": value, "reward_status": status}
            for name, value, status in [
                ("a_missing", None, "no_mixed_groups"),
                ("b_negative", -10, "available"),
                ("c_zero", 0, "available"),
                ("d_positive", 1, "available"),
            ]
        ]
    )
    assert names(result) == ["d_positive", "c_zero", "b_negative", "a_missing"]
    assert result.reward_mean_difference.iloc[:3].tolist() == [1, 0, -10]
    assert pd.isna(result.iloc[-1].reward_mean_difference)
    assert result.iloc[-1].reward_availability == "unavailable"


@pytest.mark.parametrize(
    "better,worse",
    [
        (
            {"reward_mean_difference": 2, "reward_n_mixed_groups": 1},
            {"reward_mean_difference": 1, "reward_n_mixed_groups": 6},
        ),
        ({"reward_n_mixed_groups": 5}, {"reward_n_mixed_groups": 4}),
        ({"reward_n_included_rollouts": 10}, {"reward_n_included_rollouts": 8}),
    ],
)
def test_reward_counts_break_equal_signed_differences(better, worse):
    assert names(rank(dict(feature="z", **better), dict(feature="a", **worse))) == [
        "z",
        "a",
    ]


def test_breadth_uses_fraction_then_count_independently_of_shared_prompt_status():
    result = rank(
        {"feature": "a", "prompt_positive_prompts": 5, "prompt_total_prompts": 10},
        {
            "feature": "b",
            "prompt_positive_prompts": 3,
            "prompt_total_prompts": 4,
            "promptmix_status": "no_shared_prompts",
        },
        {
            "feature": "c",
            "prompt_positive_prompts": 6,
            "prompt_total_prompts": 8,
            "promptmix_status": "shared_uncertainty_unavailable",
        },
        {"feature": "d", "prompt_positive_prompts": None},
        {"feature": "e", "prompt_positive_prompts": 0},
    )
    assert names(result) == ["c", "b", "a", "e", "d"]
    assert result.breadth_availability.tolist() == ["available"] * 4 + ["unavailable"]
    assert pd.isna(result.iloc[-1].prompt_positive_prompts)


def test_representative_is_one_whole_row_and_warnings_include_other_members():
    result = rank(
        {
            "feature": "a",
            "event_id": "combined",
            "persistence_fraction": 0.1,
            "persistence_n_retained": 1,
            "reward_mean_difference": -5,
        },
        {
            "feature": "b",
            "event_id": "combined",
            "replication_status": "not_replicated",
            "replicates": False,
            "persistence_fraction": 1,
            "persistence_n_retained": 10,
            "reward_mean_difference": 100,
            "numerical_floor_dominant": True,
        },
        {"feature": "c", "persistence_fraction": 0.5},
    )
    assert names(result) == ["c", "a"]
    selected = result.iloc[1]
    assert selected.n_members == 2
    assert selected.persistence_fraction == 0.1
    assert selected.reward_mean_difference == -5
    assert selected.numerical_floor_dominant == False
    assert selected.numerical_floor_members == (("r", "b"),)
    assert selected.numerical_floor_unavailable_members == ()


def test_contextual_and_unavailable_events_are_retained_after_behavioral_events():
    result = rank(
        {
            "feature": "a_down",
            "direction": "down",
            "member_status": "unsupported_direction",
        },
        {"feature": "b_window", "member_status": "unavailable_window"},
        {"feature": "c_missing", "omit_evidence": True},
        {"feature": "d_context", "kind": "continuous"},
        {"feature": "z_behavior", "tier": "prose"},
    )
    assert names(result) == [
        "z_behavior",
        "a_down",
        "b_window",
        "c_missing",
        "d_context",
    ]
    assert result.ranking_status.tolist() == [
        "behavioral",
        "unsupported_direction",
        "unavailable_member",
        "unavailable_evidence",
        "contextual",
    ]
    missing = result.loc[result.representative_feature.eq("c_missing")].iloc[0]
    assert pd.isna(missing.tier)
    assert missing.missing_evidence_members == (("r", "c_missing"),)
    assert missing.numerical_floor_unavailable_members == (("r", "c_missing"),)


def test_all_components_missing_does_not_prioritize_from_tier_alone():
    result = rank(
        {
            "feature": "a",
            "replication_status": "unavailable",
            "replicates": None,
            "persistence_status": "unavailable_uncertainty",
            "persistence_fraction": None,
            "reward_status": "no_mixed_groups",
            "reward_mean_difference": None,
            "prompt_positive_prompts": None,
        },
        {"feature": "z", "tier": "prose"},
    )
    assert names(result) == ["z", "a"]
    assert result.iloc[1].ranking_status == "unavailable_evidence"


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
def test_nullable_statuses_and_metrics_remain_explicitly_unavailable(missing):
    result = rank(
        {
            "feature": "a",
            "replication_status": missing,
            "replicates": missing,
            "persistence_status": missing,
            "persistence_fraction": missing,
            "reward_status": missing,
            "reward_mean_difference": missing,
            "prompt_positive_prompts": missing,
            "numerical_floor_dominant": missing,
        }
    )
    row = result.iloc[0]
    assert row.ranking_status == "unavailable_evidence"
    for name in ["replication", "persistence", "reward", "breadth"]:
        assert row[f"{name}_availability"] == "unavailable"
    assert pd.isna(row.replicates) and pd.isna(row.reward_mean_difference)
    assert row.numerical_floor_unavailable_members == (("r", "a"),)


def test_missing_tier_or_replication_status_does_not_invent_category_or_outcome():
    result = rank(
        {"feature": "tier", "tier": pd.NA},
        {"feature": "status", "replication_status": pd.NA},
    )
    assert names(result) == ["status", "tier"]
    assert result.iloc[0].replication_availability == "unavailable"
    assert result.iloc[1].ranking_status == "unavailable_evidence"


def test_missing_one_member_does_not_replace_available_representative_or_hide_warning():
    result = rank(
        {"feature": "a", "event_id": "e", "omit_evidence": True},
        {"feature": "b", "event_id": "e"},
    )
    assert names(result) == ["b"]
    assert result.iloc[0].missing_evidence_members == (("r", "a"),)


def test_ties_use_member_onset_alert_feature_then_event_identity():
    result = rank(
        {"feature": "a", "onset_step": 11, "alert_step": 13},
        {"feature": "b", "alert_step": 13},
        {"feature": "d"},
        {"feature": "c"},
    )
    assert names(result) == ["c", "d", "b", "a"]
    assert result["rank"].tolist() == [1, 2, 3, 4]


def test_cross_run_ranks_and_shuffled_inputs_are_independent_and_immutable():
    specs = [
        {"feature": "a", "run": "r", "reward_mean_difference": -100},
        {"feature": "b", "run": "r", "reward_mean_difference": -99},
        {"feature": "a", "run": "s", "reward_mean_difference": 100},
        {"feature": "b", "run": "s", "reward_mean_difference": 101},
    ]
    original = inputs(*specs)
    saved = [frame.copy(deep=True) for frame in original]
    expected = rank(*specs)
    assert list(
        zip(
            expected.run, expected["rank"], expected.representative_feature, strict=True
        )
    ) == [("r", 1, "b"), ("r", 2, "a"), ("s", 1, "b"), ("s", 2, "a")]
    for seed in [0, 1, 2]:
        reordered = [frame.sample(frac=1, random_state=seed) for frame in original]
        for frame in reordered:
            frame.index = [7] * len(frame)
        assert_frame_equal(
            rolloutdiff.rank_events(*reordered), expected, check_exact=True
        )
    for current, before in zip(original, saved, strict=True):
        assert_frame_equal(current, before, check_exact=True)


def test_empty_schema_matches_nonempty_table():
    expected = rank({"feature": "a"}).iloc[:0]
    assert_frame_equal(rolloutdiff.rank_events(*inputs()), expected)


@pytest.mark.parametrize("table", [0, 1, 2])
def test_duplicate_keys_are_rejected(table):
    tables = list(inputs({"feature": "a"}))
    tables[table] = pd.concat([tables[table], tables[table]])
    assert hasattr(rolloutdiff, "rank_events")
    with pytest.raises(ValueError, match="duplicate"):
        rolloutdiff.rank_events(*tables)


@pytest.mark.parametrize(
    "table,column,value",
    [
        (0, "n_members", 2),
        (0, "members", (("r", "unknown"),)),
        (0, "onset_end", 13),
        (1, "run", "other"),
        (2, "kind", "continuous"),
        (2, "onset_step", 9),
        (2, "replicates", False),
    ],
)
def test_conflicting_metadata_or_replication_is_rejected(table, column, value):
    tables = list(inputs({"feature": "a"}))
    tables[table].at[0, column] = value
    assert hasattr(rolloutdiff, "rank_events")
    with pytest.raises(ValueError):
        rolloutdiff.rank_events(*tables)


@pytest.mark.parametrize("event_id", [None, float("nan"), pd.NA, "", " ", "\t\n"])
def test_invalid_membership_event_id_cannot_disappear_during_grouping(event_id):
    events, members, evidence = inputs({"feature": "a"})
    orphan = members.assign(event_id=event_id, feature="orphan")
    members = pd.concat([members, orphan], ignore_index=True)
    with pytest.raises(
        ValueError, match="event members event_id must be a nonempty string"
    ):
        rolloutdiff.rank_events(events, members, evidence)


@pytest.mark.parametrize("status", ["available", "partial"])
@pytest.mark.parametrize(
    "changes",
    [
        {"persistence_fraction": 1.0, "persistence_n_retained": 0},
        {"persistence_n_retained": 11},
        {"persistence_n_retained": -1},
        {"persistence_n_retained": 1.5},
        {"persistence_n_evaluable": 21},
        {"persistence_n_evaluable": 0},
        {"persistence_n_unavailable": -1},
        {"persistence_n_unavailable": 3},
        {"persistence_n_observations": 0},
        {"persistence_fraction": 0.6},
        {"persistence_coverage": 0.25},
        {"persistence_fraction": float("inf")},
    ],
)
def test_contradictory_observed_persistence_raises_value_error(status, changes):
    spec = {"feature": "a", "persistence_status": status}
    if status == "available":
        spec.update(
            persistence_n_observations=10,
            persistence_n_unavailable=0,
            persistence_coverage=1,
        )
    events, members, evidence = inputs(spec)
    for name, value in changes.items():
        evidence.at[0, name] = value
    with pytest.raises(ValueError, match="persistence"):
        rolloutdiff.rank_events(events, members, evidence)


@pytest.mark.parametrize(
    "status,observed,unavailable", [("available", 20, 10), ("partial", 10, 0)]
)
def test_persistence_status_must_match_complete_observation_counts(
    status, observed, unavailable
):
    with pytest.raises(ValueError, match="persistence"):
        rank(
            {
                "feature": "a",
                "persistence_status": status,
                "persistence_n_observations": observed,
                "persistence_n_unavailable": unavailable,
                "persistence_coverage": 10 / observed,
            }
        )


def test_ratio_validation_tolerates_roundoff_without_rewriting_evidence():
    fraction = math.nextafter(1 / 3, math.inf)
    coverage = math.nextafter(3 / 7, 0)
    result = rank(
        {
            "feature": "a",
            "persistence_fraction": fraction,
            "persistence_coverage": coverage,
            "persistence_n_retained": 1,
            "persistence_n_evaluable": 3,
            "persistence_n_observations": 7,
            "persistence_n_unavailable": 4,
        }
    )
    assert result.iloc[0].persistence_availability == "available"
    assert result.iloc[0].persistence_fraction == fraction
    assert result.iloc[0].persistence_coverage == coverage


def test_tiny_nonzero_persistence_fraction_cannot_be_replaced_by_zero():
    with pytest.raises(ValueError, match="persistence_fraction"):
        rank(
            {
                "feature": "a",
                "persistence_n_retained": 1,
                "persistence_n_evaluable": 10**16,
                "persistence_n_observations": 2 * 10**16,
                "persistence_n_unavailable": 10**16,
                "persistence_fraction": 0,
            }
        )


@pytest.mark.parametrize("retained,unavailable", [(5, 8), (0, 10)])
def test_missing_evaluable_count_does_not_hide_impossible_present_counts(
    retained, unavailable
):
    with pytest.raises(ValueError, match="persistence"):
        rank(
            {
                "feature": "a",
                "persistence_n_evaluable": None,
                "persistence_n_retained": retained,
                "persistence_n_unavailable": unavailable,
                "persistence_n_observations": 10,
            }
        )


@pytest.mark.parametrize("missing", [None, float("nan"), pd.NA])
@pytest.mark.parametrize(
    "column",
    [
        "persistence_n_retained",
        "persistence_n_evaluable",
        "persistence_n_unavailable",
        "persistence_n_observations",
        "persistence_fraction",
        "persistence_coverage",
    ],
)
def test_missing_persistence_fields_preserve_availability_and_values(column, missing):
    result = rank({"feature": "a", column: missing})
    row = result.iloc[0]
    assert pd.isna(row[column])
    expected = (
        "available"
        if column in {"persistence_n_retained", "persistence_n_unavailable"}
        else "unavailable"
    )
    assert row.persistence_availability == expected


@pytest.mark.parametrize(
    "status", ["no_future_observations", "unavailable_uncertainty", None]
)
def test_explicitly_unavailable_persistence_is_not_reinterpreted(status):
    result = rank(
        {
            "feature": "a",
            "persistence_status": status,
            "persistence_fraction": 1,
            "persistence_n_retained": 0,
        }
    )
    row = result.iloc[0]
    assert row.persistence_availability == "unavailable"
    assert row.persistence_fraction == 1 and row.persistence_n_retained == 0
