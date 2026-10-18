import json

import pandas as pd
import pytest

from rolloutdiff.adapters import from_table, load_trl, load_verl
from rolloutdiff.schema import Rollout, validate_rollouts


def record(**changes):
    return {
        "run": "run",
        "step": 1,
        "sample_id": "s",
        "prompt_id": "p",
        "prompt": "question",
        "response": "<think>retain this</think>\nanswer",
        "reward": 1.0,
    } | changes


@pytest.mark.parametrize(
    "changes",
    [
        {"step": 1.5},
        {"step": True},
        {"step": -1},
        {"reward": float("nan")},
        {"reward": float("inf")},
        {"reward": None},
        {"reward": True},
        {"prompt_id": ""},
        {"sample_id": None},
        {"prompt": None},
        {"label_hack": 1},
    ],
)
def test_canonical_rejects_invalid_data(changes):
    with pytest.raises(ValueError):
        Rollout(**record(**changes))


def test_canonical_preserves_text_and_rejects_duplicate_identity():
    frame = validate_rollouts(pd.DataFrame([record()]))
    assert frame.loc[0, "response"] == "<think>retain this</think>\nanswer"
    with pytest.raises(ValueError, match="duplicate"):
        validate_rollouts(pd.DataFrame([record(), record(step=2)]))


def test_prompt_identity_cannot_silently_merge_distinct_prompts():
    with pytest.raises(ValueError, match="prompt_id"):
        validate_rollouts(
            pd.DataFrame([record(), record(sample_id="t", prompt="other")])
        )
    assert (
        len(
            validate_rollouts(
                pd.DataFrame([record(), record(run="other", prompt="other")])
            )
        )
        == 2
    )


def test_empty_table_has_canonical_columns():
    out = validate_rollouts(pd.DataFrame(columns=list(record())))
    assert len(out) == 0
    assert {
        "run",
        "step",
        "sample_id",
        "prompt_id",
        "prompt",
        "response",
        "reward",
    } <= set(out)


def test_trl_explicit_reward_and_excludes_labels(tmp_path):
    raw = pd.DataFrame(
        {
            "step": [2, 2],
            "prompt": ["q", "q"],
            "completion": ["<think>raw</think>answer", ""],
            "format_reward": [0.2, 0.3],
            "training_passed": [1.0, 0.0],
            "proxy_reward_hacked": [1, 0],
        }
    )
    path = tmp_path / "completions_00002.parquet"
    raw.to_parquet(path)
    with pytest.raises(ValueError, match="reward_column"):
        load_trl(tmp_path, run="trl")
    out = load_trl(tmp_path, run="trl", reward_column="training_passed")
    assert out.reward.tolist() == [1.0, 0.0]
    assert out.response.tolist() == ["<think>raw</think>answer", ""]
    assert out.prompt_id.nunique() == 1
    assert not any(c.startswith(("proxy_", "label_")) for c in out)
    assert out.sample_id.is_unique
    pd.testing.assert_frame_equal(
        out, load_trl(path, run="trl", reward_column="training_passed")
    )


def test_trl_numeric_order_and_unrelated_files_ignored(tmp_path):
    for step in [10, 2]:
        pd.DataFrame(
            {"step": [step], "prompt": ["q"], "completion": ["a"], "reward": [1]}
        ).to_parquet(tmp_path / f"completions_{step}.parquet")
    pd.DataFrame({"unrelated": [42]}).to_parquet(tmp_path / "features.parquet")
    out = load_trl(tmp_path, run="run")
    assert out.step.tolist() == [2, 10]
    assert out.sample_id.nunique() == 2


def test_chat_prompt_roles_participate_in_identity_and_messages_survive_parquet(
    tmp_path,
):
    prompts = [
        [{"role": "system", "content": "same"}],
        [{"role": "user", "content": "same"}],
    ]
    completion = [
        {
            "role": "assistant",
            "content": "a",
            "tool_calls": [
                {
                    "id": "call",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        }
    ]
    pd.DataFrame(
        {
            "step": [1, 1],
            "prompt": prompts,
            "completion": [completion, completion],
            "reward": [1, 0],
        }
    ).to_parquet(tmp_path / "completions_1.parquet")
    out = load_trl(tmp_path, run="chat")
    assert out.prompt.tolist() == ["same", "same"]
    assert out.prompt_id.nunique() == 2
    assert out.iloc[0].prompt_messages == prompts[0]
    assert out.iloc[0].response_messages == completion
    assert out.iloc[0].tool_calls[0]["function"]["name"] == "read"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_missing_or_nonfinite_reward_is_not_zero(tmp_path, bad):
    pd.DataFrame(
        {"step": [1], "prompt": ["q"], "completion": ["a"], "reward": [bad]}
    ).to_parquet(tmp_path / "completions_1.parquet")
    with pytest.raises(ValueError, match="reward"):
        load_trl(tmp_path, run="r")


def test_filename_step_disagreement_fails(tmp_path):
    pd.DataFrame(
        {"step": [2], "prompt": ["q"], "completion": ["a"], "reward": [1]}
    ).to_parquet(tmp_path / "completions_1.parquet")
    with pytest.raises(ValueError, match="step"):
        load_trl(tmp_path, run="r")


def test_verl_layout_and_stable_ids(tmp_path):
    for step in [10, 2]:
        rows = [
            {"step": step, "input": "q", "output": "a", "score": 0.5, "gts": "ignored"},
            {"step": step, "input": "q", "output": "b", "score": 1.0},
        ]
        (tmp_path / f"{step}.jsonl").write_text(
            "".join(json.dumps(x) + "\n" for x in rows)
        )
    out = load_verl(tmp_path, run="verl")
    assert out.step.tolist() == [2, 2, 10, 10]
    assert out.reward.tolist() == [0.5, 1.0, 0.5, 1.0]
    assert out.prompt_id.nunique() == 1
    assert out.sample_id.nunique() == 4
    assert "gts" not in out


def test_verl_malformed_row_reports_location_without_contents(tmp_path):
    (tmp_path / "2.jsonl").write_text('{"private": "do not echo"\n')
    with pytest.raises(ValueError, match=r"2.jsonl:1") as err:
        load_verl(tmp_path, run="r")
    assert "do not echo" not in str(err.value)


def test_explicit_column_mapping_preserves_ids_and_ignores_annotations():
    raw = pd.DataFrame(
        {
            "iteration": [3],
            "question": ["q"],
            "answer": ["a"],
            "score": [0.75],
            "id": [14],
            "problem": [27],
            "label_hack": [1],
        },
        index=[91],
    )
    out = from_table(
        raw,
        run="r",
        columns={
            "step": "iteration",
            "prompt": "question",
            "response": "answer",
            "reward": "score",
            "sample_id": "id",
            "prompt_id": "problem",
        },
    )
    assert out.iloc[0][["step", "sample_id", "prompt_id", "reward"]].tolist() == [
        3,
        "14",
        "27",
        0.75,
    ]
    assert "label_hack" not in out
    with pytest.raises(ValueError, match="column"):
        from_table(raw, run="r", columns={"step": "absent"})


def test_unsupported_content_fails_instead_of_stringifying():
    raw = pd.DataFrame(
        {
            "step": [1],
            "prompt": [[{"role": "user", "content": [{"type": "image", "url": "x"}]}]],
            "response": ["a"],
            "reward": [1],
        }
    )
    with pytest.raises(ValueError, match="content"):
        from_table(raw, run="r")


def test_text_content_blocks_are_preserved():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"},
            ],
        }
    ]
    out = from_table(
        pd.DataFrame(
            {"step": [1], "prompt": [messages], "response": ["a"], "reward": [1]}
        ),
        run="r",
    )
    assert out.iloc[0].prompt == "first\nsecond"
    assert out.iloc[0].prompt_messages == messages


def test_no_matching_dump_files_fails(tmp_path):
    with pytest.raises(ValueError, match="No"):
        load_trl(tmp_path, run="r")
    with pytest.raises(ValueError, match="No"):
        load_verl(tmp_path, run="r")


def test_missing_top_level_calls_fall_back_to_response_messages():
    call = {
        "id": "call",
        "type": "function",
        "function": {"name": "read", "arguments": "{}"},
    }
    raw = pd.DataFrame(
        [
            {
                "step": 1,
                "prompt": "q",
                "response": "a",
                "reward": 1,
                "tool_calls": [call],
            },
            {
                "step": 1,
                "prompt": "q",
                "response": [
                    {"role": "assistant", "content": None, "tool_calls": [call]}
                ],
                "reward": 0,
            },
        ]
    )
    out = from_table(raw, run="r")
    assert out.tool_calls.tolist() == [[call], [call]]
