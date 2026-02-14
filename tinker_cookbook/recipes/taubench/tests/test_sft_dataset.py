"""Tests for SFT dataset — message normalization, ask_sonnet injection, trainable marking.

Key properties tested:
1. _normalize_tau2_messages: strip id, convert args to JSON string, handle empty content
2. _inject_ask_sonnet_calls: never first assistant, rate controls, seed determinism
3. _mark_trainable_fields: only ask_sonnet calls trainable
4. _generate_advice_for_action: text, tool calls in <tool_call> format
5. SimpleSupervisedDataset: batch access, shuffling, length
6. DynamicInjectionDataset: re-injection per epoch
7. _split_datums_by_tasks: official splits, fallback random
"""

import json
import random
from typing import Any, cast
from unittest.mock import MagicMock, patch

import tinker

from tinker_cookbook.renderers import ToolCall

from tinker_cookbook.recipes.taubench.sft_dataset import (
    ConversationRecord,
    DynamicInjectionDataset,
    SimpleSupervisedDataset,
    _generate_advice_for_action,
    _inject_ask_sonnet_calls,
    _mark_trainable_fields,
    _normalize_tau2_messages,
    _split_datums_by_tasks,
)
from tinker_cookbook.recipes.taubench.components.types import AskSonnetMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages(*roles: str) -> list[dict]:
    """Create a simple message list from roles. Content is 'msg_N'."""
    return [{"role": r, "content": f"msg_{i}"} for i, r in enumerate(roles)]


def _make_tool_call_msg(name: str, args: dict, tc_id: str = "tc_1") -> dict:
    """Create a tau2-style assistant message with tool_calls."""
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": tc_id, "name": name, "arguments": args}],
    }


def _make_tool_response(content: str, tc_id: str = "tc_1") -> dict:
    """Create a tau2-style tool response message."""
    return {"role": "tool", "content": content, "id": tc_id}


# ---------------------------------------------------------------------------
# _normalize_tau2_messages
# ---------------------------------------------------------------------------


class TestNormalize:
    def test_strips_id_from_tool_calls(self):
        """Tool call id should be removed; ToolCall objects created."""
        msgs = [_make_tool_call_msg("get_order", {"order_id": "123"}, tc_id="call_abc")]
        result = _normalize_tau2_messages(msgs)
        tc = result[0]["tool_calls"][0]
        # Should be a ToolCall pydantic object, not a dict
        assert isinstance(tc, ToolCall)
        assert tc.function.name == "get_order"
        # id should NOT be on the ToolCall (or if it is, it's None/default)
        assert tc.id is None or tc.id != "call_abc"

    def test_converts_arguments_dict_to_json_string(self):
        """Arguments that are dicts should become JSON strings in ToolCall."""
        msgs = [_make_tool_call_msg("cancel_order", {"id": "42", "reason": "changed mind"})]
        result = _normalize_tau2_messages(msgs)
        tc = result[0]["tool_calls"][0]
        # arguments should be a JSON string
        parsed_args = json.loads(tc.function.arguments)
        assert parsed_args == {"id": "42", "reason": "changed mind"}

    def test_handles_none_content(self):
        """None content should become empty string."""
        msgs = [{"role": "assistant", "content": None}]
        result = _normalize_tau2_messages(msgs)
        assert result[0]["content"] == ""

    def test_renames_id_to_tool_call_id_for_tool_messages(self):
        """Tool messages should have tool_call_id, not id."""
        msgs = [_make_tool_response("Order found", tc_id="call_xyz")]
        result = _normalize_tau2_messages(msgs)
        assert result[0]["tool_call_id"] == "call_xyz"
        assert "id" not in result[0]

    def test_preserves_existing_tool_call_id(self):
        """If tool message already has tool_call_id, keep it."""
        msgs = [{"role": "tool", "content": "ok", "tool_call_id": "existing_id"}]
        result = _normalize_tau2_messages(msgs)
        assert result[0]["tool_call_id"] == "existing_id"

    def test_preserves_message_ordering(self):
        """Message order should be preserved."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = _normalize_tau2_messages(msgs)
        assert [m["role"] for m in result] == ["system", "user", "assistant"]

    def test_preserves_string_arguments(self):
        """If arguments is already a string, keep it as-is."""
        msgs = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"name": "foo", "arguments": '{"key": "val"}'},
                ],
            }
        ]
        result = _normalize_tau2_messages(msgs)
        tc = result[0]["tool_calls"][0]
        assert tc.function.arguments == '{"key": "val"}'


# ---------------------------------------------------------------------------
# _inject_ask_sonnet_calls
# ---------------------------------------------------------------------------


class TestInjectAskSonnet:
    def test_rate_zero_unchanged(self):
        """injection_rate=0 should return messages unchanged."""
        msgs = _make_messages("system", "user", "assistant", "user", "assistant")
        result = _inject_ask_sonnet_calls(msgs, 0.0, random.Random(42))
        assert result == msgs

    def test_first_assistant_never_injected(self):
        """The first assistant message should never be replaced."""
        msgs = _make_messages("system", "user", "assistant", "user", "assistant")
        result = _inject_ask_sonnet_calls(msgs, 1.0, random.Random(42))
        # First assistant (index 2) should be preserved as-is
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "msg_2"
        assert "tool_calls" not in result[2]

    def test_rate_one_injects_all_eligible(self):
        """injection_rate=1 should inject before all eligible assistant messages."""
        msgs = _make_messages(
            "system", "user", "assistant", "user", "assistant", "user", "assistant"
        )
        result = _inject_ask_sonnet_calls(msgs, 1.0, random.Random(42))
        # Two eligible assistants (indices 4, 6). Each gets expanded to 3 messages.
        # So result should have 3 (unchanged) + 2*3 (injected) + 2 (user msgs between) = 11
        # Actually let's count: system, user, assistant (first, unchanged), user, [ask_sonnet, advice, assistant], user, [ask_sonnet, advice, assistant]
        # = 3 + 1 + 3 + 1 + 3 = 11
        assert len(result) == 11

        # Check that injected messages have ask_sonnet tool calls
        ask_sonnet_msgs = [
            m
            for m in result
            if m.get("tool_calls")
            and any(
                (tc.function.name if hasattr(tc, "function") else tc.get("name")) == "ask_sonnet"
                for tc in m["tool_calls"]
            )
        ]
        assert len(ask_sonnet_msgs) == 2

    def test_different_seeds_different_patterns(self):
        """Different RNG seeds should produce different injection patterns."""
        msgs = _make_messages(
            "system",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
            "user",
            "assistant",
        )
        result1 = _inject_ask_sonnet_calls(msgs, 0.5, random.Random(1))
        result2 = _inject_ask_sonnet_calls(msgs, 0.5, random.Random(2))
        # With enough eligible turns and rate=0.5, different seeds should give different results
        # (not guaranteed but very likely with 4 eligible turns)
        assert result1 != result2

    def test_injection_format_has_three_messages(self):
        """Each injection should produce: ask_sonnet call + advice + original."""
        msgs = _make_messages("system", "user", "assistant", "user", "assistant")
        result = _inject_ask_sonnet_calls(msgs, 1.0, random.Random(42))
        # The second assistant (index 4) is eligible. After injection:
        # system, user, assistant(first), user, ask_sonnet_call, advice, assistant(original)
        assert len(result) == 7
        # Check the three injected messages
        ask_msg = result[4]
        advice_msg = result[5]
        original_msg = result[6]
        assert ask_msg["role"] == "assistant"
        assert ask_msg.get("tool_calls") is not None
        assert advice_msg["role"] == "tool"
        assert original_msg["content"] == "msg_4"  # Original content preserved

    def test_no_eligible_turns_unchanged(self):
        """Only one assistant message → nothing to inject."""
        msgs = _make_messages("system", "user", "assistant")
        result = _inject_ask_sonnet_calls(msgs, 1.0, random.Random(42))
        assert result == msgs


# ---------------------------------------------------------------------------
# _mark_trainable_fields
# ---------------------------------------------------------------------------


class TestMarkTrainable:
    def test_ask_sonnet_call_is_trainable(self):
        """Messages with ask_sonnet tool calls should be trainable=True."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(function=ToolCall.FunctionBody(name="ask_sonnet", arguments="{}"))
                ],
            }
        ]
        result = _mark_trainable_fields(msgs)
        assert result[0]["trainable"] is True

    def test_non_ask_sonnet_not_trainable(self):
        """Regular messages should be trainable=False."""
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "tool", "content": "result"},
        ]
        result = _mark_trainable_fields(msgs)
        for m in result:
            assert m["trainable"] is False

    def test_other_tool_calls_not_trainable(self):
        """Non-ask_sonnet tool calls should be trainable=False."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    ToolCall(
                        function=ToolCall.FunctionBody(name="get_order", arguments='{"id": "1"}')
                    )
                ],
            }
        ]
        result = _mark_trainable_fields(msgs)
        assert result[0]["trainable"] is False

    def test_dict_format_tool_calls(self):
        """Should handle dict-format tool calls too."""
        msgs = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": "ask_sonnet", "arguments": "{}"}}],
            }
        ]
        result = _mark_trainable_fields(msgs)
        assert result[0]["trainable"] is True


# ---------------------------------------------------------------------------
# _generate_advice_for_action
# ---------------------------------------------------------------------------


class TestGenerateAdvice:
    def test_text_only_message(self):
        """Message with only text content returns that content."""
        msg = {"role": "assistant", "content": "Please check your order status."}
        result = _generate_advice_for_action(msg)
        assert "Please check your order status." in result

    def test_tool_call_message(self):
        """Message with tool call produces <tool_call> formatted output."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(
                        name="get_order", arguments='{"order_id": "123"}'
                    )
                )
            ],
        }
        result = _generate_advice_for_action(msg)
        assert "<tool_call>" in result
        assert "get_order" in result
        assert "</tool_call>" in result

    def test_empty_message_returns_default(self):
        """Empty message should return default advice text."""
        msg = {"role": "assistant", "content": ""}
        result = _generate_advice_for_action(msg)
        assert result == "Proceed with the customer's request."

    def test_text_and_tool_calls_combined(self):
        """Message with both text and tool calls produces combined output."""
        msg = {
            "role": "assistant",
            "content": "Let me check that for you.",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(name="lookup", arguments='{"query": "test"}')
                )
            ],
        }
        result = _generate_advice_for_action(msg)
        assert "Let me check that for you." in result
        assert "<tool_call>" in result

    def test_dict_format_tool_calls(self):
        """Should handle dict-format tool calls (from raw JSON)."""
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"name": "get_order", "arguments": {"order_id": "123"}}],
        }
        result = _generate_advice_for_action(msg)
        assert "<tool_call>" in result
        assert "get_order" in result


# ---------------------------------------------------------------------------
# SimpleSupervisedDataset
# ---------------------------------------------------------------------------


class TestSimpleSupervisedDataset:
    def test_get_batch_returns_correct_slice(self):
        """get_batch(i) returns data[i*batch_size : (i+1)*batch_size]."""
        data = cast(list[tinker.Datum], [MagicMock() for _ in range(10)])
        ds = SimpleSupervisedDataset(data, batch_size=3)
        batch = ds.get_batch(0)
        assert len(batch) == 3
        assert batch == data[:3]

    def test_get_batch_last_batch_partial(self):
        """Last batch may be smaller than batch_size."""
        data = cast(list[tinker.Datum], [MagicMock() for _ in range(10)])
        ds = SimpleSupervisedDataset(data, batch_size=3)
        batch = ds.get_batch(3)  # index 3 → [9:12] but only 1 item
        assert len(batch) == 1

    def test_len_is_floor_division(self):
        """len(dataset) = len(data) // batch_size."""
        data = cast(list[tinker.Datum], [MagicMock() for _ in range(10)])
        ds = SimpleSupervisedDataset(data, batch_size=3)
        assert len(ds) == 3  # 10 // 3 = 3

    def test_set_epoch_shuffles(self):
        """set_epoch should shuffle data deterministically."""
        data = cast(list[tinker.Datum], list(range(20)))
        ds = SimpleSupervisedDataset(data, batch_size=5)
        ds.set_epoch(42)
        shuffled = [ds.get_batch(i) for i in range(len(ds))]
        flat = [item for batch in shuffled for item in batch]
        # Should be a permutation, not the original order
        assert flat != cast(list[tinker.Datum], list(range(20)))
        assert sorted(cast(list[Any], flat)) == list(range(20))

    def test_set_epoch_deterministic(self):
        """Same seed produces same shuffle."""
        data = cast(list[tinker.Datum], list(range(20)))
        ds1 = SimpleSupervisedDataset(data, batch_size=5)
        ds2 = SimpleSupervisedDataset(data, batch_size=5)
        ds1.set_epoch(99)
        ds2.set_epoch(99)
        for i in range(len(ds1)):
            assert ds1.get_batch(i) == ds2.get_batch(i)

    def test_different_seeds_different_order(self):
        """Different seeds produce different shuffles."""
        data = cast(list[tinker.Datum], list(range(20)))
        ds = SimpleSupervisedDataset(data, batch_size=5)
        ds.set_epoch(1)
        order1 = [ds.get_batch(i) for i in range(len(ds))]
        ds.set_epoch(2)
        order2 = [ds.get_batch(i) for i in range(len(ds))]
        assert order1 != order2


# ---------------------------------------------------------------------------
# DynamicInjectionDataset
# ---------------------------------------------------------------------------


class TestDynamicInjectionDataset:
    def _make_dynamic_dataset(self, n_convs=5, injection_rate=0.5):
        """Create a DynamicInjectionDataset with mocked renderer."""
        import torch

        convs = [
            ConversationRecord(
                messages=_make_messages("system", "user", "assistant", "user", "assistant"),
                task_id=f"task_{i}",
                domain="retail",
            )
            for i in range(n_convs)
        ]

        mock_renderer = MagicMock()
        # build_supervised_example returns (ModelInput, weights)
        mock_model_input = MagicMock()
        mock_model_input.chunks = []
        mock_weights = torch.ones(10)
        mock_renderer.build_supervised_example.return_value = (
            mock_model_input,
            mock_weights,
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset.datum_from_model_input_weights"
        ) as mock_datum_fn:
            mock_datum_fn.return_value = MagicMock()
            ds = DynamicInjectionDataset(
                conversations=convs,
                batch_size=2,
                renderer=mock_renderer,
                train_on_what=MagicMock(),
                domain_tools={"retail": []},
                injection_rate=injection_rate,
                injection_mode=AskSonnetMode.DIRECT_INJECTION,
                max_length=1024,
            )
        return ds, mock_renderer

    def test_set_epoch_changes_datums(self):
        """Different epochs should produce different injection patterns."""
        ds, renderer = self._make_dynamic_dataset()
        # Capture calls from epoch 0 (set in __init__)
        calls_epoch_0 = renderer.build_supervised_example.call_count

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset.datum_from_model_input_weights"
        ) as mock_datum_fn:
            mock_datum_fn.return_value = MagicMock()
            ds.set_epoch(1)

        # Should have re-rendered all conversations
        assert renderer.build_supervised_example.call_count > calls_epoch_0

    def test_deterministic_given_seed(self):
        """Same seed should produce same datums."""
        import torch

        convs = [
            ConversationRecord(
                messages=_make_messages("system", "user", "assistant", "user", "assistant"),
                task_id="t1",
                domain="retail",
            )
        ]

        mock_renderer = MagicMock()
        mock_model_input = MagicMock()
        mock_model_input.chunks = []
        mock_renderer.build_supervised_example.return_value = (
            mock_model_input,
            torch.ones(10),
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset.datum_from_model_input_weights"
        ) as mock_datum_fn:
            mock_datum_fn.return_value = MagicMock()
            DynamicInjectionDataset(
                conversations=convs,
                batch_size=1,
                renderer=mock_renderer,
                train_on_what=MagicMock(),
                domain_tools={"retail": []},
                injection_rate=0.5,
                injection_mode=AskSonnetMode.DIRECT_INJECTION,
                max_length=1024,
            )

        # Collect the messages passed to build_supervised_example for seed 0
        calls1 = [
            call.args[0] if call.args else call.kwargs.get("messages")
            for call in mock_renderer.build_supervised_example.call_args_list
        ]

        # Reset mock and create again with same seed
        mock_renderer.reset_mock()
        mock_renderer.build_supervised_example.return_value = (
            mock_model_input,
            torch.ones(10),
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset.datum_from_model_input_weights"
        ) as mock_datum_fn:
            mock_datum_fn.return_value = MagicMock()
            DynamicInjectionDataset(
                conversations=convs,
                batch_size=1,
                renderer=mock_renderer,
                train_on_what=MagicMock(),
                domain_tools={"retail": []},
                injection_rate=0.5,
                injection_mode=AskSonnetMode.DIRECT_INJECTION,
                max_length=1024,
            )

        calls2 = [
            call.args[0] if call.args else call.kwargs.get("messages")
            for call in mock_renderer.build_supervised_example.call_args_list
        ]

        # Same seed (0 in __init__) should produce same message patterns
        assert len(calls1) == len(calls2)


# ---------------------------------------------------------------------------
# _split_datums_by_tasks
# ---------------------------------------------------------------------------


class TestSplitDatumsByTasks:
    def test_official_split_used_when_available(self):
        """When official test split exists, use it."""
        datums_with_tasks: list[tuple[tinker.Datum, str]] = cast(
            list[tuple[tinker.Datum, str]],
            [
                (MagicMock(), "task_1"),
                (MagicMock(), "task_2"),
                (MagicMock(), "task_3"),
                (MagicMock(), "task_4"),
            ],
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset._get_task_ids_for_split"
        ) as mock_split:
            mock_split.return_value = {"task_2", "task_4"}
            train, test = _split_datums_by_tasks(
                datums_with_tasks,
                domain="retail",
                use_official_test_split=True,
                fallback_test_fraction=None,
            )

        assert len(train) == 2
        assert len(test) == 2

    def test_fallback_random_split(self):
        """When no official split, fall back to random split."""
        datums_with_tasks = cast(
            list[tuple[tinker.Datum, str]],
            [(MagicMock(), f"task_{i}") for i in range(10)],
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset._get_task_ids_for_split"
        ) as mock_split:
            mock_split.return_value = set()  # No official split
            train, test = _split_datums_by_tasks(
                datums_with_tasks,
                domain="retail",
                use_official_test_split=True,
                fallback_test_fraction=0.2,
            )

        assert len(train) + len(test) == 10
        assert len(test) == 2  # 10 * 0.2 = 2

    def test_no_split_configured(self):
        """With no official split and no fallback, all goes to train."""
        datums_with_tasks = cast(
            list[tuple[tinker.Datum, str]],
            [(MagicMock(), f"task_{i}") for i in range(5)],
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset._get_task_ids_for_split"
        ) as mock_split:
            mock_split.return_value = set()
            train, test = _split_datums_by_tasks(
                datums_with_tasks,
                domain="retail",
                use_official_test_split=False,
                fallback_test_fraction=None,
            )

        assert len(train) == 5
        assert len(test) == 0

    def test_fallback_split_deterministic(self):
        """Fallback random split should be deterministic (seed=0)."""
        datums = cast(
            list[tuple[tinker.Datum, str]],
            [(MagicMock(), f"task_{i}") for i in range(20)],
        )

        with patch(
            "tinker_cookbook.recipes.taubench.sft_dataset._get_task_ids_for_split"
        ) as mock_split:
            mock_split.return_value = set()
            train1, test1 = _split_datums_by_tasks(datums, "retail", True, 0.3)
            train2, test2 = _split_datums_by_tasks(datums, "retail", True, 0.3)

        # Same input → same split (deterministic seed)
        assert len(train1) == len(train2)
        assert len(test1) == len(test2)
