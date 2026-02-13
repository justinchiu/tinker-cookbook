"""Tests for bugs identified by Codex PR review.

Each test targets a specific bug and should FAIL before the fix, PASS after.
"""

import json
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.recipes.taubench.components.ask_sonnet_renderers import (
    ConditioningRenderer,
    DirectRenderer,
)
from tinker_cookbook.recipes.taubench.components.action_parser import ActionParser
from tinker_cookbook.recipes.taubench.components.types import ActionType


# ---------------------------------------------------------------------------
# Bug #9: _format_tool_call doesn't decode dict arguments that are JSON strings
# ---------------------------------------------------------------------------


class TestFormatToolCallDictArgs:
    def test_dict_tool_call_with_string_arguments_decoded(self):
        """Dict tool_calls with JSON string arguments should be decoded to objects."""
        r = DirectRenderer()
        tc = {
            "function": {
                "name": "get_order",
                "arguments": '{"order_id": "123"}',  # JSON string, not dict
            }
        }
        result = json.loads(r._format_tool_call(tc))
        # Arguments should be a decoded dict, not a string
        assert isinstance(result["arguments"], dict)
        assert result["arguments"]["order_id"] == "123"

    def test_dict_tool_call_with_dict_arguments_unchanged(self):
        """Dict tool_calls with dict arguments should pass through unchanged."""
        r = DirectRenderer()
        tc = {
            "function": {
                "name": "get_order",
                "arguments": {"order_id": "123"},
            }
        }
        result = json.loads(r._format_tool_call(tc))
        assert isinstance(result["arguments"], dict)
        assert result["arguments"]["order_id"] == "123"

    def test_dict_tool_call_top_level_with_string_arguments(self):
        """Top-level dict format with string arguments should also decode."""
        r = DirectRenderer()
        tc = {
            "name": "cancel_order",
            "arguments": '{"reason": "changed mind"}',
        }
        result = json.loads(r._format_tool_call(tc))
        assert isinstance(result["arguments"], dict)
        assert result["arguments"]["reason"] == "changed mind"


# ---------------------------------------------------------------------------
# Bug #10: ConditioningRenderer.get_tau2_action ignores tool_calls in followup
# ---------------------------------------------------------------------------


class TestConditioningRendererToolCalls:
    def test_followup_with_tool_calls_extracts_action(self):
        """When followup has tool_calls but empty content, should extract the tool call."""
        r = ConditioningRenderer()
        followup = {
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "get_order",
                        "arguments": '{"id": "42"}',
                    }
                }
            ],
        }
        action = r.get_tau2_action(sonnet_response="advice", qwen_followup=followup)
        # Should extract the tool call, not return empty string
        assert action != ""
        parsed = json.loads(action)
        assert parsed["name"] == "get_order"

    def test_followup_with_content_still_works(self):
        """When followup has content, should still use content extraction."""
        r = ConditioningRenderer()
        followup = {"content": "Hello, I can help you with that."}
        action = r.get_tau2_action(sonnet_response="advice", qwen_followup=followup)
        assert action == "Hello, I can help you with that."


# ---------------------------------------------------------------------------
# Bug #11: _build_system_with_tools doesn't filter top-level ask_sonnet
# ---------------------------------------------------------------------------


class TestFilterTopLevelAskSonnet:
    def test_top_level_ask_sonnet_filtered(self):
        """Tools in top-level format {"name": "ask_sonnet"} should be filtered."""
        r = DirectRenderer()
        tools = [
            {"name": "get_order", "description": "Get order", "parameters": {}},
            {"name": "ask_sonnet", "description": "Delegate", "parameters": {}},
        ]
        result = r._build_system_with_tools("You are helpful.", tools)
        # ask_sonnet should not appear in tools section
        if "# Available Tools" in result:
            tools_section = result.split("# Available Tools")[1]
            assert "ask_sonnet" not in tools_section
        assert "get_order" in result

    def test_nested_ask_sonnet_still_filtered(self):
        """Standard nested format should still be filtered (regression check)."""
        r = DirectRenderer()
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Get order",
                    "parameters": {},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_sonnet",
                    "description": "Delegate",
                    "parameters": {},
                },
            },
        ]
        result = r._build_system_with_tools("You are helpful.", tools)
        if "# Available Tools" in result:
            tools_section = result.split("# Available Tools")[1]
            assert "ask_sonnet" not in tools_section
        assert "get_order" in result


# ---------------------------------------------------------------------------
# Bug #12: Forced ask_sonnet uses "args" instead of "arguments"
# ---------------------------------------------------------------------------


class TestForcedAskSonnetPayload:
    def test_forced_ask_sonnet_uses_arguments_key(self):
        """The pre-computed ask_sonnet string should use 'arguments', not 'args'."""
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import (
            EpsilonAskSonnetPolicy,
        )

        with pytest.MonkeyPatch.context() as mp:
            # Mock get_tokenizer to avoid needing real model
            mock_tokenizer = MagicMock()
            mock_tokenizer.encode.return_value = [1, 2, 3]
            mp.setattr(
                "tinker_cookbook.recipes.taubench.components.epsilon_policy.get_tokenizer",
                lambda _: mock_tokenizer,
            )

            policy = EpsilonAskSonnetPolicy(
                model_name="test", max_tokens=100
            )

        # Verify the template string uses "arguments", not "args"
        encoded_str = mock_tokenizer.encode.call_args[0][0]
        assert '"arguments"' in encoded_str, f"Expected 'arguments' key, got: {encoded_str}"
        assert '"args"' not in encoded_str, f"Should not use 'args' key, got: {encoded_str}"


# ---------------------------------------------------------------------------
# Bug #13: action_parser doesn't handle list content from renderer
# ---------------------------------------------------------------------------


class TestActionParserListContent:
    def test_parse_handles_list_content(self):
        """When renderer returns content as list (think blocks), should not crash."""
        mock_renderer = MagicMock()
        # Renderer returns content as list (e.g., when thinking parts present)
        mock_renderer.parse_response.return_value = (
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me think..."},
                    {"type": "text", "text": "I'll check order 123"},
                ],
            },
            True,
        )

        parser = ActionParser(mock_renderer)
        # Should not raise TypeError
        result = parser.parse([1, 2, 3])
        assert result.action_type == ActionType.TEXT

    def test_parse_handles_string_content_normally(self):
        """String content should still work as before (regression check)."""
        mock_renderer = MagicMock()
        mock_renderer.parse_response.return_value = (
            {
                "role": "assistant",
                "content": '<tool_call>\n{"name": "get_order", "arguments": {"id": "1"}}\n</tool_call>',
            },
            True,
        )

        parser = ActionParser(mock_renderer)
        result = parser.parse([1, 2, 3])
        assert result.action_type == ActionType.TOOL_CALL
        assert result.tool_name == "get_order"


# ---------------------------------------------------------------------------
# Bug #14: RLTestSetEvaluator doesn't accept temperature kwarg
# ---------------------------------------------------------------------------


class TestRLTestSetEvaluatorAPI:
    def test_no_temperature_parameter(self):
        """Verify RLTestSetEvaluator does NOT accept temperature, so we know
        to not pass it from env.py and eval.py."""
        import inspect

        from tinker_cookbook.rl.metric_util import RLTestSetEvaluator

        sig = inspect.signature(RLTestSetEvaluator.__init__)
        assert "temperature" not in sig.parameters, (
            "RLTestSetEvaluator should not accept temperature"
        )


# ---------------------------------------------------------------------------
# Bug #16: Tool injection uses upstream create_conversation_prefix_with_tools
# ---------------------------------------------------------------------------


class TestRendererToolsKwargCompat:
    def test_create_conversation_prefix_with_tools_exists(self):
        """Renderer.create_conversation_prefix_with_tools must exist.

        Taubench injects tools into the system prompt using this upstream method
        for both RL (env.py) and SFT (sft_dataset.py).
        """
        import inspect

        from tinker_cookbook.renderers.base import Renderer

        assert hasattr(Renderer, "create_conversation_prefix_with_tools")
        sig = inspect.signature(Renderer.create_conversation_prefix_with_tools)
        assert "tools" in sig.parameters
        assert "system_prompt" in sig.parameters

    def test_openai_tools_to_tool_specs_conversion(self):
        """_openai_tools_to_tool_specs should convert OpenAI format to ToolSpec."""
        from tinker_cookbook.recipes.taubench.env import _openai_tools_to_tool_specs

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_order",
                    "description": "Get order details",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        specs = _openai_tools_to_tool_specs(tools)
        assert len(specs) == 1
        assert specs[0]["name"] == "get_order"
        assert specs[0]["description"] == "Get order details"
