#!/usr/bin/env python3
"""
Unit tests for taubench components.

Usage:
    uv run pytest tinker_cookbook/recipes/taubench/tests/test_components.py -v
"""

import json
import pytest

from tinker_cookbook.recipes.taubench.components import (
    ActionParser,
    ActionType,
    AskSonnetMode,
    AskSonnetRenderer,
    ConditioningRenderer,
    DirectInjectionRenderer,
    ExternalLLMClient,
    ExternalLLMConfig,
    MessageManager,
    ObservationType,
    ParsedAction,
    Tau2StepResult,
    get_ask_sonnet_renderer,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tokenizer():
    """Get tokenizer for testing."""
    return get_tokenizer("Qwen/Qwen3-30B-A3B-Instruct-2507")


@pytest.fixture
def renderer(tokenizer):
    """Get renderer for testing."""
    return get_renderer("qwen3", tokenizer)


@pytest.fixture
def action_parser(renderer):
    """Get action parser for testing."""
    return ActionParser(renderer)


@pytest.fixture
def message_manager():
    """Get message manager for testing."""
    return MessageManager(
        system_prompt="You are a helpful assistant.",
        external_system_prompt="You are a helpful assistant. Use <tool_call> tags.",
        initial_user_content="Hello, I need help.",
    )


# =============================================================================
# AskSonnetMode Tests
# =============================================================================


class TestAskSonnetMode:
    """Tests for AskSonnetMode enum."""

    def test_direct_injection_value(self):
        assert AskSonnetMode.DIRECT_INJECTION.value == "direct"

    def test_conditioning_value(self):
        assert AskSonnetMode.CONDITIONING.value == "conditioning"

    def test_get_renderer_direct(self):
        renderer = get_ask_sonnet_renderer(AskSonnetMode.DIRECT_INJECTION)
        assert isinstance(renderer, DirectInjectionRenderer)

    def test_get_renderer_conditioning(self):
        renderer = get_ask_sonnet_renderer(AskSonnetMode.CONDITIONING)
        assert isinstance(renderer, ConditioningRenderer)

    def test_get_renderer_invalid(self):
        with pytest.raises(ValueError, match="Unknown ask_sonnet mode"):
            get_ask_sonnet_renderer("invalid")


# =============================================================================
# DirectInjectionRenderer Tests
# =============================================================================


class TestDirectInjectionRenderer:
    """Tests for DirectInjectionRenderer."""

    @pytest.fixture
    def renderer(self):
        return DirectInjectionRenderer()

    def test_should_return_early(self, renderer):
        """Direct injection should NOT return early."""
        assert renderer.should_return_early() is False

    def test_requires_followup(self, renderer):
        """Direct injection should NOT require followup."""
        assert renderer.requires_followup() is False

    def test_format_sonnet_response_for_messages(self, renderer):
        """Sonnet's response should be added as tool result."""
        content = "I recommend checking the order status."
        msg = renderer.format_sonnet_response_for_messages(content)

        assert msg["role"] == "tool"
        assert msg["content"] == content
        assert msg["tool_call_id"] == "ask_sonnet_call"

    def test_format_sonnet_response_for_external(self, renderer):
        """Sonnet's response should be recorded as assistant message."""
        content = "I recommend checking the order status."
        msg = renderer.format_sonnet_response_for_external(content)

        assert msg["role"] == "assistant"
        assert msg["content"] == content

    def test_get_tau2_action_with_tool_call_tags(self, renderer):
        """Should extract action from <tool_call> tags."""
        sonnet_response = '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>'
        action = renderer.get_tau2_action(sonnet_response, None)

        assert action == '{"name": "get_order", "arguments": {"order_id": "123"}}'

    def test_get_tau2_action_with_raw_json(self, renderer):
        """Should handle raw JSON responses."""
        sonnet_response = '{"name": "get_user", "arguments": {"user_id": "456"}}'
        action = renderer.get_tau2_action(sonnet_response, None)

        assert action == '{"name": "get_user", "arguments": {"user_id": "456"}}'

    def test_get_tau2_action_plain_text(self, renderer):
        """Plain text should be returned as-is."""
        sonnet_response = "Hello, how can I help you today?"
        action = renderer.get_tau2_action(sonnet_response, None)

        assert action == "Hello, how can I help you today?"


# =============================================================================
# ConditioningRenderer Tests
# =============================================================================


class TestConditioningRenderer:
    """Tests for ConditioningRenderer."""

    @pytest.fixture
    def renderer(self):
        return ConditioningRenderer()

    def test_should_return_early(self, renderer):
        """Conditioning should return early."""
        assert renderer.should_return_early() is True

    def test_requires_followup(self, renderer):
        """Conditioning should require followup."""
        assert renderer.requires_followup() is True

    def test_format_sonnet_response_for_messages(self, renderer):
        """Sonnet's response should be formatted as advice."""
        content = "I suggest using the refund tool."
        msg = renderer.format_sonnet_response_for_messages(content)

        assert msg["role"] == "tool"
        assert "[Sonnet's Advice]:" in msg["content"]
        assert content in msg["content"]
        assert msg["tool_call_id"] == "ask_sonnet_call"

    def test_format_sonnet_response_for_external(self, renderer):
        """External message should indicate advice was delivered."""
        content = "I suggest using the refund tool."
        msg = renderer.format_sonnet_response_for_external(content)

        assert msg["role"] == "user"
        assert "advice was delivered" in msg["content"]

    def test_get_tau2_action_uses_followup(self, renderer):
        """Should use Qwen's followup, not Sonnet's response."""
        sonnet_response = '<tool_call>{"name": "suggest", "arguments": {}}</tool_call>'
        qwen_followup = {
            "content": '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "789"}}\n</tool_call>'
        }

        action = renderer.get_tau2_action(sonnet_response, qwen_followup)
        assert action == '{"name": "get_order", "arguments": {"order_id": "789"}}'

    def test_get_tau2_action_requires_followup(self, renderer):
        """Should raise error if no followup provided."""
        sonnet_response = "Some advice"

        with pytest.raises(ValueError, match="requires policy followup"):
            renderer.get_tau2_action(sonnet_response, None)


# =============================================================================
# ActionParser Tests
# =============================================================================


class TestActionParser:
    """Tests for ActionParser."""

    def test_parse_raw_json_tool_call(self, action_parser, tokenizer):
        """Should parse raw JSON tool calls."""
        text = '{"name": "get_user_details", "arguments": {"user_id": "123"}}'
        tokens = tokenizer.encode(text, add_special_tokens=False)

        parsed = action_parser.parse(tokens)

        assert parsed.action_type == ActionType.TOOL_CALL
        assert parsed.tool_name == "get_user_details"
        assert parsed.tool_args == {"user_id": "123"}

    def test_parse_tool_call_tags(self, action_parser, tokenizer):
        """Should parse <tool_call> tagged actions."""
        text = 'Let me help.\n<tool_call>\n{"name": "get_order", "arguments": {"order_id": "456"}}\n</tool_call>'
        tokens = tokenizer.encode(text, add_special_tokens=False)

        parsed = action_parser.parse(tokens)

        assert parsed.action_type == ActionType.TOOL_CALL
        assert parsed.tool_name == "get_order"
        assert parsed.tool_args == {"order_id": "456"}

    def test_parse_ask_sonnet(self, action_parser, tokenizer):
        """Should detect ask_sonnet calls."""
        text = '<tool_call>\n{"name": "ask_sonnet", "arguments": {}}\n</tool_call>'
        tokens = tokenizer.encode(text, add_special_tokens=False)

        parsed = action_parser.parse(tokens)

        assert parsed.action_type == ActionType.ASK_SONNET
        assert parsed.tool_name == "ask_sonnet"

    def test_parse_ask_sonnet_raw_json(self, action_parser, tokenizer):
        """Should detect ask_sonnet in raw JSON format."""
        text = '{"name": "ask_sonnet", "arguments": {}}'
        tokens = tokenizer.encode(text, add_special_tokens=False)

        parsed = action_parser.parse(tokens)

        assert parsed.action_type == ActionType.ASK_SONNET
        assert action_parser.is_ask_sonnet(parsed) is True

    def test_parse_plain_text(self, action_parser, tokenizer):
        """Should identify plain text as TEXT type."""
        text = "Hello! How can I help you today?"
        tokens = tokenizer.encode(text, add_special_tokens=False)

        parsed = action_parser.parse(tokens)

        assert parsed.action_type == ActionType.TEXT
        assert parsed.tool_name is None

    def test_is_ask_sonnet_positive(self, action_parser, tokenizer):
        """is_ask_sonnet should return True for ask_sonnet calls."""
        text = '{"name": "ask_sonnet", "arguments": {}}'
        tokens = tokenizer.encode(text, add_special_tokens=False)
        parsed = action_parser.parse(tokens)

        assert action_parser.is_ask_sonnet(parsed) is True

    def test_is_ask_sonnet_negative(self, action_parser, tokenizer):
        """is_ask_sonnet should return False for other tool calls."""
        text = '{"name": "get_order", "arguments": {}}'
        tokens = tokenizer.encode(text, add_special_tokens=False)
        parsed = action_parser.parse(tokens)

        assert action_parser.is_ask_sonnet(parsed) is False

    def test_to_tau2_action_tool_call(self, action_parser, tokenizer):
        """Should convert parsed tool call to tau2 format."""
        text = '{"name": "get_user", "arguments": {"id": "123"}}'
        tokens = tokenizer.encode(text, add_special_tokens=False)
        parsed = action_parser.parse(tokens)

        action = action_parser.to_tau2_action(parsed)
        parsed_action = json.loads(action)

        assert parsed_action["name"] == "get_user"
        assert parsed_action["arguments"] == {"id": "123"}

    def test_to_tau2_action_text(self, action_parser, tokenizer):
        """Should return plain text for TEXT actions."""
        text = "Hello, I can help you with that."
        tokens = tokenizer.encode(text, add_special_tokens=False)
        parsed = action_parser.parse(tokens)

        action = action_parser.to_tau2_action(parsed)

        assert "Hello" in action


# =============================================================================
# MessageManager Tests
# =============================================================================


class TestMessageManager:
    """Tests for MessageManager."""

    def test_init_creates_both_histories(self, message_manager):
        """Should initialize both message histories."""
        assert len(message_manager.messages) == 2  # system + user
        assert len(message_manager.external_messages) == 2

    def test_init_system_prompts(self, message_manager):
        """Should use different system prompts."""
        assert message_manager.messages[0]["content"] == "You are a helpful assistant."
        assert "tool_call" in message_manager.external_messages[0]["content"]

    def test_add_assistant_to_both(self, message_manager):
        """add_assistant should add to both histories."""
        initial_len = len(message_manager.messages)
        message_manager.add_assistant("I can help with that.", to_external=True)

        assert len(message_manager.messages) == initial_len + 1
        assert len(message_manager.external_messages) == initial_len + 1
        assert message_manager.messages[-1]["role"] == "assistant"
        assert message_manager.external_messages[-1]["role"] == "assistant"

    def test_add_assistant_to_messages_only(self, message_manager):
        """add_assistant with to_external=False should only add to messages."""
        initial_external_len = len(message_manager.external_messages)
        message_manager.add_assistant("Internal note", to_external=False)

        assert message_manager.messages[-1]["content"] == "Internal note"
        assert len(message_manager.external_messages) == initial_external_len

    def test_add_user(self, message_manager):
        """add_user should add to both histories."""
        message_manager.add_user("My order number is 12345.")

        assert message_manager.messages[-1]["role"] == "user"
        assert message_manager.messages[-1]["content"] == "My order number is 12345."
        assert message_manager.external_messages[-1]["role"] == "user"

    def test_add_user_empty_content(self, message_manager):
        """add_user should handle empty content."""
        message_manager.add_user("")

        assert message_manager.messages[-1]["content"] == "(waiting)"

    def test_add_tool_result(self, message_manager):
        """add_tool_result should format differently for each history."""
        message_manager.add_tool_result("Order found: status=shipped", tool_call_id="order_123")

        # Messages gets tool role
        assert message_manager.messages[-1]["role"] == "tool"
        assert message_manager.messages[-1]["tool_call_id"] == "order_123"

        # External gets user role with prefix
        assert message_manager.external_messages[-1]["role"] == "user"
        assert "[Tool Result]:" in message_manager.external_messages[-1]["content"]

    def test_add_tool_result_empty_content(self, message_manager):
        """add_tool_result should handle empty content."""
        message_manager.add_tool_result("")

        assert message_manager.messages[-1]["content"] == "(empty result)"

    def test_add_ask_sonnet_call(self, message_manager):
        """add_ask_sonnet_call should only add to messages."""
        initial_external_len = len(message_manager.external_messages)
        ask_sonnet_msg = {"role": "assistant", "content": '{"name": "ask_sonnet"}'}

        message_manager.add_ask_sonnet_call(ask_sonnet_msg)

        assert message_manager.messages[-1] == ask_sonnet_msg
        assert len(message_manager.external_messages) == initial_external_len

    def test_add_sonnet_response_direct(self, message_manager):
        """add_sonnet_response with DirectInjectionRenderer."""
        renderer = DirectInjectionRenderer()
        content = "I recommend the refund option."

        message_manager.add_sonnet_response(content, renderer)

        assert message_manager.messages[-1]["role"] == "tool"
        assert message_manager.external_messages[-1]["role"] == "assistant"

    def test_add_sonnet_response_conditioning(self, message_manager):
        """add_sonnet_response with ConditioningRenderer."""
        renderer = ConditioningRenderer()
        content = "I suggest using get_order first."

        message_manager.add_sonnet_response(content, renderer)

        assert "[Sonnet's Advice]:" in message_manager.messages[-1]["content"]
        assert "advice was delivered" in message_manager.external_messages[-1]["content"]

    def test_get_external_messages_for_llm(self, message_manager):
        """get_external_messages_for_llm should return simplified format."""
        message_manager.add_assistant("Test message")

        llm_messages = message_manager.get_external_messages_for_llm()

        for msg in llm_messages:
            assert "role" in msg
            assert "content" in msg
            # Should not have extra keys
            assert set(msg.keys()) == {"role", "content"}


# =============================================================================
# ExternalLLMConfig Tests
# =============================================================================


class TestExternalLLMConfig:
    """Tests for ExternalLLMConfig."""

    def test_default_values(self):
        config = ExternalLLMConfig(model="claude-sonnet-4-5-20250929")

        assert config.model == "claude-sonnet-4-5-20250929"
        assert config.temperature == 0.0
        assert config.max_tokens == 1024

    def test_custom_values(self):
        config = ExternalLLMConfig(
            model="gpt-4o",
            temperature=0.5,
            max_tokens=2048,
        )

        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


# =============================================================================
# Tau2StepResult Tests
# =============================================================================


class TestTau2StepResult:
    """Tests for Tau2StepResult dataclass."""

    def test_creation(self):
        result = Tau2StepResult(
            obs_type=ObservationType.USER_MESSAGE,
            obs_content="Hello there!",
            raw_obs="user: Hello there!",
            reward=0.5,
            terminated=False,
            truncated=False,
            info={"step": 1},
        )

        assert result.obs_type == ObservationType.USER_MESSAGE
        assert result.obs_content == "Hello there!"
        assert result.reward == 0.5
        assert result.terminated is False


class TestObservationType:
    """Tests for ObservationType enum."""

    def test_values(self):
        assert ObservationType.USER_MESSAGE.value == "user"
        assert ObservationType.TOOL_RESULT.value == "tool"
        assert ObservationType.OTHER.value == "other"
