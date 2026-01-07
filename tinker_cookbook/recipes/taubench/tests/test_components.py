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
    DirectRenderer,
    EpsilonAskSonnetPolicy,
    ExplorationMode,
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
        assert isinstance(renderer, DirectRenderer)

    def test_get_renderer_conditioning(self):
        renderer = get_ask_sonnet_renderer(AskSonnetMode.CONDITIONING)
        assert isinstance(renderer, ConditioningRenderer)

    def test_get_renderer_invalid(self):
        with pytest.raises(ValueError, match="Unknown ask_sonnet mode"):
            get_ask_sonnet_renderer("invalid")


# =============================================================================
# DirectRenderer Tests
# =============================================================================


class TestDirectRenderer:
    """Tests for DirectRenderer."""

    @pytest.fixture
    def renderer(self):
        return DirectRenderer()

    def test_should_return_early(self, renderer):
        """Direct mode should NOT return early."""
        assert renderer.should_return_early() is False

    def test_requires_followup(self, renderer):
        """Direct mode should NOT require followup."""
        assert renderer.requires_followup() is False

    def test_format_sonnet_response_for_messages(self, renderer):
        """Sonnet's response should be added as tool result with advice prefix."""
        content = "I recommend checking the order status."
        msg = renderer.format_sonnet_response_for_messages(content)

        assert msg["role"] == "tool"
        assert "[Sonnet's Advice]:" in msg["content"]
        assert content in msg["content"]
        assert msg["tool_call_id"] == "ask_sonnet_call"

    def test_render_for_advisor(self, renderer):
        """render_for_advisor should convert messages to advisor format."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "tool", "content": "Order found", "tool_call_id": "123"},
        ]
        tools = [
            {"function": {"name": "get_order", "description": "Get order", "parameters": {}}}
        ]

        result = renderer.render_for_advisor(messages, tools, "You are helpful.")

        # System should have tools added
        assert result[0]["role"] == "system"
        assert "Available Tools" in result[0]["content"]
        assert "get_order" in result[0]["content"]

        # User stays as user
        assert result[1]["role"] == "user"
        assert result[1]["content"] == "Hello"

        # Assistant stays as assistant
        assert result[2]["role"] == "assistant"
        assert result[2]["content"] == "Hi there!"

        # Tool becomes user with prefix
        assert result[3]["role"] == "user"
        assert "[Tool Result]:" in result[3]["content"]

    def test_render_for_advisor_strips_ask_sonnet_instructions(self, renderer):
        """
        CRITICAL TEST: render_for_advisor must strip ask_sonnet instructions.

        This prevents the advisor from seeing instructions about delegating to itself,
        which causes it to return meta-commentary like "I need to delegate to Claude Sonnet..."
        instead of taking actual actions.
        """
        from tinker_cookbook.recipes.taubench.components import ASK_SONNET_INSTRUCTION

        # System prompt WITH ask_sonnet instructions (what Qwen sees)
        base_prompt_with_instructions = "You are a helpful assistant." + ASK_SONNET_INSTRUCTION

        messages = [
            {"role": "system", "content": base_prompt_with_instructions},
            {"role": "user", "content": "Hello"},
        ]
        tools = [
            {"function": {"name": "get_order", "description": "Get order", "parameters": {}}}
        ]

        result = renderer.render_for_advisor(messages, tools, base_prompt_with_instructions)

        # The advisor's system prompt should NOT contain ask_sonnet instructions
        advisor_system = result[0]["content"]
        assert "ask_sonnet" not in advisor_system, (
            "Advisor should NOT see ask_sonnet instructions! "
            "This causes the advisor to return meta-commentary instead of actions."
        )
        assert "delegate" not in advisor_system.lower(), (
            "Advisor should NOT see delegation instructions!"
        )

        # But it should still have the base prompt and tool info
        assert "You are a helpful assistant" in advisor_system
        assert "get_order" in advisor_system

    def test_render_for_advisor_removes_final_ask_sonnet(self, renderer):
        """
        CRITICAL TEST: render_for_advisor must remove the final ask_sonnet turn.

        Sonnet returns empty responses when ask_sonnet is the final turn in
        certain conversation patterns. The fix: simply remove the final ask_sonnet
        turn before sending to Sonnet.

        Previous ask_sonnet calls and [Sonnet's Advice] responses are KEPT so
        Sonnet has full context of the conversation.
        """
        # Simulate a conversation with a previous ask_sonnet call
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi, I want to cancel my order"},
            # First ask_sonnet call - KEPT in rendered messages
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
            # Sonnet's advice - KEPT in rendered messages
            {"role": "tool", "content": "[Sonnet's Advice]:\n\nFirst, authenticate the user.", "tool_call_id": "ask_sonnet_call"},
            # Policy's followup action
            {"role": "assistant", "content": '<tool_call>\n{"name": "find_user", "arguments": {"email": "test@example.com"}}\n</tool_call>'},
            {"role": "tool", "content": "user_123", "tool_call_id": "tool_call"},
            {"role": "user", "content": "My order ID is 456"},
            # Second ask_sonnet call - REMOVED (it's the final turn)
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
        ]
        tools = [{"function": {"name": "find_user", "description": "Find user", "parameters": {}}}]

        result = renderer.render_for_advisor(messages, tools, "You are helpful.")

        all_content = " ".join(msg.get("content", "") for msg in result)

        # Previous Sonnet advice IS now kept (Sonnet needs full context)
        assert "[Sonnet's Advice]" in all_content, (
            "Previous Sonnet advice should be KEPT for full context."
        )

        # The FINAL ask_sonnet call should be removed (it triggers empty responses)
        last_msg = result[-1]
        assert "ask_sonnet" not in last_msg.get("content", ""), (
            "Final ask_sonnet turn should be removed to prevent empty responses."
        )

        # The actual tool calls and results should still be present
        assert "find_user" in all_content
        assert "user_123" in all_content

    def test_render_for_advisor_keeps_previous_ask_sonnet_history(self, renderer):
        """
        render_for_advisor should keep previous ask_sonnet calls and responses.

        Sonnet needs full conversation context to give good advice.
        Only the FINAL ask_sonnet turn is removed.
        """
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            # Previous ask_sonnet that returned empty
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
            {"role": "tool", "content": "[Advisor Error]: The advisor returned an empty response.", "tool_call_id": "ask_sonnet_call"},
            # Policy continued on its own
            {"role": "assistant", "content": "How can I help you?"},
            {"role": "user", "content": "I need help"},
        ]
        tools = []

        result = renderer.render_for_advisor(messages, tools, "You are helpful.")

        all_content = " ".join(msg.get("content", "") for msg in result)

        # Previous ask_sonnet history is KEPT
        assert "ask_sonnet" in all_content, (
            "Previous ask_sonnet calls should be kept for context."
        )
        assert "[Advisor Error]" in all_content, (
            "Previous advisor errors should be kept for context."
        )

        # The conversation should still make sense
        assert "Hello" in all_content
        assert "How can I help you?" in all_content

    def test_ask_sonnet_final_turn_causes_empty_response(self, renderer):
        """
        Test if having ask_sonnet tool call in final turn causes Sonnet to return empty.

        Hypothesis: Sonnet returns empty when it sees an ask_sonnet tool call
        (rendered as assistant message) as the last message.

        Run with: pytest -v -s test_components.py::TestDirectRenderer::test_ask_sonnet_final_turn_causes_empty_response
        """
        import asyncio
        from tinker_cookbook.recipes.taubench.components import ExternalLLMClient, ExternalLLMConfig

        client = ExternalLLMClient(ExternalLLMConfig(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=1024,
        ))

        base_system = "You are a helpful customer service agent for a retail company."
        tools = [
            {"function": {"name": "find_user_id_by_name_zip", "description": "Find user ID", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "zip": {"type": "string"}}}}},
            {"function": {"name": "cancel_order", "description": "Cancel an order", "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}}}},
        ]

        # Build system prompt with tools
        system_with_tools = renderer._build_system_with_tools(base_system, tools)

        # Test 1: Messages WITHOUT ask_sonnet in final turn
        messages_no_ask_sonnet = [
            {"role": "system", "content": system_with_tools},
            {"role": "user", "content": "Hi, I want to cancel my order. My name is John Smith and my zip is 12345."},
        ]

        # Test 2: Messages WITH ask_sonnet in final turn (as assistant)
        messages_with_ask_sonnet = [
            {"role": "system", "content": system_with_tools},
            {"role": "user", "content": "Hi, I want to cancel my order. My name is John Smith and my zip is 12345."},
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "arguments": {}}\n</tool_call>'},
        ]

        async def run_tests():
            print("\n" + "="*60)
            print("TEST 1: No ask_sonnet in final turn")
            print("="*60)
            result1 = await client.call_with_usage(messages_no_ask_sonnet)
            print(f"Response length: {len(result1.content)}")
            print(f"Response empty: {len(result1.content.strip()) == 0}")
            print(f"Response: {result1.content[:300]}...")

            print("\n" + "="*60)
            print("TEST 2: ask_sonnet in final turn (as assistant)")
            print("="*60)
            result2 = await client.call_with_usage(messages_with_ask_sonnet)
            print(f"Response length: {len(result2.content)}")
            print(f"Response empty: {len(result2.content.strip()) == 0}")
            print(f"Response: {result2.content[:300]}...")

            print("\n" + "="*60)
            print("RESULT:")
            print("="*60)
            print(f"Without ask_sonnet: {len(result1.content)} chars, empty={len(result1.content.strip()) == 0}")
            print(f"With ask_sonnet:    {len(result2.content)} chars, empty={len(result2.content.strip()) == 0}")

            # The hypothesis: ask_sonnet in final turn causes empty
            if len(result2.content.strip()) == 0 and len(result1.content.strip()) > 0:
                print("\n*** CONFIRMED: ask_sonnet in final turn causes empty response! ***")
            else:
                print("\n*** NOT CONFIRMED: both responses similar ***")

        asyncio.run(run_tests())

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

    def test_direct_mode_message_flow(self, renderer):
        """
        Test the expected message flow in DIRECT mode.

        In DIRECT mode, after ask_sonnet:
        1. ask_sonnet call added (assistant)
        2. Sonnet's advice added as tool result with [Sonnet's Advice] prefix
        3. Action extracted and added as assistant message (copied from Sonnet)
        4. tau2 result added

        This ensures Qwen sees the full conversation including what was sent to tau2.
        """
        # Simulate the message flow
        mm = MessageManager(
            system_prompt="You are helpful.",
            initial_user_content="Hello",
        )

        # 1. Qwen calls ask_sonnet
        ask_sonnet_msg = {"role": "assistant", "content": '{"name": "ask_sonnet", "arguments": {}}'}
        mm.add_ask_sonnet_call(ask_sonnet_msg)

        # 2. Sonnet responds with advice
        sonnet_response = '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>'
        mm.add_sonnet_response(sonnet_response, renderer)

        # 3. Action extracted and added as assistant (this is what env.py does)
        # env.py wraps tool calls in <tool_call> tags for the message
        action_str = renderer.get_tau2_action(sonnet_response, None)
        import json
        try:
            json.loads(action_str)
            assistant_content = f"<tool_call>\n{action_str}\n</tool_call>"
        except json.JSONDecodeError:
            assistant_content = action_str
        mm.add_assistant(assistant_content)

        # 4. tau2 result would be added here (simulated)
        mm.add_tool_result('{"status": "found", "order": {...}}')

        # Verify message structure
        assert len(mm.messages) == 6  # system, user, ask_sonnet, advice, action, result
        assert mm.messages[2]["role"] == "assistant"  # ask_sonnet call
        assert mm.messages[3]["role"] == "tool"  # Sonnet's advice
        assert "[Sonnet's Advice]:" in mm.messages[3]["content"]
        assert mm.messages[4]["role"] == "assistant"  # copied action
        assert "<tool_call>" in mm.messages[4]["content"]  # has proper wrapper
        assert "get_order" in mm.messages[4]["content"]
        assert mm.messages[5]["role"] == "tool"  # tau2 result


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

    def test_render_for_advisor_removes_final_ask_sonnet(self, renderer):
        """
        ConditioningRenderer should also remove final ask_sonnet turn.

        The render_for_advisor method is inherited from AskSonnetRenderer,
        so both DirectRenderer and ConditioningRenderer should have this behavior.
        """
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi, I need help"},
            {"role": "assistant", "content": "Sure, what do you need?"},
            {"role": "user", "content": "Cancel my order"},
            # Final ask_sonnet call - should be removed
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
        ]
        tools = [{"function": {"name": "cancel_order", "description": "Cancel order", "parameters": {}}}]

        result = renderer.render_for_advisor(messages, tools, "You are helpful.")

        # Final message should NOT be the ask_sonnet call
        last_msg = result[-1]
        assert "ask_sonnet" not in last_msg.get("content", ""), (
            "Final ask_sonnet turn should be removed."
        )

        # Should end with the user message
        assert last_msg["role"] == "user"
        assert "Cancel my order" in last_msg["content"]

    def test_conditioning_mode_message_flow(self, renderer):
        """
        Test the expected message flow in CONDITIONING mode.

        In CONDITIONING mode, after ask_sonnet:
        1. ask_sonnet call added (assistant)
        2. Sonnet's advice added as tool result with [Sonnet's Advice] prefix
        3. Observation returned to Qwen (should_return_early=True)
        4. Qwen produces followup action (assistant)
        5. tau2 result added

        This ensures Qwen sees Sonnet's advice and produces its own action.
        """
        # Simulate the message flow
        mm = MessageManager(
            system_prompt="You are helpful.",
            initial_user_content="Hello",
        )

        # 1. Qwen calls ask_sonnet
        ask_sonnet_msg = {"role": "assistant", "content": '{"name": "ask_sonnet", "arguments": {}}'}
        mm.add_ask_sonnet_call(ask_sonnet_msg)

        # 2. Sonnet responds with advice
        sonnet_response = "I recommend calling get_order first.\n<tool_call>\n" \
                          '{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>'
        mm.add_sonnet_response(sonnet_response, renderer)

        # 3. should_return_early=True, so observation returned to Qwen
        assert renderer.should_return_early() is True

        # 4. Qwen produces followup action (may differ from Sonnet's suggestion)
        qwen_followup = {"role": "assistant", "content": '<tool_call>\n{"name": "get_order", "arguments": {"order_id": "123"}}\n</tool_call>'}
        mm.add_assistant_message_dict(qwen_followup)

        # 5. tau2 result added
        mm.add_tool_result('{"status": "found", "order": {...}}')

        # Verify message structure
        assert len(mm.messages) == 6  # system, user, ask_sonnet, advice, followup, result
        assert mm.messages[2]["role"] == "assistant"  # ask_sonnet call
        assert mm.messages[3]["role"] == "tool"  # Sonnet's advice
        assert "[Sonnet's Advice]:" in mm.messages[3]["content"]
        assert mm.messages[4]["role"] == "assistant"  # Qwen's followup
        assert mm.messages[5]["role"] == "tool"  # tau2 result

        # Verify get_tau2_action uses followup, not Sonnet's response
        action_str = renderer.get_tau2_action(sonnet_response, qwen_followup)
        assert "get_order" in action_str


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

    def test_init_creates_messages(self, message_manager):
        """Should initialize message history."""
        assert len(message_manager.messages) == 2  # system + user

    def test_init_system_prompt(self, message_manager):
        """Should use system prompt."""
        assert message_manager.messages[0]["content"] == "You are a helpful assistant."
        assert message_manager.system_prompt == "You are a helpful assistant."

    def test_add_assistant(self, message_manager):
        """add_assistant should add to messages."""
        initial_len = len(message_manager.messages)
        message_manager.add_assistant("I can help with that.")

        assert len(message_manager.messages) == initial_len + 1
        assert message_manager.messages[-1]["role"] == "assistant"
        assert message_manager.messages[-1]["content"] == "I can help with that."

    def test_add_user(self, message_manager):
        """add_user should add to messages."""
        message_manager.add_user("My order number is 12345.")

        assert message_manager.messages[-1]["role"] == "user"
        assert message_manager.messages[-1]["content"] == "My order number is 12345."

    def test_add_user_empty_content(self, message_manager):
        """add_user should handle empty content."""
        message_manager.add_user("")

        assert message_manager.messages[-1]["content"] == "(waiting)"

    def test_add_tool_result(self, message_manager):
        """add_tool_result should add tool message."""
        message_manager.add_tool_result("Order found: status=shipped", tool_call_id="order_123")

        assert message_manager.messages[-1]["role"] == "tool"
        assert message_manager.messages[-1]["tool_call_id"] == "order_123"
        assert message_manager.messages[-1]["content"] == "Order found: status=shipped"

    def test_add_tool_result_empty_content(self, message_manager):
        """add_tool_result should handle empty content."""
        message_manager.add_tool_result("")

        assert message_manager.messages[-1]["content"] == "(empty result)"

    def test_add_ask_sonnet_call(self, message_manager):
        """add_ask_sonnet_call should add to messages."""
        initial_len = len(message_manager.messages)
        ask_sonnet_msg = {"role": "assistant", "content": '{"name": "ask_sonnet"}'}

        message_manager.add_ask_sonnet_call(ask_sonnet_msg)

        assert message_manager.messages[-1] == ask_sonnet_msg
        assert len(message_manager.messages) == initial_len + 1

    def test_add_sonnet_response_direct(self, message_manager):
        """add_sonnet_response with DirectRenderer."""
        renderer = DirectRenderer()
        content = "I recommend the refund option."

        message_manager.add_sonnet_response(content, renderer)

        assert message_manager.messages[-1]["role"] == "tool"
        assert "[Sonnet's Advice]:" in message_manager.messages[-1]["content"]

    def test_add_sonnet_response_conditioning(self, message_manager):
        """add_sonnet_response with ConditioningRenderer."""
        renderer = ConditioningRenderer()
        content = "I suggest using get_order first."

        message_manager.add_sonnet_response(content, renderer)

        assert message_manager.messages[-1]["role"] == "tool"
        assert "[Sonnet's Advice]:" in message_manager.messages[-1]["content"]

    def test_add_assistant_message_dict(self, message_manager):
        """add_assistant_message_dict should add message dict directly."""
        msg = {"role": "assistant", "content": "Hello", "tool_calls": []}

        message_manager.add_assistant_message_dict(msg)

        assert message_manager.messages[-1] == msg


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


class TestExplorationMode:
    """Tests for ExplorationMode enum."""

    def test_epsilon_greedy_value(self):
        assert ExplorationMode.EPSILON_GREEDY.value == "epsilon"

    def test_rao_blackwell_value(self):
        assert ExplorationMode.RAO_BLACKWELL.value == "rao_blackwell"


class TestRaoBlackwellExploration:
    """Tests for Rao-Blackwell exploration mode in EpsilonAskSonnetPolicy.

    Note: These tests access the ContextVar state via _rollout_ctx to simulate
    turn progression and test forcing logic. In production, the state is managed
    automatically by start_episode() and __call__().
    """

    @pytest.fixture
    def policy_rb(self):
        """Create a policy with Rao-Blackwell mode."""
        return EpsilonAskSonnetPolicy(
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
            max_tokens=1024,
            mode=ExplorationMode.RAO_BLACKWELL,
        )

    @pytest.fixture
    def policy_epsilon(self):
        """Create a policy with epsilon-greedy mode."""
        return EpsilonAskSonnetPolicy(
            model_name="Qwen/Qwen3-30B-A3B-Instruct-2507",
            max_tokens=1024,
            mode=ExplorationMode.EPSILON_GREEDY,
            initial_epsilon=1.0,  # Always force for testing
        )

    def _get_ctx(self):
        """Get the current rollout context."""
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx
        return _rollout_ctx.get()

    def _set_turn_count(self, turn: int):
        """Set the assistant turn count in context."""
        ctx = self._get_ctx()
        if ctx:
            ctx['assistant_turn_count'] = turn

    def _set_last_ask_sonnet(self, value: bool):
        """Set the last_action_was_ask_sonnet flag in context."""
        ctx = self._get_ctx()
        if ctx:
            ctx['last_action_was_ask_sonnet'] = value

    def _append_forced_turn(self, turn: int):
        """Append a turn to forced_on_turns list in context."""
        ctx = self._get_ctx()
        if ctx:
            ctx['forced_on_turns'].append(turn)

    def test_policy_mode_initialization(self, policy_rb, policy_epsilon):
        """Test that policies are initialized with correct mode."""
        assert policy_rb.mode == ExplorationMode.RAO_BLACKWELL
        assert policy_epsilon.mode == ExplorationMode.EPSILON_GREEDY

    def test_start_episode_sets_rollout_idx(self, policy_rb):
        """Test that start_episode sets the rollout index in context."""
        policy_rb.start_episode(rollout_idx=5)
        ctx = self._get_ctx()
        assert ctx['rollout_idx'] == 5
        assert ctx['assistant_turn_count'] == 0

    def test_rb_rollout_0_never_forces(self, policy_rb):
        """Test that rollout 0 (baseline) never forces ask_sonnet."""
        policy_rb.start_episode(rollout_idx=0)
        for turn in range(10):
            self._set_turn_count(turn)
            assert policy_rb._should_force() is False, f"Rollout 0 should never force (turn {turn})"

    def test_rb_rollout_n_forces_on_turn_n(self, policy_rb):
        """Test that rollout N forces ask_sonnet on turn N only."""
        for rollout_idx in range(1, 12):
            policy_rb.start_episode(rollout_idx=rollout_idx)
            for turn in range(12):
                self._set_turn_count(turn)
                expected = (turn == rollout_idx)
                actual = policy_rb._should_force()
                assert actual == expected, (
                    f"Rollout {rollout_idx}, turn {turn}: "
                    f"expected should_force={expected}, got {actual}"
                )

    def test_rb_first_turn_never_forces(self, policy_rb):
        """Test that first turn (greeting) is never forced even for rollout 1+."""
        # Rollout 1 should force on turn 1, not turn 0
        policy_rb.start_episode(rollout_idx=1)
        self._set_turn_count(0)
        assert policy_rb._should_force() is False

        self._set_turn_count(1)
        assert policy_rb._should_force() is True

    def test_epsilon_first_turn_never_forces(self, policy_epsilon):
        """Test that epsilon-greedy never forces on first turn."""
        policy_epsilon.start_episode(rollout_idx=0)
        self._set_turn_count(0)
        # Even with epsilon=1.0, first turn should not force
        assert policy_epsilon._should_force() is False

    def test_epsilon_subsequent_turns_can_force(self, policy_epsilon):
        """Test that epsilon-greedy can force on subsequent turns."""
        policy_epsilon.start_episode(rollout_idx=0)
        self._set_turn_count(1)
        # With epsilon=1.0, should always force on non-first turns
        assert policy_epsilon._should_force() is True

    def test_metrics_include_mode(self, policy_rb):
        """Test that metrics include exploration mode."""
        metrics = policy_rb.get_metrics_and_reset()
        assert "epsilon_policy/mode" in metrics
        assert metrics["epsilon_policy/mode"] == "rao_blackwell"

    def test_forced_on_turns_tracking(self, policy_rb):
        """Test that forced_on_turns tracks which turns were forced."""
        policy_rb.start_episode(rollout_idx=3)
        assert policy_rb.forced_on_turns == []

        # Simulate forcing on turn 3
        self._set_turn_count(3)
        self._append_forced_turn(3)

        assert policy_rb.forced_on_turns == [3]

    def test_forced_on_turns_reset_on_new_episode(self, policy_rb):
        """Test that forced_on_turns resets on new episode."""
        policy_rb.start_episode(rollout_idx=3)
        self._append_forced_turn(3)
        assert policy_rb.forced_on_turns == [3]

        # Start new episode - should reset
        policy_rb.start_episode(rollout_idx=5)
        assert policy_rb.forced_on_turns == []

    def test_forced_on_turns_multiple_epsilon(self, policy_epsilon):
        """Test that epsilon mode can force multiple turns."""
        policy_epsilon.start_episode(rollout_idx=0)

        # Simulate multiple forced turns (epsilon=1.0 always forces after turn 0)
        self._set_turn_count(1)
        self._append_forced_turn(1)
        self._set_turn_count(3)
        self._append_forced_turn(3)

        assert policy_epsilon.forced_on_turns == [1, 3]

    def test_no_consecutive_ask_sonnet_forcing(self, policy_epsilon):
        """Test that consecutive ask_sonnet forcing is prevented."""
        policy_epsilon.start_episode(rollout_idx=0)
        self._set_turn_count(1)

        # First call should force (epsilon=1.0)
        assert policy_epsilon._should_force() is True

        # Simulate that ask_sonnet was taken
        self._set_last_ask_sonnet(True)
        self._set_turn_count(2)

        # Should NOT force on next turn (consecutive prevention)
        assert policy_epsilon._should_force() is False

        # After a non-ask_sonnet action, can force again
        self._set_last_ask_sonnet(False)
        self._set_turn_count(3)
        assert policy_epsilon._should_force() is True

    def test_no_consecutive_ask_sonnet_rao_blackwell(self, policy_rb):
        """Test that Rao-Blackwell also prevents consecutive ask_sonnet."""
        # Rollout 2 should force on turn 2
        policy_rb.start_episode(rollout_idx=2)
        self._set_turn_count(2)
        assert policy_rb._should_force() is True

        # But if last action was ask_sonnet, don't force
        self._set_last_ask_sonnet(True)
        assert policy_rb._should_force() is False
