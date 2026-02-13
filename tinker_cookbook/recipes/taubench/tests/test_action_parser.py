"""Tests for ActionParser — parsing model output into structured actions.

Key properties tested:
1. Three extraction formats: renderer tool_calls, <tool_call> XML tags, raw JSON
2. ask_sonnet detection by type AND content regex fallback
3. to_tau2_action produces correct JSON for tools, plain text for messages
4. Malformed JSON doesn't crash (graceful degradation to TEXT)
5. parse() delegates to renderer.parse_response correctly
"""

import json
from unittest.mock import MagicMock

import pytest

from tinker_cookbook.recipes.taubench.components.types import ActionType, ParsedAction
from tinker_cookbook.recipes.taubench.components.action_parser import ActionParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_renderer():
    """A mock renderer whose parse_response we control."""
    return MagicMock()


@pytest.fixture
def parser(mock_renderer):
    return ActionParser(renderer=mock_renderer)


def _make_tool_call_obj(name: str, arguments: str):
    """Create a mock tool call object with .function.name/.function.arguments."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


# ---------------------------------------------------------------------------
# Extraction format 1: renderer tool_calls field
# ---------------------------------------------------------------------------


class TestExtractFromRendererToolCalls:
    def test_tool_call_from_renderer(self, parser, mock_renderer):
        """When renderer returns tool_calls, extract name/args from it."""
        tc = _make_tool_call_obj("get_order", '{"order_id": "123"}')
        mock_renderer.parse_response.return_value = (
            {"content": "", "tool_calls": [tc]},
            True,
        )
        parsed = parser.parse(action_tokens=[1, 2, 3])
        assert parsed.action_type == ActionType.TOOL_CALL
        assert parsed.tool_name == "get_order"
        assert parsed.tool_args == {"order_id": "123"}
        assert parsed.parse_success is True

    def test_ask_sonnet_from_renderer(self, parser, mock_renderer):
        """ask_sonnet tool in renderer tool_calls → ASK_SONNET type."""
        tc = _make_tool_call_obj("ask_sonnet", "{}")
        mock_renderer.parse_response.return_value = (
            {"content": "", "tool_calls": [tc]},
            True,
        )
        parsed = parser.parse(action_tokens=[1, 2])
        assert parsed.action_type == ActionType.ASK_SONNET
        assert parsed.tool_name == "ask_sonnet"


# ---------------------------------------------------------------------------
# Extraction format 2: <tool_call> XML tags in content
# ---------------------------------------------------------------------------


class TestExtractFromXmlTags:
    def test_tool_call_from_xml_tags(self, parser, mock_renderer):
        content = '<tool_call>\n{"name": "cancel_order", "arguments": {"id": "99"}}\n</tool_call>'
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TOOL_CALL
        assert parsed.tool_name == "cancel_order"
        assert parsed.tool_args == {"id": "99"}

    def test_ask_sonnet_from_xml_tags(self, parser, mock_renderer):
        content = '<tool_call>\n{"name": "ask_sonnet", "arguments": {}}\n</tool_call>'
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.ASK_SONNET
        assert parsed.tool_name == "ask_sonnet"

    def test_malformed_json_in_xml_tags_falls_through(self, parser, mock_renderer):
        """Malformed JSON in <tool_call> tags → falls through to TEXT."""
        content = "<tool_call>\n{not valid json}\n</tool_call>"
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TEXT
        assert parsed.tool_name is None


# ---------------------------------------------------------------------------
# Extraction format 3: raw JSON
# ---------------------------------------------------------------------------


class TestExtractFromRawJson:
    def test_raw_json_with_name_field(self, parser, mock_renderer):
        content = '{"name": "get_user", "arguments": {"user_id": "42"}}'
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TOOL_CALL
        assert parsed.tool_name == "get_user"
        assert parsed.tool_args == {"user_id": "42"}

    def test_raw_json_without_name_field_is_text(self, parser, mock_renderer):
        """JSON without 'name' key is not a tool call."""
        content = '{"key": "value"}'
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TEXT
        assert parsed.tool_name is None

    def test_raw_json_malformed_is_text(self, parser, mock_renderer):
        """Invalid JSON that looks like it might be JSON → TEXT."""
        content = '{not json at all}'
        mock_renderer.parse_response.return_value = (
            {"content": content},
            True,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TEXT


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------


class TestPlainText:
    def test_plain_text_message(self, parser, mock_renderer):
        mock_renderer.parse_response.return_value = (
            {"content": "Hello, how can I help you?"},
            True,
        )
        parsed = parser.parse(action_tokens=[1, 2])
        assert parsed.action_type == ActionType.TEXT
        assert parsed.raw_content == "Hello, how can I help you?"
        assert parsed.tool_name is None
        assert parsed.tool_args is None

    def test_empty_content_is_text(self, parser, mock_renderer):
        mock_renderer.parse_response.return_value = (
            {"content": ""},
            False,
        )
        parsed = parser.parse(action_tokens=[1])
        assert parsed.action_type == ActionType.TEXT
        assert parsed.parse_success is False


# ---------------------------------------------------------------------------
# is_ask_sonnet — detection with regex fallback
# ---------------------------------------------------------------------------


class TestIsAskSonnet:
    def test_true_when_action_type_is_ask_sonnet(self, parser):
        parsed = ParsedAction(
            raw_content="",
            action_type=ActionType.ASK_SONNET,
            tool_name="ask_sonnet",
        )
        assert parser.is_ask_sonnet(parsed) is True

    def test_false_for_regular_tool_call(self, parser):
        parsed = ParsedAction(
            raw_content='{"name": "get_order"}',
            action_type=ActionType.TOOL_CALL,
            tool_name="get_order",
        )
        assert parser.is_ask_sonnet(parsed) is False

    def test_regex_fallback_detects_ask_sonnet_in_content(self, parser):
        """Even if extraction failed, content regex catches ask_sonnet."""
        parsed = ParsedAction(
            raw_content='Some text with "name": "ask_sonnet" in it',
            action_type=ActionType.TEXT,  # extraction failed
            tool_name=None,
        )
        assert parser.is_ask_sonnet(parsed) is True

    def test_false_for_plain_text(self, parser):
        parsed = ParsedAction(
            raw_content="Hello, how can I help you?",
            action_type=ActionType.TEXT,
        )
        assert parser.is_ask_sonnet(parsed) is False


# ---------------------------------------------------------------------------
# to_tau2_action — conversion to tau2 gym format
# ---------------------------------------------------------------------------


class TestToTau2Action:
    def test_tool_call_produces_json(self, parser):
        parsed = ParsedAction(
            raw_content="",
            action_type=ActionType.TOOL_CALL,
            tool_name="get_order",
            tool_args={"order_id": "123"},
            original_message={"content": ""},
        )
        result = parser.to_tau2_action(parsed)
        data = json.loads(result)
        assert data["name"] == "get_order"
        assert data["arguments"] == {"order_id": "123"}

    def test_ask_sonnet_produces_json(self, parser):
        parsed = ParsedAction(
            raw_content="",
            action_type=ActionType.ASK_SONNET,
            tool_name="ask_sonnet",
            tool_args={},
            original_message={"content": ""},
        )
        result = parser.to_tau2_action(parsed)
        data = json.loads(result)
        assert data["name"] == "ask_sonnet"

    def test_plain_text_returned_as_is(self, parser):
        parsed = ParsedAction(
            raw_content="I'll check that for you.",
            action_type=ActionType.TEXT,
            original_message={"content": "I'll check that for you."},
        )
        result = parser.to_tau2_action(parsed)
        assert result == "I'll check that for you."

    def test_text_with_residual_tool_call_tags_stripped(self, parser):
        """If content has <tool_call> tags but extraction failed, tags get extracted."""
        content = 'Some text <tool_call>\n{"name": "x", "arguments": {}}\n</tool_call>'
        parsed = ParsedAction(
            raw_content=content,
            action_type=ActionType.TEXT,
            tool_name=None,
            tool_args=None,
            original_message={"content": content},
        )
        result = parser.to_tau2_action(parsed)
        # Should extract the JSON from tool_call tags
        data = json.loads(result)
        assert data["name"] == "x"

    def test_tool_call_with_none_args_falls_to_content(self, parser):
        """TOOL_CALL with None tool_args should fall back to content parsing."""
        content = '<tool_call>\n{"name": "check", "arguments": {"a": 1}}\n</tool_call>'
        parsed = ParsedAction(
            raw_content=content,
            action_type=ActionType.TOOL_CALL,
            tool_name="check",
            tool_args=None,  # args extraction failed
            original_message={"content": content},
        )
        result = parser.to_tau2_action(parsed)
        data = json.loads(result)
        assert data["name"] == "check"

    def test_plain_text_strips_orphaned_tool_call_tags(self, parser):
        """Plain text with tool_call tags that contain no valid JSON → stripped."""
        content = "Hello <tool_call>broken</tool_call> world"
        parsed = ParsedAction(
            raw_content=content,
            action_type=ActionType.TEXT,
            original_message={"content": content},
        )
        result = parser.to_tau2_action(parsed)
        assert result == "Hello  world"


# ---------------------------------------------------------------------------
# parse_response delegation
# ---------------------------------------------------------------------------


class TestParseResponseDelegation:
    def test_passes_tokens_to_renderer(self, parser, mock_renderer):
        mock_renderer.parse_response.return_value = ({"content": "hi"}, True)
        parser.parse(action_tokens=[10, 20, 30])
        mock_renderer.parse_response.assert_called_once_with([10, 20, 30])

    def test_parse_success_propagated(self, parser, mock_renderer):
        mock_renderer.parse_response.return_value = ({"content": "x"}, False)
        parsed = parser.parse(action_tokens=[1])
        assert parsed.parse_success is False
