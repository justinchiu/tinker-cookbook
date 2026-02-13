"""ActionParser - Parses model outputs and detects special actions."""

import json
import logging
import re
from typing import TYPE_CHECKING

from tinker_cookbook.recipes.taubench.components.types import (
    ActionType,
    ParsedAction,
)

if TYPE_CHECKING:
    from tinker_cookbook.renderers import Renderer

logger = logging.getLogger(__name__)


class ActionParser:
    """Parses model actions and detects special cases like ask_sonnet."""

    def __init__(self, renderer: "Renderer"):
        self.renderer = renderer

    def parse(self, action_tokens: list[int]) -> ParsedAction:
        """
        Parse model output tokens into a structured action.

        Args:
            action_tokens: Token IDs from the model

        Returns:
            ParsedAction with type, content, and tool info if applicable
        """
        # Use renderer to parse response
        message, parse_success = self.renderer.parse_response(action_tokens)
        content = message.get("content", "")

        # Check for tool calls in different formats
        tool_name, tool_args = self._extract_tool_call(message)

        # Determine action type
        if tool_name == "ask_sonnet":
            action_type = ActionType.ASK_SONNET
        elif tool_name is not None:
            action_type = ActionType.TOOL_CALL
        else:
            action_type = ActionType.TEXT

        return ParsedAction(
            raw_content=content,
            action_type=action_type,
            tool_name=tool_name,
            tool_args=tool_args,
            parse_success=parse_success,
            original_message=message,
        )

    def _extract_tool_call(self, message: dict) -> tuple[str | None, dict | None]:
        """
        Extract tool call info from message in various formats.

        Returns:
            Tuple of (tool_name, tool_args) or (None, None) if no tool call
        """
        # Check 1: tool_calls field populated by renderer
        if "tool_calls" in message and message["tool_calls"]:
            tool_call = message["tool_calls"][0]
            name = tool_call.function.name
            args_str = tool_call.function.arguments
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            return name, args

        content = message.get("content", "")

        # Check 2: <tool_call> tags in content
        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL,
        )
        if tool_call_match:
            try:
                tool_json = json.loads(tool_call_match.group(1))
                name = tool_json.get("name")
                args = tool_json.get("arguments", {})
                return name, args
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call JSON from <tool_call> tags")

        # Check 3: Raw JSON at start of content
        raw_json_match = re.match(r"^\s*(\{.*\})\s*$", content, flags=re.DOTALL)
        if raw_json_match:
            try:
                tool_json = json.loads(raw_json_match.group(1))
                if "name" in tool_json:
                    name = tool_json.get("name")
                    args = tool_json.get("arguments", {})
                    return name, args
            except json.JSONDecodeError:
                pass

        return None, None

    def is_ask_sonnet(self, parsed: ParsedAction) -> bool:
        """Check if parsed action is an ask_sonnet call."""
        if parsed.action_type == ActionType.ASK_SONNET:
            return True

        # Additional check: look for "name": "ask_sonnet" pattern in content
        # This catches edge cases where extraction might have failed
        if re.search(r'"name"\s*:\s*"ask_sonnet"', parsed.raw_content):
            return True

        return False

    def to_tau2_action(self, parsed: ParsedAction) -> str:
        """
        Convert parsed action to tau2 gym action string.

        Tau2 expects either:
        - JSON format for tool calls: {"name": "tool_name", "arguments": {...}}
        - Plain text for messages
        """
        if parsed.action_type in (ActionType.TOOL_CALL, ActionType.ASK_SONNET):
            if parsed.tool_name and parsed.tool_args is not None:
                return json.dumps({"name": parsed.tool_name, "arguments": parsed.tool_args})

        # Check for <tool_call> tags in content and extract
        message = parsed.original_message
        content = message.get("content", "")

        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL,
        )
        if tool_call_match:
            return tool_call_match.group(1)

        # Plain text - strip any <tool_call> tags just in case
        return re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            content,
            flags=re.DOTALL,
        ).strip()
