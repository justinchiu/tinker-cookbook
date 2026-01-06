"""AskSonnetRenderer hierarchy - Different modes for handling ask_sonnet interactions."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from tinker_cookbook.recipes.taubench.components.types import AskSonnetMode

if TYPE_CHECKING:
    from tinker_cookbook.recipes.taubench.env import Tau2Env
    from tinker_cookbook.rl.types import StepResult

logger = logging.getLogger(__name__)

# Pattern to match and strip ASK_SONNET_INSTRUCTION from system prompts
# This is critical: the advisor should NOT see instructions about delegating to itself
_ASK_SONNET_INSTRUCTION_PATTERN = re.compile(
    r"\n\nIMPORTANT: You have access to a special tool called `ask_sonnet`.*?"
    r"for subsequent turns if needed\.",
    re.DOTALL
)


class AskSonnetRenderer(ABC):
    """
    Abstract base class for rendering ask_sonnet interactions.

    Handles both:
    - Preparing messages for advisor API call (render_for_advisor)
    - Processing advisor's response (format_sonnet_response_for_messages, get_tau2_action)

    Different renderers handle the ask_sonnet flow differently:
    - DirectRenderer: Sonnet's response is used directly as tau2 action
    - ConditioningRenderer: Sonnet's response is advice; policy decides what to do
    """

    def render_for_advisor(
        self,
        messages: list[dict],
        tools: list[dict],
        base_system_prompt: str,
    ) -> list[dict]:
        """
        Convert messages to advisor-compatible format.

        Args:
            messages: Policy's message list
            tools: List of tool definitions (OpenAI function format)
            base_system_prompt: Base system prompt (may include ask_sonnet instructions)

        Returns:
            List of messages formatted for advisor API call
        """
        # CRITICAL: Strip ask_sonnet instructions from system prompt!
        # The advisor should NOT see instructions about delegating to itself,
        # otherwise it gets confused and returns meta-commentary like
        # "I need to delegate this to Claude Sonnet..." instead of taking action.
        clean_system_prompt = _ASK_SONNET_INSTRUCTION_PATTERN.sub("", base_system_prompt)

        result = []

        # Track indices to skip (ask_sonnet calls and their responses)
        skip_indices = set()
        for i, msg in enumerate(messages):
            # Skip ask_sonnet tool calls
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "ask_sonnet" in content:
                    skip_indices.add(i)
                    # Also skip the next message if it's the tool result
                    if i + 1 < len(messages):
                        next_msg = messages[i + 1]
                        if next_msg.get("role") == "tool" and (
                            "[Sonnet's Advice]" in next_msg.get("content", "") or
                            "Advisor Error" in next_msg.get("content", "")
                        ):
                            skip_indices.add(i + 1)

        for i, msg in enumerate(messages):
            if i in skip_indices:
                continue
            role = msg.get("role", "")

            if role == "system":
                # Build system prompt with tool descriptions (using cleaned prompt)
                result.append({
                    "role": "system",
                    "content": self._build_system_with_tools(clean_system_prompt, tools),
                })

            elif role == "tool":
                # Convert tool result to user message
                # Note: ask_sonnet responses are already filtered out by skip_indices above
                content = msg.get("content", "")
                result.append({
                    "role": "user",
                    "content": f"[Tool Result]: {content}" if content else "[Tool Result]: (empty)",
                })

            elif role == "assistant":
                # Convert tool_calls to <tool_call> text format
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls", [])

                if tool_calls:
                    parts = [content] if content else []
                    for tc in tool_calls:
                        tc_json = self._format_tool_call(tc)
                        parts.append(f"<tool_call>\n{tc_json}\n</tool_call>")
                    content = "\n".join(parts)

                result.append({
                    "role": "assistant",
                    "content": content,
                })

            elif role == "user":
                result.append({
                    "role": "user",
                    "content": msg.get("content", ""),
                })

            else:
                # Unknown role - pass through
                logger.warning("Unknown message role: %s", role)
                result.append({
                    "role": role,
                    "content": msg.get("content", ""),
                })

        return result

    def _build_system_with_tools(self, base_prompt: str, tools: list[dict]) -> str:
        """Build system prompt with tool descriptions in text."""
        # Filter out ask_sonnet tool (advisor shouldn't see it)
        filtered_tools = [t for t in tools if t.get("function", {}).get("name") != "ask_sonnet"]

        if not filtered_tools:
            return base_prompt

        tool_descriptions = []
        for tool in filtered_tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(
                f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
            )
        tools_text = "\n".join(tool_descriptions)

        return f"""{base_prompt}

# Available Tools

You have access to the following tools. To use a tool, respond with a JSON object in this exact format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

{tools_text}

# Response Format

You MUST respond with EITHER:
1. A tool call using the <tool_call> format above, OR
2. A text message to send to the user

Do NOT respond with empty content. Always provide a response."""

    def _format_tool_call(self, tc: Any) -> str:
        """Format a tool call to JSON string."""
        # Handle pydantic ToolCall objects
        if hasattr(tc, "function"):
            name = tc.function.name
            args = tc.function.arguments
            # Parse arguments if it's a string
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
        # Handle dict format
        elif isinstance(tc, dict):
            func = tc.get("function", tc)
            name = func.get("name", "unknown")
            args = func.get("arguments", {})
        else:
            return str(tc)

        return json.dumps({"name": name, "arguments": args})

    @abstractmethod
    def format_sonnet_response_for_messages(self, content: str) -> dict:
        """
        Format Sonnet's response to add to policy's messages as tool result.

        Args:
            content: Sonnet's response content

        Returns:
            Message dict to add to messages
        """
        pass

    @abstractmethod
    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        """
        Get the action string to send to tau2 gym.

        Args:
            sonnet_response: Sonnet's response content
            qwen_followup: Qwen's followup message (for conditioning mode)

        Returns:
            Action string for tau2 gym
        """
        pass

    @abstractmethod
    def should_return_early(self) -> bool:
        """
        Whether this mode should return early after ask_sonnet (wait for policy followup).

        Returns:
            True if should return early, False if should continue to tau2
        """
        pass

    @abstractmethod
    def requires_followup(self) -> bool:
        """
        Whether this mode requires a policy followup turn before tau2 action.

        Returns:
            True if requires followup, False otherwise
        """
        pass

    def _extract_action_from_content(self, content: str) -> str:
        """
        Extract tau2 action from content.

        Handles both <tool_call> wrapped JSON and raw JSON/text.

        Args:
            content: Content string

        Returns:
            Action string for tau2
        """
        # Check for <tool_call> tags
        tool_call_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            content,
            flags=re.DOTALL
        )
        if tool_call_match:
            return tool_call_match.group(1)

        # Check for raw JSON
        raw_json_match = re.match(r'^\s*(\{.*\})\s*$', content, flags=re.DOTALL)
        if raw_json_match:
            try:
                parsed = json.loads(raw_json_match.group(1))
                if "name" in parsed:
                    return raw_json_match.group(1)
            except json.JSONDecodeError:
                pass

        # Plain text - strip any <tool_call> tags just in case
        return re.sub(
            r"<tool_call>.*?</tool_call>",
            "",
            content,
            flags=re.DOTALL
        ).strip()


class ConditioningRenderer(AskSonnetRenderer):
    """
    Conditioning mode: Sonnet's response is advice; policy decides what to do.

    Flow:
    1. Policy calls ask_sonnet
    2. Sonnet responds with advice
    3. Policy sees Sonnet's advice as tool result
    4. Policy generates its own action based on the advice
    5. Policy's action is sent to tau2

    This allows the policy to learn from Sonnet's reasoning.
    """

    def format_sonnet_response_for_messages(self, content: str) -> dict:
        """Sonnet's response is added as advice (tool result)."""
        return {
            "role": "tool",
            "content": f"[Sonnet's Advice]:\n{content}",
            "tool_call_id": "ask_sonnet_call",
        }

    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        """Use policy's followup as the action, not Sonnet's response."""
        if qwen_followup is None:
            raise ValueError("Conditioning mode requires policy followup")
        content = qwen_followup.get("content", "")
        return self._extract_action_from_content(content)

    def should_return_early(self) -> bool:
        """Return early to get policy's followup."""
        return True

    def requires_followup(self) -> bool:
        """Followup required from policy."""
        return True


class DirectRenderer(ConditioningRenderer):
    """
    Direct mode: After the ask_sonnet tool, Sonnet's response is sent directly to tau2.

    Inherits from ConditioningRenderer to use the same message format
    ([Sonnet's Advice] prefix), but sends the action directly to tau2
    instead of waiting for a policy followup.

    Flow:
    1. Policy calls ask_sonnet
    2. Sonnet responds with action/message
    3. Sonnet's response is sent directly to tau2
    4. tau2 result is returned to policy
    """

    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        """Use Sonnet's response directly as the action."""
        return self._extract_action_from_content(sonnet_response)

    def should_return_early(self) -> bool:
        """Don't return early - send directly to tau2."""
        return False

    def requires_followup(self) -> bool:
        """No followup required."""
        return False


def get_ask_sonnet_renderer(mode: AskSonnetMode) -> AskSonnetRenderer:
    """
    Factory function to get the appropriate renderer for a mode.

    Args:
        mode: AskSonnetMode enum value

    Returns:
        Appropriate AskSonnetRenderer instance
    """
    if mode == AskSonnetMode.DIRECT_INJECTION:
        return DirectRenderer()
    elif mode == AskSonnetMode.CONDITIONING:
        return ConditioningRenderer()
    else:
        raise ValueError(f"Unknown ask_sonnet mode: {mode}")
