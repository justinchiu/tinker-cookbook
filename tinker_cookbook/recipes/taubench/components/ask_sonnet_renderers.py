"""AskSonnetRenderer hierarchy - Different modes for handling ask_sonnet interactions."""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from tinker_cookbook.recipes.taubench.components.types import AskSonnetMode

if TYPE_CHECKING:
    from tinker_cookbook.recipes.taubench.env import Tau2Env
    from tinker_cookbook.rl.types import StepResult

logger = logging.getLogger(__name__)


class AskSonnetRenderer(ABC):
    """
    Abstract base class for rendering ask_sonnet interactions.

    Different renderers handle the ask_sonnet flow differently:
    - DirectInjectionRenderer: Sonnet's response is used directly as tau2 action
    - ConditioningRenderer: Sonnet's response is advice; policy decides what to do
    """

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
    def format_sonnet_response_for_external(self, content: str) -> dict:
        """
        Format Sonnet's response to add to external_llm_messages.

        Args:
            content: Sonnet's response content

        Returns:
            Message dict to add to external_messages
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


class DirectInjectionRenderer(AskSonnetRenderer):
    """
    Direct injection mode: Sonnet's response is used directly as the tau2 action.

    Flow:
    1. Policy calls ask_sonnet
    2. Sonnet responds with action/message
    3. Sonnet's response is sent directly to tau2
    4. Policy doesn't see Sonnet's response during the episode
    """

    def format_sonnet_response_for_messages(self, content: str) -> dict:
        """Sonnet's response is added as tool result."""
        return {
            "role": "tool",
            "content": content,
            "tool_call_id": "ask_sonnet_call",
        }

    def format_sonnet_response_for_external(self, content: str) -> dict:
        """Sonnet's response is recorded as its own assistant message."""
        return {
            "role": "assistant",
            "content": content,
        }

    def get_tau2_action(self, sonnet_response: str, qwen_followup: dict | None) -> str:
        """Use Sonnet's response directly as the action."""
        return self._extract_action_from_content(sonnet_response)

    def should_return_early(self) -> bool:
        """Don't return early - continue to tau2."""
        return False

    def requires_followup(self) -> bool:
        """No followup required."""
        return False


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

    def format_sonnet_response_for_external(self, content: str) -> dict:
        """Record that advice was delivered."""
        return {
            "role": "user",
            "content": "[Your previous advice was delivered. Waiting for agent's next action.]",
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


def get_ask_sonnet_renderer(mode: AskSonnetMode) -> AskSonnetRenderer:
    """
    Factory function to get the appropriate renderer for a mode.

    Args:
        mode: AskSonnetMode enum value

    Returns:
        Appropriate AskSonnetRenderer instance
    """
    if mode == AskSonnetMode.DIRECT_INJECTION:
        return DirectInjectionRenderer()
    elif mode == AskSonnetMode.CONDITIONING:
        return ConditioningRenderer()
    else:
        raise ValueError(f"Unknown ask_sonnet mode: {mode}")
