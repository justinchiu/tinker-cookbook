"""MessageManager - Manages parallel message histories for policy and external LLM."""

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tinker_cookbook.recipes.taubench.components.ask_sonnet_renderers import AskSonnetRenderer

logger = logging.getLogger(__name__)


class MessageManager:
    """
    Manages parallel message histories for the policy model and external LLM.

    Two histories are maintained:
    - messages: Policy's view (Qwen) - includes ask_sonnet as a tool
    - external_messages: External LLM's view (Sonnet) - tools in system prompt, no ask_sonnet
    """

    def __init__(
        self,
        system_prompt: str,
        external_system_prompt: str,
        initial_user_content: str,
    ):
        """
        Initialize message histories.

        Args:
            system_prompt: System prompt for policy model
            external_system_prompt: System prompt for external LLM (with tools injected)
            initial_user_content: Initial user message (or placeholder)
        """
        self.messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_user_content},
        ]

        self.external_messages: list[dict] = [
            {"role": "system", "content": external_system_prompt},
            {"role": "user", "content": initial_user_content},
        ]

    def add_assistant(self, content: str, to_external: bool = True) -> None:
        """
        Add an assistant message to histories.

        Args:
            content: Assistant message content (may include <tool_call> tags)
            to_external: Whether to also add to external_messages
        """
        msg = {"role": "assistant", "content": content}
        self.messages.append(msg)

        if to_external:
            self.external_messages.append({
                "role": "assistant",
                "content": content,
            })

        logger.debug(
            "Added assistant message (messages=%d, external=%d)",
            len(self.messages),
            len(self.external_messages) if to_external else "n/a"
        )

    def add_assistant_message_dict(self, message: dict, to_external: bool = True) -> None:
        """
        Add an assistant message dict directly to histories.

        Args:
            message: Full message dict from renderer
            to_external: Whether to also add to external_messages
        """
        self.messages.append(message)

        if to_external:
            self.external_messages.append({
                "role": "assistant",
                "content": message.get("content", ""),
            })

        logger.debug(
            "Added assistant message dict (messages=%d, external=%d)",
            len(self.messages),
            len(self.external_messages) if to_external else "n/a"
        )

    def add_user(self, content: str) -> None:
        """
        Add a user message to both histories.

        Args:
            content: User message content
        """
        # Ensure non-empty content (Anthropic requirement)
        if not content:
            content = "(waiting)"

        msg = {"role": "user", "content": content}
        self.messages.append(msg)
        self.external_messages.append(msg)

        logger.debug(
            "Added user message (messages=%d, external=%d)",
            len(self.messages),
            len(self.external_messages)
        )

    def add_tool_result(
        self,
        content: str,
        tool_call_id: str = "tool_call",
        external_format: str | None = None,
    ) -> None:
        """
        Add a tool result to histories.

        Args:
            content: Tool result content
            tool_call_id: ID of the tool call this responds to
            external_format: Optional different format for external_messages
                           If None, adds as user message with [Tool Result] prefix
        """
        # Ensure non-empty content (Anthropic requirement)
        if not content:
            content = "(empty result)"

        # For policy's view: standard tool message
        tool_msg = {
            "role": "tool",
            "content": content,
            "tool_call_id": tool_call_id,
        }
        self.messages.append(tool_msg)

        # For external LLM's view: user message with tool result
        if external_format is not None:
            ext_msg = {"role": "user", "content": external_format}
        else:
            ext_msg = {"role": "user", "content": f"[Tool Result]: {content}"}
        self.external_messages.append(ext_msg)

        logger.debug(
            "Added tool result (messages=%d, external=%d)",
            len(self.messages),
            len(self.external_messages)
        )

    def add_ask_sonnet_call(self, message: dict) -> None:
        """
        Add an ask_sonnet tool call to policy's messages.

        Args:
            message: The assistant message containing the ask_sonnet call
        """
        self.messages.append(message)
        logger.debug("Added ask_sonnet call (messages=%d)", len(self.messages))

    def add_sonnet_response(
        self,
        content: str,
        renderer: "AskSonnetRenderer",
    ) -> None:
        """
        Add Sonnet's response using the appropriate renderer.

        Args:
            content: Sonnet's response content
            renderer: AskSonnetRenderer to format the response
        """
        # Format for policy's messages
        policy_msg = renderer.format_sonnet_response_for_messages(content)
        self.messages.append(policy_msg)

        # Format for external messages
        external_msg = renderer.format_sonnet_response_for_external(content)
        self.external_messages.append(external_msg)

        logger.debug(
            "Added Sonnet response (messages=%d, external=%d)",
            len(self.messages),
            len(self.external_messages)
        )

    def get_external_messages_for_llm(self) -> list[dict]:
        """
        Get external_messages formatted for LLM API call.

        Returns simple role/content dicts.
        """
        return [
            {"role": msg["role"], "content": msg.get("content", "")}
            for msg in self.external_messages
        ]
