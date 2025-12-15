"""ExternalLLMClient - Handles external LLM (e.g., Sonnet) interactions."""

import logging

import litellm

from tinker_cookbook.recipes.taubench.components.types import ExternalLLMConfig

logger = logging.getLogger(__name__)


class ExternalLLMClient:
    """
    Client for calling external LLM (e.g., Claude Sonnet).

    Uses text-based tool calling (tools in system prompt) rather than API tool_use.
    """

    def __init__(self, config: ExternalLLMConfig):
        """
        Initialize the external LLM client.

        Args:
            config: Configuration for the external LLM
        """
        self.model = config.model
        self.temperature = config.temperature
        self.max_tokens = config.max_tokens

    async def call(self, messages: list[dict]) -> str:
        """
        Call the external LLM with the given messages.

        Args:
            messages: List of message dicts (role, content)

        Returns:
            Response content string
        """
        logger.info(
            "Calling external LLM (%s) with %d messages",
            self.model,
            len(messages),
        )

        try:
            response = await litellm.acompletion(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            self._log_error(messages, e)
            raise

        content = response.choices[0].message.content or ""

        logger.info(
            "External LLM (%s) response: %s",
            self.model,
            content[:100] + "..." if len(content) > 100 else content
        )

        return content

    def _log_error(self, messages: list[dict], error: Exception) -> None:
        """Log detailed error info for debugging."""
        print("\n" + "=" * 80)
        print("ERROR calling external LLM - dumping messages:")
        print("=" * 80)
        for i, msg in enumerate(messages):
            print(f"\n--- Message {i} ---")
            print(f"  role: {msg.get('role')}")
            content = msg.get('content', '')
            preview = repr(content[:200]) + "..." if len(content) > 200 else repr(content)
            print(f"  content: {preview}")
        print("=" * 80 + "\n")
        logger.error("External LLM call failed: %s", error)
