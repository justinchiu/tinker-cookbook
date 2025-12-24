"""RolloutLogger - Logs full conversation histories from tau2 rollouts."""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RolloutLogger:
    """
    Logger for tau2 rollout conversations.

    Saves full message histories (policy view and external LLM view) to JSON files.
    Each episode gets its own file with metadata.
    """

    log_dir: str
    enabled: bool = True
    _episode_count: int = field(default=0, init=False)

    def __post_init__(self):
        if self.enabled and self.log_dir:
            self.rollout_path = Path(self.log_dir) / "rollouts"
            self.rollout_path.mkdir(parents=True, exist_ok=True)
            logger.info("RolloutLogger initialized at %s", self.rollout_path)

    def log_episode(
        self,
        domain: str,
        task_id: str,
        reward: float,
        messages: list[dict],
        external_messages: list[dict] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Log a completed episode.

        Args:
            domain: The tau2 domain (e.g., "retail", "airline")
            task_id: The task ID within the domain
            reward: Final reward for the episode
            messages: Policy's message history
            external_messages: External LLM's message history (if applicable)
            metadata: Additional metadata (token costs, ask_sonnet_count, etc.)

        Returns:
            Path to the saved file, or None if logging is disabled
        """
        if not self.enabled or not self.log_dir:
            return None

        self._episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{domain}_{task_id}_{timestamp}.json"
        filepath = self.rollout_path / filename

        # Serialize messages - handle ToolCall objects
        def serialize_message(msg: dict) -> dict:
            result = {}
            for k, v in msg.items():
                if k == "tool_calls" and v:
                    result[k] = [
                        {"name": tc.function.name, "arguments": tc.function.arguments}
                        if hasattr(tc, "function") else tc
                        for tc in v
                    ]
                else:
                    result[k] = v
            return result

        episode_data = {
            "timestamp": datetime.now().isoformat(),
            "domain": domain,
            "task_id": task_id,
            "reward": reward,
            "messages": [serialize_message(m) for m in messages],
        }

        if external_messages:
            episode_data["external_messages"] = [serialize_message(m) for m in external_messages]

        if metadata:
            episode_data["metadata"] = metadata

        with open(filepath, "w") as f:
            json.dump(episode_data, f, indent=2)

        logger.info(
            "Logged episode %d: %s task=%s reward=%.2f -> %s",
            self._episode_count, domain, task_id, reward, filename
        )

        return str(filepath)

    def get_episode_count(self) -> int:
        """Get the number of episodes logged so far."""
        return self._episode_count

    @staticmethod
    def load_episode(filepath: str) -> dict:
        """Load a previously logged episode."""
        with open(filepath) as f:
            return json.load(f)

    def list_episodes(self) -> list[Path]:
        """List all logged episode files."""
        if not self.enabled or not self.log_dir:
            return []
        return sorted(self.rollout_path.glob("*.json"))
