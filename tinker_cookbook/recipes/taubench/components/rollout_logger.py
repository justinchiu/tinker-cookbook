"""RolloutLogger - Logs full conversation histories from tau2 rollouts."""

import hashlib
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

    Saves full message histories (policy view) to JSON files.
    Supports sampling to only log a limited number of successes and failures.
    """

    log_dir: str
    enabled: bool = True
    # Subdirectory name (e.g., "rollouts" or "eval_rollouts")
    subdir: str = "rollouts"
    # Sampling limits per iteration (0 = log all, >0 = sample)
    max_success_per_iter: int = 3  # Log up to N successful episodes per iteration
    max_failure_per_iter: int = 3  # Log up to N failed episodes per iteration
    _episode_count: int = field(default=0, init=False)
    _iter_success_count: int = field(default=0, init=False)
    _iter_failure_count: int = field(default=0, init=False)
    _current_iter: int = field(default=0, init=False)

    def __post_init__(self):
        if self.enabled and self.log_dir:
            self.rollout_path = Path(self.log_dir) / self.subdir
            self.rollout_path.mkdir(parents=True, exist_ok=True)
            if self.max_success_per_iter == 0 and self.max_failure_per_iter == 0:
                logger.info("RolloutLogger initialized at %s (logging all)", self.rollout_path)
            else:
                logger.info(
                    "RolloutLogger initialized at %s (sampling: %d success, %d failure per iter)",
                    self.rollout_path, self.max_success_per_iter, self.max_failure_per_iter
                )

    def start_iteration(self, iteration: int) -> None:
        """Call at the start of each training iteration to reset sampling counters."""
        self._current_iter = iteration
        self._iter_success_count = 0
        self._iter_failure_count = 0
        logger.debug("RolloutLogger: starting iteration %d", iteration)

    def log_episode(
        self,
        domain: str,
        task_id: str,
        reward: float,
        messages: list[dict],
        metadata: dict[str, Any] | None = None,
    ) -> str | None:
        """
        Log a completed episode (with sampling).

        Args:
            domain: The tau2 domain (e.g., "retail", "airline")
            task_id: The task ID within the domain
            reward: Final reward for the episode
            messages: Policy's message history
            metadata: Additional metadata (token costs, ask_sonnet_count, etc.)

        Returns:
            Path to the saved file, or None if logging is disabled/skipped
        """
        if not self.enabled or not self.log_dir:
            return None

        # Check sampling limits
        is_success = reward > 0.5  # Consider reward > 0.5 as success

        if is_success:
            if self.max_success_per_iter > 0 and self._iter_success_count >= self.max_success_per_iter:
                return None  # Skip - already logged enough successes this iteration
            self._iter_success_count += 1
        else:
            if self.max_failure_per_iter > 0 and self._iter_failure_count >= self.max_failure_per_iter:
                return None  # Skip - already logged enough failures this iteration
            self._iter_failure_count += 1

        self._episode_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        status = "success" if is_success else "failure"
        # Use short hash of task_id in filename (full task_id is saved in JSON content)
        task_hash = hashlib.md5(task_id.encode()).hexdigest()[:8]
        filename = f"iter{self._current_iter:04d}_{status}_{domain}_{task_hash}_{timestamp}.json"

        # Ensure directory exists (safety check)
        if not hasattr(self, 'rollout_path'):
            self.rollout_path = Path(self.log_dir) / self.subdir
        self.rollout_path.mkdir(parents=True, exist_ok=True)

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
