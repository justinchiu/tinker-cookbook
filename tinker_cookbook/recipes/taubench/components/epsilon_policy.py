"""
Epsilon-greedy policy wrapper for ask_sonnet exploration during RL training.

With probability epsilon, forces an ask_sonnet tool call instead of sampling
from the base policy. This enables exploration of when ask_sonnet is helpful,
with the policy learning from reward signals.
"""

import logging
import random
from contextvars import ContextVar
from dataclasses import dataclass, field

import tinker

from tinker_cookbook.completers import (
    StopCondition,
    TinkerTokenCompleter,
    TokenCompleter,
    TokensWithLogprobs,
)
from tinker_cookbook.recipes.taubench.components.types import ExplorationMode
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Context variable for tracking rollout state in async concurrent rollouts
_rollout_ctx: ContextVar[dict | None] = ContextVar("rollout_ctx", default=None)

logger = logging.getLogger(__name__)


@dataclass
class EpsilonAskSonnetMetrics:
    """Tracks epsilon-ask-sonnet action statistics."""

    # Per-episode counters (reset each episode)
    episode_forced_ask_sonnet: int = 0
    episode_policy_ask_sonnet: int = 0
    episode_policy_other: int = 0
    episode_turns: int = 0

    # Cumulative counters (across all episodes)
    total_forced_ask_sonnet: int = 0
    total_policy_ask_sonnet: int = 0
    total_policy_other: int = 0
    total_turns: int = 0
    total_episodes: int = 0

    def record_forced_ask_sonnet(self) -> None:
        self.episode_forced_ask_sonnet += 1
        self.total_forced_ask_sonnet += 1
        self.episode_turns += 1
        self.total_turns += 1

    def record_policy_ask_sonnet(self) -> None:
        self.episode_policy_ask_sonnet += 1
        self.total_policy_ask_sonnet += 1
        self.episode_turns += 1
        self.total_turns += 1

    def record_policy_other(self) -> None:
        self.episode_policy_other += 1
        self.total_policy_other += 1
        self.episode_turns += 1
        self.total_turns += 1

    def end_episode(self) -> None:
        """Update episode count and reset per-episode counters."""
        self.total_episodes += 1
        self.episode_forced_ask_sonnet = 0
        self.episode_policy_ask_sonnet = 0
        self.episode_policy_other = 0
        self.episode_turns = 0

    def get_metrics(self) -> dict[str, float]:
        """Get metrics dict for logging."""
        total = self.total_turns or 1  # Avoid division by zero
        return {
            "epsilon_policy/forced_ask_sonnet_total": self.total_forced_ask_sonnet,
            "epsilon_policy/policy_ask_sonnet_total": self.total_policy_ask_sonnet,
            "epsilon_policy/policy_other_total": self.total_policy_other,
            "epsilon_policy/forced_ask_sonnet_rate": self.total_forced_ask_sonnet
            / total,
            "epsilon_policy/policy_ask_sonnet_rate": self.total_policy_ask_sonnet
            / total,
            "epsilon_policy/policy_other_rate": self.total_policy_other / total,
            "epsilon_policy/total_turns": self.total_turns,
            "epsilon_policy/total_episodes": self.total_episodes,
        }


def linear_decay(
    initial: float,
    final: float,
    current_step: int,
    decay_steps: int,
) -> float:
    """Linear decay from initial to final over decay_steps."""
    if current_step >= decay_steps:
        return final
    progress = current_step / decay_steps
    return initial + (final - initial) * progress


def exponential_decay(
    initial: float,
    final: float,
    current_step: int,
    decay_rate: float,
) -> float:
    """Exponential decay from initial towards final."""
    return final + (initial - final) * (decay_rate**current_step)


@dataclass
class EpsilonAskSonnetPolicy(TokenCompleter):
    """
    Epsilon-greedy policy wrapper that occasionally forces ask_sonnet calls.

    With probability epsilon, forces an ask_sonnet tool call instead of
    sampling from the model. Uses compute_logprobs to get the model's actual
    logprobs for the forced tokens, ensuring correct importance ratios.
    """

    model_name: str
    max_tokens: int
    temperature: float = 1.0

    # Epsilon parameters
    initial_epsilon: float = 0.3
    final_epsilon: float = 0.05
    decay_steps: int = 1000
    decay_type: str = "linear"
    decay_rate: float = 0.995

    # Random seed
    seed: int = 42

    # Exploration mode
    mode: ExplorationMode = ExplorationMode.EPSILON_GREEDY

    # Sampling client - set this before each rollout batch
    sampling_client: tinker.SamplingClient = field(default=None, repr=False)

    # Internal state
    _rng: random.Random = field(default=None, init=False, repr=False)
    _tokenizer: object = field(default=None, init=False, repr=False)
    _ask_sonnet_tokens: list[int] = field(default=None, init=False, repr=False)
    _current_step: int = field(default=0, init=False)
    _metrics: EpsilonAskSonnetMetrics = field(
        default_factory=EpsilonAskSonnetMetrics, init=False
    )

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._tokenizer = get_tokenizer(self.model_name)

        # Pre-compute ask_sonnet tokens
        ask_sonnet_str = (
            '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'
        )
        self._ask_sonnet_tokens = self._tokenizer.encode(
            ask_sonnet_str, add_special_tokens=False
        )

        logger.info(
            "EpsilonAskSonnetPolicy initialized: mode=%s, initial_epsilon=%.3f, "
            "final_epsilon=%.3f, decay_type=%s, decay_steps=%d",
            self.mode.value,
            self.initial_epsilon,
            self.final_epsilon,
            self.decay_type,
            self.decay_steps,
        )

    @property
    def current_epsilon(self) -> float:
        """Get current epsilon value based on decay schedule."""
        if self.decay_type == "linear":
            return linear_decay(
                self.initial_epsilon,
                self.final_epsilon,
                self._current_step,
                self.decay_steps,
            )
        elif self.decay_type == "exponential":
            return exponential_decay(
                self.initial_epsilon,
                self.final_epsilon,
                self._current_step,
                self.decay_rate,
            )
        else:
            raise ValueError(f"Unknown decay_type: {self.decay_type}")

    @property
    def metrics(self) -> EpsilonAskSonnetMetrics:
        return self._metrics

    def start_episode(self, rollout_idx: int = 0) -> None:
        """Reset turn counter. Uses ContextVar for async concurrency safety."""
        ctx = {
            "rollout_idx": rollout_idx,
            "assistant_turn_count": 0,
            "forced_on_turns": [],
            "last_action_was_ask_sonnet": False,
        }
        _rollout_ctx.set(ctx)

    def end_episode(self) -> None:
        """Update metrics at end of episode."""
        ctx = _rollout_ctx.get()
        if ctx and ctx["forced_on_turns"]:
            logger.info(
                "Episode ended: rollout=%d, forced_on_turns=%s, total_turns=%d",
                ctx["rollout_idx"],
                ctx["forced_on_turns"],
                ctx["assistant_turn_count"],
            )
        self._metrics.end_episode()

    @property
    def forced_on_turns(self) -> list[int]:
        """Get list of turns where ask_sonnet was forced this episode."""
        ctx = _rollout_ctx.get()
        return ctx["forced_on_turns"].copy() if ctx else []

    def step(self) -> None:
        """Call after each training step to update epsilon decay."""
        self._current_step += 1

    def _is_ask_sonnet_action(self, tokens: list[int]) -> bool:
        """Check if the action contains an ask_sonnet tool call."""
        try:
            text = self._tokenizer.decode(tokens)
            return "ask_sonnet" in text and "<tool_call>" in text
        except Exception:
            return False

    def _should_force(self) -> bool:
        """Determine if we should force ask_sonnet based on exploration mode."""
        ctx = _rollout_ctx.get()
        if not ctx:
            return False

        turn_count = ctx["assistant_turn_count"]
        is_first_turn = turn_count == 0


        # Never force consecutive ask_sonnet calls
        if ctx["last_action_was_ask_sonnet"]:
            return False

        if self.mode == ExplorationMode.EPSILON_GREEDY:
            return not is_first_turn and self._rng.random() < self.current_epsilon

        elif self.mode == ExplorationMode.RAO_BLACKWELL:
            rollout_idx = ctx["rollout_idx"]
            if rollout_idx == 0:
                return False  # Baseline rollout - never force
            already_forced = len(ctx["forced_on_turns"]) > 0
            if already_forced:
                return False
            return turn_count >= rollout_idx

        return False

    async def _get_logprobs_for_forced_action(
        self, observation: tinker.ModelInput
    ) -> list[float]:
        """Get the model's logprobs for the ask_sonnet tokens."""
        ob_tokens = observation.to_ints()
        full_tokens = ob_tokens + self._ask_sonnet_tokens
        full_prompt = tinker.ModelInput.from_ints(tokens=full_tokens)

        all_logprobs = await self.sampling_client.compute_logprobs_async(full_prompt)

        ask_sonnet_logprobs = all_logprobs[-len(self._ask_sonnet_tokens) :]
        return [lp if lp is not None else 0.0 for lp in ask_sonnet_logprobs]

    async def __call__(
        self, model_input: tinker.ModelInput, stop: StopCondition
    ) -> TokensWithLogprobs:
        """Sample action, potentially forcing ask_sonnet based on exploration mode."""
        if self.sampling_client is None:
            raise ValueError(
                "sampling_client must be set before calling epsilon policy"
            )

        ctx = _rollout_ctx.get()
        if ctx is None:
            raise ValueError("start_episode() must be called before sampling")

        should_force = self._should_force()

        if should_force:
            forced_turn = ctx["assistant_turn_count"]
            logger.debug(
                "Forcing ask_sonnet (mode=%s, turn=%d, rollout=%d, step=%d)",
                self.mode.value,
                forced_turn,
                ctx["rollout_idx"],
                self._current_step,
            )

            logprobs = await self._get_logprobs_for_forced_action(model_input)

            ctx["forced_on_turns"].append(forced_turn)
            ctx["assistant_turn_count"] += 1
            ctx["last_action_was_ask_sonnet"] = True
            self._metrics.record_forced_ask_sonnet()

            return TokensWithLogprobs(
                tokens=self._ask_sonnet_tokens.copy(),
                maybe_logprobs=logprobs,
            )

        # Sample from the model
        base_policy = TinkerTokenCompleter(
            self.sampling_client, self.max_tokens, self.temperature
        )
        result = await base_policy(model_input, stop)
        ctx["assistant_turn_count"] += 1

        is_ask_sonnet = self._is_ask_sonnet_action(result.tokens)
        ctx["last_action_was_ask_sonnet"] = is_ask_sonnet

        if is_ask_sonnet:
            self._metrics.record_policy_ask_sonnet()
            if ctx["assistant_turn_count"] == 1:
                logger.warning("Policy called ask_sonnet on first turn (greeting)")
        else:
            self._metrics.record_policy_other()

        return result

    def get_metrics_and_reset(self) -> dict[str, float | str]:
        """Get current metrics including epsilon value and mode."""
        metrics = self._metrics.get_metrics()
        metrics["epsilon_policy/current_epsilon"] = self.current_epsilon
        metrics["epsilon_policy/current_step"] = self._current_step
        metrics["epsilon_policy/mode"] = self.mode.value
        return metrics
