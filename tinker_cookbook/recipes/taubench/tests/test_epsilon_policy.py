"""Tests for EpsilonAskSonnetPolicy — exploration strategies for ask_sonnet.

Key properties tested:
1. Decay functions: linear and exponential produce correct values at boundaries
2. Metrics: counters increment correctly, end_episode resets per-episode counters
3. Epsilon-greedy: first turn never forced, epsilon=1.0 forces all eligible turns,
   consecutive prevention
4. Rao-Blackwell: rollout 0 never forces, rollout N forces on turn N, only once
5. Episode lifecycle: start/end via ContextVar, forced_on_turns tracking
6. __call__: forces ask_sonnet tokens with real logprobs, delegates to base policy
"""

import random
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.taubench.components.types import ExplorationMode
from tinker_cookbook.recipes.taubench.components.epsilon_policy import (
    EpsilonAskSonnetMetrics,
    EpsilonAskSonnetPolicy,
    linear_decay,
    exponential_decay,
)


# ---------------------------------------------------------------------------
# Decay functions — pure math
# ---------------------------------------------------------------------------


class TestLinearDecay:
    def test_at_start(self):
        assert linear_decay(initial=1.0, final=0.0, current_step=0, decay_steps=100) == 1.0

    def test_at_end(self):
        assert linear_decay(initial=1.0, final=0.0, current_step=100, decay_steps=100) == 0.0

    def test_midpoint(self):
        result = linear_decay(initial=1.0, final=0.0, current_step=50, decay_steps=100)
        assert abs(result - 0.5) < 1e-9

    def test_past_end_clamps_to_final(self):
        result = linear_decay(initial=1.0, final=0.0, current_step=200, decay_steps=100)
        assert result == 0.0

    def test_increasing_schedule(self):
        """Linear decay also works for increasing schedules (final > initial)."""
        result = linear_decay(initial=0.0, final=1.0, current_step=50, decay_steps=100)
        assert abs(result - 0.5) < 1e-9

    def test_quarter_point(self):
        result = linear_decay(initial=0.3, final=0.05, current_step=25, decay_steps=100)
        expected = 0.3 + (0.05 - 0.3) * 0.25  # 0.3 - 0.0625 = 0.2375
        assert abs(result - expected) < 1e-9


class TestExponentialDecay:
    def test_at_start(self):
        result = exponential_decay(initial=1.0, final=0.0, current_step=0, decay_rate=0.5)
        assert abs(result - 1.0) < 1e-9

    def test_after_one_step(self):
        result = exponential_decay(initial=1.0, final=0.0, current_step=1, decay_rate=0.5)
        assert abs(result - 0.5) < 1e-9

    def test_approaches_final(self):
        """After many steps, should approach final value."""
        result = exponential_decay(initial=1.0, final=0.0, current_step=100, decay_rate=0.5)
        assert result < 1e-20  # Very close to 0

    def test_with_nonzero_final(self):
        result = exponential_decay(initial=1.0, final=0.1, current_step=1, decay_rate=0.5)
        # final + (initial - final) * rate^step = 0.1 + 0.9 * 0.5 = 0.55
        assert abs(result - 0.55) < 1e-9


# ---------------------------------------------------------------------------
# Metrics tracking
# ---------------------------------------------------------------------------


class TestEpsilonAskSonnetMetrics:
    def test_initial_state(self):
        m = EpsilonAskSonnetMetrics()
        assert m.episode_forced_ask_sonnet == 0
        assert m.total_forced_ask_sonnet == 0
        assert m.total_episodes == 0

    def test_record_forced(self):
        m = EpsilonAskSonnetMetrics()
        m.record_forced_ask_sonnet()
        assert m.episode_forced_ask_sonnet == 1
        assert m.total_forced_ask_sonnet == 1
        assert m.episode_turns == 1
        assert m.total_turns == 1

    def test_record_policy_ask_sonnet(self):
        m = EpsilonAskSonnetMetrics()
        m.record_policy_ask_sonnet()
        assert m.episode_policy_ask_sonnet == 1
        assert m.total_policy_ask_sonnet == 1

    def test_record_policy_other(self):
        m = EpsilonAskSonnetMetrics()
        m.record_policy_other()
        assert m.episode_policy_other == 1
        assert m.total_policy_other == 1

    def test_end_episode_resets_per_episode_counters(self):
        m = EpsilonAskSonnetMetrics()
        m.record_forced_ask_sonnet()
        m.record_policy_other()
        m.end_episode()

        # Per-episode counters reset
        assert m.episode_forced_ask_sonnet == 0
        assert m.episode_policy_ask_sonnet == 0
        assert m.episode_policy_other == 0
        assert m.episode_turns == 0

        # Totals preserved
        assert m.total_forced_ask_sonnet == 1
        assert m.total_policy_other == 1
        assert m.total_turns == 2
        assert m.total_episodes == 1

    def test_get_metrics_dict(self):
        m = EpsilonAskSonnetMetrics()
        m.record_forced_ask_sonnet()
        m.record_policy_other()
        metrics = m.get_metrics()

        assert "epsilon_policy/forced_ask_sonnet_total" in metrics
        assert metrics["epsilon_policy/forced_ask_sonnet_total"] == 1
        assert metrics["epsilon_policy/total_turns"] == 2
        rate = metrics["epsilon_policy/forced_ask_sonnet_rate"]
        assert isinstance(rate, float)
        assert abs(rate - 0.5) < 1e-9

    def test_get_metrics_no_division_by_zero(self):
        m = EpsilonAskSonnetMetrics()
        metrics = m.get_metrics()
        # Should not crash even with 0 turns
        assert metrics["epsilon_policy/forced_ask_sonnet_rate"] == 0.0


# ---------------------------------------------------------------------------
# EpsilonAskSonnetPolicy — construction and epsilon property
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer that returns predictable tokens."""
    tokenizer = MagicMock()
    # ask_sonnet_str = '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'
    tokenizer.encode.return_value = [100, 200, 300, 400, 500]
    tokenizer.decode.return_value = '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'
    return tokenizer


@pytest.fixture
def policy(mock_tokenizer):
    """Create policy with mocked tokenizer."""
    with patch(
        "tinker_cookbook.recipes.taubench.components.epsilon_policy.get_tokenizer",
        return_value=mock_tokenizer,
    ):
        return EpsilonAskSonnetPolicy(
            model_name="test-model",
            max_tokens=1024,
            temperature=0.7,
            initial_epsilon=0.3,
            final_epsilon=0.05,
            decay_steps=100,
            decay_type="linear",
            seed=42,
            mode=ExplorationMode.EPSILON_GREEDY,
        )


class TestPolicyConstruction:
    def test_initial_epsilon(self, policy):
        assert abs(policy.current_epsilon - 0.3) < 1e-9

    def test_epsilon_after_steps(self, policy):
        for _ in range(50):
            policy.step()
        expected = 0.3 + (0.05 - 0.3) * 0.5  # 0.175
        assert abs(policy.current_epsilon - expected) < 1e-9

    def test_epsilon_at_end(self, policy):
        for _ in range(100):
            policy.step()
        assert abs(policy.current_epsilon - 0.05) < 1e-9

    def test_ask_sonnet_tokens_precomputed(self, policy):
        assert policy._ask_sonnet_tokens == [100, 200, 300, 400, 500]


# ---------------------------------------------------------------------------
# _should_force — epsilon-greedy mode
# ---------------------------------------------------------------------------


class TestShouldForceEpsilonGreedy:
    def test_no_context_returns_false(self, policy):
        """Without start_episode, should_force returns False."""
        assert policy._should_force() is False

    def test_first_turn_never_forced(self, policy):
        """Turn 0 (greeting) should never be forced, even with epsilon=1.0."""
        policy._rng = random.Random(0)  # Deterministic
        # Temporarily set epsilon very high
        policy.initial_epsilon = 1.0
        policy.final_epsilon = 1.0
        policy.start_episode(rollout_idx=0)
        # Turn count is 0 → should never force
        assert policy._should_force() is False

    def test_epsilon_1_forces_non_first_turn(self, policy):
        """With epsilon=1.0, non-first turns should always be forced."""
        policy.initial_epsilon = 1.0
        policy.final_epsilon = 1.0
        policy.start_episode(rollout_idx=0)
        # Simulate having passed turn 0
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 1
        ctx["last_action_was_ask_sonnet"] = False
        assert policy._should_force() is True

    def test_epsilon_0_never_forces(self, policy):
        """With epsilon=0.0, should never force."""
        policy.initial_epsilon = 0.0
        policy.final_epsilon = 0.0
        policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 5
        ctx["last_action_was_ask_sonnet"] = False
        assert policy._should_force() is False

    def test_consecutive_prevention(self, policy):
        """Should not force if last action was already ask_sonnet."""
        policy.initial_epsilon = 1.0
        policy.final_epsilon = 1.0
        policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 2
        ctx["last_action_was_ask_sonnet"] = True  # Last was ask_sonnet
        assert policy._should_force() is False


# ---------------------------------------------------------------------------
# _should_force — Rao-Blackwell mode
# ---------------------------------------------------------------------------


class TestShouldForceRaoBlackwell:
    @pytest.fixture
    def rb_policy(self, mock_tokenizer):
        with patch(
            "tinker_cookbook.recipes.taubench.components.epsilon_policy.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            return EpsilonAskSonnetPolicy(
                model_name="test-model",
                max_tokens=1024,
                temperature=0.7,
                mode=ExplorationMode.RAO_BLACKWELL,
            )

    def test_rollout_0_never_forces(self, rb_policy):
        """Rollout 0 is the baseline — never force."""
        rb_policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        for turn in range(10):
            ctx["assistant_turn_count"] = turn
            ctx["last_action_was_ask_sonnet"] = False
            assert rb_policy._should_force() is False

    def test_rollout_n_forces_on_turn_n(self, rb_policy):
        """Rollout N should force on turn N."""
        rb_policy.start_episode(rollout_idx=3)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        # Turns before target
        for turn in range(3):
            ctx["assistant_turn_count"] = turn
            ctx["last_action_was_ask_sonnet"] = False
            assert rb_policy._should_force() is False, f"Should not force on turn {turn}"

        # Target turn
        ctx["assistant_turn_count"] = 3
        ctx["last_action_was_ask_sonnet"] = False
        assert rb_policy._should_force() is True

    def test_rollout_forces_only_once(self, rb_policy):
        """After forcing once, should not force again."""
        rb_policy.start_episode(rollout_idx=1)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 1
        ctx["last_action_was_ask_sonnet"] = False
        assert rb_policy._should_force() is True

        # Simulate having forced
        ctx["forced_on_turns"].append(1)
        ctx["assistant_turn_count"] = 2
        ctx["last_action_was_ask_sonnet"] = False
        assert rb_policy._should_force() is False

    def test_postpones_if_consecutive(self, rb_policy):
        """If target turn would be consecutive, postpone."""
        rb_policy.start_episode(rollout_idx=2)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 2
        ctx["last_action_was_ask_sonnet"] = True  # Previous was ask_sonnet
        assert rb_policy._should_force() is False

        # Next turn should force (postponed)
        ctx["assistant_turn_count"] = 3
        ctx["last_action_was_ask_sonnet"] = False
        assert rb_policy._should_force() is True


# ---------------------------------------------------------------------------
# Episode lifecycle
# ---------------------------------------------------------------------------


class TestEpisodeLifecycle:
    def test_start_episode_sets_context(self, policy):
        policy.start_episode(rollout_idx=5)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        assert ctx["rollout_idx"] == 5
        assert ctx["assistant_turn_count"] == 0
        assert ctx["forced_on_turns"] == []
        assert ctx["last_action_was_ask_sonnet"] is False

    def test_forced_on_turns_returns_copy(self, policy):
        policy.start_episode(rollout_idx=0)
        turns = policy.forced_on_turns
        assert turns == []
        # Modifying returned list shouldn't affect internal state
        turns.append(99)
        assert policy.forced_on_turns == []

    def test_end_episode_updates_metrics(self, policy):
        policy.start_episode(rollout_idx=0)
        policy.end_episode()
        assert policy.metrics.total_episodes == 1

    def test_get_metrics_and_reset(self, policy):
        policy.start_episode(rollout_idx=0)
        policy.end_episode()
        metrics = policy.get_metrics_and_reset()
        assert "epsilon_policy/current_epsilon" in metrics
        assert "epsilon_policy/mode" in metrics
        assert metrics["epsilon_policy/mode"] == "epsilon"
        assert metrics["epsilon_policy/total_episodes"] == 1


# ---------------------------------------------------------------------------
# __call__ — forcing and delegation
# ---------------------------------------------------------------------------


class TestCall:
    @pytest.mark.asyncio
    async def test_forces_ask_sonnet_tokens(self, policy):
        """When _should_force returns True, should return ask_sonnet tokens with logprobs."""
        mock_sampling_client = MagicMock()
        mock_sampling_client.compute_logprobs_async = AsyncMock(
            return_value=[0.0] * 10 + [-0.1, -0.2, -0.3, -0.4, -0.5]
        )
        policy.sampling_client = mock_sampling_client

        # Force epsilon=1.0 so it always forces (except first turn)
        policy.initial_epsilon = 1.0
        policy.final_epsilon = 1.0

        policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 1  # Past first turn

        mock_input = MagicMock()
        mock_input.to_ints.return_value = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        result = await policy(mock_input, stop=["<|end|>"])

        assert result.tokens == [100, 200, 300, 400, 500]
        assert result.maybe_logprobs == [-0.1, -0.2, -0.3, -0.4, -0.5]
        assert policy.metrics.episode_forced_ask_sonnet == 1

    @pytest.mark.asyncio
    async def test_delegates_to_base_policy_when_not_forcing(self, policy):
        """When not forcing, should delegate to TinkerTokenCompleter."""
        mock_sampling_client = MagicMock()
        policy.sampling_client = mock_sampling_client

        # Set epsilon=0 so it never forces
        policy.initial_epsilon = 0.0
        policy.final_epsilon = 0.0

        policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 1

        mock_input = MagicMock()

        # Mock TinkerTokenCompleter
        from tinker_cookbook.completers import TokensWithLogprobs

        mock_result = TokensWithLogprobs(tokens=[10, 20, 30], maybe_logprobs=[-1.0, -2.0, -3.0])

        with patch(
            "tinker_cookbook.recipes.taubench.components.epsilon_policy.TinkerTokenCompleter"
        ) as MockCompleter:
            mock_completer_instance = AsyncMock(return_value=mock_result)
            MockCompleter.return_value = mock_completer_instance

            # Mock tokenizer.decode to say it's not ask_sonnet
            policy._tokenizer.decode.return_value = "some non-ask_sonnet text"

            result = await policy(mock_input, stop=["<|end|>"])

        assert result.tokens == [10, 20, 30]
        assert policy.metrics.episode_policy_other == 1

    @pytest.mark.asyncio
    async def test_raises_without_sampling_client(self, policy):
        policy.start_episode(rollout_idx=0)
        mock_input = MagicMock()
        with pytest.raises(ValueError, match="sampling_client must be set"):
            await policy(mock_input, stop=["<|end|>"])

    @pytest.mark.asyncio
    async def test_raises_without_start_episode(self, policy):
        policy.sampling_client = MagicMock()
        # Reset context to None
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        _rollout_ctx.set(None)

        mock_input = MagicMock()
        with pytest.raises(ValueError, match="start_episode"):
            await policy(mock_input, stop=["<|end|>"])

    @pytest.mark.asyncio
    async def test_tracks_policy_ask_sonnet(self, policy):
        """When policy voluntarily calls ask_sonnet, it's tracked as policy_ask_sonnet."""
        mock_sampling_client = MagicMock()
        policy.sampling_client = mock_sampling_client

        policy.initial_epsilon = 0.0
        policy.final_epsilon = 0.0

        policy.start_episode(rollout_idx=0)
        from tinker_cookbook.recipes.taubench.components.epsilon_policy import _rollout_ctx

        ctx = _rollout_ctx.get()
        assert ctx is not None
        ctx["assistant_turn_count"] = 1

        mock_input = MagicMock()

        from tinker_cookbook.completers import TokensWithLogprobs

        mock_result = TokensWithLogprobs(tokens=[10, 20], maybe_logprobs=[-1.0, -2.0])

        with patch(
            "tinker_cookbook.recipes.taubench.components.epsilon_policy.TinkerTokenCompleter"
        ) as MockCompleter:
            mock_completer_instance = AsyncMock(return_value=mock_result)
            MockCompleter.return_value = mock_completer_instance

            # Tokenizer says it IS ask_sonnet
            policy._tokenizer.decode.return_value = (
                '<tool_call>\n{"name": "ask_sonnet"}\n</tool_call>'
            )

            await policy(mock_input, stop=["<|end|>"])

        assert policy.metrics.episode_policy_ask_sonnet == 1
        # last_action_was_ask_sonnet should be set
        ctx = _rollout_ctx.get()
        assert ctx is not None
        assert ctx["last_action_was_ask_sonnet"] is True
