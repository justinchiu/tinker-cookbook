"""Tests for Tau2Env lifecycle — context enforcement, ask_sonnet modes, termination.

Key properties tested:
1. Pre-step context too long → immediate termination with reward=0
2. Post-step response too long → termination with reward=0
3. Consecutive ask_sonnet calls → termination with reward=0
4. Non-consecutive ask_sonnet calls → proceed normally
5. Token costs tracked separately (policy, sonnet, user-sim)
6. Direct mode: Sonnet response sent to tau2 immediately
7. Conditioning mode: returns early, waits for policy followup
8. Empty advisor response → error observation, episode continues
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from tinker_cookbook.recipes.taubench.components.types import (
    AskSonnetMode,
    ObservationType,
    Tau2StepResult,
)
from tinker_cookbook.recipes.taubench.env import Tau2Env


def _make_env(**overrides) -> Tau2Env:
    """Create a Tau2Env with all dependencies mocked out.

    Returns a Tau2Env created via __new__ with mocked internals.
    """
    env = Tau2Env.__new__(Tau2Env)

    # Core attributes
    env.renderer = MagicMock()
    env.domain = "retail"
    env.task_id = "test_task"
    env.max_context_length = overrides.get("max_context_length", None)
    env._context_exceeded = False
    env._current_obs_length = overrides.get("current_obs_length", 100)
    env.rollout_logger = None

    # ask_sonnet tracking
    env.ask_sonnet_call_count = 0
    env._last_action_was_ask_sonnet = overrides.get("last_action_was_ask_sonnet", False)
    env.empty_advisor_responses = 0
    env.ask_sonnet_mode = overrides.get("ask_sonnet_mode", AskSonnetMode.DIRECT_INJECTION)

    # Token tracking
    env.sonnet_input_tokens = 0
    env.sonnet_output_tokens = 0
    env.policy_input_tokens = 0
    env.policy_output_tokens = 0
    env.tau2_user_input_tokens = 0
    env.tau2_user_output_tokens = 0
    env.tau2_user_cost_usd = 0.0

    # Mocked components
    env.action_parser = MagicMock()
    env.gym = MagicMock()
    env.messages = MagicMock()
    env.messages.messages = [{"role": "system", "content": "sys"}]
    env.messages.system_prompt = "sys"
    env._advisor_system_prompt = "sys"
    env.tools = []

    # External LLM
    env.external_llm = overrides.get("external_llm", None)
    env.ask_sonnet_renderer = overrides.get("ask_sonnet_renderer", None)

    # Renderer mock defaults
    mock_obs = MagicMock()
    mock_obs.length = overrides.get("next_obs_length", 100)
    env.renderer.build_generation_prompt.return_value = mock_obs
    env.renderer.get_stop_sequences.return_value = ["<|end|>"]

    return env


# ---------------------------------------------------------------------------
# Pre-step context enforcement
# ---------------------------------------------------------------------------


class TestPreStepContextEnforcement:
    @pytest.mark.asyncio
    async def test_pre_step_too_long_terminates(self):
        """If current obs length exceeds max, terminate immediately with reward=0."""
        env = _make_env(max_context_length=500, current_obs_length=600)
        result = await env.step([1, 2, 3])
        assert result.episode_done is True
        assert result.reward == 0.0
        assert env._context_exceeded is True

    @pytest.mark.asyncio
    async def test_pre_step_within_limit_proceeds(self):
        """If current obs length is within max, proceed normally."""
        env = _make_env(max_context_length=500, current_obs_length=400)

        # Setup action parser to return text action
        parsed = MagicMock()
        parsed.original_message = {"content": "Hello"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = False
        env.action_parser.to_tau2_action.return_value = "Hello"

        # Mock tau2 gym step
        env.gym.step = AsyncMock(
            return_value=Tau2StepResult(
                obs_type=ObservationType.USER_MESSAGE,
                obs_content="Thanks!",
                raw_obs="user: Thanks!",
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        )

        result = await env.step([1, 2, 3])
        assert result.episode_done is False
        assert env._context_exceeded is False

    @pytest.mark.asyncio
    async def test_no_max_context_length_never_terminates_early(self):
        """With max_context_length=None, pre-step check is skipped."""
        env = _make_env(max_context_length=None, current_obs_length=999999)

        parsed = MagicMock()
        parsed.original_message = {"content": "Hi"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = False
        env.action_parser.to_tau2_action.return_value = "Hi"

        env.gym.step = AsyncMock(
            return_value=Tau2StepResult(
                obs_type=ObservationType.USER_MESSAGE,
                obs_content="Hello",
                raw_obs="user: Hello",
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        )

        result = await env.step([1, 2])
        assert result.episode_done is False


# ---------------------------------------------------------------------------
# Post-step context enforcement
# ---------------------------------------------------------------------------


class TestPostStepContextEnforcement:
    @pytest.mark.asyncio
    async def test_post_step_too_long_terminates(self):
        """If next obs after tau2 step exceeds max, terminate with reward=0."""
        env = _make_env(max_context_length=500, current_obs_length=400, next_obs_length=600)

        parsed = MagicMock()
        parsed.original_message = {"content": "action"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = False
        env.action_parser.to_tau2_action.return_value = "action"

        env.gym.step = AsyncMock(
            return_value=Tau2StepResult(
                obs_type=ObservationType.USER_MESSAGE,
                obs_content="Long response...",
                raw_obs="user: Long response...",
                reward=0.5,
                terminated=False,
                truncated=False,
                info={},
            )
        )

        result = await env.step([1, 2])
        assert result.episode_done is True
        assert result.reward == 0.0
        assert env._context_exceeded is True


# ---------------------------------------------------------------------------
# Consecutive ask_sonnet termination
# ---------------------------------------------------------------------------


class TestConsecutiveAskSonnet:
    @pytest.mark.asyncio
    async def test_consecutive_ask_sonnet_terminates(self):
        """Two ask_sonnet calls in a row → termination with reward=0."""
        env = _make_env(
            last_action_was_ask_sonnet=True,
            external_llm=MagicMock(),
            ask_sonnet_renderer=MagicMock(),
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask_sonnet call"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        result = await env.step([1, 2])
        assert result.episode_done is True
        assert result.reward == 0.0
        # Messages should have the error added
        env.messages.add_tool_result.assert_called()

    @pytest.mark.asyncio
    async def test_non_consecutive_ask_sonnet_proceeds(self):
        """ask_sonnet after a normal action → proceed normally."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = "I suggest checking the order status"
        mock_llm_result.input_tokens = 100
        mock_llm_result.output_tokens = 50
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_renderer = MagicMock()
        mock_renderer.render_for_advisor.return_value = [{"role": "user", "content": "hi"}]
        mock_renderer.should_return_early.return_value = True  # Conditioning mode
        mock_renderer.format_sonnet_response_for_messages.return_value = {
            "role": "tool",
            "content": "[Sonnet's Advice]: check order",
            "tool_call_id": "ask_sonnet_call",
        }

        env = _make_env(
            last_action_was_ask_sonnet=False,
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask_sonnet call"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        result = await env.step([1, 2])
        assert result.episode_done is False
        assert env.ask_sonnet_call_count == 1


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


class TestTokenTracking:
    @pytest.mark.asyncio
    async def test_policy_output_tokens_tracked(self):
        """Policy output tokens (action length) should be tracked."""
        env = _make_env()

        parsed = MagicMock()
        parsed.original_message = {"content": "Hello"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = False
        env.action_parser.to_tau2_action.return_value = "Hello"

        env.gym.step = AsyncMock(
            return_value=Tau2StepResult(
                obs_type=ObservationType.USER_MESSAGE,
                obs_content="Hi",
                raw_obs="user: Hi",
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        )

        await env.step([10, 20, 30, 40, 50])
        assert env.policy_output_tokens == 5

    @pytest.mark.asyncio
    async def test_sonnet_tokens_tracked(self):
        """Sonnet input/output tokens tracked when ask_sonnet is called."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = "advice"
        mock_llm_result.input_tokens = 200
        mock_llm_result.output_tokens = 75
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_renderer = MagicMock()
        mock_renderer.render_for_advisor.return_value = []
        mock_renderer.should_return_early.return_value = True

        env = _make_env(
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        await env.step([1])
        assert env.sonnet_input_tokens == 200
        assert env.sonnet_output_tokens == 75


# ---------------------------------------------------------------------------
# Direct mode
# ---------------------------------------------------------------------------


class TestDirectMode:
    @pytest.mark.asyncio
    async def test_direct_sends_to_tau2(self):
        """In direct mode, Sonnet's response is sent to tau2 immediately."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = '{"name": "get_order", "arguments": {"id": "123"}}'
        mock_llm_result.input_tokens = 50
        mock_llm_result.output_tokens = 20
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_ask_renderer = MagicMock()
        mock_ask_renderer.render_for_advisor.return_value = []
        mock_ask_renderer.should_return_early.return_value = False  # Direct mode
        mock_ask_renderer.get_tau2_action.return_value = (
            '{"name": "get_order", "arguments": {"id": "123"}}'
        )

        env = _make_env(
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_ask_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        env.gym.step = AsyncMock(
            return_value=Tau2StepResult(
                obs_type=ObservationType.TOOL_RESULT,
                obs_content="Order found",
                raw_obs="tool: Order found",
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        )

        result = await env.step([1])
        # Should have sent to tau2 (gym.step called)
        env.gym.step.assert_awaited_once()
        assert result.episode_done is False
        # _last_action_was_ask_sonnet reset because Sonnet took an action
        assert env._last_action_was_ask_sonnet is False


# ---------------------------------------------------------------------------
# Conditioning mode
# ---------------------------------------------------------------------------


class TestConditioningMode:
    @pytest.mark.asyncio
    async def test_conditioning_returns_early(self):
        """In conditioning mode, should return observation for policy to decide."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = "I suggest looking up the order"
        mock_llm_result.input_tokens = 80
        mock_llm_result.output_tokens = 30
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_ask_renderer = MagicMock()
        mock_ask_renderer.render_for_advisor.return_value = []
        mock_ask_renderer.should_return_early.return_value = True  # Conditioning

        env = _make_env(
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_ask_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        result = await env.step([1])
        assert result.episode_done is False
        # Should NOT have called tau2 gym
        env.gym.step.assert_not_called()
        # _last_action_was_ask_sonnet should be True (waiting for followup)
        assert env._last_action_was_ask_sonnet is True


# ---------------------------------------------------------------------------
# Empty advisor response
# ---------------------------------------------------------------------------


class TestEmptyAdvisorResponse:
    @pytest.mark.asyncio
    async def test_empty_response_continues_episode(self):
        """Empty advisor response → error observation, episode continues."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = ""  # Empty!
        mock_llm_result.input_tokens = 50
        mock_llm_result.output_tokens = 0
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_ask_renderer = MagicMock()
        mock_ask_renderer.render_for_advisor.return_value = []

        env = _make_env(
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_ask_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        result = await env.step([1])
        assert result.episode_done is False
        assert env.empty_advisor_responses == 1
        # Error message should have been added
        env.messages.add_tool_result.assert_called()
        call_args = env.messages.add_tool_result.call_args
        assert "Advisor Error" in str(call_args)

    @pytest.mark.asyncio
    async def test_whitespace_only_response_is_empty(self):
        """Whitespace-only advisor response treated as empty."""
        mock_external_llm = MagicMock()
        mock_llm_result = MagicMock()
        mock_llm_result.content = "   \n  "  # Whitespace only
        mock_llm_result.input_tokens = 50
        mock_llm_result.output_tokens = 5
        mock_external_llm.call_with_usage = AsyncMock(return_value=mock_llm_result)

        mock_ask_renderer = MagicMock()
        mock_ask_renderer.render_for_advisor.return_value = []

        env = _make_env(
            external_llm=mock_external_llm,
            ask_sonnet_renderer=mock_ask_renderer,
        )

        parsed = MagicMock()
        parsed.original_message = {"content": "ask"}
        env.action_parser.parse.return_value = parsed
        env.action_parser.is_ask_sonnet.return_value = True

        result = await env.step([1])
        assert result.episode_done is False
        assert env.empty_advisor_responses == 1
