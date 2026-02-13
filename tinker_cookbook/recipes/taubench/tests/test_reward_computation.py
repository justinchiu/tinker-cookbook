"""Tests for reward computation — penalty composition in compute_group_rewards.

Key properties tested:
1. Zero penalties → zero penalty regardless of ask_sonnet count
2. Each penalty type applied correctly in isolation
3. All penalties stack additively
4. Metrics dict has expected keys
5. Return value is (-total_penalty, metrics) for each env
"""

from unittest.mock import MagicMock

import pytest

from tinker_cookbook.recipes.taubench.env import Tau2EnvGroupBuilder


def _make_mock_env(
    ask_sonnet_call_count: int = 0,
    empty_advisor_responses: int = 0,
    sonnet_input_tokens: int = 0,
    sonnet_output_tokens: int = 0,
    policy_input_tokens: int = 0,
    policy_output_tokens: int = 0,
    tau2_user_input_tokens: int = 0,
    tau2_user_output_tokens: int = 0,
    tau2_user_cost_usd: float = 0.0,
):
    """Create a mock Tau2Env with specified attributes."""
    env = MagicMock()
    env.ask_sonnet_call_count = ask_sonnet_call_count
    env.empty_advisor_responses = empty_advisor_responses
    env.sonnet_input_tokens = sonnet_input_tokens
    env.sonnet_output_tokens = sonnet_output_tokens
    env.policy_input_tokens = policy_input_tokens
    env.policy_output_tokens = policy_output_tokens
    env.tau2_user_input_tokens = tau2_user_input_tokens
    env.tau2_user_output_tokens = tau2_user_output_tokens
    env.tau2_user_cost_usd = tau2_user_cost_usd
    return env


def _make_builder(**kwargs) -> Tau2EnvGroupBuilder:
    """Create a Tau2EnvGroupBuilder with default values, overriding kwargs."""
    defaults = dict(
        domain="retail",
        task_id="t1",
        renderer=MagicMock(),
        num_envs=1,
        ask_sonnet_penalty=0.0,
        sonnet_token_penalty_per_1k=0.0,
        tau2_user_token_penalty_per_1k=0.0,
        tau2_user_cost_penalty=0.0,
    )
    defaults.update(kwargs)
    return Tau2EnvGroupBuilder(**defaults)


class TestZeroPenalties:
    @pytest.mark.asyncio
    async def test_no_penalties_returns_zero(self):
        """With all penalties=0, reward adjustment is 0 regardless of ask_sonnet count."""
        builder = _make_builder()
        env = _make_mock_env(ask_sonnet_call_count=5, sonnet_input_tokens=10000)
        results = await builder.compute_group_rewards([], [env])

        assert len(results) == 1
        reward, metrics = results[0]
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_no_penalties_with_zero_usage(self):
        """Zero usage + zero penalties → zero adjustment."""
        builder = _make_builder()
        env = _make_mock_env()
        results = await builder.compute_group_rewards([], [env])
        reward, _ = results[0]
        assert reward == 0.0


class TestAskSonnetPenalty:
    @pytest.mark.asyncio
    async def test_per_call_penalty(self):
        """ask_sonnet_penalty * count should be the total penalty."""
        builder = _make_builder(ask_sonnet_penalty=0.1)
        env = _make_mock_env(ask_sonnet_call_count=3)
        results = await builder.compute_group_rewards([], [env])
        reward, metrics = results[0]
        assert abs(reward - (-0.3)) < 1e-9
        assert abs(metrics["ask_sonnet_penalty"] - 0.3) < 1e-9

    @pytest.mark.asyncio
    async def test_zero_calls_no_penalty(self):
        """No ask_sonnet calls → no penalty even with nonzero rate."""
        builder = _make_builder(ask_sonnet_penalty=0.5)
        env = _make_mock_env(ask_sonnet_call_count=0)
        results = await builder.compute_group_rewards([], [env])
        reward, _ = results[0]
        assert reward == 0.0


class TestSonnetTokenPenalty:
    @pytest.mark.asyncio
    async def test_sonnet_token_penalty(self):
        """sonnet_token_penalty_per_1k * (input + output) / 1000."""
        builder = _make_builder(sonnet_token_penalty_per_1k=0.01)
        env = _make_mock_env(sonnet_input_tokens=2000, sonnet_output_tokens=1000)
        results = await builder.compute_group_rewards([], [env])
        reward, metrics = results[0]
        # 0.01 * 3000/1000 = 0.03
        assert abs(reward - (-0.03)) < 1e-9
        assert abs(metrics["sonnet_token_penalty"] - 0.03) < 1e-9


class TestTau2UserTokenPenalty:
    @pytest.mark.asyncio
    async def test_tau2_user_token_penalty(self):
        """tau2_user_token_penalty_per_1k * (input + output) / 1000."""
        builder = _make_builder(tau2_user_token_penalty_per_1k=0.005)
        env = _make_mock_env(tau2_user_input_tokens=4000, tau2_user_output_tokens=1000)
        results = await builder.compute_group_rewards([], [env])
        reward, metrics = results[0]
        # 0.005 * 5000/1000 = 0.025
        assert abs(reward - (-0.025)) < 1e-9
        assert abs(metrics["tau2_user_token_penalty"] - 0.025) < 1e-9


class TestTau2UserCostPenalty:
    @pytest.mark.asyncio
    async def test_tau2_user_cost_penalty(self):
        """tau2_user_cost_penalty * tau2_user_cost_usd."""
        builder = _make_builder(tau2_user_cost_penalty=10.0)
        env = _make_mock_env(tau2_user_cost_usd=0.05)
        results = await builder.compute_group_rewards([], [env])
        reward, metrics = results[0]
        # 10.0 * 0.05 = 0.5
        assert abs(reward - (-0.5)) < 1e-9
        assert abs(metrics["tau2_user_cost_penalty"] - 0.5) < 1e-9


class TestStackedPenalties:
    @pytest.mark.asyncio
    async def test_all_penalties_stack_additively(self):
        """All penalty types should add together."""
        builder = _make_builder(
            ask_sonnet_penalty=0.1,
            sonnet_token_penalty_per_1k=0.01,
            tau2_user_token_penalty_per_1k=0.005,
            tau2_user_cost_penalty=10.0,
        )
        env = _make_mock_env(
            ask_sonnet_call_count=2,
            sonnet_input_tokens=1000,
            sonnet_output_tokens=500,
            tau2_user_input_tokens=2000,
            tau2_user_output_tokens=500,
            tau2_user_cost_usd=0.03,
        )
        results = await builder.compute_group_rewards([], [env])
        reward, metrics = results[0]

        # ask_sonnet: 0.1 * 2 = 0.2
        # sonnet tokens: 0.01 * 1500/1000 = 0.015
        # tau2 user tokens: 0.005 * 2500/1000 = 0.0125
        # tau2 user cost: 10.0 * 0.03 = 0.3
        expected_penalty = 0.2 + 0.015 + 0.0125 + 0.3
        assert abs(reward - (-expected_penalty)) < 1e-9
        assert abs(metrics["total_cost_penalty"] - expected_penalty) < 1e-9


class TestMultipleEnvs:
    @pytest.mark.asyncio
    async def test_separate_results_per_env(self):
        """Each env gets its own (reward, metrics) tuple."""
        builder = _make_builder(ask_sonnet_penalty=0.1)
        env1 = _make_mock_env(ask_sonnet_call_count=1)
        env2 = _make_mock_env(ask_sonnet_call_count=3)
        results = await builder.compute_group_rewards([], [env1, env2])
        assert len(results) == 2
        assert abs(results[0][0] - (-0.1)) < 1e-9
        assert abs(results[1][0] - (-0.3)) < 1e-9


class TestMetricsKeys:
    @pytest.mark.asyncio
    async def test_metrics_has_expected_keys(self):
        """Metrics dict should contain all tracking fields."""
        builder = _make_builder(ask_sonnet_penalty=0.1)
        env = _make_mock_env(ask_sonnet_call_count=1)
        results = await builder.compute_group_rewards([], [env])
        _, metrics = results[0]

        expected_keys = [
            "ask_sonnet_count",
            "ask_sonnet_penalty",
            "empty_advisor_responses",
            "sonnet_input_tokens",
            "sonnet_output_tokens",
            "policy_input_tokens",
            "policy_output_tokens",
            "total_sonnet_tokens",
            "total_policy_tokens",
            "tau2_user_input_tokens",
            "tau2_user_output_tokens",
            "tau2_user_cost_usd",
            "sonnet_token_penalty",
            "tau2_user_token_penalty",
            "tau2_user_cost_penalty",
            "total_cost_penalty",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_token_totals_computed(self):
        """total_sonnet_tokens and total_policy_tokens should be sums."""
        builder = _make_builder()
        env = _make_mock_env(
            sonnet_input_tokens=100,
            sonnet_output_tokens=50,
            policy_input_tokens=200,
            policy_output_tokens=80,
        )
        results = await builder.compute_group_rewards([], [env])
        _, metrics = results[0]
        assert metrics["total_sonnet_tokens"] == 150
        assert metrics["total_policy_tokens"] == 280
