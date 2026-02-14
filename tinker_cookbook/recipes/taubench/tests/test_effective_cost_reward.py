"""Tests for effective cost reward in compute_group_rewards.

The effective cost reward formula:
    effective_cost = policy_output_tokens + sonnet_cost_multiplier * (sonnet_input + sonnet_output)
    total_reward = task_success * max(0, effective_cost_budget - effective_cost)

Where task_success = 1 if sum(step_rewards) > 0.5 else 0.

Key properties tested:
1. budget=None → old penalty system (backward compat)
2. Failed task → reward=0 regardless of tokens
3. Success with 0 tokens → reward=budget
4. Success decreases with policy output tokens
5. Sonnet tokens cost sonnet_cost_multiplier× more
6. Cost > budget → reward=0 (clamped, not negative)
7. Non-default budget and multiplier work
8. Metrics dict has expected effective cost keys
9. Multiple envs computed independently
"""

from typing import Any
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
) -> Any:
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


def _make_builder(**kwargs: Any) -> Tau2EnvGroupBuilder:
    """Create a Tau2EnvGroupBuilder with default values, overriding kwargs."""
    defaults: dict[str, Any] = dict(
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


def _make_mock_trajectory(step_rewards: list[float]) -> Any:
    """Create a mock trajectory with given step rewards."""
    traj = MagicMock()
    transitions = []
    for r in step_rewards:
        t = MagicMock()
        t.reward = r
        transitions.append(t)
    traj.transitions = transitions
    return traj


class TestBudgetNoneUsesOldPenalty:
    @pytest.mark.asyncio
    async def test_default_budget_uses_penalty_system(self):
        """With effective_cost_budget=None, old penalty system is used."""
        builder = _make_builder(ask_sonnet_penalty=0.1)
        env = _make_mock_env(ask_sonnet_call_count=2)
        traj = _make_mock_trajectory([0.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, metrics = results[0]
        # Old penalty: -0.1 * 2 = -0.2
        assert abs(reward - (-0.2)) < 1e-9
        assert "effective_cost" not in metrics


class TestFailureAlwaysZero:
    @pytest.mark.asyncio
    async def test_failure_gives_zero_reward(self):
        """Failed task (sum(step_rewards) <= 0.5) → reward=0 regardless of tokens."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env = _make_mock_env(policy_output_tokens=100)
        # Step rewards sum to 0.0 → failure
        traj = _make_mock_trajectory([0.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, _ = results[0]
        assert reward == 0.0

    @pytest.mark.asyncio
    async def test_failure_with_many_tokens(self):
        """Failed task with lots of tokens still gets 0."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env = _make_mock_env(policy_output_tokens=5000, sonnet_input_tokens=1000)
        traj = _make_mock_trajectory([0.0, 0.0, 0.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, _ = results[0]
        assert reward == 0.0


class TestSuccessZeroCostGetsBudget:
    @pytest.mark.asyncio
    async def test_success_zero_cost(self):
        """Success with 0 tokens → reward = budget."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env = _make_mock_env(policy_output_tokens=0, sonnet_input_tokens=0, sonnet_output_tokens=0)
        # Step rewards sum to 1.0 > 0.5 → success
        traj = _make_mock_trajectory([1.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, _ = results[0]
        # total_reward = 1 * max(0, 10000 - 0) = 10000
        # step_reward_sum = 1.0, so final_reward = 10000 - 1.0 = 9999.0
        assert abs(reward - 9999.0) < 1e-6


class TestSuccessDecreasesWithPolicyTokens:
    @pytest.mark.asyncio
    async def test_more_tokens_lower_reward(self):
        """More policy output tokens → lower reward."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env_few = _make_mock_env(policy_output_tokens=100)
        env_many = _make_mock_env(policy_output_tokens=5000)
        traj = _make_mock_trajectory([1.0])

        results_few = await builder.compute_group_rewards([traj], [env_few])
        results_many = await builder.compute_group_rewards([traj], [env_many])

        reward_few = results_few[0][0]
        reward_many = results_many[0][0]
        assert reward_few > reward_many


class TestSonnetTokensMultiplied:
    @pytest.mark.asyncio
    async def test_sonnet_cost_multiplier(self):
        """Sonnet tokens cost sonnet_cost_multiplier× more than policy tokens."""
        builder = _make_builder(effective_cost_budget=100000.0, sonnet_cost_multiplier=30.0)

        # 100 policy output tokens → effective_cost = 100
        env_policy = _make_mock_env(policy_output_tokens=100)
        traj = _make_mock_trajectory([1.0])
        results_policy = await builder.compute_group_rewards([traj], [env_policy])
        reward_policy = results_policy[0][0]

        # 100 sonnet tokens → effective_cost = 30 * 100 = 3000
        env_sonnet = _make_mock_env(
            sonnet_input_tokens=50, sonnet_output_tokens=50, policy_output_tokens=0
        )
        results_sonnet = await builder.compute_group_rewards([traj], [env_sonnet])
        reward_sonnet = results_sonnet[0][0]

        # Policy-only should have much higher reward (lower cost)
        assert reward_policy > reward_sonnet


class TestMaxZeroClamp:
    @pytest.mark.asyncio
    async def test_cost_exceeds_budget(self):
        """Cost > budget → reward=0 (not negative)."""
        builder = _make_builder(effective_cost_budget=100.0)
        env = _make_mock_env(policy_output_tokens=200)  # cost=200 > budget=100
        traj = _make_mock_trajectory([1.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, _ = results[0]
        # total_reward = 1 * max(0, 100 - 200) = 0
        # step_reward_sum = 1.0, final_reward = 0 - 1.0 = -1.0
        assert abs(reward - (-1.0)) < 1e-6


class TestCustomParams:
    @pytest.mark.asyncio
    async def test_non_default_budget_and_multiplier(self):
        """Non-default budget and multiplier work correctly."""
        builder = _make_builder(effective_cost_budget=5000.0, sonnet_cost_multiplier=10.0)
        env = _make_mock_env(
            policy_output_tokens=1000, sonnet_input_tokens=100, sonnet_output_tokens=100
        )
        traj = _make_mock_trajectory([1.0])
        results = await builder.compute_group_rewards([traj], [env])
        reward, metrics = results[0]

        # effective_cost = 1000 + 10 * (100 + 100) = 1000 + 2000 = 3000
        # total_reward = 1 * max(0, 5000 - 3000) = 2000
        # step_reward_sum = 1.0, final_reward = 2000 - 1.0 = 1999.0
        assert abs(reward - 1999.0) < 1e-6
        assert abs(metrics["effective_cost"] - 3000.0) < 1e-6


class TestMetricsIncludeEffectiveCost:
    @pytest.mark.asyncio
    async def test_effective_cost_metrics(self):
        """Metrics dict should include effective cost keys."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env = _make_mock_env(
            policy_output_tokens=100, sonnet_input_tokens=50, sonnet_output_tokens=50
        )
        traj = _make_mock_trajectory([1.0])
        results = await builder.compute_group_rewards([traj], [env])
        _, metrics = results[0]

        expected_keys = [
            "effective_cost",
            "effective_cost_budget",
            "task_success",
        ]
        for key in expected_keys:
            assert key in metrics, f"Missing key: {key}"


class TestMultipleEnvsIndependent:
    @pytest.mark.asyncio
    async def test_each_env_independent(self):
        """Each env is computed independently."""
        builder = _make_builder(effective_cost_budget=10000.0)
        env1 = _make_mock_env(policy_output_tokens=100)
        env2 = _make_mock_env(policy_output_tokens=5000)
        traj1 = _make_mock_trajectory([1.0])
        traj2 = _make_mock_trajectory([0.0])

        results = await builder.compute_group_rewards([traj1, traj2], [env1, env2])
        assert len(results) == 2

        # env1: success, low cost → positive reward
        reward1, _ = results[0]
        # env2: failure → zero total
        reward2, _ = results[1]

        assert reward1 > 0
        assert reward2 <= 0  # zero total - step rewards = negative or zero final
