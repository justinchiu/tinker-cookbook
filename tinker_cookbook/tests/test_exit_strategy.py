"""Tests for ExIt strategy plumbing.

Key properties tested:
1. TrajectoryGroup works with strategy_id=None (backward compat)
2. Advantages scaled by weight / count when strategy_weights is set
3. ValueError when strategy_id not in weights dict
4. context_transform replaces observations when set
5. None context_transform means identity (no change)
"""

import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import assemble_training_data, compute_advantages
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup, Transition

import pytest


def _make_trajectory(ob_tokens: list[int], ac_tokens: list[int]) -> Trajectory:
    """Create a single-step trajectory."""
    ob = tinker.ModelInput.from_ints(ob_tokens)
    ac = TokensWithLogprobs(tokens=ac_tokens, maybe_logprobs=[0.0] * len(ac_tokens))
    transition = Transition(ob=ob, ac=ac, reward=0.0, episode_done=True)
    return Trajectory(transitions=[transition], final_ob=ob)


def _make_group(rewards: list[float], strategy_id: str | None = None) -> TrajectoryGroup:
    """Create a TrajectoryGroup with given rewards and optional strategy_id."""
    return TrajectoryGroup(
        trajectories_G=[_make_trajectory([1, 2], [3]) for _ in rewards],
        final_rewards_G=rewards,
        metrics_G=[{} for _ in rewards],
        strategy_id=strategy_id,
    )


class TestStrategyIdOptional:
    def test_none_strategy_id(self):
        """TrajectoryGroup works with strategy_id=None (backward compat)."""
        group = _make_group([1.0, 2.0])
        assert group.strategy_id is None
        # Should still compute total rewards correctly
        assert group.get_total_rewards() == [1.0, 2.0]

    def test_explicit_strategy_id(self):
        """TrajectoryGroup stores the strategy_id."""
        group = _make_group([1.0, 2.0], strategy_id="main")
        assert group.strategy_id == "main"


class TestDataclassSubclassCanOverrideStrategyId:
    def test_frozen_dataclass_subclass_with_strategy_id_field(self):
        """A frozen dataclass subclass of EnvGroupBuilder can define strategy_id as a field."""
        from dataclasses import dataclass
        from collections.abc import Sequence
        from tinker_cookbook.rl.types import Env, EnvGroupBuilder

        @dataclass(frozen=True)
        class MyBuilder(EnvGroupBuilder):
            strategy_id: str | None = None

            async def make_envs(self) -> Sequence[Env]:
                return []

        builder = MyBuilder(strategy_id="fast")
        assert builder.strategy_id == "fast"

        builder_none = MyBuilder()
        assert builder_none.strategy_id is None

    def test_chz_subclass_with_strategy_id_field(self):
        """A chz subclass of EnvGroupBuilder can define strategy_id as a field."""
        import chz
        from collections.abc import Sequence
        from tinker_cookbook.rl.types import Env, EnvGroupBuilder

        @chz.chz
        class MyChzBuilder(EnvGroupBuilder):
            strategy_id: str | None = None

            async def make_envs(self) -> Sequence[Env]:
                return []

        builder = MyChzBuilder(strategy_id="slow")
        assert builder.strategy_id == "slow"

        builder_none = MyChzBuilder()
        assert builder_none.strategy_id is None


class TestStrategyWeightScaling:
    def test_single_strategy_weight(self):
        """Advantages are scaled by weight / count for a single strategy."""
        group1 = _make_group([1.0, 3.0], strategy_id="main")
        group2 = _make_group([2.0, 4.0], strategy_id="main")
        groups = [group1, group2]

        advantages_P = compute_advantages(groups)
        strategy_weights = {"main": 2.0}
        data_D, _ = assemble_training_data(groups, advantages_P, strategy_weights=strategy_weights)

        # With 2 groups of strategy "main" and weight=2.0,
        # scaling factor = 2.0 / 2 = 1.0 (no change)
        # Verify we get data (doesn't crash)
        assert len(data_D) > 0

    def test_multi_strategy_different_weights(self):
        """Different strategy weights scale advantages differently."""
        group_a = _make_group([0.0, 10.0], strategy_id="fast")
        group_b = _make_group([0.0, 10.0], strategy_id="slow")

        # Both groups have same rewards, so same raw advantages
        advantages_P = compute_advantages([group_a, group_b])

        # fast gets weight 2.0 (1 group → scale=2.0/1=2.0)
        # slow gets weight 0.5 (1 group → scale=0.5/1=0.5)
        strategy_weights = {"fast": 2.0, "slow": 0.5}
        data_D, metadata_D = assemble_training_data(
            [group_a, group_b], advantages_P, strategy_weights=strategy_weights
        )

        # Extract advantages from each group's datums
        group_0_advs = []
        group_1_advs = []
        for datum, meta in zip(data_D, metadata_D, strict=True):
            adv_tensor = datum.loss_fn_inputs["advantages"].to_torch()
            mask_tensor = datum.loss_fn_inputs["mask"].to_torch()
            # Get only the non-zero (action) advantages
            masked_advs = adv_tensor[mask_tensor > 0]
            if meta["group_idx"] == 0:
                group_0_advs.append(masked_advs)
            else:
                group_1_advs.append(masked_advs)

        # The "fast" group's action advantages should be 4× the "slow" group's
        # because 2.0/0.5 = 4.0
        fast_max = torch.cat(group_0_advs).abs().max().item()
        slow_max = torch.cat(group_1_advs).abs().max().item()
        assert abs(fast_max / slow_max - 4.0) < 1e-5


class TestStrategyWeightsMissingRaises:
    def test_missing_strategy_id_raises(self):
        """ValueError when strategy_id not in weights dict."""
        group = _make_group([1.0, 2.0], strategy_id="unknown")
        advantages_P = compute_advantages([group])
        strategy_weights = {"main": 1.0}

        with pytest.raises(ValueError, match="unknown"):
            assemble_training_data([group], advantages_P, strategy_weights=strategy_weights)

    def test_none_strategy_id_with_weights_raises(self):
        """ValueError when strategy_id is None but strategy_weights is set."""
        group = _make_group([1.0, 2.0])  # strategy_id=None
        advantages_P = compute_advantages([group])
        strategy_weights = {"main": 1.0}

        with pytest.raises(ValueError):
            assemble_training_data([group], advantages_P, strategy_weights=strategy_weights)


class TestContextTransform:
    def test_context_transform_applied(self):
        """Observations are replaced by context_transform output."""
        ob_tokens = [10, 20, 30]
        ac_tokens = [40]
        traj = _make_trajectory(ob_tokens, ac_tokens)
        group = _make_group([1.0, 2.0], strategy_id="main")
        # Replace trajectories with our known trajectory
        group.trajectories_G = [traj, traj]

        # Transform that replaces observation with shorter one
        def transform(ob: tinker.ModelInput) -> tinker.ModelInput:
            return tinker.ModelInput.from_ints([99])

        advantages_P = compute_advantages([group])
        data_D, _ = assemble_training_data([group], advantages_P, context_transform=transform)

        # With transform, each datum should have 1 ob token + 1 ac token = 2, minus 1 = 1
        for datum in data_D:
            assert datum.model_input.length == 1

    def test_no_context_transform_default(self):
        """None context_transform means identity (no change)."""
        ob_tokens = [10, 20, 30]
        ac_tokens = [40]
        traj = _make_trajectory(ob_tokens, ac_tokens)
        group = _make_group([1.0], strategy_id=None)
        group.trajectories_G = [traj]

        advantages_P = compute_advantages([group])

        # Without transform
        data_no_transform, _ = assemble_training_data([group], advantages_P)
        # With explicit None transform
        data_none_transform, _ = assemble_training_data(
            [group], advantages_P, context_transform=None
        )

        assert len(data_no_transform) == len(data_none_transform)
        for d1, d2 in zip(data_no_transform, data_none_transform, strict=True):
            assert d1.model_input.length == d2.model_input.length
