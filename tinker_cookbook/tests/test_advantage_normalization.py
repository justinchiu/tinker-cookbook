"""Tests for advantage normalization in compute_advantages.

Key properties tested:
1. Default (normalize=False): advantages are mean-centered only
2. normalize=True: advantages have mean≈0, std≈1
3. Constant rewards with normalize=True don't crash (no div-by-zero)
4. Groups are normalized independently
"""

import torch

from tinker_cookbook.rl.data_processing import compute_advantages
from tinker_cookbook.rl.types import Trajectory, TrajectoryGroup


def _make_trajectory_group(total_rewards: list[float]) -> TrajectoryGroup:
    """Create a TrajectoryGroup with the given total rewards.

    Since compute_advantages uses get_total_rewards() which sums per-step rewards
    + final_reward, we put all reward in final_rewards_G and use empty trajectories.
    """
    return TrajectoryGroup(
        trajectories_G=[Trajectory(transitions=[], final_ob=None) for _ in total_rewards],  # type: ignore[arg-type]
        final_rewards_G=total_rewards,
        metrics_G=[{} for _ in total_rewards],
    )


class TestDefaultOnlyCenters:
    def test_rewards_1_3_5(self):
        """rewards [1,3,5] → advantages [-2,0,2] (mean-centered)."""
        group = _make_trajectory_group([1.0, 3.0, 5.0])
        [advantages] = compute_advantages([group])
        expected = torch.tensor([-2.0, 0.0, 2.0])
        assert torch.allclose(advantages, expected)

    def test_default_is_not_standardized(self):
        """Default mode should NOT standardize to std=1."""
        group = _make_trajectory_group([1.0, 3.0, 5.0])
        [advantages] = compute_advantages([group])
        # std should be ~1.63, not 1.0
        assert advantages.std().item() > 1.1


class TestNormalizeTrueStandardizes:
    def test_mean_near_zero(self):
        """With normalize=True, mean of advantages should be ~0."""
        group = _make_trajectory_group([1.0, 3.0, 5.0])
        [advantages] = compute_advantages([group], normalize_advantages=True)
        assert abs(advantages.mean().item()) < 1e-6

    def test_std_near_one(self):
        """With normalize=True, std of advantages should be ~1."""
        group = _make_trajectory_group([1.0, 3.0, 5.0])
        [advantages] = compute_advantages([group], normalize_advantages=True)
        assert abs(advantages.std().item() - 1.0) < 1e-5

    def test_different_scale(self):
        """Normalization works for rewards with large variance."""
        group = _make_trajectory_group([0.0, 100.0, 200.0])
        [advantages] = compute_advantages([group], normalize_advantages=True)
        assert abs(advantages.mean().item()) < 1e-5
        assert abs(advantages.std().item() - 1.0) < 1e-4


class TestConstantRewardsNoCrash:
    def test_all_equal_returns_zeros(self):
        """All-equal rewards with normalize=True should produce all zeros (no NaN)."""
        group = _make_trajectory_group([5.0, 5.0, 5.0])
        [advantages] = compute_advantages([group], normalize_advantages=True)
        assert torch.allclose(advantages, torch.zeros(3))
        assert not torch.isnan(advantages).any()

    def test_single_element(self):
        """Single-element group with normalize=True should return zero."""
        group = _make_trajectory_group([7.0])
        [advantages] = compute_advantages([group], normalize_advantages=True)
        assert torch.allclose(advantages, torch.zeros(1))


class TestGroupsNormalizedIndependently:
    def test_two_groups_different_distributions(self):
        """Two groups with different reward distributions are each normalized independently."""
        group1 = _make_trajectory_group([1.0, 3.0, 5.0])
        group2 = _make_trajectory_group([100.0, 200.0, 300.0])
        advantages = compute_advantages([group1, group2], normalize_advantages=True)

        assert len(advantages) == 2
        for adv in advantages:
            assert abs(adv.mean().item()) < 1e-5
            assert abs(adv.std().item() - 1.0) < 1e-4
