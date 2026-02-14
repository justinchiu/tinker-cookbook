"""Tests for data processing — specifically SequenceAccumulator instance isolation.

The SequenceAccumulator used class-level mutable attributes, meaning state leaked
between calls to trajectory_to_data(). After the fix (converting to a @dataclass
with instance attributes), two sequential calls should produce independent results.
"""

import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition


def _make_simple_trajectory(ob_tokens: list[int], ac_tokens: list[int]) -> Trajectory:
    """Create a trajectory with one transition: ob -> ac."""
    ob = tinker.ModelInput.from_ints(ob_tokens)
    ac = TokensWithLogprobs(
        tokens=ac_tokens,
        maybe_logprobs=[0.0] * len(ac_tokens),
    )
    transition = Transition(
        ob=ob,
        ac=ac,
        reward=1.0,
        episode_done=True,
    )
    return Trajectory(transitions=[transition], final_ob=ob)


class TestSequenceAccumulatorInstanceIsolation:
    def test_sequential_calls_independent(self):
        """Two sequential calls to trajectory_to_data produce independent results.

        Before the fix, the class-level lists would leak state between calls.
        """
        traj1 = _make_simple_trajectory([10, 20, 30], [40, 50])
        traj2 = _make_simple_trajectory([100, 200], [300])

        data1 = trajectory_to_data(traj1, 1.0)
        data2 = trajectory_to_data(traj2, 2.0)

        # Each should produce exactly one datum
        assert len(data1) == 1
        assert len(data2) == 1

        # The datums should have different lengths
        # traj1: 3 ob tokens + 2 ac tokens = 5 total, minus 1 for shift = 4
        # traj2: 2 ob tokens + 1 ac token = 3 total, minus 1 for shift = 2
        assert data1[0].model_input.length == 4
        assert data2[0].model_input.length == 2

    def test_no_state_leak_in_advantages(self):
        """Advantages from one call don't leak into the next."""
        traj1 = _make_simple_trajectory([1, 2], [3])
        traj2 = _make_simple_trajectory([4, 5], [6])

        data1 = trajectory_to_data(traj1, 10.0)
        data2 = trajectory_to_data(traj2, 20.0)

        adv1 = data1[0].loss_fn_inputs["advantages"].to_torch()
        adv2 = data2[0].loss_fn_inputs["advantages"].to_torch()

        # traj1 advantages should only have values 0 (ob) and 10 (ac)
        # traj2 advantages should only have values 0 (ob) and 20 (ac)
        assert not torch.any(adv1 == 20.0)
        assert not torch.any(adv2 == 10.0)
