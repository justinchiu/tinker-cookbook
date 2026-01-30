import tinker
import torch

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.recipes.math_efficiency.efficient_env import (
    EfficientGsm8kDataset,
    ExItStrategy,
    ExItStrategyConfig,
    ANSWER_HINT_TEXT,
)
from tinker_cookbook.rl.data_processing import assemble_training_data
from tinker_cookbook.rl.problem_env import ProblemGroupBuilder
from tinker_cookbook.rl.types import StrategyId, Trajectory, TrajectoryGroup, Transition


class _TestStrategy(StrategyId):
    A = "A"
    B = "B"


def _make_traj(ob_tokens: list[int], ac_tokens: list[int], logprobs: list[float]) -> Trajectory:
    transition = Transition(
        ob=tinker.ModelInput.from_ints(ob_tokens),
        ac=TokensWithLogprobs(tokens=ac_tokens, maybe_logprobs=logprobs),
        reward=0.0,
        episode_done=True,
        metrics={},
        logs={},
    )
    return Trajectory(transitions=[transition], final_ob=tinker.ModelInput.empty())


def test_problem_group_builder_logging_tags_strategy():
    builder = ProblemGroupBuilder(
        env_thunk=lambda: None,
        num_envs=1,
        dataset_name="test",
        strategy_id=_TestStrategy.A,
    )
    assert builder.logging_tags() == ["test", "strategy/A"]


def test_strategy_weight_scaling():
    traj_a1 = _make_traj([1], [2], [-0.1])
    traj_a2 = _make_traj([1], [3], [-0.2])
    traj_b1 = _make_traj([1], [4], [-0.3])

    group_a = TrajectoryGroup(
        trajectories_G=[traj_a1, traj_a2],
        final_rewards_G=[0.0, 0.0],
        metrics_G=[{}, {}],
        strategy_id=_TestStrategy.A,
    )
    group_b = TrajectoryGroup(
        trajectories_G=[traj_b1],
        final_rewards_G=[0.0],
        metrics_G=[{}],
        strategy_id=_TestStrategy.B,
    )

    advantages_P = [torch.tensor([1.0, 3.0]), torch.tensor([2.0])]
    strategy_weights = {_TestStrategy.A: 1.0, _TestStrategy.B: 1.0}

    data_D, metadata_D = assemble_training_data(
        [group_a, group_b],
        advantages_P,
        strategy_weights=strategy_weights,
    )

    def _action_adv(datum: tinker.Datum) -> float:
        adv = datum.loss_fn_inputs["advantages"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch() > 0
        return float(adv[mask].item())

    advs = {}
    for datum, meta in zip(data_D, metadata_D, strict=True):
        advs[(meta["group_idx"], meta["traj_idx"])] = _action_adv(datum)

    assert advs[(0, 0)] == 0.5
    assert advs[(0, 1)] == 1.5
    assert advs[(1, 0)] == 2.0


class _DummyRenderer:
    def build_generation_prompt(self, messages):
        return tinker.ModelInput.from_ints([len(messages)])


def test_strategy_configs_builders_context_transform():
    dataset = EfficientGsm8kDataset.__new__(EfficientGsm8kDataset)
    dataset.renderer = _DummyRenderer()
    dataset.max_tokens = 10
    dataset.strategy_configs = [
        ExItStrategyConfig(
            strategy_id=ExItStrategy.IID,
            sampling_prefix=[],
            training_prefix=[],
        ),
        ExItStrategyConfig(
            strategy_id=ExItStrategy.ANSWER_HINT,
            sampling_prefix=[],
            training_prefix=[],
        ),
        ExItStrategyConfig(
            strategy_id=ExItStrategy.PROMPT_AUG,
            sampling_prefix=[{"role": "system", "content": "be concise"}],
            training_prefix=[],
        ),
    ]

    row = {"question": "2+2?", "answer": "#### 4"}
    builders = dataset._make_env_group_builders(row, group_size=2)
    assert len(builders) == 3

    iid_builder, answer_builder, aug_builder = builders
    assert iid_builder.strategy_id == ExItStrategy.IID
    assert iid_builder.context_transform is None

    assert answer_builder.strategy_id == ExItStrategy.ANSWER_HINT
    assert answer_builder.context_transform is not None
    env = answer_builder.env_thunk()
    assert ANSWER_HINT_TEXT.format(answer="4") in env.get_question()

    assert aug_builder.strategy_id == ExItStrategy.PROMPT_AUG
    assert aug_builder.context_transform is not None

    transformed = aug_builder.context_transform(tinker.ModelInput.empty(), 0)
    assert isinstance(transformed, tinker.ModelInput)
    assert transformed.length == 1
