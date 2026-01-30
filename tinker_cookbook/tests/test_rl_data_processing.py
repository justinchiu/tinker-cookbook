import torch
import tinker

from tinker_cookbook.completers import TokensWithLogprobs
from tinker_cookbook.rl.data_processing import trajectory_to_data
from tinker_cookbook.rl.types import Trajectory, Transition


def _flatten_model_input(model_input: tinker.ModelInput) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if isinstance(chunk, tinker.types.EncodedTextChunk):
            tokens.extend(chunk.tokens)
        else:
            tokens.extend([0] * chunk.length)
    return tokens


def test_context_transform_swaps_prompt():
    ob = tinker.ModelInput.from_ints([1, 2, 3])
    ac = TokensWithLogprobs(tokens=[4, 5], maybe_logprobs=[-0.1, -0.2])
    transition = Transition(
        ob=ob,
        ac=ac,
        reward=0.0,
        episode_done=True,
        metrics={},
        logs={},
    )
    traj = Trajectory(transitions=[transition], final_ob=tinker.ModelInput.empty())

    def transform(_ob: tinker.ModelInput, _turn_idx: int) -> tinker.ModelInput:
        return tinker.ModelInput.from_ints([9, 9])

    data = trajectory_to_data(traj, traj_advantage=1.0, context_transform=transform)
    assert len(data) == 1
    datum = data[0]

    full_sequence = datum.model_input.append_int(
        int(datum.loss_fn_inputs["target_tokens"].data[-1])
    )
    assert _flatten_model_input(full_sequence) == [9, 9, 4, 5]

    logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
    mask = datum.loss_fn_inputs["mask"].to_torch() > 0
    assert torch.allclose(logprobs[mask], torch.tensor([-0.1, -0.2]))


def test_context_transform_prefix_merging_single_datum():
    ob1 = tinker.ModelInput.from_ints([1])
    ac1 = TokensWithLogprobs(tokens=[2], maybe_logprobs=[-0.1])
    ob2 = tinker.ModelInput.from_ints([1, 2, 3])
    ac2 = TokensWithLogprobs(tokens=[4], maybe_logprobs=[-0.2])

    traj = Trajectory(
        transitions=[
            Transition(
                ob=ob1,
                ac=ac1,
                reward=0.0,
                episode_done=False,
                metrics={},
                logs={},
            ),
            Transition(
                ob=ob2,
                ac=ac2,
                reward=0.0,
                episode_done=True,
                metrics={},
                logs={},
            ),
        ],
        final_ob=tinker.ModelInput.empty(),
    )

    data = trajectory_to_data(
        traj,
        traj_advantage=1.0,
        context_transform=lambda ob, _turn_idx: ob,
    )
    assert len(data) == 1
    datum = data[0]

    full_sequence = datum.model_input.append_int(
        int(datum.loss_fn_inputs["target_tokens"].data[-1])
    )
    assert _flatten_model_input(full_sequence) == [1, 2, 3, 4]


def test_context_transform_breaks_prefix_splits_datums():
    ob1 = tinker.ModelInput.from_ints([1])
    ac1 = TokensWithLogprobs(tokens=[2], maybe_logprobs=[-0.1])
    ob2 = tinker.ModelInput.from_ints([1, 2, 3])
    ac2 = TokensWithLogprobs(tokens=[4], maybe_logprobs=[-0.2])

    traj = Trajectory(
        transitions=[
            Transition(
                ob=ob1,
                ac=ac1,
                reward=0.0,
                episode_done=False,
                metrics={},
                logs={},
            ),
            Transition(
                ob=ob2,
                ac=ac2,
                reward=0.0,
                episode_done=True,
                metrics={},
                logs={},
            ),
        ],
        final_ob=tinker.ModelInput.empty(),
    )

    def transform(ob: tinker.ModelInput, turn_idx: int) -> tinker.ModelInput:
        if turn_idx == 1:
            return tinker.ModelInput.from_ints([8])
        return ob

    data = trajectory_to_data(traj, traj_advantage=1.0, context_transform=transform)
    assert len(data) == 2

    full_sequence_0 = data[0].model_input.append_int(
        int(data[0].loss_fn_inputs["target_tokens"].data[-1])
    )
    full_sequence_1 = data[1].model_input.append_int(
        int(data[1].loss_fn_inputs["target_tokens"].data[-1])
    )
    assert _flatten_model_input(full_sequence_0) == [1, 2]
    assert _flatten_model_input(full_sequence_1) == [8, 4]
