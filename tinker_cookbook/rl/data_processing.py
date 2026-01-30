"""
Data processing functions for RL training.

Contains functions for computing advantages, converting trajectories to training data,
and assembling training batches.
"""

import logging
from dataclasses import dataclass, field
from typing import Callable, List

import tinker
import torch
from tinker import TensorData
from tinker_cookbook.rl.types import StrategyId, Trajectory, TrajectoryGroup
from tinker_cookbook.supervised.common import (
    create_rightshifted_model_input_and_leftshifted_targets,
)
from tinker_cookbook.utils.misc_utils import all_same, safezip

logger = logging.getLogger(__name__)


def compute_advantages(
    trajectory_groups_P: List[TrajectoryGroup],
    normalize_advantages: bool = False,
) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups.

    Args:
        trajectory_groups_P: List of trajectory groups
        normalize_advantages: If True, standardize advantages (mean=0, std=1).
            If False (default), only center by mean.
    """
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        mean = rewards_G.mean()
        advantages_G = rewards_G - mean
        # Optionally normalize by std for unit variance
        if normalize_advantages:
            std = rewards_G.std()
            advantages_G = advantages_G / (std + 1e-8)
        advantages_P.append(advantages_G)

    return advantages_P


FlatObElem = int | tinker.ModelInputChunk
FlatOb = list[FlatObElem]


def _is_prefix(seq1: FlatOb, seq2: FlatOb) -> bool:
    """
    Check if seq1 is a prefix of seq2.
    """
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def _flat_ob_token_len(flat_ob: FlatOb) -> int:
    out = 0
    for elem in flat_ob:
        if isinstance(elem, int):
            out += 1
        else:
            out += elem.length
    return out


def _flat_ob_to_model_input(flat_ob: FlatOb) -> tinker.ModelInput:
    out: list[tinker.ModelInputChunk] = []
    current_text_chunk: list[int] = []

    def flush_text_chunk():
        if current_text_chunk:
            out.append(tinker.EncodedTextChunk(tokens=current_text_chunk))
            current_text_chunk.clear()

    for elem in flat_ob:
        if isinstance(elem, int):
            current_text_chunk.append(elem)
        else:
            flush_text_chunk()
            out.append(elem)
    flush_text_chunk()
    return tinker.ModelInput(chunks=out)


def _flatten_chunks(chunks: list[tinker.ModelInputChunk]) -> FlatOb:
    out: FlatOb = []
    for chunk in chunks:
        if isinstance(chunk, tinker.EncodedTextChunk):
            out.extend(chunk.tokens)
        else:
            out.append(chunk)
    return out


@dataclass
class SequenceAccumulator:
    """Accumulates tokens, logprobs, advantages, and mask for building training Datums."""

    full_sequence: FlatOb = field(default_factory=list)
    sampled_logprobs: list[float] = field(default_factory=list)
    advantages: list[float] = field(default_factory=list)
    mask: list[float] = field(default_factory=list)

    def clear(self):
        self.full_sequence = []
        self.sampled_logprobs = []
        self.advantages = []
        self.mask = []


def trajectory_to_data(
    traj: Trajectory,
    traj_advantage: float,
    context_transform: Callable[[tinker.ModelInput, int], tinker.ModelInput] | None = None,
) -> list[tinker.Datum]:
    """
    Return one or more Datum objects corresponding to the trajectory.
    If the sequence grows by appending, i.e., each successive observation contains
    the previous observation+action as a prefix, then we can return a single Datum.
    However, if we get a sequence that's not an extension of the previous sequence,
    then that results in a new Datum.

    For example, let O1 denote a chunk of observation tokens, and let A1 denote an action.

    Then let's say ob_ac_pairs is as follows.

    (O1, A1)
    (O1+A1+O2, A2)
    (O3, A3)

    Then we will merge the first two observation-action pairs into a single Datum,
    and the last observation-action pair into a separate Datum.
    """
    acc = SequenceAccumulator()

    def make_datum_from_state():
        all_tokens_T = _flat_ob_to_model_input(acc.full_sequence)
        input_tokens_T, target_tokens_T = create_rightshifted_model_input_and_leftshifted_targets(
            list(all_tokens_T.chunks)
        )
        sampled_logprobs_T = acc.sampled_logprobs[1:]
        advantages_T = acc.advantages[1:]
        mask_T = acc.mask[1:]
        assert (
            input_tokens_T.length
            == len(target_tokens_T)
            == len(sampled_logprobs_T)
            == len(advantages_T)
            == len(mask_T)
        )
        return tinker.Datum(
            model_input=input_tokens_T,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens_T)),
                "logprobs": TensorData.from_torch(torch.tensor(sampled_logprobs_T)),
                "advantages": TensorData.from_torch(torch.tensor(advantages_T)),
                "mask": TensorData.from_torch(torch.tensor(mask_T)),
            },
        )

    data: list[tinker.Datum] = []
    for i_transition, transition in enumerate(traj.transitions):
        ob = transition.ob
        if context_transform is not None:
            ob = context_transform(ob, i_transition)
        ob_flat = _flatten_chunks(ob.chunks)
        ac_with_logprobs = transition.ac
        if len(acc.full_sequence) == 0:
            delta_ob_flat = ob_flat
        elif _is_prefix(acc.full_sequence, ob_flat):
            delta_ob_flat = ob_flat[len(acc.full_sequence) :]
        else:
            data.append(make_datum_from_state())
            acc.clear()
            delta_ob_flat = ob_flat
        delta_ob_len = _flat_ob_token_len(delta_ob_flat)
        acc.full_sequence.extend(delta_ob_flat)
        acc.full_sequence.extend(ac_with_logprobs.tokens)
        acc.sampled_logprobs.extend(
            [0.0] * delta_ob_len + ac_with_logprobs.logprobs
        )
        acc.advantages.extend(
            [0] * delta_ob_len + [traj_advantage] * len(ac_with_logprobs.tokens)
        )
        acc.mask.extend([0.0] * delta_ob_len + [1.0] * len(ac_with_logprobs.tokens))

    if acc.full_sequence:
        data.append(make_datum_from_state())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P: List[torch.Tensor],
    strategy_weights: dict[StrategyId, float] | None = None,
) -> tuple[List[tinker.Datum], List[dict[str, int]]]:
    """Convert trajectories to training data format."""
    data_D: list[tinker.Datum] = []
    metadata_D: list[dict[str, int]] = []

    strategy_counts: dict[StrategyId, int] = {}
    if strategy_weights is not None:
        for traj_group in trajectory_groups_P:
            if traj_group.strategy_id is None:
                raise ValueError("strategy_weights provided but strategy_id is missing")
            strategy_counts[traj_group.strategy_id] = strategy_counts.get(
                traj_group.strategy_id, 0
            ) + len(traj_group.trajectories_G)
        missing = [sid for sid in strategy_counts if sid not in strategy_weights]
        if missing:
            raise ValueError(f"strategy_weights missing entries for: {missing}")
        unused = [sid for sid in strategy_weights if sid not in strategy_counts]
        if unused:
            logger.warning("strategy_weights contains unused entries: %s", unused)

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P)
    ):
        weight_scale = 1.0
        if strategy_weights is not None:
            assert traj_group.strategy_id is not None
            weight_scale = (
                strategy_weights[traj_group.strategy_id]
                / strategy_counts[traj_group.strategy_id]
            )
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G)
        ):
            # Build the full sequence from the trajectory
            new_data = trajectory_to_data(
                traj,
                float(traj_advantage) * weight_scale,
                context_transform=traj_group.context_transform,
            )
            data_D.extend(new_data)
            metadata_D.extend([dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data])

    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient")
        return trajectory_groups_P[0:1]  # return singleton list in case empty
        # list will cause problems
    return new_groups
