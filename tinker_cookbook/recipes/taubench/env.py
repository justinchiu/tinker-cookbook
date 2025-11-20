import gymnasium as gym
from tau2.gym.gym_agent import AgentGymEnv

from tinker import ModelInput
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.types import (
    Action,
    Env,
    Observation,
    StepResult,
)


class Tau2Env(Env):
    def __init__(self, renderer: Renderer, domain: str, task_id: str):
        self.renderer = renderer
        self.domain = domain
        self.task_id = task_id

        self.env = AgentGymEnv(domain=domain, task_id=task_id)
        obs, info = self.env.reset()
        self.messages = [
            {"role": "system", "content": "You are a helpful assistant helping a user with their task."},
            {"role": "user", "content": obs},
        ]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Return the initial observation as a tokenized prompt
        model_input = self.renderer.build_generation_prompt(self.messages)
        return model_input, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        # Parse the action to get the assistant's message
        assistant_message, parse_success = self.renderer.parse_response(action)

        # Add assistant's response to conversation
        self.messages.append(assistant_message)

        # Convert the assistant's response to a string for the gym environment
        action_str = assistant_message["content"]

        # Step the gym environment
        obs, reward, terminated, truncated, info = self.env.step(action_str)

        # Update conversation with new observation if there is one
        if obs and not (terminated or truncated):
            self.messages.append({"role": "user", "content": obs})

        # Build next observation
        next_obs = self.renderer.build_generation_prompt(self.messages) if not (terminated or truncated) else None

        # Return step result
        return StepResult(
            next_observation=next_obs,
            next_stop_condition=self.stop_condition if not (terminated or truncated) else None,
            episode_done=(terminated or truncated),
            reward=reward,
        )


def construct_tau2_env(domain: str, task_id: str, model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"):
    # Use a default model and renderer for now
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    return Tau2Env(renderer, domain=domain, task_id=task_id)


# Dataset classes following the twenty_questions pattern

import math
from dataclasses import dataclass
from functools import partial
from typing import Literal, Sequence

import chz
import tau2.registry as reg
import tinker
from tinker_cookbook import renderers
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
)


@dataclass(frozen=True)
class Tau2EnvGroupBuilder(EnvGroupBuilder):
    """Group builder for tau2 environments."""
    domain: str
    task_id: str
    renderer: renderers.Renderer
    num_envs: int

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task."""
        return [
            Tau2Env(self.renderer, self.domain, self.task_id)
            for _ in range(self.num_envs)
        ]

    def logging_tags(self) -> list[str]:
        """Return tags for logging/aggregation."""
        # Return task ID as tag for aggregating metrics
        return ["tau2", self.domain, self.task_id[:20]]  # Truncate task ID if too long


@dataclass(frozen=True)
class Tau2Dataset(RLDataset):
    """RL Dataset for tau2 environments."""
    tasks: list
    renderer: renderers.Renderer
    domain: str
    batch_size: int
    group_size: int

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.tasks))

        return [
            Tau2EnvGroupBuilder(
                domain=self.domain,
                task_id=task.id,
                renderer=self.renderer,
                num_envs=self.group_size
            )
            for task in self.tasks[batch_start:batch_end]
        ]

    def __len__(self) -> int:
        """Return number of batches."""
        return math.ceil(len(self.tasks) / self.batch_size)


@chz.chz
class Tau2DatasetBuilder(RLDatasetBuilder):
    """Builder for tau2 RL datasets."""
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str | None = None
    group_size: int = 1
    domain: Literal["telecom", "airline", "retail", "mock"] = "telecom"
    task_set: Literal["default", "full", "small"] = "default"
    seed: int = 0
    test_group_size: int = 32

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        """Build train and test datasets."""
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)

        # Use recommended renderer if not specified
        if self.renderer_name is None:
            renderer_name = get_recommended_renderer_name(self.model_name_for_tokenizer)
        else:
            renderer_name = self.renderer_name

        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        # Get tasks based on domain and task set
        train_tasks, test_tasks = self._get_train_and_test_tasks()

        # Create train and test datasets
        train_dataset = Tau2Dataset(
            tasks=train_tasks,
            renderer=renderer,
            domain=self.domain,
            batch_size=self.batch_size,
            group_size=self.group_size,
        )

        test_dataset = Tau2Dataset(
            tasks=test_tasks,
            renderer=renderer,
            domain=self.domain,
            batch_size=len(test_tasks),  # Single batch for test
            group_size=self.test_group_size,
        )

        return train_dataset, test_dataset

    def _get_train_and_test_tasks(self):
        """Get tasks for the specified domain and task set, split for train/test."""
        # Use the registry's task loader directly to get the correct tasks
        tasks_loader = reg.registry.get_tasks_loader(self.domain)
        all_tasks = tasks_loader()

        # For task_set filtering, just use simple size limits
        if self.task_set == "small":
            # Use first 20 tasks for small set
            all_tasks = all_tasks[:20]
        elif self.task_set == "full":
            # Use all tasks
            pass
        else:  # default
            # Use first 50 tasks for default
            all_tasks = all_tasks[:50]

        # Split tasks for train and test
        # Take first 10% for test (or at least 1 task)
        test_size = max(1, len(all_tasks) // 10)
        test_tasks = all_tasks[:test_size]
        train_tasks = all_tasks[test_size:]

        # Shuffle train tasks if seed is provided
        if self.seed:
            import random
            random.Random(self.seed).shuffle(train_tasks)

        return train_tasks, test_tasks
