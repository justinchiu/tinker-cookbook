import asyncio
import gymnasium as gym
from tau2.gym.gym_agent import AgentGymEnv
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

import json
import logging
import os

# Configure LiteLLM (uses standard logging module, not loguru)
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)  # Only show warnings and errors

logger = logging.getLogger(__name__)

from tinker import ModelInput
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.evaluators import EvaluatorBuilder
from tinker_cookbook.renderers import Message, Renderer, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import (
    Action,
    Env,
    Observation,
    StepResult,
)


class Tau2Env(Env):
    def __init__(self, renderer: Renderer, domain: str, task_id: str, max_context_length: int | None = None):
        self.renderer = renderer
        self.domain = domain
        self.task_id = task_id
        self.max_context_length = max_context_length
        self._context_exceeded = False

        # Keeping the old Sonnet ID here for reference in case Tau2 toggles back:
        # user_llm="claude-sonnet-4-5-20250929"
        self.env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            user_llm="gpt-4.1-2025-04-14",
            user_llm_args={"temperature": 0.0, "max_tokens": 1024}
        )
        # Note: reset() is synchronous and may block, but we can't make __init__ async
        # For now, we'll leave this as-is since it only happens once per env
        obs, info = self.env.reset()

        domain_policy = self.env._get_policy()
        system_prompt = SYSTEM_PROMPT.format(domain_policy=domain_policy, agent_instruction=AGENT_INSTRUCTION)

        # Get tools from tau2 gym and convert to standard OpenAI format
        tools = self.env._get_tools()
        tool_jsons = [x.model_dump_json() for x in tools]
        tau2_tools = [json.loads(x) for x in tool_jsons]

        # Convert tau2 format to OpenAI format
        self.tools = []
        for tool in tau2_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("short_desc", "") or tool.get("long_desc", ""),
                    "parameters": tool.get("params", {})
                }
            }
            self.tools.append(openai_tool)

        # Store messages without manually injecting tools - let the renderer handle it
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        # Return the initial observation as a tokenized prompt with tools injected
        model_input = self.renderer.build_generation_prompt(self.messages, tools=self.tools)
        return model_input, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        # Parse the action to get the assistant's message
        assistant_message, parse_success = self.renderer.parse_response(action)

        # Add assistant's response to conversation
        self.messages.append(assistant_message)

        # Convert the assistant's response to the format expected by tau2 gym
        # Tau2 gym expects either:
        # - JSON format for tool calls: {"name": "tool_name", "arguments": {...}}
        # - Plain text for messages
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            # Send the first tool call as JSON (tau2 gym handles one tool at a time)
            # The renderer returns tau2 format: {"id": "...", "name": "...", "arguments": {...}}
            # We just need to extract name and arguments for tau2 gym
            tool_call = assistant_message["tool_calls"][0]

            # Build tau2 format (just name + arguments)
            tau2_tool_call = {
                "name": tool_call["name"],
                "arguments": tool_call["arguments"]
            }
            action_str = json.dumps(tau2_tool_call)
        else:
            # Send plain text, stripping any <tool_call> tags if present
            import re
            action_str = re.sub(r"<tool_call>.*?</tool_call>", "", assistant_message["content"], flags=re.DOTALL).strip()

        """
        # Debug: print what we're sending to tau2 gym
        print(f"\n[DEBUG] Sending to tau2 gym:")
        print(f"  action_str: {repr(action_str[:200])}..." if len(action_str) > 200 else f"  action_str: {repr(action_str)}")
        print(f"  assistant_message has tool_calls: {'tool_calls' in assistant_message and assistant_message['tool_calls']}")
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            print(f"  Number of tool_calls: {len(assistant_message['tool_calls'])}")
        """

        # Step the gym environment (wrap in thread to avoid blocking)
        obs, reward, terminated, truncated, info = await asyncio.to_thread(
            self.env.step, action_str
        )

        """
        # Debug: print observation details
        print(f"\n[DEBUG] Tau2 gym returned:")
        print(f"  obs type: {type(obs)}")
        print(f"  obs value: {repr(obs[:200])}..." if isinstance(obs, str) and len(obs) > 200 else f"  obs value: {repr(obs)}")
        print(f"  reward: {reward}")
        print(f"  terminated: {terminated}, truncated: {truncated}")
        """

        # Update conversation with new observation if there is one
        if obs and not (terminated or truncated):
            # Parse observation from tau2 gym format ("role: content")
            # Observations can be:
            # - "user: <text>" - user messages
            # - "tool: {...}" - tool results
            if obs.startswith("user: "):
                self.messages.append({"role": "user", "content": obs[6:]})  # Strip "user: " prefix
            elif obs.startswith("tool: "):
                self.messages.append({"role": "tool", "content": obs[6:]})  # Strip "tool: " prefix
            else:
                # Fallback: add as-is (shouldn't happen but just in case)
                self.messages.append({"role": "user", "content": obs})

        # Build next observation with tools injected - always provide it (like twenty_questions)
        next_obs = self.renderer.build_generation_prompt(self.messages, tools=self.tools)

        # Check if context length exceeded
        episode_done = terminated or truncated
        if self.max_context_length is not None and next_obs.length > self.max_context_length:
            logger.warning(
                "Context length %d exceeded max %d for task %s, terminating episode with reward=0",
                next_obs.length,
                self.max_context_length,
                self.task_id,
            )
            self._context_exceeded = True
            episode_done = True
            reward = 0.0

        # Return step result
        return StepResult(
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,  # Always provide stop condition
            episode_done=episode_done,
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
    actual_domain: str = None  # The real domain for when domain="all"
    max_context_length: int | None = None  # Max context length before failing episode

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task.

        Uses asyncio.to_thread to create envs in parallel since Tau2Env.__init__
        blocks (it calls AgentGymEnv.reset() which starts a thread and waits).
        """
        # Use actual_domain if provided (for domain="all"), otherwise use domain
        env_domain = self.actual_domain or self.domain

        # Create envs in parallel using thread pool to avoid blocking the event loop
        return list(await asyncio.gather(*[
            asyncio.to_thread(Tau2Env, self.renderer, env_domain, self.task_id, self.max_context_length)
            for _ in range(self.num_envs)
        ]))

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
    max_context_length: int | None = None  # Max context length before failing episode

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.tasks))

        return [
            Tau2EnvGroupBuilder(
                domain=self.domain,
                task_id=task.id,
                renderer=self.renderer,
                num_envs=self.group_size,
                actual_domain=getattr(task, '_actual_domain', None),  # Pass the actual domain if available
                max_context_length=self.max_context_length,
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
    domain: Literal["telecom", "airline", "retail", "mock", "telecom-workflow", "all"] = "all"
    seed: int = 0
    test_group_size: int = 1
    num_epochs: int = 1  # Number of epochs to train for

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

        # Repeat tasks for num_epochs
        train_tasks_with_epochs = train_tasks * self.num_epochs

        # Create train and test datasets
        train_dataset = Tau2Dataset(
            tasks=train_tasks_with_epochs,
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
        """Get tasks for the specified domain, honoring official train/test splits."""

        TRAIN_SPLIT = "train"
        TEST_SPLIT = "test"

        def load_tasks_for_domain(domain_name: str, split_name: str | None) -> list:
            tasks_loader = reg.registry.get_tasks_loader(domain_name)
            if split_name is None:
                tasks = tasks_loader()
            else:
                try:
                    tasks = tasks_loader(task_split_name=split_name)
                except TypeError:
                    logger.warning(
                        "Domain %s does not support split '%s'; using default set",
                        domain_name,
                        split_name,
                    )
                    tasks = tasks_loader()
                except ValueError as exc:
                    logger.warning(
                        "Domain %s missing split '%s' (%s); falling back to base",
                        domain_name,
                        split_name,
                        exc,
                    )
                    tasks = tasks_loader()

            for task in tasks:
                setattr(task, "_actual_domain", domain_name)
            return tasks

        if self.domain == "all":
            domains = ["telecom", "airline", "retail", "telecom-workflow"]
        else:
            domains = [self.domain]

        train_tasks: list = []
        test_tasks: list = []
        for domain_name in domains:
            train_tasks.extend(load_tasks_for_domain(domain_name, TRAIN_SPLIT))
            test_tasks.extend(load_tasks_for_domain(domain_name, TEST_SPLIT))

        import random

        rng = random.Random(self.seed)
        rng.shuffle(train_tasks)
        rng.shuffle(test_tasks)

        logger.info("=" * 60)
        if self.domain == "all":
            logger.info("TAU2 MULTI-DOMAIN DATASET (train split=train, test split=test)")
        else:
            logger.info(
                "TAU2 DATASET - %s domain (train split=train, test split=test)",
                self.domain,
            )

        logger.info("TEST TASKS (%d tasks for evaluation):", len(test_tasks))
        for task in test_tasks[:5]:
            logger.info("  TEST: %s", task.id)
        if len(test_tasks) > 5:
            logger.info("  ... and %d more test tasks", len(test_tasks) - 5)

        logger.info("TRAIN TASKS (%d tasks for training):", len(train_tasks))
        for task in train_tasks[:5]:
            logger.info("  TRAIN: %s", task.id)
        if len(train_tasks) > 5:
            logger.info("  ... and %d more train tasks", len(train_tasks) - 5)
        logger.info("=" * 60)

        return train_tasks, test_tasks


def build_tau_eval_builders(
    *,
    enabled: bool,
    model_name: str,
    renderer_name: str,
    domain: str,
    num_tasks: int | None,
    batch_size: int,
    group_size: int,
    max_tokens: int,
    temperature: float,
    task_seed: int,
    eval_name: str,
    max_context_length: int | None = None,
) -> list[EvaluatorBuilder]:
    """Construct Tau2 rollout evaluators for supervised recipes."""

    if not enabled:
        return []

    eval_dataset_builder = Tau2DatasetBuilder(
        batch_size=max(1, batch_size),
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        group_size=max(1, group_size),
        domain=domain,
        seed=task_seed,
        test_group_size=max(1, group_size),
        num_epochs=1,
    )

    # Build datasets ahead of time so evaluator builders stay lightweight at runtime
    _, raw_test_dataset = asyncio.run(eval_dataset_builder())
    tasks = list(raw_test_dataset.tasks)
    if num_tasks is not None:
        tasks = tasks[: max(1, num_tasks)]

    if not tasks:
        raise ValueError("Tau2 evaluation enabled but no tasks were loaded.")

    eval_dataset = Tau2Dataset(
        tasks=tasks,
        renderer=raw_test_dataset.renderer,
        domain=raw_test_dataset.domain,
        batch_size=min(len(tasks), max(1, batch_size)),
        group_size=max(1, group_size),
        max_context_length=max_context_length,
    )

    logger = logging.getLogger(__name__)
    logger.info(
        "Enabling Tau2 rollout eval '%s' on %d tasks (domain=%s, group_size=%d)",
        eval_name,
        len(tasks),
        domain,
        group_size,
    )

    def builder() -> RLTestSetEvaluator:
        return RLTestSetEvaluator(
            dataset=eval_dataset,
            max_tokens=max_tokens,
            temperature=temperature,
            name=eval_name,
        )

    return [builder]
