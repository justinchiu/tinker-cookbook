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

        self.env = AgentGymEnv(
            domain=domain,
            task_id=task_id
            # Use default user_llm (GPT-4) for now
            # user_llm="gpt-5-nano",  # This model returns empty responses
            # user_llm_args={"temperature": 1, "max_tokens": 1024}
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
            tool_call = assistant_message["tool_calls"][0]
            action_str = json.dumps(tool_call)
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

        # Return step result
        return StepResult(
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,  # Always provide stop condition
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
    actual_domain: str = None  # The real domain for when domain="all"

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task."""
        # Use actual_domain if provided (for domain="all"), otherwise use domain
        env_domain = self.actual_domain or self.domain
        return [
            Tau2Env(self.renderer, env_domain, self.task_id)
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
                num_envs=self.group_size,
                actual_domain=getattr(task, '_actual_domain', None)  # Pass the actual domain if available
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
        """Get tasks for the specified domain, split for train/test."""
        # Handle "all" domain to load tasks from all available domains
        if self.domain == "all":
            all_tasks = []
            domains = ["telecom", "airline", "retail", "telecom-workflow"]  # Exclude mock by default

            for domain in domains:
                tasks_loader = reg.registry.get_tasks_loader(domain)
                domain_tasks = tasks_loader()
                # Wrap tasks with domain info for "all" mode
                for task in domain_tasks:
                    # Store the actual domain with the task
                    task._actual_domain = domain
                all_tasks.extend(domain_tasks)  # Always use ALL tasks
        else:
            # Single domain - use the registry's task loader directly
            tasks_loader = reg.registry.get_tasks_loader(self.domain)
            all_tasks = tasks_loader()  # Always use ALL tasks
            # Mark tasks with their domain for consistency
            for task in all_tasks:
                task._actual_domain = self.domain

        # Shuffle all tasks before splitting (seeded permutation)
        import random
        random.Random(self.seed).shuffle(all_tasks)

        # Use all tasks for both train and test (no split)
        test_tasks = all_tasks
        train_tasks = all_tasks

        # Log the task ID split for debugging
        import logging
        logger = logging.getLogger(__name__)

        logger.info("="*60)
        if self.domain == "all":
            logger.info(f"TAU2 MULTI-DOMAIN DATASET - telecom, airline, retail (ALL TASKS)")
            # Count tasks by domain
            from collections import defaultdict
            domain_counts = defaultdict(int)
            for task in all_tasks:
                # Extract domain from task ID (format: domain_category_scenario)
                domain_prefix = task.id.split('_')[0] if '_' in task.id else "unknown"
                if domain_prefix in ["mobile", "service", "roaming", "bill", "internet", "mms"]:
                    domain_counts["telecom"] += 1
                elif domain_prefix in ["flight", "baggage", "seat", "booking", "meal"]:
                    domain_counts["airline"] += 1
                elif domain_prefix in ["return", "order", "product", "shipping", "payment"]:
                    domain_counts["retail"] += 1
                else:
                    domain_counts["other"] += 1

            logger.info(f"Task distribution across domains:")
            for domain, count in sorted(domain_counts.items()):
                logger.info(f"  {domain}: {count} tasks")
        else:
            logger.info(f"TAU2 DATASET - {self.domain} domain (ALL TASKS)")

        logger.info("="*60)
        logger.info(f"TEST TASKS ({len(test_tasks)} tasks for evaluation):")
        for task in test_tasks:
            logger.info(f"  TEST: {task.id}")

        logger.info(f"TRAIN TASKS ({len(train_tasks)} tasks for training):")
        for i, task in enumerate(train_tasks):
            if i < 5:  # Show first 5 to avoid spam
                logger.info(f"  TRAIN: {task.id}")
            elif i == 5:
                logger.info(f"  ... and {len(train_tasks) - 5} more train tasks")
                break
        logger.info("="*60)

        return train_tasks, test_tasks
