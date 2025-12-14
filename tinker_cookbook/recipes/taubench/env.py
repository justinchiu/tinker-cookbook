import asyncio
import gymnasium as gym
from tau2.gym.gym_agent import AgentGymEnv
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

import json
import logging
import os

import litellm

from tinker_cookbook.recipes.taubench.message_converters import (
    convert_messages_for_openai,
)

# Configure LiteLLM (uses standard logging module, not loguru)
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)  # Only show warnings and errors

# Special tool for delegating to external LLM
ASK_SONNET_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_sonnet",
        "description": "Delegate this turn to Claude Sonnet. Sonnet will see the full conversation and respond on your behalf.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}

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
    def __init__(
        self,
        renderer: Renderer,
        domain: str,
        task_id: str,
        max_context_length: int | None = None,
        external_llm_model: str | None = None,
        external_llm_temperature: float = 0.0,
        external_llm_max_tokens: int = 1024,
    ):
        self.renderer = renderer
        self.domain = domain
        self.task_id = task_id
        self.max_context_length = max_context_length
        self._context_exceeded = False

        # External LLM configuration for ask_sonnet
        self.external_llm_model = external_llm_model
        self.external_llm_temperature = external_llm_temperature
        self.external_llm_max_tokens = external_llm_max_tokens

        # Separate message history for external LLM (keeps raw tool_calls with IDs)
        # This is needed because Anthropic requires tool_use_id to match tool_result
        self.external_llm_messages: list[dict] = []

        # Track ask_sonnet calls for reward computation
        self.ask_sonnet_call_count: int = 0

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

        # Add ask_sonnet instruction if external LLM is configured
        if self.external_llm_model is not None:
            ask_sonnet_instruction = """

IMPORTANT: You have access to a special tool called `ask_sonnet` that delegates the current turn to a more capable AI assistant (Claude Sonnet). Use this tool when:
- You are unsure how to proceed with a complex request
- You need help understanding the customer's needs
- You want to verify your approach before taking an action
- The task requires careful reasoning or nuanced judgment

When you call `ask_sonnet`, Claude Sonnet will see the full conversation and respond on your behalf. Use this tool liberally when uncertain - it's better to ask for help than to make mistakes."""
            system_prompt = system_prompt + ask_sonnet_instruction

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

        # Add ask_sonnet tool if external LLM is configured
        if self.external_llm_model is not None:
            self.tools.append(ASK_SONNET_TOOL)

        # Store messages without manually injecting tools - let the renderer handle it
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": obs},
        ]

        # Initialize external LLM messages with same initial state
        self.external_llm_messages = [
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

    async def _call_external_llm(self) -> dict:
        """
        Call external LLM (e.g., Sonnet) with the current conversation context.
        Returns an assistant message dict with content in Qwen format (tool calls as <tool_call> tags).
        Also updates self.external_llm_messages with raw format for future calls.
        """
        if self.external_llm_model is None:
            raise ValueError("external_llm_model not configured but ask_sonnet was called")

        # Build tools for external LLM (excluding ask_sonnet)
        external_tools = [t for t in self.tools if t["function"]["name"] != "ask_sonnet"]

        # Use external_llm_messages which preserves raw format (including tool_call IDs)
        messages_for_llm = convert_messages_for_openai(self.external_llm_messages)

        logger.info(
            "Calling external LLM (%s) with %d messages and %d tools",
            self.external_llm_model,
            len(messages_for_llm),
            len(external_tools),
        )
        # Debug: log messages being sent with content details
        for i, msg in enumerate(messages_for_llm):
            content = msg.get("content")
            content_preview = None
            if content:
                content_preview = content[:50] + "..." if len(content) > 50 else content
            logger.debug(
                "Message %d: role=%s, content=%r, has_tool_calls=%s",
                i, msg.get("role"), content_preview,
                "tool_calls" in msg and bool(msg.get("tool_calls"))
            )

        try:
            response = await litellm.acompletion(
                model=self.external_llm_model,
                messages=messages_for_llm,
                tools=external_tools if external_tools else None,
                max_tokens=self.external_llm_max_tokens,
                temperature=self.external_llm_temperature,
            )
        except Exception as e:
            # Print detailed debug info on error
            print("\n" + "=" * 80)
            print("ERROR calling external LLM - dumping messages:")
            print("=" * 80)
            for i, msg in enumerate(messages_for_llm):
                print(f"\n--- Message {i} ---")
                print(f"  role: {msg.get('role')}")
                content = msg.get('content')
                if content is not None:
                    print(f"  content type: {type(content).__name__}")
                    print(f"  content empty: {not content}")
                    print(f"  content repr: {repr(content[:200]) if len(str(content)) > 200 else repr(content)}")
                else:
                    print(f"  content: None")
                if msg.get('tool_calls'):
                    print(f"  tool_calls: {len(msg['tool_calls'])} calls")
                    for j, tc in enumerate(msg['tool_calls']):
                        print(f"    [{j}] id={tc.get('id')}, name={tc.get('function', {}).get('name')}")
                if msg.get('tool_call_id'):
                    print(f"  tool_call_id: {msg['tool_call_id']}")
            print("=" * 80 + "\n")
            raise

        choice = response.choices[0]
        message = choice.message

        # Store raw response in external_llm_messages (preserves tool_calls with IDs)
        # Convert litellm message to dict and fix empty content (Anthropic rejects empty content)
        raw_message = message.model_dump()
        if not raw_message.get("content") and raw_message.get("tool_calls"):
            # Remove empty/None content for tool-call-only messages
            raw_message.pop("content", None)
        self.external_llm_messages.append(raw_message)

        # Convert to Qwen format for training (tool calls as <tool_call> tags in content)
        result: dict = {"role": "assistant", "content": message.content or ""}

        if message.tool_calls:
            tool_call_strs = []
            for tc in message.tool_calls:
                tool_call_dict = {
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                }
                tool_call_strs.append(f"<tool_call>\n{json.dumps(tool_call_dict)}\n</tool_call>")

            if result["content"]:
                result["content"] = result["content"] + "\n" + "\n".join(tool_call_strs)
            else:
                result["content"] = "\n".join(tool_call_strs)

        logger.info(
            "External LLM (%s) response: %s",
            self.external_llm_model,
            result["content"][:100] + "..." if len(result.get("content", "")) > 100 else result.get("content", "")
        )

        return result

    async def step(self, action: Action) -> StepResult:
        # Parse the action to get the assistant's message
        assistant_message, parse_success = self.renderer.parse_response(action)

        # Check if this is an ask_sonnet action - delegate to external LLM
        sonnet_response = None
        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            tool_call = assistant_message["tool_calls"][0]
            if tool_call.function.name == "ask_sonnet":
                logger.info("ask_sonnet called, delegating to external LLM")
                self.ask_sonnet_call_count += 1

                # Add the ask_sonnet call to messages first
                self.messages.append(assistant_message)

                # Call external LLM
                sonnet_response = await self._call_external_llm()

                # Add Sonnet's response as a tool result
                tool_result_msg = {
                    "role": "tool",
                    "content": sonnet_response.get("content", ""),
                    "tool_call_id": "ask_sonnet_call",
                }
                self.messages.append(tool_result_msg)

                # Use Sonnet's response for the tau2 gym action
                assistant_message = sonnet_response

        # Add assistant's response to conversation (if not ask_sonnet, which already added messages above)
        if sonnet_response is None:
            self.messages.append(assistant_message)

        # Convert the assistant's response to the format expected by tau2 gym
        # Tau2 gym expects either:
        # - JSON format for tool calls: {"name": "tool_name", "arguments": {...}}
        # - Plain text for messages
        import re

        # Check for tool calls - either in tool_calls field (from renderer.parse_response)
        # or in <tool_call> tags in content (from external LLM)
        tool_call_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", assistant_message.get("content", ""), flags=re.DOTALL)

        if "tool_calls" in assistant_message and assistant_message["tool_calls"]:
            # From renderer.parse_response - pydantic ToolCall with nested function.name, function.arguments
            tool_call = assistant_message["tool_calls"][0]
            tau2_tool_call = {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments)
            }
            action_str = json.dumps(tau2_tool_call)
        elif tool_call_match:
            # From external LLM - <tool_call> tags in content
            # Parse the JSON inside the tags (already has name + arguments)
            action_str = tool_call_match.group(1)
        else:
            # Plain text - strip any <tool_call> tags just in case
            action_str = re.sub(r"<tool_call>.*?</tool_call>", "", assistant_message.get("content", ""), flags=re.DOTALL).strip()

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
                user_content = obs[6:]  # Strip "user: " prefix
                # Anthropic requires non-empty content
                if not user_content:
                    user_content = "(waiting)"
                user_msg = {"role": "user", "content": user_content}
                self.messages.append(user_msg)
                self.external_llm_messages.append(user_msg)
            elif obs.startswith("tool: "):
                tool_content = obs[6:]  # Strip "tool: " prefix

                # Get the tool_call_id from the raw external_llm_messages (has actual IDs)
                tool_call_id = None
                for msg in reversed(self.external_llm_messages):
                    if msg.get("role") == "assistant" and msg.get("tool_calls"):
                        # Raw format from litellm has tool_calls as list of dicts
                        tool_call_id = msg["tool_calls"][0].get("id")
                        break

                # Anthropic requires non-empty content for tool results
                if not tool_content:
                    tool_content = "(empty result)"

                tool_msg = {
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call_id or "unknown",
                }
                self.messages.append(tool_msg)
                self.external_llm_messages.append(tool_msg)
            else:
                # Fallback: add as-is (shouldn't happen but just in case)
                fallback_msg = {"role": "user", "content": obs}
                self.messages.append(fallback_msg)
                self.external_llm_messages.append(fallback_msg)

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
    # External LLM configuration for ask_sonnet
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    # Penalty per ask_sonnet call (subtracted from reward)
    ask_sonnet_penalty: float = 0.0

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task.

        Uses asyncio.to_thread to create envs in parallel since Tau2Env.__init__
        blocks (it calls AgentGymEnv.reset() which starts a thread and waits).
        """
        # Use actual_domain if provided (for domain="all"), otherwise use domain
        env_domain = self.actual_domain or self.domain

        # Create envs in parallel using thread pool to avoid blocking the event loop
        return list(await asyncio.gather(*[
            asyncio.to_thread(
                Tau2Env,
                self.renderer,
                env_domain,
                self.task_id,
                self.max_context_length,
                self.external_llm_model,
                self.external_llm_temperature,
                self.external_llm_max_tokens,
            )
            for _ in range(self.num_envs)
        ]))

    async def compute_group_rewards(
        self, trajectory_group: list, env_group: Sequence[Env]
    ) -> list[tuple[float, dict]]:
        """Compute rewards with penalty for ask_sonnet calls.

        The penalty is subtracted from the final reward for each ask_sonnet call.
        This encourages the model to learn to solve tasks itself rather than
        always delegating to the external LLM.
        """
        results = []
        for env in env_group:
            tau2_env = env  # type: Tau2Env
            ask_sonnet_count = tau2_env.ask_sonnet_call_count
            penalty = self.ask_sonnet_penalty * ask_sonnet_count

            metrics = {
                "ask_sonnet_count": ask_sonnet_count,
                "ask_sonnet_penalty": penalty,
            }
            results.append((-penalty, metrics))

        return results

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
    # External LLM configuration for ask_sonnet
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    # Penalty per ask_sonnet call
    ask_sonnet_penalty: float = 0.0

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
                external_llm_model=self.external_llm_model,
                external_llm_temperature=self.external_llm_temperature,
                external_llm_max_tokens=self.external_llm_max_tokens,
                ask_sonnet_penalty=self.ask_sonnet_penalty,
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
    max_context_length: int | None = 16384  # Fail episode if context exceeds this
    # External LLM configuration for ask_sonnet
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    # Penalty per ask_sonnet call (subtracted from reward)
    ask_sonnet_penalty: float = 0.0

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
            max_context_length=self.max_context_length,
            external_llm_model=self.external_llm_model,
            external_llm_temperature=self.external_llm_temperature,
            external_llm_max_tokens=self.external_llm_max_tokens,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
        )

        test_dataset = Tau2Dataset(
            tasks=test_tasks,
            renderer=renderer,
            domain=self.domain,
            batch_size=len(test_tasks),  # Single batch for test
            group_size=self.test_group_size,
            max_context_length=self.max_context_length,
            external_llm_model=self.external_llm_model,
            external_llm_temperature=self.external_llm_temperature,
            external_llm_max_tokens=self.external_llm_max_tokens,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
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
    # External LLM (ask_sonnet) configuration
    external_llm_model: str | None = None,
    external_llm_temperature: float = 0.0,
    external_llm_max_tokens: int = 1024,
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
        external_llm_model=external_llm_model,
        external_llm_temperature=external_llm_temperature,
        external_llm_max_tokens=external_llm_max_tokens,
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
        external_llm_model=external_llm_model,
        external_llm_temperature=external_llm_temperature,
        external_llm_max_tokens=external_llm_max_tokens,
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
