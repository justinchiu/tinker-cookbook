"""Tau2 environment for RL training with optional ask_sonnet support."""

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from typing import Literal, Sequence

import chz
import tau2.registry as reg
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.eval.evaluators import EvaluatorBuilder
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.recipes.taubench.components import (
    ActionParser,
    AskSonnetMode,
    AskSonnetRenderer,
    ExternalLLMClient,
    ExternalLLMConfig,
    MessageManager,
    ObservationType,
    Tau2GymWrapper,
    get_ask_sonnet_renderer,
)
from tinker_cookbook.renderers import Renderer, get_renderer
from tinker_cookbook.rl.metric_util import RLTestSetEvaluator
from tinker_cookbook.rl.types import (
    Action,
    Env,
    EnvGroupBuilder,
    Observation,
    RLDataset,
    RLDatasetBuilder,
    StepResult,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

logger = logging.getLogger(__name__)

# Special tool for delegating to external LLM
ASK_SONNET_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_sonnet",
        "description": "Delegate this turn to Claude Sonnet. Sonnet will see the full conversation and respond on your behalf.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

ASK_SONNET_INSTRUCTION = """

IMPORTANT: You have access to a special tool called `ask_sonnet` that delegates the current turn to a more capable AI assistant (Claude Sonnet). Use this tool when:
- You are unsure how to proceed with a complex request
- You need help understanding the customer's needs
- You want to verify your approach before taking an action
- The task requires careful reasoning or nuanced judgment

When you call `ask_sonnet`, Claude Sonnet will see the full conversation and respond on your behalf. Use this tool liberally when uncertain - it's better to ask for help than to make mistakes.

NOTE: Always greet the customer yourself on the first turn. Do not use `ask_sonnet` for the initial greeting - handle it directly, then use `ask_sonnet` for subsequent turns if needed."""


class Tau2Env(Env):
    """
    Tau2 environment for RL training.

    Wraps the tau2 gym environment with:
    - Proper message history management
    - Optional ask_sonnet support (delegate to external LLM)
    - Multiple ask_sonnet modes (direct injection, conditioning)
    """

    def __init__(
        self,
        renderer: Renderer,
        domain: str,
        task_id: str,
        max_context_length: int | None = None,
        # External LLM configuration
        external_llm_model: str | None = None,
        external_llm_temperature: float = 0.0,
        external_llm_max_tokens: int = 1024,
        # Ask sonnet mode
        ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION,
    ):
        self.renderer = renderer
        self.domain = domain
        self.task_id = task_id
        self.max_context_length = max_context_length
        self._context_exceeded = False

        # Track ask_sonnet calls for reward computation
        self.ask_sonnet_call_count: int = 0

        # State for conditioning mode
        self._awaiting_followup = False
        self._pending_sonnet_response: str | None = None

        # Initialize tau2 gym wrapper
        self.gym = Tau2GymWrapper(domain, task_id)

        # Get initial observation and system prompt
        initial_obs = self.gym.get_initial_observation()
        domain_policy = self.gym.env._get_policy()
        system_prompt = SYSTEM_PROMPT.format(
            domain_policy=domain_policy,
            agent_instruction=AGENT_INSTRUCTION,
        )

        # Get tools and add ask_sonnet if external LLM configured
        self.tools = self.gym.get_tools()

        # Setup external LLM if configured
        self.external_llm: ExternalLLMClient | None = None
        self.ask_sonnet_renderer: AskSonnetRenderer | None = None

        if external_llm_model is not None:
            # Add ask_sonnet instruction to system prompt
            system_prompt = system_prompt + ASK_SONNET_INSTRUCTION

            # Add ask_sonnet tool
            self.tools.append(ASK_SONNET_TOOL)

            # Create external LLM client
            self.external_llm = ExternalLLMClient(
                ExternalLLMConfig(
                    model=external_llm_model,
                    temperature=external_llm_temperature,
                    max_tokens=external_llm_max_tokens,
                )
            )

            # Create ask_sonnet renderer
            self.ask_sonnet_renderer = get_ask_sonnet_renderer(ask_sonnet_mode)

            # Build external system prompt with tools (excluding ask_sonnet)
            external_tools = [t for t in self.tools if t["function"]["name"] != "ask_sonnet"]
            external_system_prompt = self._build_system_prompt_with_tools(
                system_prompt, external_tools
            )
        else:
            external_system_prompt = system_prompt

        # Initialize action parser
        self.action_parser = ActionParser(renderer)

        # Initialize message manager
        initial_user_content = initial_obs if initial_obs else "(Customer connected, please greet them)"
        self.messages = MessageManager(
            system_prompt=system_prompt,
            external_system_prompt=external_system_prompt,
            initial_user_content=initial_user_content,
        )

    def _build_system_prompt_with_tools(self, system_prompt: str, tools: list[dict]) -> str:
        """Build system prompt with tools described in text (for external LLM)."""
        tool_descriptions = []
        for tool in tools:
            func = tool.get("function", tool)
            name = func.get("name", "unknown")
            desc = func.get("description", "")
            params = func.get("parameters", {})
            tool_descriptions.append(
                f"- {name}: {desc}\n  Parameters: {json.dumps(params, indent=2)}"
            )
        tools_text = "\n".join(tool_descriptions)

        return f"""{system_prompt}

# Available Tools

You have access to the following tools. To use a tool, respond with a JSON object in the following format:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

{tools_text}"""

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for the policy."""
        model_input = self.renderer.build_generation_prompt(
            self.messages.messages, tools=self.tools
        )
        return model_input, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Step the environment with a policy action.

        Handles:
        - Regular actions (tool calls, text)
        - ask_sonnet calls (delegate to external LLM)
        - Conditioning mode followup
        """
        # Handle conditioning mode followup
        if self._awaiting_followup:
            return await self._handle_followup(action)

        # Parse action
        parsed = self.action_parser.parse(action)

        # Check for ask_sonnet
        if self.action_parser.is_ask_sonnet(parsed) and self.external_llm is not None:
            return await self._handle_ask_sonnet(parsed)

        # Handle regular action
        return await self._handle_regular_action(parsed)

    async def _handle_ask_sonnet(self, parsed) -> StepResult:
        """Handle an ask_sonnet call."""
        logger.info("ask_sonnet called, delegating to external LLM")
        self.ask_sonnet_call_count += 1

        # Add ask_sonnet call to messages
        self.messages.add_ask_sonnet_call(parsed.original_message)

        # Call external LLM
        external_messages = self.messages.get_external_messages_for_llm()
        sonnet_response = await self.external_llm.call(external_messages)

        # Add Sonnet's response using the renderer
        self.messages.add_sonnet_response(sonnet_response, self.ask_sonnet_renderer)

        # Check if we should return early (conditioning mode)
        if self.ask_sonnet_renderer.should_return_early():
            self._awaiting_followup = True
            self._pending_sonnet_response = sonnet_response

            # Build observation for policy to see Sonnet's response
            next_obs = self.renderer.build_generation_prompt(
                self.messages.messages, tools=self.tools
            )
            return StepResult(
                next_observation=next_obs,
                next_stop_condition=self.stop_condition,
                episode_done=False,
                reward=0.0,
            )

        # Direct injection: use Sonnet's response as the action
        action_str = self.ask_sonnet_renderer.get_tau2_action(sonnet_response, None)
        return await self._send_to_tau2(action_str)

    async def _handle_followup(self, action: Action) -> StepResult:
        """Handle policy's followup after ask_sonnet in conditioning mode."""
        self._awaiting_followup = False

        # Parse followup action
        parsed = self.action_parser.parse(action)

        # Add followup to messages
        self.messages.add_assistant_message_dict(parsed.original_message, to_external=True)

        # Get tau2 action from followup
        action_str = self.ask_sonnet_renderer.get_tau2_action(
            self._pending_sonnet_response,
            parsed.original_message,
        )
        self._pending_sonnet_response = None

        return await self._send_to_tau2(action_str)

    async def _handle_regular_action(self, parsed) -> StepResult:
        """Handle a regular action (not ask_sonnet)."""
        # Add to messages
        self.messages.add_assistant_message_dict(parsed.original_message, to_external=True)

        # Convert to tau2 action
        action_str = self.action_parser.to_tau2_action(parsed)

        return await self._send_to_tau2(action_str)

    async def _send_to_tau2(self, action_str: str) -> StepResult:
        """Send action to tau2 gym and process the result."""
        # Step the gym
        result = await self.gym.step(action_str)

        # Process observation
        if result.raw_obs and not (result.terminated or result.truncated):
            self._process_observation(result)

        # Build next observation
        next_obs = self.renderer.build_generation_prompt(
            self.messages.messages, tools=self.tools
        )

        # Check context length
        episode_done = result.terminated or result.truncated
        reward = result.reward

        if self.max_context_length is not None and next_obs.length > self.max_context_length:
            logger.warning(
                "Context length %d exceeded max %d for task %s, terminating",
                next_obs.length,
                self.max_context_length,
                self.task_id,
            )
            self._context_exceeded = True
            episode_done = True
            reward = 0.0

        return StepResult(
            next_observation=next_obs,
            next_stop_condition=self.stop_condition,
            episode_done=episode_done,
            reward=reward,
        )

    def _process_observation(self, result) -> None:
        """Process tau2 observation and update message histories."""
        if result.obs_type == ObservationType.USER_MESSAGE:
            self.messages.add_user(result.obs_content)
        elif result.obs_type == ObservationType.TOOL_RESULT:
            self.messages.add_tool_result(result.obs_content)
        else:
            # Fallback: treat as user message
            self.messages.add_user(result.raw_obs)


# =============================================================================
# Helper functions
# =============================================================================


def construct_tau2_env(
    domain: str,
    task_id: str,
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507",
) -> Tau2Env:
    """Construct a Tau2Env with default settings."""
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    return Tau2Env(renderer, domain=domain, task_id=task_id)


# =============================================================================
# Dataset classes
# =============================================================================


@dataclass(frozen=True)
class Tau2EnvGroupBuilder(EnvGroupBuilder):
    """Group builder for tau2 environments."""

    domain: str
    task_id: str
    renderer: renderers.Renderer
    num_envs: int
    actual_domain: str = None
    max_context_length: int | None = None
    # External LLM configuration
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION
    # Penalty per ask_sonnet call
    ask_sonnet_penalty: float = 0.0

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task."""
        env_domain = self.actual_domain or self.domain

        return list(
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        Tau2Env,
                        self.renderer,
                        env_domain,
                        self.task_id,
                        self.max_context_length,
                        self.external_llm_model,
                        self.external_llm_temperature,
                        self.external_llm_max_tokens,
                        self.ask_sonnet_mode,
                    )
                    for _ in range(self.num_envs)
                ]
            )
        )

    async def compute_group_rewards(
        self, trajectory_group: list, env_group: Sequence[Env]
    ) -> list[tuple[float, dict]]:
        """Compute rewards with penalty for ask_sonnet calls."""
        results = []
        for env in env_group:
            tau2_env: Tau2Env = env
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
        return ["tau2", self.domain, self.task_id[:20]]


@dataclass(frozen=True)
class Tau2Dataset(RLDataset):
    """RL Dataset for tau2 environments."""

    tasks: list
    renderer: renderers.Renderer
    domain: str
    batch_size: int
    group_size: int
    max_context_length: int | None = None
    # External LLM configuration
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION
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
                actual_domain=getattr(task, "_actual_domain", None),
                max_context_length=self.max_context_length,
                external_llm_model=self.external_llm_model,
                external_llm_temperature=self.external_llm_temperature,
                external_llm_max_tokens=self.external_llm_max_tokens,
                ask_sonnet_mode=self.ask_sonnet_mode,
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
    num_epochs: int = 1
    max_context_length: int | None = 16384
    # External LLM configuration
    external_llm_model: str | None = None
    external_llm_temperature: float = 0.0
    external_llm_max_tokens: int = 1024
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION
    ask_sonnet_penalty: float = 0.0

    async def __call__(self) -> tuple[RLDataset, RLDataset]:
        """Build train and test datasets."""
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)

        if self.renderer_name is None:
            renderer_name = get_recommended_renderer_name(self.model_name_for_tokenizer)
        else:
            renderer_name = self.renderer_name

        renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

        train_tasks, test_tasks = self._get_train_and_test_tasks()
        train_tasks_with_epochs = train_tasks * self.num_epochs

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
            ask_sonnet_mode=self.ask_sonnet_mode,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
        )

        test_dataset = Tau2Dataset(
            tasks=test_tasks,
            renderer=renderer,
            domain=self.domain,
            batch_size=len(test_tasks),
            group_size=self.test_group_size,
            max_context_length=self.max_context_length,
            external_llm_model=self.external_llm_model,
            external_llm_temperature=self.external_llm_temperature,
            external_llm_max_tokens=self.external_llm_max_tokens,
            ask_sonnet_mode=self.ask_sonnet_mode,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
        )

        return train_dataset, test_dataset

    def _get_train_and_test_tasks(self):
        """Get tasks for the specified domain, honoring official train/test splits."""
        import random

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


# =============================================================================
# Evaluator builders
# =============================================================================


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
    # External LLM configuration
    external_llm_model: str | None = None,
    external_llm_temperature: float = 0.0,
    external_llm_max_tokens: int = 1024,
    ask_sonnet_mode: AskSonnetMode = AskSonnetMode.DIRECT_INJECTION,
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
        ask_sonnet_mode=ask_sonnet_mode,
    )

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
        ask_sonnet_mode=ask_sonnet_mode,
    )

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
