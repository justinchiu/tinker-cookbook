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
    ASK_SONNET_INSTRUCTION,
    ASK_SONNET_TOOL,
    ActionParser,
    AskSonnetMode,
    AskSonnetRenderer,
    ExternalLLMClient,
    ExternalLLMConfig,
    MessageManager,
    ObservationType,
    RolloutLogger,
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
        # User simulator LLM
        user_llm: str | None = None,
        # Logging
        rollout_logger: RolloutLogger | None = None,
    ):
        self.renderer = renderer
        self.domain = domain
        self.rollout_logger = rollout_logger
        self.task_id = task_id
        self.max_context_length = max_context_length
        self._context_exceeded = False
        self._current_obs_length = 0  # Track current observation length for pre-step check

        # Track ask_sonnet calls for reward computation
        self.ask_sonnet_call_count: int = 0
        self._last_action_was_ask_sonnet: bool = False  # Track consecutive ask_sonnet calls
        self.empty_advisor_responses: int = 0  # Track empty responses from advisor

        # Token cost tracking
        self.sonnet_input_tokens: int = 0
        self.sonnet_output_tokens: int = 0
        self.policy_input_tokens: int = 0
        self.policy_output_tokens: int = 0
        # Tau2 user-simulator token/cost tracking (available only once episode terminates)
        self.tau2_user_input_tokens: int = 0
        self.tau2_user_output_tokens: int = 0
        self.tau2_user_cost_usd: float = 0.0

        # Store ask_sonnet mode for step logic
        self.ask_sonnet_mode = ask_sonnet_mode

        # Initialize tau2 gym wrapper
        self.gym = Tau2GymWrapper(domain, task_id, user_llm=user_llm)
        logger.debug(
            "Tau2Env initialized for task %s with max_context_length=%s",
            task_id,
            max_context_length,
        )

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

        # Initialize action parser
        self.action_parser = ActionParser(renderer)

        # Initialize message manager
        initial_user_content = initial_obs if initial_obs else "(Customer connected, please greet them)"
        self.messages = MessageManager(
            system_prompt=system_prompt,
            initial_user_content=initial_user_content,
        )

    @property
    def stop_condition(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        """Get the initial observation for the policy."""
        model_input = self.renderer.build_generation_prompt(
            self.messages.messages, tools=self.tools
        )
        # Track current observation length for pre-step check
        self._current_obs_length = model_input.length

        # Check if initial observation already exceeds max_context_length
        if (
            self.max_context_length is not None
            and model_input.length > self.max_context_length
        ):
            logger.warning(
                "Initial observation length %d exceeds max %d for task %s, will terminate on first step",
                model_input.length,
                self.max_context_length,
                self.task_id,
            )

        return model_input, self.stop_condition

    async def step(self, action: Action) -> StepResult:
        """
        Step the environment with a policy action.

        Logic:
        1. Pre-check: terminate if current observation exceeded max_context_length
        2. Parse action
        3. If ask_sonnet + conditioning: call Sonnet, return observation (don't send to tau2)
        4. If ask_sonnet + direct: call Sonnet, send Sonnet's response to tau2
        5. Otherwise: send action to tau2
        """
        # Pre-check: terminate immediately if current observation exceeded max_context_length
        # This prevents the policy from being called on too-long observations
        if (
            self.max_context_length is not None
            and self._current_obs_length > self.max_context_length
        ):
            logger.warning(
                "Pre-step context length %d exceeded max %d for task %s, terminating",
                self._current_obs_length,
                self.max_context_length,
                self.task_id,
            )
            self._context_exceeded = True
            # Return current observation (can't build a new one without stepping tau2)
            current_obs = self.renderer.build_generation_prompt(
                self.messages.messages, tools=self.tools
            )
            return StepResult(
                next_observation=current_obs,
                next_stop_condition=self.stop_condition,
                episode_done=True,
                reward=0.0,
            )

        # Track policy output tokens (the action)
        action_tokens = action if isinstance(action, list) else []
        self.policy_output_tokens += len(action_tokens)

        # Parse action
        parsed = self.action_parser.parse(action)

        # Check for ask_sonnet
        if self.action_parser.is_ask_sonnet(parsed) and self.external_llm is not None:
            # Check for consecutive ask_sonnet calls (policy not following advice)
            if self._last_action_was_ask_sonnet:
                logger.warning(
                    "Consecutive ask_sonnet calls detected for task %s - terminating episode",
                    self.task_id,
                )
                # Add the second ask_sonnet call to messages so it's visible in logged trajectory
                self.messages.add_ask_sonnet_call(parsed.original_message)
                self.messages.add_tool_result(
                    "[Error]: Consecutive ask_sonnet calls are not allowed. "
                    "You must act on the advisor's advice before asking again.",
                    tool_call_id="ask_sonnet_call",
                )
                self._log_episode(reward=0.0)
                current_obs = self.renderer.build_generation_prompt(
                    self.messages.messages, tools=self.tools
                )
                return StepResult(
                    next_observation=current_obs,
                    next_stop_condition=self.stop_condition,
                    episode_done=True,
                    reward=0.0,
                )

            logger.info("ask_sonnet called, delegating to external LLM")
            self.ask_sonnet_call_count += 1
            self._last_action_was_ask_sonnet = True

            # Add ask_sonnet call to messages
            self.messages.add_ask_sonnet_call(parsed.original_message)

            # Call external LLM with usage tracking
            advisor_messages = self.ask_sonnet_renderer.render_for_advisor(
                self.messages.messages,
                self.tools,
                self.messages.system_prompt,
            )
            result = await self.external_llm.call_with_usage(advisor_messages)
            sonnet_response = result.content

            # Track Sonnet token usage
            self.sonnet_input_tokens += result.input_tokens
            self.sonnet_output_tokens += result.output_tokens

            # Handle empty advisor response
            if not sonnet_response or not sonnet_response.strip():
                self.empty_advisor_responses += 1
                logger.warning("Advisor returned empty response, providing error feedback to policy")
                # Return an error observation so the policy can continue
                error_msg = "[Advisor Error]: The advisor returned an empty response. Please proceed without advisor help."
                self.messages.add_tool_result(error_msg, tool_call_id="ask_sonnet_call")
                next_obs = self.renderer.build_generation_prompt(
                    self.messages.messages, tools=self.tools
                )
                self._current_obs_length = next_obs.length
                return StepResult(
                    next_observation=next_obs,
                    next_stop_condition=self.stop_condition,
                    episode_done=False,
                    reward=0.0,
                )

            # Add Sonnet's response using the renderer
            self.messages.add_sonnet_response(sonnet_response, self.ask_sonnet_renderer)

            if self.ask_sonnet_renderer.should_return_early():
                # Conditioning: return observation, wait for policy followup
                next_obs = self.renderer.build_generation_prompt(
                    self.messages.messages, tools=self.tools
                )
                # Track observation length for pre-step check on next iteration
                self._current_obs_length = next_obs.length
                return StepResult(
                    next_observation=next_obs,
                    next_stop_condition=self.stop_condition,
                    episode_done=False,
                    reward=0.0,
                )
            else:
                # Direct: send Sonnet's response to tau2 immediately
                self._last_action_was_ask_sonnet = False  # Reset - Sonnet took an action
                action_str = self.ask_sonnet_renderer.get_tau2_action(sonnet_response, None)
                # Add assistant message with proper <tool_call> format
                # (action_str is raw JSON, need to wrap it for the message)
                try:
                    json.loads(action_str)  # Check if it's valid JSON (tool call)
                    assistant_content = f"<tool_call>\n{action_str}\n</tool_call>"
                except json.JSONDecodeError:
                    # Plain text response
                    assistant_content = action_str
                self.messages.add_assistant(assistant_content)
                return await self._send_to_tau2(action_str)

        # Not ask_sonnet: add to messages and send to tau2
        self._last_action_was_ask_sonnet = False  # Reset flag
        self.messages.add_assistant_message_dict(parsed.original_message)
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

        # Track observation length for pre-step check on next iteration
        self._current_obs_length = next_obs.length

        # Track policy input tokens (the observation/prompt length)
        self.policy_input_tokens += next_obs.length

        # Check context length
        episode_done = result.terminated or result.truncated
        reward = result.reward

        if episode_done:
            self._maybe_capture_tau2_user_costs(result.info)

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

        # Log episode if done
        if episode_done:
            self._log_episode(reward)

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

    def _maybe_capture_tau2_user_costs(self, info: dict) -> None:
        """Capture Tau2 user-simulator token/cost totals from the final `simulation_run` info."""
        sim_run = info.get("simulation_run")
        if not sim_run:
            return

        try:
            sim = json.loads(sim_run) if isinstance(sim_run, str) else sim_run
        except Exception:
            logger.warning("Failed to parse tau2 simulation_run info")
            return

        if not isinstance(sim, dict) or not sim:
            return

        # Prefer the aggregate if present; otherwise fall back to summing message-level cost.
        user_cost = sim.get("user_cost")
        if isinstance(user_cost, (int, float)):
            self.tau2_user_cost_usd = float(user_cost)
        else:
            total_cost = 0.0
            for msg in sim.get("messages", []) or []:
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "user":
                    continue
                cost = msg.get("cost")
                if isinstance(cost, (int, float)):
                    total_cost += float(cost)
            self.tau2_user_cost_usd = total_cost

        prompt_tokens = 0
        completion_tokens = 0
        for msg in sim.get("messages", []) or []:
            if not isinstance(msg, dict):
                continue
            if msg.get("role") != "user":
                continue
            usage = msg.get("usage")
            if not isinstance(usage, dict):
                continue
            prompt_tokens += int(usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("completion_tokens") or 0)

        self.tau2_user_input_tokens = prompt_tokens
        self.tau2_user_output_tokens = completion_tokens

    def get_token_costs(self) -> dict:
        """Get token cost metrics for this episode."""
        return {
            "sonnet_input_tokens": self.sonnet_input_tokens,
            "sonnet_output_tokens": self.sonnet_output_tokens,
            "policy_input_tokens": self.policy_input_tokens,
            "policy_output_tokens": self.policy_output_tokens,
            "total_sonnet_tokens": self.sonnet_input_tokens + self.sonnet_output_tokens,
            "total_policy_tokens": self.policy_input_tokens + self.policy_output_tokens,
            "tau2_user_input_tokens": self.tau2_user_input_tokens,
            "tau2_user_output_tokens": self.tau2_user_output_tokens,
            "tau2_user_cost_usd": self.tau2_user_cost_usd,
        }

    def _log_episode(self, reward: float) -> None:
        """Log full conversation histories at episode end."""
        if not self.rollout_logger:
            return

        metadata = {
            "ask_sonnet_count": self.ask_sonnet_call_count,
            "empty_advisor_responses": self.empty_advisor_responses,
            "context_exceeded": self._context_exceeded,
            **self.get_token_costs(),
        }

        self.rollout_logger.log_episode(
            domain=self.domain,
            task_id=self.task_id,
            reward=reward,
            messages=self.messages.messages,
            metadata=metadata,
        )


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
    # User simulator LLM (default: gpt-4.1, use gpt-4o-mini for cheaper/faster)
    user_llm: str | None = None
    # Penalty per ask_sonnet call
    ask_sonnet_penalty: float = 0.0
    # Optional token/cost penalties (applied to final reward)
    sonnet_token_penalty_per_1k: float = 0.0
    tau2_user_token_penalty_per_1k: float = 0.0
    tau2_user_cost_penalty: float = 0.0
    # Logging
    rollout_logger: RolloutLogger | None = None

    async def make_envs(self) -> Sequence[Env]:
        """Create a group of tau2 environments with the same task."""
        env_domain = self.actual_domain or self.domain

        return list(
            await asyncio.gather(
                *[
                    asyncio.to_thread(
                        Tau2Env,
                        renderer=self.renderer,
                        domain=env_domain,
                        task_id=self.task_id,
                        max_context_length=self.max_context_length,
                        external_llm_model=self.external_llm_model,
                        external_llm_temperature=self.external_llm_temperature,
                        external_llm_max_tokens=self.external_llm_max_tokens,
                        ask_sonnet_mode=self.ask_sonnet_mode,
                        user_llm=self.user_llm,
                        rollout_logger=self.rollout_logger,
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
            ask_sonnet_penalty = self.ask_sonnet_penalty * ask_sonnet_count
            sonnet_token_penalty = self.sonnet_token_penalty_per_1k * (
                (tau2_env.sonnet_input_tokens + tau2_env.sonnet_output_tokens) / 1000.0
            )
            tau2_user_token_penalty = self.tau2_user_token_penalty_per_1k * (
                (tau2_env.tau2_user_input_tokens + tau2_env.tau2_user_output_tokens) / 1000.0
            )
            tau2_user_cost_penalty = self.tau2_user_cost_penalty * tau2_env.tau2_user_cost_usd
            total_penalty = (
                ask_sonnet_penalty + sonnet_token_penalty + tau2_user_token_penalty + tau2_user_cost_penalty
            )

            metrics = {
                "ask_sonnet_count": ask_sonnet_count,
                "ask_sonnet_penalty": ask_sonnet_penalty,
                "empty_advisor_responses": tau2_env.empty_advisor_responses,
                # Token costs
                "sonnet_input_tokens": tau2_env.sonnet_input_tokens,
                "sonnet_output_tokens": tau2_env.sonnet_output_tokens,
                "policy_input_tokens": tau2_env.policy_input_tokens,
                "policy_output_tokens": tau2_env.policy_output_tokens,
                "total_sonnet_tokens": tau2_env.sonnet_input_tokens + tau2_env.sonnet_output_tokens,
                "total_policy_tokens": tau2_env.policy_input_tokens + tau2_env.policy_output_tokens,
                "tau2_user_input_tokens": tau2_env.tau2_user_input_tokens,
                "tau2_user_output_tokens": tau2_env.tau2_user_output_tokens,
                "tau2_user_cost_usd": tau2_env.tau2_user_cost_usd,
                "sonnet_token_penalty": sonnet_token_penalty,
                "tau2_user_token_penalty": tau2_user_token_penalty,
                "tau2_user_cost_penalty": tau2_user_cost_penalty,
                "total_cost_penalty": total_penalty,
            }
            results.append((-total_penalty, metrics))

        return results

    def logging_tags(self) -> list[str]:
        """Return tags for logging/aggregation."""
        domain_tag = self.actual_domain or self.domain
        return ["tau2", domain_tag, self.task_id[:20]]


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
    # User simulator LLM (default: gpt-4.1, use gpt-4o-mini for cheaper/faster)
    user_llm: str | None = None
    ask_sonnet_penalty: float = 0.0
    sonnet_token_penalty_per_1k: float = 0.0
    tau2_user_token_penalty_per_1k: float = 0.0
    tau2_user_cost_penalty: float = 0.0
    # Logging
    rollout_logger: RolloutLogger | None = None

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        """Get a batch of environment group builders."""
        # Start new iteration for rollout logger (reset sampling counters)
        if self.rollout_logger:
            self.rollout_logger.start_iteration(index)

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
                user_llm=self.user_llm,
                ask_sonnet_penalty=self.ask_sonnet_penalty,
                sonnet_token_penalty_per_1k=self.sonnet_token_penalty_per_1k,
                tau2_user_token_penalty_per_1k=self.tau2_user_token_penalty_per_1k,
                tau2_user_cost_penalty=self.tau2_user_cost_penalty,
                rollout_logger=self.rollout_logger,
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
    # User simulator LLM (can use gpt-4o-mini for cheaper/faster)
    user_llm: str = "gpt-4.1"
    ask_sonnet_penalty: float = 0.0
    sonnet_token_penalty_per_1k: float = 0.0
    tau2_user_token_penalty_per_1k: float = 0.0
    tau2_user_cost_penalty: float = 0.0
    # Logging
    rollout_logger: RolloutLogger | None = None  # For training (sampled)
    eval_rollout_logger: RolloutLogger | None = None  # For evaluation (all)

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
            user_llm=self.user_llm,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
            sonnet_token_penalty_per_1k=self.sonnet_token_penalty_per_1k,
            tau2_user_token_penalty_per_1k=self.tau2_user_token_penalty_per_1k,
            tau2_user_cost_penalty=self.tau2_user_cost_penalty,
            rollout_logger=self.rollout_logger,
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
            user_llm=self.user_llm,
            ask_sonnet_penalty=self.ask_sonnet_penalty,
            sonnet_token_penalty_per_1k=self.sonnet_token_penalty_per_1k,
            tau2_user_token_penalty_per_1k=self.tau2_user_token_penalty_per_1k,
            tau2_user_cost_penalty=self.tau2_user_cost_penalty,
            rollout_logger=self.eval_rollout_logger,  # Use eval logger (logs all)
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
    # Logging
    log_dir: str | None = None,
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

    # Create rollout logger if log_dir specified
    rollout_logger = RolloutLogger(log_dir=log_dir, enabled=bool(log_dir)) if log_dir else None

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
        rollout_logger=rollout_logger,
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
