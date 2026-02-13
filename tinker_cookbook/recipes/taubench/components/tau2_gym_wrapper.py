"""Tau2GymWrapper - Thin wrapper around tau2 gym with cleaner interface."""

import asyncio
import json
import logging

from tau2.gym.gym_agent import AgentGymEnv

from tinker_cookbook.recipes.taubench.components.types import (
    ObservationType,
    Tau2StepResult,
)

logger = logging.getLogger(__name__)


class Tau2GymWrapper:
    """
    Wraps tau2 gym environment with a cleaner interface.

    Handles:
    - Environment initialization
    - Async stepping
    - Observation parsing
    - Tool extraction
    """

    def __init__(self, domain: str, task_id: str, user_llm: str | None = None):
        self.domain = domain
        self.task_id = task_id

        self.env = AgentGymEnv(
            domain=domain,
            task_id=task_id,
            user_llm=user_llm,
        )

    def get_initial_observation(self) -> str:
        obs, _ = self.env.reset()
        return obs

    def get_system_prompt(self) -> str:
        return self.env._get_system_prompt()

    def get_tools(self) -> list[dict]:
        """Get tools from the gym in OpenAI format."""
        tools = self.env._get_tools()
        tool_jsons = [x.model_dump_json() for x in tools]
        tau2_tools = [json.loads(x) for x in tool_jsons]

        openai_tools = []
        for tool in tau2_tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("short_desc", "") or tool.get("long_desc", ""),
                    "parameters": tool.get("params", {}),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    async def step(self, action: str) -> Tau2StepResult:
        """Step the gym environment with an action."""
        try:
            obs, reward, terminated, truncated, info = await asyncio.to_thread(
                self.env.step, action
            )
        except json.JSONDecodeError as e:
            logger.warning(
                "JSONDecodeError parsing action: %s. Action was: %r",
                e,
                action[:200],
            )
            obs = f"tool: Error: Invalid JSON in action - {e}"
            reward = 0.0
            terminated = False
            truncated = False
            info = {}

        obs_type, obs_content = self._parse_observation(obs, terminated=terminated)

        logger.info(
            "Tau2 returned: obs=%r (len=%d), terminated=%s, truncated=%s",
            obs[:100] + "..." if len(obs) > 100 else obs,
            len(obs),
            terminated,
            truncated,
        )

        return Tau2StepResult(
            obs_type=obs_type,
            obs_content=obs_content,
            raw_obs=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _parse_observation(self, obs: str, terminated: bool = False) -> tuple[ObservationType, str]:
        """Parse tau2 observation into type and content."""
        if obs.startswith("user: "):
            return ObservationType.USER_MESSAGE, obs[6:]
        elif obs.startswith("tool: "):
            return ObservationType.TOOL_RESULT, obs[6:]
        else:
            if not (terminated and obs == ""):
                logger.warning(
                    "Unexpected obs format (not user: or tool:): %r",
                    obs[:100] + "..." if len(obs) > 100 else obs,
                )
            return ObservationType.OTHER, obs
