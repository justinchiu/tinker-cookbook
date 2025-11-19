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
    def __init__(self, renderer: Renderer):
        self.env = AgentGymEnv(domain="telecom", task_id="[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]")
        self.renderer = renderer
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


def construct_tau2_env():
    # Use a default model and renderer for now
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)
    return Tau2Env(renderer)
