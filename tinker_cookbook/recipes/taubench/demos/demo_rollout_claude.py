#!/usr/bin/env python3
"""
Demo rollout script for taubench using Claude Sonnet 4.5 as both agent and user LLM.

Uses litellm to call Claude API (same as tau2 internally).
"""

import asyncio
import json
import logging
import sys
from dataclasses import dataclass

import chz
import litellm
import tau2.registry as reg
from tau2.gym.gym_agent import AgentGymEnv
from tau2.agent.llm_agent import AGENT_INSTRUCTION, SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure LiteLLM logging
litellm_logger = logging.getLogger("LiteLLM")
litellm_logger.setLevel(logging.WARNING)


@chz.chz
class Config:
    """Configuration for Claude-based demo rollout."""
    # Claude model for agent (the one being evaluated/trained)
    agent_model: str = "claude-sonnet-4-5-20250929"
    # Claude model for user simulation
    user_model: str = "claude-sonnet-4-5-20250929"

    domain: str = "retail"
    task_id: str | None = None  # Will pick first task from domain if not specified

    max_tokens: int = 1024
    temperature: float = 1.0
    max_steps: int = 10

    # Output file for rollout data (optional)
    output_file: str | None = None


def convert_tools_to_openai_format(tools) -> list[dict]:
    """Convert tau2 tools to OpenAI function calling format."""
    tool_jsons = [x.model_dump_json() for x in tools]
    tau2_tools = [json.loads(x) for x in tool_jsons]

    openai_tools = []
    for tool in tau2_tools:
        openai_tool = {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("short_desc", "") or tool.get("long_desc", ""),
                "parameters": tool.get("params", {})
            }
        }
        openai_tools.append(openai_tool)
    return openai_tools


async def call_claude_agent(
    messages: list[dict],
    tools: list[dict],
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    """Call Claude via litellm to get agent response."""
    response = await litellm.acompletion(
        model=model,
        messages=messages,
        tools=tools,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    # Extract the assistant message
    choice = response.choices[0]
    message = choice.message

    result = {"role": "assistant", "content": message.content or ""}

    # Handle tool calls
    if message.tool_calls:
        result["tool_calls"] = []
        for tc in message.tool_calls:
            result["tool_calls"].append({
                "id": tc.id,
                "name": tc.function.name,
                "arguments": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
            })

    return result


async def run_rollout(config: Config):
    """Run a full episode using Claude for both agent and user."""

    print("=" * 80)
    print("TAUBENCH ROLLOUT WITH CLAUDE SONNET 4.5")
    print("=" * 80)

    # Get task_id from registry if not specified
    task_id = config.task_id
    if task_id is None:
        tasks = reg.registry.get_tasks_loader(config.domain)()
        task_id = tasks[0].id
        print(f"No task_id specified, using first task: {task_id}")

    print(f"\nAgent model: {config.agent_model}")
    print(f"User model: {config.user_model}")
    print(f"Domain: {config.domain}")
    print(f"Task ID: {task_id}")

    # Create tau2 environment with Claude as user LLM
    print("\n" + "-" * 80)
    print("Creating Tau2 environment...")
    print("-" * 80)

    env = AgentGymEnv(
        domain=config.domain,
        task_id=task_id,
        user_llm=config.user_model,
        user_llm_args={"temperature": config.temperature, "max_tokens": config.max_tokens},
        max_steps=config.max_steps,
    )

    # Reset environment to get initial observation
    obs, info = env.reset()

    # Get tools and policy
    tools = env._get_tools()
    openai_tools = convert_tools_to_openai_format(tools)
    domain_policy = env._get_policy()

    print(f"\nTotal tools available: {len(openai_tools)}")
    for tool in openai_tools[:5]:
        print(f"  - {tool['function']['name']}")
    if len(openai_tools) > 5:
        print(f"  ... and {len(openai_tools) - 5} more")

    # Build system prompt (same as tau2 does)
    system_prompt = SYSTEM_PROMPT.format(
        domain_policy=domain_policy,
        agent_instruction=AGENT_INSTRUCTION
    )

    # Initialize conversation for Claude agent
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": obs},
    ]

    print("\n" + "=" * 80)
    print("INITIAL OBSERVATION")
    print("=" * 80)
    print(f"\n{obs[:500]}..." if len(obs) > 500 else f"\n{obs}")

    # Rollout data for saving
    rollout_data = {
        "config": {
            "agent_model": config.agent_model,
            "user_model": config.user_model,
            "domain": config.domain,
            "task_id": task_id,
        },
        "system_prompt": system_prompt,
        "turns": [],
    }

    # Run rollout loop
    step = 0
    terminated = False
    total_reward = 0.0

    while not terminated and step < config.max_steps:
        step += 1
        print("\n" + "=" * 80)
        print(f"STEP {step}")
        print("=" * 80)

        # Get agent response from Claude
        print("\nCalling Claude agent...")
        agent_response = await call_claude_agent(
            messages=messages,
            tools=openai_tools,
            model=config.agent_model,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )

        print(f"\nAgent response:")
        if agent_response.get("tool_calls"):
            for tc in agent_response["tool_calls"]:
                print(f"  Tool call: {tc['name']}({json.dumps(tc['arguments'])})")
        if agent_response.get("content"):
            content = agent_response["content"]
            print(f"  Content: {content[:200]}..." if len(content) > 200 else f"  Content: {content}")

        # Convert to tau2 action format
        if agent_response.get("tool_calls"):
            tc = agent_response["tool_calls"][0]
            action_str = json.dumps({"name": tc["name"], "arguments": tc["arguments"]})
        else:
            action_str = agent_response.get("content", "")

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action_str)
        total_reward = reward  # Reward is cumulative in tau2

        print(f"\nEnvironment response:")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Observation: {obs[:200]}..." if len(obs) > 200 else f"  Observation: {obs}")

        # Record turn
        turn_data = {
            "step": step,
            "agent_response": agent_response,
            "action_str": action_str,
            "observation": obs,
            "reward": reward,
            "terminated": terminated,
        }
        rollout_data["turns"].append(turn_data)

        # Update conversation for next turn
        messages.append(agent_response)
        if not terminated and obs:
            # Parse observation role
            if obs.startswith("user: "):
                messages.append({"role": "user", "content": obs[6:]})
            elif obs.startswith("tool: "):
                # Add tool result message
                if agent_response.get("tool_calls"):
                    tc = agent_response["tool_calls"][0]
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": obs[6:],
                    })
                else:
                    messages.append({"role": "user", "content": obs})
            else:
                messages.append({"role": "user", "content": obs})

    print("\n" + "=" * 80)
    print("ROLLOUT COMPLETE")
    print("=" * 80)
    print(f"\nTotal steps: {step}")
    print(f"Final reward: {total_reward}")
    print(f"Terminated: {terminated}")

    # Save rollout data if output file specified
    if config.output_file:
        rollout_data["final_reward"] = total_reward
        rollout_data["total_steps"] = step
        with open(config.output_file, "w") as f:
            json.dump(rollout_data, f, indent=2)
        print(f"\nRollout saved to: {config.output_file}")

    return rollout_data


async def main():
    """Main entry point."""
    config = chz.entrypoint(Config)
    try:
        await run_rollout(config)
    except Exception as e:
        logger.error(f"Rollout failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
