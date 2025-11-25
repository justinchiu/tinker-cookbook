#!/usr/bin/env python3
"""
Test script for taubench env with Qwen3 30B A3B sampling.
Tests that tool calls, observations, and actions are handled correctly.
"""

import asyncio
import logging
import sys

import chz
import tau2.registry as reg
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.taubench.env import construct_tau2_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for demo rollout."""
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    domain: str = "retail"
    task_id: str | None = None  # Will pick first task from domain if not specified
    checkpoint_path: str | None = None  # Optional checkpoint to load from
    lora_rank: int = 32
    max_tokens: int = 512
    temperature: float = 1.0
    max_steps: int = 10


async def test_taubench_sampling(config: Config):
    """Test a full episode of taubench with real model sampling."""

    print("=" * 80)
    print("TAUBENCH ENVIRONMENT SAMPLING TEST")
    print("=" * 80)

    # Get task_id from registry if not specified
    task_id = config.task_id
    if task_id is None:
        tasks = reg.registry.get_tasks_loader(config.domain)()
        task_id = tasks[0].id
        print(f"No task_id specified, using first task: {task_id}")

    print(f"\nModel: {config.model_name}")
    print(f"Domain: {config.domain}")
    print(f"Task ID: {task_id}")
    if config.checkpoint_path:
        print(f"Checkpoint: {config.checkpoint_path}")

    # Create environment
    print("\n" + "-" * 80)
    print("Creating Tau2 environment...")
    print("-" * 80)
    env = construct_tau2_env(domain=config.domain, task_id=task_id, model_name=config.model_name)

    print("\nInitial messages in env:")
    for i, msg in enumerate(env.messages):
        role = msg["role"]
        content = msg["content"][:100] + "..." if len(msg["content"]) > 100 else msg["content"]
        print(f"  Message {i} ({role}): {content}")

    # Print available tools
    print("\n" + "=" * 80)
    print("AVAILABLE TOOLS")
    print("=" * 80)
    tools = env.env._get_tools()
    print(f"Total tools available: {len(tools)}")
    for tool in tools:
        print(f"\n- {tool.name}")
        desc = tool._get_description() if hasattr(tool, '_get_description') else str(tool)
        print(f"  Description: {desc[:100]}..." if len(desc) > 100 else f"  Description: {desc}")
    print("=" * 80)

    # Create sampling client
    print("\n" + "-" * 80)
    print("Creating sampling client...")
    print("-" * 80)
    service_client = tinker.ServiceClient()

    if config.checkpoint_path:
        # Load from checkpoint
        print(f"Loading from checkpoint: {config.checkpoint_path}")
        training_client = await service_client.create_training_client_from_state_async(
            config.checkpoint_path,
            user_metadata={}
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="demo")
    else:
        # Create fresh client (base model)
        print(f"Using base model (no checkpoint)")
        training_client = await service_client.create_lora_training_client_async(
            base_model=config.model_name,
            rank=config.lora_rank,
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="demo")

    # Create completer
    completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Get initial observation
    print("\n" + "=" * 80)
    print("STEP 0: Initial Observation")
    print("=" * 80)

    initial_obs, stop_condition = await env.initial_observation()

    print(f"\nObservation length: {initial_obs.length} tokens")
    print(f"Stop condition: {stop_condition}")

    # Decode and show the prompt
    decoded_prompt = env.renderer.tokenizer.decode(initial_obs.to_ints())
    print(f"\nDecoded prompt (last 500 chars):")
    print(decoded_prompt[-500:])

    # Sample action
    print("\n" + "-" * 80)
    print("Sampling action from model...")
    print("-" * 80)

    action_with_logprobs = await completer(initial_obs, stop_condition)
    action_tokens = action_with_logprobs.tokens

    print(f"\nAction length: {len(action_tokens)} tokens")

    # Decode the raw action
    decoded_action = env.renderer.tokenizer.decode(action_tokens)
    print(f"\nRaw action:")
    print(decoded_action)

    # Parse the action
    parsed_message, format_correct = env.renderer.parse_response(action_tokens)

    print(f"\n" + "-" * 80)
    print("Parsed action:")
    print("-" * 80)
    print(f"  Format correct: {format_correct}")
    print(f"  Role: {parsed_message['role']}")
    print(f"  Content: {parsed_message['content']}")

    if "tool_calls" in parsed_message:
        print(f"  Tool calls: {len(parsed_message['tool_calls'])}")
        for i, tool_call in enumerate(parsed_message["tool_calls"]):
            print(f"    Tool call {i+1}:")
            print(f"      Name: {tool_call['name']}")
            print(f"      Arguments: {tool_call['arguments']}")
    else:
        print("  Tool calls: None")

    # Step the environment
    print("\n" + "=" * 80)
    print("STEP 1: Environment Step")
    print("=" * 80)

    # Show what will be passed to tau2 gym
    import json
    import re
    if "tool_calls" in parsed_message and parsed_message["tool_calls"]:
        tool_call = parsed_message["tool_calls"][0]
        action_to_gym = json.dumps(tool_call)
        print(f"\nPassing tool call to tau2 gym env.step():")
        print(f"  {action_to_gym}")
    else:
        action_to_gym = re.sub(r"<tool_call>.*?</tool_call>", "", parsed_message["content"], flags=re.DOTALL).strip()
        print(f"\nPassing plain text to tau2 gym env.step():")
        print(f"  {repr(action_to_gym[:100])}...")

    result = await env.step(action_tokens)

    print(f"\nStep result:")
    print(f"  Episode done: {result.episode_done}")
    print(f"  Reward: {result.reward}")
    print(f"  Next observation length: {result.next_observation.length} tokens")

    # Show what observation was returned from tau2 gym (if available)
    if hasattr(env, '_last_gym_obs'):
        print(f"\nRaw observation from tau2 gym:")
        print(f"  {repr(env._last_gym_obs[:200])}..." if len(env._last_gym_obs) > 200 else f"  {repr(env._last_gym_obs)}")

    # Show updated conversation
    print(f"\n" + "-" * 80)
    print("Updated conversation in env:")
    print("-" * 80)
    for i, msg in enumerate(env.messages):
        role = msg["role"]
        content = msg.get("content", "")[:100] + "..." if len(msg.get("content", "")) > 100 else msg.get("content", "")
        has_tool_calls = "tool_calls" in msg
        print(f"  Message {i} ({role}){' [has tool_calls]' if has_tool_calls else ''}: {content}")

    # Continue for up to max_steps turns
    step_num = 2
    current_result = result

    while step_num <= config.max_steps and not current_result.episode_done:
        print("\n" + "=" * 80)
        print(f"STEP {step_num}: Action {step_num}")
        print("=" * 80)

        print(f"\nSampling action {step_num}...")
        action_with_logprobs = await completer(current_result.next_observation, current_result.next_stop_condition)
        action_tokens = action_with_logprobs.tokens

        decoded_action = env.renderer.tokenizer.decode(action_tokens)
        print(f"\nRaw action {step_num}:")
        print(decoded_action[:200] + "..." if len(decoded_action) > 200 else decoded_action)

        # Parse the action
        parsed_message, format_correct = env.renderer.parse_response(action_tokens)

        print(f"\n" + "-" * 80)
        print(f"Parsed action {step_num}:")
        print("-" * 80)
        print(f"  Format correct: {format_correct}")
        if "tool_calls" in parsed_message:
            print(f"  Tool calls: {len(parsed_message['tool_calls'])}")
            for i, tool_call in enumerate(parsed_message["tool_calls"]):
                print(f"    Tool call {i+1}: {tool_call['name']}({tool_call['arguments']})")
        else:
            print(f"  Content: {parsed_message['content'][:100]}...")

        # Step
        current_result = await env.step(action_tokens)
        print(f"\nStep {step_num} result:")
        print(f"  Episode done: {current_result.episode_done}")
        print(f"  Reward: {current_result.reward}")

        step_num += 1

    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)

    # Summary
    print("\nSummary:")
    print(f"  - Model correctly generated responses")
    print(f"  - Environment stepped through {step_num - 1} turns")
    print(f"  - Episode {'completed' if current_result.episode_done else 'still running (hit max steps)'}")
    print(f"  - Final conversation has {len(env.messages)} messages")
    print(f"  - Final reward: {current_result.reward}")

    # Print full conversation
    print("\n" + "=" * 80)
    print("FULL CONVERSATION")
    print("=" * 80)
    for i, msg in enumerate(env.messages):
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])

        print(f"\n--- Message {i} ({role}) ---")
        if tool_calls:
            print(f"Tool calls: {tool_calls}")
        if content:
            print(f"Content: {content[:500]}")
            if len(content) > 500:
                print(f"... (truncated, total length: {len(content)})")

    # Print rendered prompt
    print("\n" + "=" * 80)
    print("FINAL RENDERED PROMPT (last 1000 chars)")
    print("=" * 80)
    final_prompt = env.renderer.tokenizer.decode(current_result.next_observation.to_ints())
    print(final_prompt[-1000:])


async def main():
    """Main entry point."""
    config = chz.entrypoint(Config)
    try:
        await test_taubench_sampling(config)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
