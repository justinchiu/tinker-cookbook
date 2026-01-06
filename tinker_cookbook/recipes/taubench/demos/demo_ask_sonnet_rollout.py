#!/usr/bin/env python3
"""
Demo rollout with Qwen agent and forced ask_sonnet calls.

Shows the message flow in both DIRECT and CONDITIONING modes.

Usage:
    uv run python -m tinker_cookbook.recipes.taubench.demo_ask_sonnet_rollout \
        mode=direct
    uv run python -m tinker_cookbook.recipes.taubench.demo_ask_sonnet_rollout \
        mode=conditioning
"""

import asyncio
import json
import logging
import sys

import chz
import tau2.registry as reg
import tinker
from tinker_cookbook.completers import TinkerTokenCompleter
from tinker_cookbook.recipes.taubench.env import Tau2Env
from tinker_cookbook.recipes.taubench.components import (
    AskSonnetMode,
    ActionParser,
    ActionType,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@chz.chz
class Config:
    """Configuration for ask_sonnet demo rollout."""
    model_name: str = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    domain: str = "retail"
    task_id: str | None = None

    # Ask sonnet configuration
    mode: str = "direct"  # "direct" or "conditioning"
    external_llm_model: str = "gpt-5.2"

    # Sampling params
    lora_rank: int = 32
    max_tokens: int = 512
    temperature: float = 1.0

    # Force ask_sonnet on turn N (0 = never force, let model decide)
    force_ask_sonnet_on_turn: int = 2  # Force on second turn (after greeting)

    max_steps: int = 6


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_messages(messages: list[dict], title: str = "Messages"):
    """Pretty print messages."""
    print(f"\n--- {title} ({len(messages)} messages) ---")
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        tool_call_id = msg.get("tool_call_id", "")

        print(f"\n[{i}] {role.upper()}" + (f" (tool_call_id={tool_call_id})" if tool_call_id else ""))

        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, 'function'):
                    print(f"    TOOL_CALL: {tc.function.name}({tc.function.arguments})")
                else:
                    name = tc.get("name", tc.get("function", {}).get("name", "?"))
                    args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                    print(f"    TOOL_CALL: {name}({args})")

        if content:
            display_content = content[:500] + "..." if len(content) > 500 else content
            print(f"    {display_content}")


async def run_demo(config: Config):
    """Run demo rollout with ask_sonnet."""

    # Parse mode
    if config.mode == "direct":
        ask_sonnet_mode = AskSonnetMode.DIRECT_INJECTION
    elif config.mode == "conditioning":
        ask_sonnet_mode = AskSonnetMode.CONDITIONING
    else:
        raise ValueError(f"Unknown mode: {config.mode}")

    print_separator(f"ASK_SONNET DEMO - {config.mode.upper()} MODE")

    # Get task_id
    task_id = config.task_id
    if task_id is None:
        tasks = reg.registry.get_tasks_loader(config.domain)()
        task_id = tasks[0].id
        print(f"Using first task: {task_id}")

    print(f"\nModel: {config.model_name}")
    print(f"Domain: {config.domain}")
    print(f"Task ID: {task_id}")
    print(f"Mode: {config.mode}")
    print(f"External LLM: {config.external_llm_model}")
    print(f"Force ask_sonnet on turn: {config.force_ask_sonnet_on_turn}")

    # Create tokenizer and renderer
    tokenizer = get_tokenizer(config.model_name)
    renderer_name = get_recommended_renderer_name(config.model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Create environment with ask_sonnet support
    print_separator("Creating Environment")
    env = Tau2Env(
        renderer=renderer,
        domain=config.domain,
        task_id=task_id,
        external_llm_model=config.external_llm_model,
        ask_sonnet_mode=ask_sonnet_mode,
    )

    print(f"Tools available: {len(env.tools)}")
    for tool in env.tools:
        print(f"  - {tool['function']['name']}")

    # Create sampling client
    print_separator("Creating Sampling Client")
    service_client = tinker.ServiceClient()
    training_client = await service_client.create_lora_training_client_async(
        base_model=config.model_name,
        rank=config.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async(name="demo")

    completer = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )

    # Get initial observation
    print_separator("STEP 0: Initial Observation")
    obs, stop_condition = await env.initial_observation()

    print(f"Observation length: {obs.length} tokens")
    print_messages(env.messages.messages, "Initial Messages")

    # Run rollout
    step = 0
    current_obs = obs
    current_stop = stop_condition
    episode_done = False

    while not episode_done and step < config.max_steps:
        step += 1
        print_separator(f"STEP {step}")

        # Check if we should force ask_sonnet
        force_ask_sonnet = (step == config.force_ask_sonnet_on_turn)

        if force_ask_sonnet:
            print("\n*** FORCING ask_sonnet call ***")
            # Create ask_sonnet action tokens
            ask_sonnet_text = '<tool_call>\n{"name": "ask_sonnet", "arguments": {}}\n</tool_call>'
            action_tokens = tokenizer.encode(ask_sonnet_text, add_special_tokens=False)
            print(f"Injected action: {ask_sonnet_text}")
        else:
            # Sample from model
            print("\nSampling from Qwen...")
            action_with_logprobs = await completer(current_obs, current_stop)
            action_tokens = action_with_logprobs.tokens

            decoded = tokenizer.decode(action_tokens)
            print(f"Qwen action: {decoded[:300]}..." if len(decoded) > 300 else f"Qwen action: {decoded}")

        # Parse and show action type
        parsed = env.action_parser.parse(action_tokens)
        print(f"\nAction type: {parsed.action_type.value}")
        if parsed.tool_name:
            print(f"Tool name: {parsed.tool_name}")

        # Step environment
        print("\nStepping environment...")
        result = await env.step(action_tokens)

        print(f"Episode done: {result.episode_done}")
        print(f"Reward: {result.reward}")

        # Show updated messages
        print_messages(env.messages.messages, f"Messages after step {step}")

        # Update for next iteration
        current_obs = result.next_observation
        current_stop = result.next_stop_condition
        episode_done = result.episode_done

        # If this was ask_sonnet in conditioning mode, we need another step for followup
        if force_ask_sonnet and ask_sonnet_mode == AskSonnetMode.CONDITIONING and not episode_done:
            print_separator(f"STEP {step}.5: Qwen Followup (CONDITIONING)")
            print("\nSampling followup from Qwen...")

            action_with_logprobs = await completer(current_obs, current_stop)
            action_tokens = action_with_logprobs.tokens

            decoded = tokenizer.decode(action_tokens)
            print(f"Qwen followup: {decoded[:300]}..." if len(decoded) > 300 else f"Qwen followup: {decoded}")

            # Step with followup
            result = await env.step(action_tokens)

            print(f"Episode done: {result.episode_done}")
            print(f"Reward: {result.reward}")

            print_messages(env.messages.messages, f"Messages after followup")

            current_obs = result.next_observation
            current_stop = result.next_stop_condition
            episode_done = result.episode_done

    # Final summary
    print_separator("ROLLOUT COMPLETE")
    print(f"\nTotal steps: {step}")
    print(f"Final reward: {result.reward}")
    print(f"ask_sonnet calls: {env.ask_sonnet_call_count}")
    print(f"Sonnet tokens: {env.sonnet_input_tokens} in / {env.sonnet_output_tokens} out")

    print_separator("FINAL CONVERSATION")
    print_messages(env.messages.messages, "Final Messages")

    # Show rendered prompt
    print_separator("FINAL RENDERED PROMPT (last 1500 chars)")
    final_prompt = tokenizer.decode(current_obs.to_ints())
    print(final_prompt[-1500:])


async def main():
    """Main entry point."""
    config = chz.entrypoint(Config)
    try:
        await run_demo(config)
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
