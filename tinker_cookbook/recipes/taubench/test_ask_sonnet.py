#!/usr/bin/env python3
"""
Test script for ask_sonnet feature.

Usage:
    uv run python -m tinker_cookbook.recipes.taubench.test_ask_sonnet
"""

import asyncio
import json
import logging

import tau2.registry as reg
import tinker

from tinker_cookbook.recipes.taubench.env import Tau2Env, ASK_SONNET_TOOL
from tinker_cookbook.renderers import get_renderer, ToolCall
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_first_task_id(domain: str) -> str:
    """Get the first task_id for a domain from the registry."""
    tasks = reg.registry.get_tasks_loader(domain)()
    return tasks[0].id


def print_rendered_prompt(renderer, tokenizer, messages, tools, title="RENDERED PROMPT"):
    """Helper to print the rendered prompt with tools."""
    model_input = renderer.build_generation_prompt(messages, tools=tools)
    decoded = tokenizer.decode(model_input.to_ints())

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Total tokens: {model_input.length}")
    print("-" * 80)
    print(decoded)
    print("=" * 80)


def print_messages(messages, title="MESSAGES"):
    """Helper to print messages."""
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg.get("content", "")
        content_preview = content[:200] + "..." if len(content) > 200 else content
        has_tool_calls = "tool_calls" in msg and msg["tool_calls"]
        tool_info = ""
        if has_tool_calls:
            tc = msg["tool_calls"][0]
            # Handle both pydantic ToolCall and dict formats
            if hasattr(tc, 'function'):
                tool_info = f" -> {tc.function.name}({tc.function.arguments[:50]}...)"
            elif isinstance(tc, dict):
                func = tc.get("function", tc)
                tool_info = f" -> {func.get('name')}({str(func.get('arguments', ''))[:50]}...)"
        tool_call_id = msg.get("tool_call_id", "")
        tool_call_id_info = f" [tool_call_id={tool_call_id}]" if tool_call_id else ""
        print(f"  [{i}] {role}{' [has tool_calls]' if has_tool_calls else ''}{tool_info}{tool_call_id_info}")
        print(f"      content: {content_preview}")
    print("-" * 80)


def print_both_histories(env, title="MESSAGE HISTORIES"):
    """Print both self.messages and self.external_llm_messages side by side."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    print_messages(env.messages, "self.messages (Qwen's view)")
    print_messages(env.external_llm_messages, "self.external_llm_messages (Sonnet's view)")


def print_tool_call(tc: ToolCall, indent: str = "    "):
    """Print detailed tool call info."""
    print(f"{indent}id: {tc.id}")
    print(f"{indent}function.name: {tc.function.name}")
    print(f"{indent}function.arguments: {tc.function.arguments}")


def print_assistant_message(msg: dict, title: str = "ASSISTANT MESSAGE"):
    """Print detailed assistant message info."""
    print(f"\n{title}:")
    print(f"  role: {msg.get('role')}")
    print(f"  content: {msg.get('content', '')[:500]}")
    if "tool_calls" in msg and msg["tool_calls"]:
        print(f"  tool_calls ({len(msg['tool_calls'])}):")
        for i, tc in enumerate(msg["tool_calls"]):
            print(f"    [{i}]:")
            print_tool_call(tc, indent="      ")


async def test_ask_sonnet_multi_turn(num_turns: int = 3):
    """Test ask_sonnet with multiple turns to see tool calls and responses.

    Args:
        num_turns: Number of turns to run with external LLM.
    """
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    external_llm = "claude-sonnet-4-5-20250929"
    domain = "retail"

    print("=" * 80)
    print(f"Testing ask_sonnet with {num_turns} turns")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"External LLM: {external_llm}")

    # Get a real task_id from registry
    task_id = get_first_task_id(domain)
    print(f"Domain: {domain}")
    print(f"Task ID: {task_id}")

    # Setup renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    print(f"Renderer: {renderer_name}")

    # Create env with external LLM configured
    env = Tau2Env(
        renderer=renderer,
        domain=domain,
        task_id=task_id,
        external_llm_model=external_llm,
        external_llm_temperature=0.0,
        external_llm_max_tokens=1024,
    )

    print(f"\nTools available: {len(env.tools)}")
    print_both_histories(env, "INITIAL STATE")

    # Get initial observation
    obs, stop_condition = await env.initial_observation()
    print(f"\nInitial observation: {obs.length} tokens")

    # Run multiple turns, each delegating to external LLM
    for turn in range(num_turns):
        print("\n" + "=" * 80)
        print(f"TURN {turn + 1}/{num_turns}")
        print("=" * 80)

        # Build ask_sonnet action
        ask_sonnet_json = json.dumps({"name": "ask_sonnet", "args": {}})
        action_str = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(action_str, add_special_tokens=False)

        print(f"\nSending ask_sonnet action...")

        # Step the environment - this calls Sonnet
        result = await env.step(action_tokens)

        print(f"\nStep result:")
        print(f"  episode_done: {result.episode_done}")
        print(f"  reward: {result.reward}")
        print(f"  next_observation.length: {result.next_observation.length} tokens")
        print(f"  ask_sonnet_call_count: {env.ask_sonnet_call_count}")

        # Print both message histories to compare
        print_both_histories(env, f"MESSAGE HISTORIES AFTER TURN {turn + 1}")

        if result.episode_done:
            print(f"\nEpisode ended after turn {turn + 1}")
            break

    print("\n" + "=" * 80)
    print("[DONE] Multi-turn test completed!")
    print("=" * 80)


async def test_ask_sonnet(run_step: bool = False):
    """Test ask_sonnet feature with external LLM.

    Args:
        run_step: If True, actually call env.step() which will call the external LLM.
                  Default is False to avoid API costs during basic testing.
    """
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    external_llm = "claude-sonnet-4-5-20250929"  # or "gpt-4o" etc.
    domain = "retail"

    print("=" * 80)
    print("Testing ask_sonnet feature")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"External LLM: {external_llm}")

    # Get a real task_id from registry
    task_id = get_first_task_id(domain)
    print(f"Domain: {domain}")
    print(f"Task ID: {task_id}")

    # Setup renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    print(f"Renderer: {renderer_name}")

    # Create env with external LLM configured
    print("\n" + "-" * 80)
    print("Creating Tau2Env with external_llm_model...")
    print("-" * 80)

    env = Tau2Env(
        renderer=renderer,
        domain=domain,
        task_id=task_id,
        external_llm_model=external_llm,
        external_llm_temperature=0.0,
        external_llm_max_tokens=1024,
    )

    # Check that ask_sonnet tool is in the tools list
    tool_names = [t["function"]["name"] for t in env.tools]
    print(f"\nTools available ({len(env.tools)}):")
    for name in tool_names:
        print(f"  - {name}")

    assert "ask_sonnet" in tool_names, "ask_sonnet tool should be in tools list!"
    print("\n[PASS] ask_sonnet tool is present")

    # Print messages before rendering
    print_messages(env.messages, "MESSAGES IN ENV (before rendering)")

    # Print rendered prompt with tools
    print_rendered_prompt(renderer, tokenizer, env.messages, env.tools, "RENDERED PROMPT WITH TOOLS")

    # Get initial observation
    print("\n" + "-" * 80)
    print("Getting initial observation...")
    print("-" * 80)

    obs, stop_condition = await env.initial_observation()
    print(f"Observation length: {obs.length} tokens")

    if not run_step:
        print("\n[SKIP] Skipping env.step() to avoid API costs (set run_step=True to test)")
        print("\n" + "=" * 80)
        print("[PASS] Test completed (basic checks only)!")
        print("=" * 80)
        return

    # Craft an ask_sonnet action
    print("\n" + "-" * 80)
    print("Crafting ask_sonnet action...")
    print("-" * 80)

    # Build the action string that the model would output (no args needed)
    ask_sonnet_json = json.dumps({"name": "ask_sonnet", "args": {}})
    action_str = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"

    print(f"Action string:\n{action_str}")

    # Tokenize it
    action_tokens = tokenizer.encode(action_str, add_special_tokens=False)
    print(f"Action tokens: {len(action_tokens)} tokens")

    # Step the environment - this should trigger external LLM call
    print("\n" + "-" * 80)
    print("Calling env.step() - should delegate to external LLM...")
    print("-" * 80)

    result = await env.step(action_tokens)

    print(f"\nStep result:")
    print(f"  Episode done: {result.episode_done}")
    print(f"  Reward: {result.reward}")
    print(f"  Next observation length: {result.next_observation.length} tokens")

    # Print messages after step
    print_messages(env.messages, "MESSAGES AFTER STEP")

    print("\n" + "=" * 80)
    print("[PASS] Test completed!")
    print("=" * 80)


async def test_without_external_llm():
    """Test that ask_sonnet tool is NOT added when external_llm_model is None."""
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    domain = "retail"

    print("\n" + "=" * 80)
    print("Testing env WITHOUT external_llm_model")
    print("=" * 80)

    # Get a real task_id from registry
    task_id = get_first_task_id(domain)

    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    env = Tau2Env(
        renderer=renderer,
        domain=domain,
        task_id=task_id,
        external_llm_model=None,  # Not configured
    )

    tool_names = [t["function"]["name"] for t in env.tools]
    print(f"\nTools available ({len(env.tools)}):")
    for name in tool_names[:5]:
        print(f"  - {name}")
    if len(tool_names) > 5:
        print(f"  ... and {len(tool_names) - 5} more")

    assert "ask_sonnet" not in tool_names, "ask_sonnet should NOT be in tools when external_llm_model is None!"
    print(f"\n[PASS] ask_sonnet correctly not in tools (external_llm_model=None)")


async def test_direct_qwen_multi_turn(num_turns: int = 3):
    """Test multiple turns where Qwen responds directly (no ask_sonnet)."""
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    external_llm = "claude-sonnet-4-5-20250929"
    domain = "retail"

    print("\n" + "=" * 80)
    print(f"TEST: Direct Qwen responses ({num_turns} turns, NO ask_sonnet)")
    print("=" * 80)

    task_id = get_first_task_id(domain)
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer=tokenizer)

    env = Tau2Env(
        renderer=renderer,
        domain=domain,
        task_id=task_id,
        external_llm_model=external_llm,
    )

    print_both_histories(env, "INITIAL STATE")

    obs, stop_condition = await env.initial_observation()

    # Direct responses Qwen might make
    direct_responses = [
        "Hello! I'd be happy to help you with those exchanges. Could you please provide your name and zip code so I can look up your account?<|im_end|>",
        "<tool_call>\n{\"name\": \"find_user_id_by_name_zip\", \"arguments\": {\"first_name\": \"Yusuf\", \"last_name\": \"Rossi\", \"zip\": \"19122\"}}\n</tool_call><|im_end|}",
        "<tool_call>\n{\"name\": \"get_user_details\", \"arguments\": {\"user_id\": \"yusuf_rossi_9620\"}}\n</tool_call><|im_end|}",
    ]

    for turn in range(num_turns):
        print("\n" + "-" * 80)
        print(f"TURN {turn + 1}/{num_turns} (Direct Qwen response)")
        print("-" * 80)

        action_str = direct_responses[turn % len(direct_responses)]
        action_tokens = tokenizer.encode(action_str, add_special_tokens=False)

        result = await env.step(action_tokens)

        print(f"\nStep result: episode_done={result.episode_done}, ask_sonnet_count={env.ask_sonnet_call_count}")
        print_both_histories(env, f"AFTER TURN {turn + 1}")

        if result.episode_done:
            print(f"\nEpisode ended after turn {turn + 1}")
            break

    print("\n" + "=" * 80)
    print(f"FINAL: messages={len(env.messages)}, external={len(env.external_llm_messages)}")
    print("Both should be EQUAL for direct responses")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    # Suppress debug logging for cleaner output
    logging.getLogger("tau2").setLevel(logging.WARNING)
    logging.getLogger("tinker_cookbook.recipes.taubench.env").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    if len(sys.argv) > 1 and sys.argv[1] == "ask_sonnet":
        # Test with ask_sonnet
        asyncio.run(test_ask_sonnet_multi_turn(num_turns=3))
    elif len(sys.argv) > 1 and sys.argv[1] == "direct":
        # Test direct Qwen responses
        asyncio.run(test_direct_qwen_multi_turn(num_turns=3))
    else:
        # Run both tests
        print("\n" + "#" * 80)
        print("# RUNNING BOTH TESTS")
        print("#" * 80)

        print("\n\n>>> TEST 1: Direct Qwen responses (no ask_sonnet)")
        asyncio.run(test_direct_qwen_multi_turn(num_turns=3))

        print("\n\n>>> TEST 2: ask_sonnet delegated responses")
        asyncio.run(test_ask_sonnet_multi_turn(num_turns=3))
