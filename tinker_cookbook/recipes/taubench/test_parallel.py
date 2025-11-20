#!/usr/bin/env python3
"""
Test script to demonstrate that tau2 environments actually run in parallel
after the asyncio.to_thread() fix.
"""

import asyncio
import time
from datetime import datetime
from tinker_cookbook.recipes.taubench.env import construct_tau2_env
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.model_info import get_recommended_renderer_name
import logging

# Suppress tau2/LiteLLM logs for cleaner output
logging.getLogger("tau2").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


async def run_single_env_step(env_id: int, env):
    """Run a single step on an environment and track timing."""
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Env {env_id}: Starting step...")

    # Get initial observation
    obs, stop_condition = await env.initial_observation()

    # Simulate an assistant response (just use a simple message)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Create a simple assistant response
    assistant_response = "I'll help you with that. Can you provide more details about your issue?"
    test_messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": assistant_response},
    ]

    # Build the action (tokenized response)
    action = renderer.build_generation_prompt(test_messages[:-1] + [test_messages[-1]])

    # Take a step (this will make the blocking GPT-4 API call)
    result = await env.step(action)

    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Env {env_id}: Step completed in {elapsed:.2f}s, reward={result.reward}, done={result.episode_done}")

    return elapsed


async def test_parallel_execution():
    """Test that multiple environments run in parallel."""
    print("=" * 70)
    print("TESTING PARALLEL EXECUTION OF TAU2 ENVIRONMENTS")
    print("=" * 70)

    # Number of environments to test
    num_envs = 4

    # Create environments - use a real telecom task ID
    print(f"\nCreating {num_envs} tau2 environments...")
    envs = []
    # Use the first available telecom task
    task_id = "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"
    for i in range(num_envs):
        env = construct_tau2_env(
            domain="telecom",
            task_id=task_id,
            model_name="meta-llama/Llama-3.1-8B-Instruct"
        )
        envs.append(env)

    print(f"Created {num_envs} environments\n")

    # Run all environments in parallel
    print("RUNNING ENVIRONMENTS IN PARALLEL (with asyncio.to_thread fix):")
    print("-" * 60)

    start_time = time.time()

    # Create tasks for all environments
    tasks = [run_single_env_step(i, env) for i, env in enumerate(envs)]

    # Run all tasks concurrently
    step_times = await asyncio.gather(*tasks)

    total_parallel_time = time.time() - start_time

    print("-" * 60)
    print(f"\nPARALLEL EXECUTION RESULTS:")
    print(f"  Total time for {num_envs} environments: {total_parallel_time:.2f}s")
    print(f"  Average time per environment: {total_parallel_time/num_envs:.2f}s")
    print(f"  Individual step times: {[f'{t:.2f}s' for t in step_times]}")

    # Calculate what sequential would look like
    sequential_time = sum(step_times)
    print(f"\nCOMPARISON:")
    print(f"  Estimated sequential time: {sequential_time:.2f}s")
    print(f"  Actual parallel time: {total_parallel_time:.2f}s")
    print(f"  Speedup: {sequential_time/total_parallel_time:.2f}x")

    # Check if we got actual parallelism
    if total_parallel_time < sequential_time * 0.8:  # Allow 20% overhead
        print(f"\n✅ SUCCESS: Environments ran in parallel!")
        print(f"   Parallel execution took only {total_parallel_time/sequential_time*100:.1f}% of sequential time")
    else:
        print(f"\n⚠️  WARNING: Environments may not be running in parallel")
        print(f"   Expected significant speedup but parallel time is {total_parallel_time/sequential_time*100:.1f}% of sequential")

    print("=" * 70)


async def test_blocking_detection():
    """Quick test to show when operations are blocking vs non-blocking."""
    print("\nTEST: Checking if operations are truly async...")
    print("-" * 60)

    # Use a real telecom task ID
    task_id = "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"
    env = construct_tau2_env(
        domain="telecom",
        task_id=task_id,
        model_name="meta-llama/Llama-3.1-8B-Instruct"
    )

    async def monitor_task():
        """Task that prints dots to show the event loop is not blocked."""
        for i in range(10):
            print(".", end="", flush=True)
            await asyncio.sleep(0.5)
        print()

    print("Starting environment step (dots show event loop is responsive):")

    # Start both the monitor and the env step
    monitor = asyncio.create_task(monitor_task())
    step_task = asyncio.create_task(run_single_env_step(0, env))

    # Wait for both to complete
    await asyncio.gather(monitor, step_task, return_exceptions=True)

    print("If you saw dots printing during the step, the event loop was not blocked!")
    print("-" * 60)


async def main():
    """Run all tests."""
    # First show that operations are non-blocking
    await test_blocking_detection()

    # Then test parallel execution
    await test_parallel_execution()


if __name__ == "__main__":
    print("\nStarting parallel execution test for tau2 environments...")
    print("This will make real API calls to GPT-4, so it may take a moment.\n")

    asyncio.run(main())