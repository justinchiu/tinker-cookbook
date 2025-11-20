#!/usr/bin/env python3
"""
Simple demonstration that tau2 environments now run in parallel
after the asyncio.to_thread() fix.
"""

import asyncio
import time
from datetime import datetime


def blocking_operation(env_id: int, duration: float = 2.0):
    """Simulates tau2's blocking GPT-4 API call."""
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Env {env_id}: Starting blocking operation...")
    time.sleep(duration)  # Simulate API call delay
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Env {env_id}: Completed blocking operation")
    return f"Result from env {env_id}"


async def broken_async_step(env_id: int):
    """How tau2 WAS working - blocking the event loop."""
    start = time.time()
    result = blocking_operation(env_id)  # THIS BLOCKS THE EVENT LOOP!
    elapsed = time.time() - start
    return elapsed, result


async def fixed_async_step(env_id: int):
    """How tau2 works NOW - using asyncio.to_thread to avoid blocking."""
    start = time.time()
    result = await asyncio.to_thread(blocking_operation, env_id)  # Non-blocking!
    elapsed = time.time() - start
    return elapsed, result


async def run_broken_version():
    """Demonstrate the problem - environments run sequentially."""
    print("\n" + "=" * 70)
    print("BEFORE FIX: Environments block each other (sequential execution)")
    print("=" * 70)

    start = time.time()
    # Try to run 4 environments "in parallel" (but they actually block)
    results = await asyncio.gather(
        broken_async_step(0),
        broken_async_step(1),
        broken_async_step(2),
        broken_async_step(3),
    )

    total_time = time.time() - start
    print(f"\nResults: {[r[1] for r in results]}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Expected if parallel: ~2s")
    print(f"Actual (sequential): ~{total_time:.1f}s ❌ BLOCKING!")


async def run_fixed_version():
    """Demonstrate the fix - environments run in parallel."""
    print("\n" + "=" * 70)
    print("AFTER FIX: Environments run in parallel (with asyncio.to_thread)")
    print("=" * 70)

    start = time.time()
    # Run 4 environments truly in parallel
    results = await asyncio.gather(
        fixed_async_step(0),
        fixed_async_step(1),
        fixed_async_step(2),
        fixed_async_step(3),
    )

    total_time = time.time() - start
    print(f"\nResults: {[r[1] for r in results]}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Expected if parallel: ~2s")
    print(f"Actual (parallel): ~{total_time:.1f}s ✅ PARALLEL!")


async def visual_demo():
    """Visual demonstration showing event loop responsiveness."""
    print("\n" + "=" * 70)
    print("VISUAL DEMONSTRATION: Event loop responsiveness")
    print("=" * 70)

    async def heartbeat():
        """Shows the event loop is responsive."""
        for i in range(10):
            print("♥", end="", flush=True)
            await asyncio.sleep(0.4)
        print()

    print("\nBROKEN VERSION (heartbeat will freeze):")
    print("Starting heartbeat: ", end="")
    heartbeat_task = asyncio.create_task(heartbeat())
    broken_task = asyncio.create_task(broken_async_step(99))

    try:
        # The heartbeat will freeze while broken_async_step blocks
        await asyncio.wait_for(asyncio.gather(heartbeat_task, broken_task), timeout=5)
    except asyncio.TimeoutError:
        print("\n(Timed out - event loop was blocked!)")

    print("\nFIXED VERSION (heartbeat continues):")
    print("Starting heartbeat: ", end="")
    heartbeat_task = asyncio.create_task(heartbeat())
    fixed_task = asyncio.create_task(fixed_async_step(99))

    # The heartbeat will continue while fixed_async_step runs
    await asyncio.gather(heartbeat_task, fixed_task)


async def main():
    """Run all demonstrations."""
    print("\nDemonstrating tau2 async fix...")
    print("This simulates what happens with tau2's blocking GPT-4 API calls.\n")

    # Show the problem
    await run_broken_version()

    # Show the solution
    await run_fixed_version()

    # Visual demonstration
    await visual_demo()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("The asyncio.to_thread() wrapper prevents tau2's synchronous operations")
    print("from blocking the event loop, allowing true parallel execution of")
    print("multiple environments. This dramatically speeds up RL training!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())