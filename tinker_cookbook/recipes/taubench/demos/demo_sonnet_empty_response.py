"""Demo: Sonnet empty response patterns.

This script demonstrates when Sonnet returns empty responses based on
the final message being an `ask_sonnet` tool call.

Key finding: The issue is specifically `ask_sonnet` as the final turn
combined with certain conversation patterns. Removing just the final
`ask_sonnet` turn before sending to Sonnet should fix the issue.

Usage:
    uv run python -m tinker_cookbook.recipes.taubench.demos.demo_sonnet_empty_response
"""

import asyncio
from tinker_cookbook.recipes.taubench.components import (
    ExternalLLMClient,
    ExternalLLMConfig,
)


async def run_test(client: ExternalLLMClient, name: str, messages: list[dict]) -> tuple[str, int]:
    """Run a single test and return status."""
    result = await client.call_with_usage(messages)
    length = len(result.content or "")
    status = "EMPTY" if length < 5 else f"{length} chars"
    return name, length, status


async def main():
    config = ExternalLLMConfig(
        model="claude-sonnet-4-5-20250929",
        temperature=0.0,
        max_tokens=512,
    )
    client = ExternalLLMClient(config)

    print("=" * 70)
    print("DEMO: Sonnet Empty Response Patterns")
    print("=" * 70)

    # =========================================================================
    # SECTION 1: Final turn is ask_sonnet
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 1: Final turn is ask_sonnet")
    print("=" * 70)

    tests_with_ask_sonnet = [
        # Working cases
        ("1a. Simple: user → ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Help me return my order"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
        ("1b. Substantive user msg before ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "I need to return my order #12345"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),

        # Failing cases
        ("1c. Short user msg before ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "success"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
        ("1d. Vague user msg before ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What next?"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
        ("1e. Tool result before ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "[Tool Result]: user_123"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
        ("1f. Prior tool_call before ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": '<tool_call>{"name": "get_user"}</tool_call>'},
            {"role": "user", "content": "user_123"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
        ("1g. [Sonnet's Advice] in history + ask_sonnet", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
            {"role": "user", "content": "[Sonnet's Advice]: Do X"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Now what?"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
        ]),
    ]

    for name, messages in tests_with_ask_sonnet:
        _, length, status = await run_test(client, name, messages)
        indicator = "✓" if length >= 5 else "✗"
        print(f"  {indicator} {name}: {status}")

    # =========================================================================
    # SECTION 2: Same patterns but WITHOUT ask_sonnet at end
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 2: Same patterns but final turn is user (not ask_sonnet)")
    print("=" * 70)

    tests_without_ask_sonnet = [
        ("2a. Short user msg at end", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "success"},
        ]),
        ("2b. Vague user msg at end", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "What next?"},
        ]),
        ("2c. Tool result in history, user at end", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "[Tool Result]: user_123"},
            {"role": "assistant", "content": "Found you!"},
            {"role": "user", "content": "Now help me return my order"},
        ]),
        ("2d. Prior tool_call, user at end", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": '<tool_call>{"name": "get_user"}</tool_call>'},
            {"role": "user", "content": "user_123"},
            {"role": "assistant", "content": "Found you!"},
            {"role": "user", "content": "Now help me return my order"},
        ]),
        ("2e. [Sonnet's Advice] in history, user at end", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
            {"role": "user", "content": "[Sonnet's Advice]: Do X"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Now help me return my order"},
        ]),
    ]

    for name, messages in tests_without_ask_sonnet:
        _, length, status = await run_test(client, name, messages)
        indicator = "✓" if length >= 5 else "✗"
        print(f"  {indicator} {name}: {status}")

    # =========================================================================
    # SECTION 3: The fix - remove final ask_sonnet turn
    # =========================================================================
    print("\n" + "=" * 70)
    print("SECTION 3: The fix - remove final ask_sonnet turn before calling Sonnet")
    print("=" * 70)

    # Take a failing case and show that removing the final turn fixes it
    failing_messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": '<tool_call>{"name": "get_user"}</tool_call>'},
        {"role": "user", "content": "user_123"},
        {"role": "assistant", "content": '<tool_call>{"name": "ask_sonnet"}</tool_call>'},
    ]

    print("\n  Original (with ask_sonnet at end):")
    _, length, status = await run_test(client, "original", failing_messages)
    print(f"    Result: {status}")

    # Fix: remove the final ask_sonnet turn
    fixed_messages = failing_messages[:-1]  # Remove last message
    print("\n  Fixed (final ask_sonnet removed):")
    _, length, status = await run_test(client, "fixed", fixed_messages)
    print(f"    Result: {status}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
Key findings:
1. Sonnet returns EMPTY when ask_sonnet is the final turn AND:
   - The prior user message is short/vague (e.g., "success", "What next?")
   - There's tool_call or [Tool Result] in the conversation history

2. Sonnet responds normally when:
   - The final turn is a user message (not ask_sonnet)
   - ask_sonnet is the first tool call with a substantive user message

3. THE FIX: Simply remove the final ask_sonnet turn before calling Sonnet.
   Don't need complex filtering of [Sonnet's Advice] or prior tool calls.
""")


if __name__ == "__main__":
    asyncio.run(main())
