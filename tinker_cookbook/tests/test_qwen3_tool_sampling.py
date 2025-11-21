#!/usr/bin/env python3
"""
Test script for Qwen3 tool calling with actual model sampling.
Based on test_qwen3_tool_calling from test_renderers.py but with real sampling.
"""

import asyncio
import json
import logging

import tinker
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define tools (same as in test_renderers.py)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string", "description": "City name"}},
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool1",
            "description": "Tool 1",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tool2",
            "description": "Tool 2",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def test_tool_call_generation(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    user_prompt: str,
    test_name: str,
):
    """Test generating and parsing a tool call response."""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")

    # Create conversation with system message (tools will be injected into it)
    messages: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": user_prompt},
    ]

    print(f"\nUser: {user_prompt}")

    # Build generation prompt with tools
    model_input = renderer.build_generation_prompt(messages, tools=TOOLS)
    prompt_str = tokenizer.decode(model_input.to_ints())
    print(f"\nGeneration Prompt:")
    print(prompt_str)
    print(f"\nPrompt length: {len(model_input.to_ints())} tokens")

    # Sample from model
    print("\nSampling from model...")
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=512,
            temperature=1,
            stop=renderer.get_stop_sequences(),
        ),
    )

    # Get response tokens
    response_tokens = response.sequences[0].tokens

    # Decode raw response
    raw_response = tokenizer.decode(response_tokens)
    print(f"\nRaw Response ({len(response_tokens)} tokens):")
    print(raw_response)

    # Parse response using renderer
    parsed_message, format_correct = renderer.parse_response(response_tokens)

    print(f"\nParsing Results:")
    print(f"  Format Correct: {format_correct}")
    print(f"  Role: {parsed_message['role']}")
    print(f"  Content: {parsed_message['content']}")

    # Check for tool calls
    if "tool_calls" in parsed_message:
        print(f"  Tool Calls Found: {len(parsed_message['tool_calls'])}")
        for i, tool_call in enumerate(parsed_message["tool_calls"]):
            print(f"    Tool Call {i+1}:")
            print(f"      Name: {tool_call['name']}")
            print(f"      Arguments: {tool_call['arguments']}")
    else:
        print("  Tool Calls: None")

    return parsed_message, format_correct


async def test_tool_response_parsing(
    sampling_client: tinker.SamplingClient,
    renderer,
    tokenizer,
    conversation_history: list[Message],
    test_name: str,
):
    """Test parsing a response after tool results are provided."""
    print(f"\n{'='*80}")
    print(f"Test: {test_name}")
    print(f"{'='*80}")

    print("\nConversation History:")
    for msg in conversation_history:
        role = msg["role"]
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            print(f"  {role}: {content} [Tool calls: {[tc['name'] for tc in tool_calls]}]")
        else:
            print(f"  {role}: {content}")

    # Build generation prompt with tools
    model_input = renderer.build_generation_prompt(conversation_history, tools=TOOLS)
    prompt_str = tokenizer.decode(model_input.to_ints())
    print(f"\nGeneration Prompt (last 500 chars):")
    print(prompt_str[-500:])
    print(f"\nPrompt length: {len(model_input.to_ints())} tokens")

    # Sample from model
    print("\nSampling from model...")
    response = await sampling_client.sample_async(
        prompt=model_input,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=512,
            temperature=0.7,
            stop=renderer.get_stop_sequences(),
        ),
    )

    # Get response tokens
    response_tokens = response.sequences[0].tokens

    # Decode raw response
    raw_response = tokenizer.decode(response_tokens)
    print(f"\nRaw Response ({len(response_tokens)} tokens):")
    print(raw_response)

    # Parse response using renderer
    parsed_message, format_correct = renderer.parse_response(response_tokens)

    print(f"\nParsing Results:")
    print(f"  Format Correct: {format_correct}")
    print(f"  Role: {parsed_message['role']}")
    print(f"  Content: {parsed_message['content']}")

    # Check for tool calls
    if "tool_calls" in parsed_message:
        print(f"  Tool Calls Found: {len(parsed_message['tool_calls'])}")
        for i, tool_call in enumerate(parsed_message["tool_calls"]):
            print(f"    Tool Call {i+1}:")
            print(f"      Name: {tool_call['name']}")
            print(f"      Arguments: {tool_call['arguments']}")
    else:
        print("  Tool Calls: None")

    return parsed_message, format_correct


async def main():
    """Main test function."""
    # Model setup
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    print(f"🚀 Testing Qwen3 Tool Calling with model: {model_name}")

    # Create clients
    service_client = tinker.ServiceClient()
    sampling_client = service_client.create_sampling_client(
        base_model=model_name, model_path=None
    )

    # Get tokenizer and renderer
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    print(f"📦 Using renderer: {renderer_name}")
    print(f"🛑 Stop sequences: {renderer.get_stop_sequences()}")

    # Test 1: Simple tool call request
    await test_tool_call_generation(
        sampling_client,
        renderer,
        tokenizer,
        user_prompt="What's the weather in San Francisco?",
        test_name="Simple weather query",
    )

    # Test 2: Multiple tool calls
    await test_tool_call_generation(
        sampling_client,
        renderer,
        tokenizer,
        user_prompt="Use tool1 and tool2",
        test_name="Multiple tool calls",
    )

    # Test 3: Tool response scenario
    # Simulate a conversation where tools were called and we got results
    conversation_with_tool_result: list[Message] = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in San Francisco?"},
        {
            "role": "assistant",
            "content": "Let me check the weather for you.",
            "tool_calls": [{"name": "get_weather", "arguments": {"location": "San Francisco"}}],
        },
        {"role": "tool", "content": "Temperature: 72°F, Conditions: Sunny"},
        {"role": "user", "content": "Thanks! What about New York?"},
    ]

    await test_tool_response_parsing(
        sampling_client,
        renderer,
        tokenizer,
        conversation_history=conversation_with_tool_result,
        test_name="Follow-up after tool result",
    )

    print(f"\n{'='*80}")
    print("✅ All tests completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
