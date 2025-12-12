"""
Message format converters for different LLM API providers.

This module provides utilities to convert between internal message formats
and provider-specific formats (Anthropic, OpenAI).
"""

import json
from typing import Any


def convert_messages_for_anthropic(
    messages: list[dict],
    ToolCall: type | None = None,
) -> tuple[str | None, list[dict]]:
    """
    Convert internal message format to Anthropic's format.

    Anthropic format differences:
    - System message is separate (not in messages array)
    - Tool calls are 'tool_use' content blocks in assistant messages
    - Tool results are 'tool_result' content blocks in user messages
    - Content can be string or array of content blocks

    Args:
        messages: List of messages in internal format
        ToolCall: Optional ToolCall class for isinstance checks

    Returns: (system_message, converted_messages)
    """
    # Import lazily if not provided
    if ToolCall is None:
        from tinker_cookbook.renderers import ToolCall

    system_content = None
    converted = []

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        # Extract system message separately
        if role == "system":
            system_content = content
            continue

        # Handle tool response messages -> becomes user message with tool_result
        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "unknown")
            converted.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_call_id,
                        "content": content,
                    }
                ]
            })
            continue

        # Handle assistant messages with tool calls
        if role == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
            content_blocks = []

            # Add text content if present
            if content:
                content_blocks.append({"type": "text", "text": content})

            # Add tool_use blocks
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tc.id or f"call_{id(tc)}",
                        "name": tc.function.name,
                        "input": json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else tc.function.arguments,
                    }
                elif isinstance(tc, dict):
                    # Dict format (e.g., from litellm response)
                    func = tc.get("function", tc)
                    args = func.get("arguments", "{}")
                    tool_use_block = {
                        "type": "tool_use",
                        "id": tc.get("id", f"call_{id(tc)}"),
                        "name": func.get("name", "unknown"),
                        "input": json.loads(args) if isinstance(args, str) else args,
                    }
                content_blocks.append(tool_use_block)

            converted.append({"role": "assistant", "content": content_blocks})
            continue

        # Regular user/assistant message
        converted.append({"role": role, "content": content})

    return system_content, converted


def convert_messages_for_openai(
    messages: list[dict],
    ToolCall: type | None = None,
) -> list[dict]:
    """
    Convert internal message format to OpenAI's format.

    OpenAI format:
    - Tool calls have 'tool_calls' array with function objects
    - Tool results are 'tool' role messages with tool_call_id

    Args:
        messages: List of messages in internal format
        ToolCall: Optional ToolCall class for isinstance checks

    Returns: Converted messages list
    """
    # Import lazily if not provided
    if ToolCall is None:
        from tinker_cookbook.renderers import ToolCall

    converted = []
    for msg in messages:
        new_msg = {"role": msg["role"], "content": msg.get("content", "")}

        # Convert pydantic ToolCall objects to dict format
        if "tool_calls" in msg and msg["tool_calls"]:
            new_msg["tool_calls"] = []
            for tc in msg["tool_calls"]:
                if isinstance(tc, ToolCall):
                    new_msg["tool_calls"].append({
                        "id": tc.id or f"call_{id(tc)}",
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    })
                elif isinstance(tc, dict):
                    new_msg["tool_calls"].append(tc)

        # Handle tool_call_id for tool response messages
        if "tool_call_id" in msg:
            new_msg["tool_call_id"] = msg["tool_call_id"]

        converted.append(new_msg)

    return converted


def convert_tools_for_anthropic(tools: list[dict]) -> list[dict]:
    """
    Convert tools from OpenAI format to Anthropic format.

    OpenAI format:
        {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}

    Anthropic format:
        {"name": "...", "description": "...", "input_schema": {...}}

    Args:
        tools: List of tools in OpenAI format

    Returns: List of tools in Anthropic format
    """
    converted = []
    for tool in tools:
        func = tool.get("function", {})
        converted.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def convert_anthropic_to_tinker(
    response_message: Any,
    ToolCall: type | None = None,
) -> dict:
    """
    Convert an Anthropic response message to tinker format.

    Args:
        response_message: The Anthropic response message object
        ToolCall: Optional ToolCall class for creating tool call objects

    Returns: Tinker message dict with role, content, and optional tool_calls
    """
    if ToolCall is None:
        from tinker_cookbook.renderers import ToolCall

    result: dict = {"role": "assistant", "content": ""}
    tool_calls = []

    # Anthropic responses have content as a list of blocks
    content_parts = []
    for block in getattr(response_message, "content", []):
        if hasattr(block, "type"):
            if block.type == "text":
                content_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        function=ToolCall.FunctionBody(
                            name=block.name,
                            arguments=json.dumps(block.input) if isinstance(block.input, dict) else block.input,
                        ),
                        id=block.id,
                    )
                )

    result["content"] = "\n".join(content_parts)

    if tool_calls:
        result["tool_calls"] = tool_calls

    return result


def convert_openai_to_tinker(
    response_message: Any,
    ToolCall: type | None = None,
) -> dict:
    """
    Convert an OpenAI response message to tinker format.

    Args:
        response_message: The OpenAI response message object
        ToolCall: Optional ToolCall class for creating tool call objects

    Returns: Tinker message dict with role, content, and optional tool_calls
    """
    if ToolCall is None:
        from tinker_cookbook.renderers import ToolCall

    result: dict = {"role": "assistant", "content": response_message.content or ""}

    # Handle tool calls
    if response_message.tool_calls:
        result["tool_calls"] = [
            ToolCall(
                function=ToolCall.FunctionBody(
                    name=tc.function.name,
                    arguments=tc.function.arguments if isinstance(tc.function.arguments, str) else json.dumps(tc.function.arguments),
                ),
                id=tc.id,
            )
            for tc in response_message.tool_calls
        ]

    return result
