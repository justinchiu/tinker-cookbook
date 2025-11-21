from datetime import date

import pytest
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import Message, get_renderer
from transformers.models.auto.tokenization_auto import AutoTokenizer


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        # "Qwen/Qwen3-30B-A3B", TODO: This was broken, will address in another PR.
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
    ],
)
def test_generation_against_hf_chat_templates(model_name: str):
    """Test generation prompt against HF chat templates"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    # gpt_oss HF matches gpt_oss_medium_reasoning and not the default gpt_oss
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer)
    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
        {"role": "user", "content": "What is the capital of France?"},
    ]

    if model_name.startswith("meta"):
        today = date.today().strftime("%d %b %Y")
        system_msg: Message = {
            "role": "system",
            "content": f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n",
        }
        aug_convo = [system_msg] + convo
    elif model_name.startswith("Qwen"):
        aug_convo = convo
    elif model_name.startswith("deepseek-ai"):
        aug_convo = convo
    elif model_name.startswith("openai"):
        # Thinking field should not be rendered in this case as it is not the last message.
        convo[1]["thinking"] = "The user is sharing a greeting. We should respond politely."
        aug_convo = convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_tokens = cookbook_renderer.build_generation_prompt(aug_convo).to_ints()
    hf_tokens = tokenizer.apply_chat_template(convo, add_generation_prompt=True)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize(
    "model_name",
    [
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen3-30B-A3B",
        "deepseek-ai/DeepSeek-V3.1",
        "openai/gpt-oss-20b",
    ],
)
def test_supervised_example_against_hf_chat_templates(model_name: str):
    """Test supervised example against HF chat templates"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # not using get_tokenizer(model_name)
    # because we want to test against the original tokenizer from HF, not the mirror
    render_name = (
        get_recommended_renderer_name(model_name)
        if not model_name.startswith("openai")
        else "gpt_oss_medium_reasoning"
    )
    cookbook_renderer = get_renderer(render_name, tokenizer)
    convo: list[Message] = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm fine, thank you!"},
    ]

    if model_name.startswith("meta"):
        today = date.today().strftime("%d %b %Y")
        system_msg: Message = {
            "role": "system",
            "content": f"Cutting Knowledge Date: December 2023\nToday Date: {today}\n\n",
        }
        aug_convo = [system_msg] + convo
    elif model_name.startswith("Qwen"):
        # HF includes thinking tags in assistant content for supervised examples.
        aug_convo = convo.copy()
        aug_convo[1]["content"] = "<think>\n\n</think>\n\n I'm fine, thank you!"
    elif model_name.startswith("deepseek-ai"):
        aug_convo = convo
    elif model_name.startswith("openai"):
        # Test thinking field for GPT-OSS is rendered.
        convo[1]["thinking"] = "The user is sharing a greeting. We should respond politely."
        aug_convo = convo
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    cookbook_tokens_tensor, _ = cookbook_renderer.build_supervised_example(aug_convo)
    cookbook_tokens = cookbook_tokens_tensor.tolist()
    hf_output = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
    hf_tokens = tokenizer.encode(hf_output.rstrip("\n"), add_special_tokens=False)

    assert cookbook_tokens == hf_tokens, (
        f"Cookbook tokens: {cookbook_tokens}\n"
        f"Cookbook string: {tokenizer.decode(cookbook_tokens)}\n"
        f"HF tokens: {hf_tokens}\n"
        f"HF string: {tokenizer.decode(hf_tokens)}"
    )


@pytest.mark.parametrize(
    "model_name,renderer_name",
    [
        ("Qwen/Qwen3-30B-A3B", "qwen3"),
        ("meta-llama/Llama-3.2-1B-Instruct", "llama3"),
        ("openai/gpt-oss-20b", "gpt_oss_medium_reasoning"),
    ],
)
def test_eot_parsing(model_name: str, renderer_name: str):
    """Test EOT token parsing behavior for different renderers using real tokenizers."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    renderer = get_renderer(renderer_name, tokenizer)

    # Get the appropriate EOT token for each renderer
    if renderer_name == "llama3":
        eot_token = "<|eot_id|>"
    elif renderer_name == "qwen3":
        eot_token = "<|im_end|>"
    elif renderer_name.startswith("gpt_oss"):
        eot_token = "<|return|>"
    else:
        raise ValueError(f"Unknown renderer: {renderer_name}")

    # Test case 1: Normal case with single EOT - should parse correctly
    test_response_with_eot = f"53 + 18 = 71{eot_token}"
    response_tokens = tokenizer.encode(test_response_with_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is True

    # Test case 2: No EOT token - should have format=False
    test_response_no_eot = "53 + 18 = 71"
    response_tokens_no_eot = tokenizer.encode(test_response_no_eot, add_special_tokens=False)

    message, format_correct = renderer.parse_response(response_tokens_no_eot)
    assert message["role"] == "assistant"
    assert message["content"] == "53 + 18 = 71"
    assert format_correct is False

    # Test case 3: Double EOT token - should raise ValueError
    test_response_double_eot = f"53 + 18 = 71{eot_token}{eot_token}"
    response_tokens_double_eot = tokenizer.encode(
        test_response_double_eot, add_special_tokens=False
    )

    with pytest.raises(ValueError, match="expected to split into 1 or 2 pieces"):
        _ = renderer.parse_response(response_tokens_double_eot)


def test_qwen3_tool_calling():
    """Test Qwen3 renderer against HF chat template for tool calling scenarios."""
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    renderer_name = get_recommended_renderer_name(model_name)
    cookbook_renderer = get_renderer(renderer_name, tokenizer)

    print(f"\n{'='*80}")
    print(f"Testing Qwen3 Tool Calling - Model: {model_name}")
    print(f"{'='*80}\n")

    # Define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool1",
                "description": "Tool 1",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "tool2",
                "description": "Tool 2",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]

    # Test 1: Single tool response
    print("Test 1: Single tool response message")
    print("-" * 80)
    messages_1 = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "Let me check", "tool_calls": [{"name": "get_weather", "arguments": '{"location": "SF"}'}]},
        {"role": "tool", "content": "Temperature is 72F"},
        {"role": "assistant", "content": "It's 72F in SF"},
        {"role": "user", "content": "Thanks"},
    ]

    hf_output_1 = tokenizer.apply_chat_template(messages_1, tools=tools, tokenize=False, add_generation_prompt=True)
    cookbook_tokens_1 = cookbook_renderer.build_generation_prompt(messages_1).to_ints()  # Exclude last for generation
    cookbook_output_1 = tokenizer.decode(cookbook_tokens_1)

    print("HF Template Output:")
    print(hf_output_1)
    print("\nCookbook Renderer Output:")
    print(cookbook_output_1)
    print("\n")

    # Test 2: Consecutive tool responses (grouping)
    print("Test 2: Consecutive tool response messages")
    print("-" * 80)
    messages_2 = [
        {"role": "user", "content": "Check both"},
        {"role": "assistant", "content": "Checking", "tool_calls": [{"name": "tool1", "arguments": '{}'}, {"name": "tool2", "arguments": '{}'}]},
        {"role": "tool", "content": "Result 1"},
        {"role": "tool", "content": "Result 2"},
        {"role": "assistant", "content": "Both done"},
        {"role": "user", "content": "Thanks"},
    ]

    hf_output_2 = tokenizer.apply_chat_template(messages_2, tools=tools, tokenize=False, add_generation_prompt=True)
    cookbook_tokens_2 = cookbook_renderer.build_generation_prompt(messages_2).to_ints()
    cookbook_output_2 = tokenizer.decode(cookbook_tokens_2)

    print("HF Template Output:")
    print(hf_output_2)
    print("\nCookbook Renderer Output:")
    print(cookbook_output_2)
    print("\n")

    # Test 3: Tool call without tool response yet (generation scenario)
    print("Test 3: Tool call with generation prompt")
    print("-" * 80)
    messages_3 = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "Let me check", "tool_calls": [{"name": "get_weather", "arguments": '{"location": "SF"}'}]},
        {"role": "tool", "content": "Temperature is 72F"},
        {"role": "user", "content": "Thanks"},
    ]

    hf_output_3 = tokenizer.apply_chat_template(messages_3, tools=tools, tokenize=False, add_generation_prompt=True)
    cookbook_tokens_3 = cookbook_renderer.build_generation_prompt(messages_3).to_ints()
    cookbook_output_3 = tokenizer.decode(cookbook_tokens_3)

    print("HF Template Output:")
    print(hf_output_3)
    print("\nCookbook Renderer Output:")
    print(cookbook_output_3)
    print("\n")


if __name__ == "__main__":
    # test_against_hf_chat_templates("meta-llama/Llama-3.2-1B-Instruct")
    # test_against_hf_chat_templates("Qwen/Qwen2.5-VL-3B-Instruct")
    # test_generation_against_hf_chat_templates("openai/gpt-oss-20b")
    # test_supervised_example_against_hf_chat_templates("openai/gpt-oss-20b")
    # test_eot_parsing("Qwen/Qwen3-30B-A3B", "qwen3")
    test_qwen3_tool_calling()
