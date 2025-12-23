"""
Test the SFT dataloader with dynamic ask_sonnet injection.

Usage:
    uv run python -m tinker_cookbook.recipes.taubench.tests.test_dataloader
"""

import random
from tinker_cookbook.recipes.taubench.sft_dataset import (
    DynamicInjectionDataset,
    ConversationRecord,
    _inject_ask_sonnet_calls,
    _mark_trainable_fields,
    _normalize_tau2_messages,
)
from tinker_cookbook.recipes.taubench.components import AskSonnetMode
from tinker_cookbook.recipes.taubench.env import ASK_SONNET_TOOL
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.renderers import get_renderer, TrainOnWhat


def create_mock_conversation(num_turns: int = 5) -> list[dict]:
    """Create a mock conversation with alternating user/assistant messages."""
    messages = []
    messages.append({"role": "assistant", "content": "Hello! How can I help you today?"})

    for i in range(num_turns):
        messages.append({"role": "user", "content": f"User message {i+1}"})
        messages.append({"role": "assistant", "content": f"Assistant response {i+1}"})

    return messages


def test_injection_randomization():
    """Test that injection patterns differ across epochs."""
    print("=" * 60)
    print("TEST: Injection randomization across epochs")
    print("=" * 60)

    messages = create_mock_conversation(10)

    # Run injection with different seeds
    injection_patterns = []
    for seed in range(5):
        rng = random.Random(seed)
        injected = _inject_ask_sonnet_calls(
            [m.copy() for m in messages],
            injection_rate=0.5,
            rng=rng,
            mode=AskSonnetMode.CONDITIONING,
        )

        # Find which positions have ask_sonnet calls
        ask_sonnet_positions = []
        for i, msg in enumerate(injected):
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if hasattr(tc, 'function') and tc.function.name == "ask_sonnet":
                        ask_sonnet_positions.append(i)

        injection_patterns.append(tuple(ask_sonnet_positions))
        print(f"Seed {seed}: ask_sonnet at positions {ask_sonnet_positions}")

    # Check that patterns differ
    unique_patterns = set(injection_patterns)
    print(f"\nUnique patterns: {len(unique_patterns)} / {len(injection_patterns)}")

    assert len(unique_patterns) > 1, "Expected different injection patterns across seeds!"
    print("PASS: Injection patterns differ across seeds")


def test_dynamic_injection_dataset():
    """Test DynamicInjectionDataset re-randomizes on set_epoch."""
    print("\n" + "=" * 60)
    print("TEST: DynamicInjectionDataset epoch randomization")
    print("=" * 60)

    # Setup
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Create mock conversations with real tau2 domain for system prompt
    domain = "retail"  # Use real tau2 domain
    conversations = []
    for i in range(10):
        messages = create_mock_conversation(5)
        conv = ConversationRecord(
            messages=messages,
            task_id=f"task_{i}",
            domain=domain,
        )
        conversations.append(conv)

    # Get real tools from tau2
    from tinker_cookbook.recipes.taubench.sft_dataset import _get_tau2_tools
    domain_tools = {domain: _get_tau2_tools(domain) + [ASK_SONNET_TOOL]}

    # Create dataset with larger max_length to avoid truncation
    dataset = DynamicInjectionDataset(
        conversations=conversations,
        batch_size=2,
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        domain_tools=domain_tools,
        injection_rate=0.5,
        injection_mode=AskSonnetMode.CONDITIONING,
        max_length=16384,  # Large enough to avoid truncation
    )

    print(f"Dataset size: {len(dataset)} batches, {len(dataset.datums)} datums")

    # Get token lengths for epoch 0
    epoch0_lengths = [d.model_input.length for d in dataset.datums]
    print(f"Epoch 0 token lengths: {epoch0_lengths[:5]}...")

    # Set epoch 1 and check lengths changed
    dataset.set_epoch(1)
    epoch1_lengths = [d.model_input.length for d in dataset.datums]
    print(f"Epoch 1 token lengths: {epoch1_lengths[:5]}...")

    # Set epoch 2
    dataset.set_epoch(2)
    epoch2_lengths = [d.model_input.length for d in dataset.datums]
    print(f"Epoch 2 token lengths: {epoch2_lengths[:5]}...")

    # Lengths should differ because different messages get ask_sonnet injected
    # (and thus different amounts of advice text)
    assert epoch0_lengths != epoch1_lengths or epoch1_lengths != epoch2_lengths, \
        "Expected token lengths to vary across epochs due to different injection patterns"

    print("PASS: Dataset re-randomizes on set_epoch")


def test_batch_retrieval():
    """Test that batch retrieval works correctly."""
    print("\n" + "=" * 60)
    print("TEST: Batch retrieval")
    print("=" * 60)

    # Setup
    model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    tokenizer = get_tokenizer(model_name)
    renderer_name = get_recommended_renderer_name(model_name)
    renderer = get_renderer(renderer_name, tokenizer)

    # Create mock conversations with real tau2 domain
    domain = "retail"
    conversations = []
    for i in range(10):
        messages = create_mock_conversation(3)
        conv = ConversationRecord(
            messages=messages,
            task_id=f"task_{i}",
            domain=domain,
        )
        conversations.append(conv)

    from tinker_cookbook.recipes.taubench.sft_dataset import _get_tau2_tools
    domain_tools = {domain: _get_tau2_tools(domain) + [ASK_SONNET_TOOL]}

    dataset = DynamicInjectionDataset(
        conversations=conversations,
        batch_size=3,
        renderer=renderer,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        domain_tools=domain_tools,
        injection_rate=0.5,
        injection_mode=AskSonnetMode.CONDITIONING,
        max_length=16384,
    )

    print(f"Dataset: {len(conversations)} conversations, batch_size=3, {len(dataset)} batches")

    # Get all batches
    for i in range(len(dataset)):
        batch = dataset.get_batch(i)
        print(f"Batch {i}: {len(batch)} datums")
        assert len(batch) <= 3, f"Batch size exceeded: {len(batch)}"
        for datum in batch:
            assert datum.model_input is not None
            assert datum.loss_fn_inputs is not None

    print("PASS: Batch retrieval works correctly")


def test_conditioning_format():
    """Test that conditioning mode produces expected message structure."""
    print("\n" + "=" * 60)
    print("TEST: Conditioning mode message format")
    print("=" * 60)

    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "I need help"},
        {"role": "assistant", "content": "Sure, let me help you."},
    ]

    rng = random.Random(42)
    # Force injection by using rate=1.0
    injected = _inject_ask_sonnet_calls(
        [m.copy() for m in messages],
        injection_rate=1.0,
        rng=rng,
        mode=AskSonnetMode.CONDITIONING,
    )

    print("Original messages:")
    for i, m in enumerate(messages):
        print(f"  {i}: [{m['role']}] {m.get('content', '')[:50]}")

    print("\nInjected messages:")
    for i, m in enumerate(injected):
        role = m['role']
        content = m.get('content', '')[:50]
        tool_calls = m.get('tool_calls', [])
        tool_call_id = m.get('tool_call_id', '')

        if tool_calls:
            tc_names = [tc.function.name if hasattr(tc, 'function') else '?' for tc in tool_calls]
            print(f"  {i}: [{role}] tool_calls={tc_names}")
        elif tool_call_id:
            print(f"  {i}: [{role}] tool_call_id={tool_call_id}, content={content}...")
        else:
            print(f"  {i}: [{role}] {content}")

    # Verify structure: after user message, should have:
    # 1. assistant with ask_sonnet tool call
    # 2. tool with [Sonnet's Advice]
    # 3. assistant with the actual response

    # Find the ask_sonnet sequence
    found_sequence = False
    for i in range(len(injected) - 2):
        msg1, msg2, msg3 = injected[i], injected[i+1], injected[i+2]

        # Check for ask_sonnet tool call
        if msg1.get('tool_calls'):
            tc = msg1['tool_calls'][0]
            if hasattr(tc, 'function') and tc.function.name == 'ask_sonnet':
                # Check tool response has advice
                if msg2['role'] == 'tool' and '[Sonnet\'s Advice]' in msg2.get('content', ''):
                    # Check followup is assistant
                    if msg3['role'] == 'assistant':
                        found_sequence = True
                        print(f"\nFound conditioning sequence at positions {i}, {i+1}, {i+2}")
                        break

    assert found_sequence, "Expected to find ask_sonnet -> advice -> followup sequence"
    print("PASS: Conditioning format is correct")


def test_system_prompt_with_tau2():
    """Test that the tau2 system prompt is added with ask_sonnet instruction."""
    print("\n" + "=" * 60)
    print("TEST: Tau2 system prompt with ask_sonnet instruction")
    print("=" * 60)

    from tinker_cookbook.recipes.taubench.sft_dataset import (
        _add_system_prompt_with_ask_sonnet,
        _get_cached_tau2_system_prompt,
    )
    from tinker_cookbook.recipes.taubench.components import ASK_SONNET_INSTRUCTION

    # Test with retail domain
    domain = "retail"
    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "I need help"},
    ]

    # Get tau2 system prompt
    tau2_prompt = _get_cached_tau2_system_prompt(domain)
    print(f"Tau2 system prompt length: {len(tau2_prompt)} chars")
    print(f"First 200 chars: {tau2_prompt[:200]}...")

    # Add system prompt with ask_sonnet
    with_system = _add_system_prompt_with_ask_sonnet(messages.copy(), domain)

    assert with_system[0]["role"] == "system", "First message should be system"
    assert tau2_prompt in with_system[0]["content"], "Should contain tau2 system prompt"
    assert "ask_sonnet" in with_system[0]["content"], "Should contain ask_sonnet instruction"
    assert len(with_system) == len(messages) + 1, "Should have one more message (system)"

    print(f"System message length: {len(with_system[0]['content'])} chars")
    print("PASS: System prompt correctly includes tau2 prompt + ask_sonnet instruction")


def test_mark_trainable_fields():
    """Test that _mark_trainable_fields correctly sets trainable field on messages."""
    print("\n" + "=" * 60)
    print("TEST: _mark_trainable_fields function")
    print("=" * 60)

    messages = [
        {"role": "assistant", "content": "Hello!"},  # First assistant - not eligible
        {"role": "user", "content": "I need help"},
        {"role": "assistant", "content": "Sure, let me help you."},  # Eligible
        {"role": "user", "content": "Thanks"},
        {"role": "assistant", "content": "You're welcome."},  # Eligible
    ]

    rng = random.Random(42)
    # First inject ask_sonnet calls
    injected = _inject_ask_sonnet_calls(
        [m.copy() for m in messages],
        injection_rate=1.0,
        rng=rng,
        mode=AskSonnetMode.CONDITIONING,
    )

    # Then mark trainable fields
    marked = _mark_trainable_fields(injected)

    print("Messages after marking trainable fields:")
    trainable_count = 0
    ask_sonnet_trainable_count = 0
    for i, m in enumerate(marked):
        trainable = m.get("trainable", "NOT SET")
        role = m["role"]
        tool_calls = m.get("tool_calls", [])

        is_ask_sonnet_call = False
        if tool_calls:
            for tc in tool_calls:
                if hasattr(tc, 'function') and tc.function.name == "ask_sonnet":
                    is_ask_sonnet_call = True

        print(f"  {i}: [{role}] trainable={trainable}, is_ask_sonnet={is_ask_sonnet_call}")

        if trainable == True:
            trainable_count += 1
            if is_ask_sonnet_call:
                ask_sonnet_trainable_count += 1

    # Verify: only ask_sonnet calls should be trainable
    assert trainable_count > 0, "Expected at least one trainable message"
    assert trainable_count == ask_sonnet_trainable_count, \
        f"Only ask_sonnet calls should be trainable, but {trainable_count} trainable vs {ask_sonnet_trainable_count} ask_sonnet calls"

    # Verify all messages have trainable field set
    for m in marked:
        assert "trainable" in m, f"Message missing trainable field: {m}"

    print(f"\nTrainable messages: {trainable_count} (all are ask_sonnet calls)")
    print("PASS: _mark_trainable_fields correctly marks only ask_sonnet calls as trainable")


def test_mark_trainable_no_injection():
    """Test that _mark_trainable_fields works on messages without ask_sonnet calls."""
    print("\n" + "=" * 60)
    print("TEST: _mark_trainable_fields with no ask_sonnet calls")
    print("=" * 60)

    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "I need help"},
        {"role": "assistant", "content": "Sure."},
    ]

    # No injection, just mark trainable - should mark all as not trainable
    result = _mark_trainable_fields([m.copy() for m in messages])

    print("Messages with trainable field (no ask_sonnet):")
    for i, m in enumerate(result):
        trainable = m.get("trainable", "NOT SET")
        print(f"  {i}: [{m['role']}] trainable={trainable}")
        assert trainable == False, f"Expected trainable=False when no ask_sonnet, got {trainable}"

    print("PASS: All messages marked as not trainable when no ask_sonnet calls")


def test_system_prompt_trainable_field():
    """Test that system prompt gets trainable=False when _mark_trainable_fields is called after adding it."""
    print("\n" + "=" * 60)
    print("TEST: System prompt gets trainable field")
    print("=" * 60)

    from tinker_cookbook.recipes.taubench.sft_dataset import _add_system_prompt_with_ask_sonnet
    from tinker_cookbook.renderers import ToolCall

    # Simulate the flow: inject -> add system prompt -> mark trainable
    # Create a message with ask_sonnet tool call
    messages = [
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "I need help"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                ToolCall(
                    function=ToolCall.FunctionBody(name="ask_sonnet", arguments="{}")
                )
            ]
        },
    ]

    # Add system prompt (no trainable fields yet)
    with_system = _add_system_prompt_with_ask_sonnet(messages, "retail")

    # Now mark trainable fields
    result = _mark_trainable_fields(with_system)

    print("Messages after adding system prompt and marking trainable:")
    for i, m in enumerate(result):
        trainable = m.get("trainable", "NOT SET")
        print(f"  {i}: [{m['role']}] trainable={trainable}")

    # System message should be first and have trainable=False
    assert result[0]["role"] == "system", "First message should be system"
    assert result[0].get("trainable") == False, "System message should have trainable=False"

    # All messages should have trainable field
    for m in result:
        assert "trainable" in m, f"Message missing trainable field: {m['role']}"

    # The ask_sonnet call should be trainable=True
    ask_sonnet_msg = result[3]  # After system, assistant, user
    assert ask_sonnet_msg.get("trainable") == True, "ask_sonnet call should be trainable"

    print("PASS: System prompt correctly gets trainable=False")


def test_mark_trainable_single_assistant():
    """Test that _mark_trainable_fields works on messages without any ask_sonnet calls."""
    print("\n" + "=" * 60)
    print("TEST: _mark_trainable_fields with single assistant message")
    print("=" * 60)

    # Only one assistant message - after injection there would be no ask_sonnet calls
    messages = [
        {"role": "assistant", "content": "Hello!"},  # First assistant - not eligible for injection
        {"role": "user", "content": "Thanks, bye!"},
    ]

    rng = random.Random(42)
    # First inject (this won't add any ask_sonnet since only first assistant)
    injected = _inject_ask_sonnet_calls(
        [m.copy() for m in messages],
        injection_rate=1.0,  # Would inject if there were eligible turns
        rng=rng,
        mode=AskSonnetMode.CONDITIONING,
    )

    # Then mark trainable
    result = _mark_trainable_fields(injected)

    print("Messages with trainable field (single assistant, no ask_sonnet):")
    for i, m in enumerate(result):
        trainable = m.get("trainable", "NOT SET")
        print(f"  {i}: [{m['role']}] trainable={trainable}")
        assert "trainable" in m, f"Message missing trainable field: {m}"
        assert trainable == False, f"Expected trainable=False, got {trainable}"

    print("PASS: All messages marked as not trainable when no ask_sonnet calls")


def main():
    print("Testing SFT Dataloader with Dynamic Injection")
    print("=" * 60)

    test_injection_randomization()
    test_dynamic_injection_dataset()
    test_batch_retrieval()
    test_conditioning_format()
    test_system_prompt_with_tau2()
    test_mark_trainable_fields()
    test_mark_trainable_no_injection()
    test_system_prompt_trainable_field()
    test_mark_trainable_single_assistant()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
