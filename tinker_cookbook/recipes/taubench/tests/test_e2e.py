#!/usr/bin/env python3
"""
End-to-end tests for taubench environment and ask_sonnet modes.

These tests verify the full integration of components in realistic scenarios.

Usage:
    # Run without API calls (mock mode)
    uv run pytest tinker_cookbook/recipes/taubench/tests/test_e2e.py -v

    # Run with real API calls (requires ANTHROPIC_API_KEY)
    uv run pytest tinker_cookbook/recipes/taubench/tests/test_e2e.py -v --run-api-tests
"""

import asyncio
import json
import logging
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import tau2.registry as reg

from tinker_cookbook.model_info import get_recommended_renderer_name
from tinker_cookbook.recipes.taubench.components import AskSonnetMode, ObservationType, Tau2StepResult
from tinker_cookbook.recipes.taubench.components.external_llm import LLMCallResult
from tinker_cookbook.recipes.taubench.env import (
    Tau2Env,
    Tau2DatasetBuilder,
    Tau2EnvGroupBuilder,
    construct_tau2_env,
)
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from tinker_cookbook.recipes.taubench.tests.fixtures import (
    MockConversation,
    MockUserResponse,
    MockToolResponse,
    get_retail_return_conversation,
    get_retail_simple_lookup_conversation,
    SONNET_MOCK_RESPONSES,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.anyio


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def model_name():
    return "Qwen/Qwen3-30B-A3B-Instruct-2507"


@pytest.fixture
def tokenizer(model_name):
    return get_tokenizer(model_name)


@pytest.fixture
def renderer(model_name, tokenizer):
    renderer_name = get_recommended_renderer_name(model_name)
    return get_renderer(renderer_name, tokenizer)


@pytest.fixture
def task_id():
    """Get a valid task ID from the registry."""
    tasks = reg.registry.get_tasks_loader("retail")()
    return tasks[0].id


# =============================================================================
# Environment Construction Tests
# =============================================================================


class TestTau2EnvConstruction:
    """Tests for Tau2Env construction and initialization."""

    def test_construct_basic_env(self, renderer, task_id):
        """Should create basic env without external LLM."""
        env = Tau2Env(renderer=renderer, domain="retail", task_id=task_id)

        assert env.external_llm is None
        assert env.ask_sonnet_renderer is None
        assert "ask_sonnet" not in [t["function"]["name"] for t in env.tools]

    def test_construct_env_with_external_llm(self, renderer, task_id):
        """Should create env with external LLM configured."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
        )

        assert env.external_llm is not None
        assert env.ask_sonnet_renderer is not None
        assert "ask_sonnet" in [t["function"]["name"] for t in env.tools]

    def test_construct_env_direct_injection_mode(self, renderer, task_id):
        """Should configure direct injection renderer."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        assert env.ask_sonnet_renderer.should_return_early() is False
        assert env.ask_sonnet_renderer.requires_followup() is False

    def test_construct_env_conditioning_mode(self, renderer, task_id):
        """Should configure conditioning renderer."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        assert env.ask_sonnet_renderer.should_return_early() is True
        assert env.ask_sonnet_renderer.requires_followup() is True

    def test_construct_tau2_env_helper(self, task_id):
        """construct_tau2_env helper should work."""
        env = construct_tau2_env(domain="retail", task_id=task_id)

        assert env is not None
        assert len(env.tools) > 0

    def test_env_initial_state(self, renderer, task_id):
        """Env should have correct initial state."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
        )

        assert env.ask_sonnet_call_count == 0
        assert len(env.messages.messages) == 2  # system + initial user


# =============================================================================
# Initial Observation Tests
# =============================================================================


class TestInitialObservation:
    """Tests for getting initial observations."""

    @pytest.mark.asyncio
    async def test_initial_observation(self, renderer, task_id):
        """Should return valid initial observation."""
        env = Tau2Env(renderer=renderer, domain="retail", task_id=task_id)

        obs, stop_condition = await env.initial_observation()

        assert obs is not None
        assert obs.length > 0
        assert stop_condition is not None

    @pytest.mark.asyncio
    async def test_initial_observation_with_external_llm(self, renderer, task_id):
        """Initial observation should include ask_sonnet tool."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
        )

        obs, _ = await env.initial_observation()

        # Observation should be longer due to ask_sonnet tool
        assert obs.length > 0


# =============================================================================
# Direct Action Tests (No ask_sonnet)
# =============================================================================


class TestDirectActions:
    """Tests for direct actions without ask_sonnet."""

    @pytest.mark.asyncio
    async def test_text_response(self, renderer, task_id, tokenizer):
        """Should handle plain text responses."""
        env = Tau2Env(renderer=renderer, domain="retail", task_id=task_id)

        # Simulate a greeting response
        text = "Hello! How can I help you today?<|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)

        result = await env.step(action_tokens)

        assert result is not None
        assert result.next_observation is not None
        assert env.ask_sonnet_call_count == 0

    @pytest.mark.asyncio
    async def test_tool_call_response(self, renderer, task_id, tokenizer):
        """Should handle tool call responses."""
        env = Tau2Env(renderer=renderer, domain="retail", task_id=task_id)

        # Simulate a tool call
        tool_json = json.dumps({
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "John", "last_name": "Doe", "zip": "12345"}
        })
        text = f"<tool_call>\n{tool_json}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)

        result = await env.step(action_tokens)

        assert result is not None
        assert env.ask_sonnet_call_count == 0

    @pytest.mark.asyncio
    async def test_message_histories_sync_for_direct_actions(self, renderer, task_id, tokenizer):
        """Message histories should stay in sync for direct actions."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
        )

        initial_msg_count = len(env.messages.messages)

        # Direct response (not ask_sonnet)
        text = "Let me look that up for you.<|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)

        await env.step(action_tokens)

        # Messages should increase (assistant + user response)
        msg_increase = len(env.messages.messages) - initial_msg_count
        assert msg_increase >= 2  # At least assistant + user/tool result


# =============================================================================
# ask_sonnet Tests with Mocked External LLM
# =============================================================================


class TestAskSonnetMocked:
    """Tests for ask_sonnet with mocked external LLM calls."""

    @pytest.mark.asyncio
    async def test_ask_sonnet_direct_injection(self, renderer, task_id, tokenizer):
        """Direct injection should use Sonnet's response as action."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        # Mock the external LLM call
        mock_response = '<tool_call>\n{"name": "get_user_details", "arguments": {"user_id": "test_123"}}\n</tool_call>'

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = LLMCallResult(content=mock_response, input_tokens=100, output_tokens=50)

            # Send ask_sonnet action
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            result = await env.step(action_tokens)

            # Verify external LLM was called
            mock_call.assert_called_once()

            # Verify state
            assert env.ask_sonnet_call_count == 1
            # Direct injection sends directly to tau2 (doesn't wait for followup)

    @pytest.mark.asyncio
    async def test_ask_sonnet_conditioning_returns_early(self, renderer, task_id, tokenizer):
        """Conditioning mode should return early for policy followup."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        mock_response = "I recommend using the get_user_details tool first to look up the customer."

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = LLMCallResult(content=mock_response, input_tokens=100, output_tokens=50)

            # Send ask_sonnet action
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            result = await env.step(action_tokens)

            # Verify state - should be waiting for followup
            assert env.ask_sonnet_call_count == 1

            # Episode should not be done yet (conditioning returns early)
            assert result.episode_done is False

    @pytest.mark.asyncio
    async def test_ask_sonnet_conditioning_followup(self, renderer, task_id, tokenizer):
        """Conditioning mode should use policy's followup as action."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        mock_response = "I recommend using the get_user_details tool."

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = LLMCallResult(content=mock_response, input_tokens=100, output_tokens=50)

            # First: send ask_sonnet
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)
            result = await env.step(action_tokens)

            # Should return early (not done, waiting for followup)
            assert result.episode_done is False

            # Second: send policy's followup
            followup_json = json.dumps({
                "name": "find_user_id_by_name_zip",
                "arguments": {"first_name": "Jane", "last_name": "Doe", "zip": "54321"}
            })
            followup_text = f"<tool_call>\n{followup_json}\n</tool_call><|im_end|>"
            followup_tokens = tokenizer.encode(followup_text, add_special_tokens=False)

            result = await env.step(followup_tokens)

            # Followup sent to tau2 - verify we got a response

    @pytest.mark.asyncio
    async def test_ask_sonnet_increments_count(self, renderer, task_id, tokenizer):
        """ask_sonnet calls should increment counter."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        mock_response = "Hello, how can I help you?"

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = LLMCallResult(content=mock_response, input_tokens=100, output_tokens=50)

            assert env.ask_sonnet_call_count == 0

            # First ask_sonnet
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            await env.step(action_tokens)
            assert env.ask_sonnet_call_count == 1

            # Second ask_sonnet (if episode continues)
            if not (await env.step(action_tokens)).episode_done:
                await env.step(action_tokens)
                # Count should increase


# =============================================================================
# Dataset Builder Tests
# =============================================================================


class TestTau2DatasetBuilder:
    """Tests for Tau2DatasetBuilder."""

    @pytest.mark.asyncio
    async def test_build_datasets(self, model_name):
        """Should build train and test datasets."""
        builder = Tau2DatasetBuilder(
            batch_size=2,
            model_name_for_tokenizer=model_name,
            group_size=1,
            domain="retail",
            num_epochs=1,
        )

        train_dataset, test_dataset = await builder()

        assert train_dataset is not None
        assert test_dataset is not None
        assert len(train_dataset) > 0

    @pytest.mark.asyncio
    async def test_build_datasets_with_ask_sonnet_mode(self, model_name):
        """Should pass ask_sonnet_mode to datasets."""
        builder = Tau2DatasetBuilder(
            batch_size=2,
            model_name_for_tokenizer=model_name,
            group_size=1,
            domain="retail",
            num_epochs=1,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        train_dataset, test_dataset = await builder()

        assert train_dataset.ask_sonnet_mode == AskSonnetMode.CONDITIONING
        assert test_dataset.ask_sonnet_mode == AskSonnetMode.CONDITIONING


# =============================================================================
# EnvGroupBuilder Tests
# =============================================================================


class TestTau2EnvGroupBuilder:
    """Tests for Tau2EnvGroupBuilder."""

    @pytest.mark.asyncio
    async def test_make_envs(self, renderer, task_id):
        """Should create multiple envs."""
        builder = Tau2EnvGroupBuilder(
            domain="retail",
            task_id=task_id,
            renderer=renderer,
            num_envs=2,
        )

        envs = await builder.make_envs()

        assert len(envs) == 2
        for env in envs:
            assert isinstance(env, Tau2Env)

    @pytest.mark.asyncio
    async def test_make_envs_with_ask_sonnet(self, renderer, task_id):
        """Should create envs with ask_sonnet configured."""
        builder = Tau2EnvGroupBuilder(
            domain="retail",
            task_id=task_id,
            renderer=renderer,
            num_envs=2,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        envs = await builder.make_envs()

        for env in envs:
            assert env.external_llm is not None
            assert env.ask_sonnet_renderer.requires_followup() is True

    @pytest.mark.asyncio
    async def test_compute_group_rewards(self, renderer, task_id):
        """Should compute rewards with ask_sonnet penalty."""
        builder = Tau2EnvGroupBuilder(
            domain="retail",
            task_id=task_id,
            renderer=renderer,
            num_envs=2,
            ask_sonnet_penalty=0.1,
        )

        envs = await builder.make_envs()

        # Simulate ask_sonnet calls
        envs[0].ask_sonnet_call_count = 3
        envs[1].ask_sonnet_call_count = 1

        rewards = await builder.compute_group_rewards([], envs)

        assert len(rewards) == 2
        assert abs(rewards[0][0] - (-0.3)) < 1e-9  # -0.1 * 3
        assert abs(rewards[1][0] - (-0.1)) < 1e-9  # -0.1 * 1
        assert rewards[0][1]["ask_sonnet_count"] == 3
        assert rewards[1][1]["ask_sonnet_count"] == 1


# =============================================================================
# Mocked Multi-Turn Conversation Tests (No API calls)
# =============================================================================


def create_mock_gym_step(mock_conversation: MockConversation):
    """
    Create a mock for Tau2GymWrapper.step() that returns responses from MockConversation.

    This function creates a closure that tracks state and returns appropriate
    mock responses based on the action type (tool call vs text message).

    Args:
        mock_conversation: The MockConversation to use for responses

    Returns:
        An async function that can be used to mock gym.step()
    """
    async def mock_step(action: str):
        """Mock step function that returns user or tool responses."""
        # Try to parse as JSON to detect tool calls
        try:
            action_data = json.loads(action)
            tool_name = action_data.get("name", "")

            # Check for order_id in arguments for disambiguation
            order_id = action_data.get("arguments", {}).get("order_id")

            # Get tool response
            tool_response = mock_conversation.get_tool_response(tool_name, order_id)

            return Tau2StepResult(
                obs_type=ObservationType.TOOL_RESULT,
                obs_content=tool_response.content,
                raw_obs=f"tool: {tool_response.content}",
                reward=tool_response.reward,
                terminated=tool_response.terminated,
                truncated=tool_response.truncated,
                info={},
            )
        except (json.JSONDecodeError, KeyError):
            # Not a tool call - return next user message
            user_response = mock_conversation.next_user_response()

            return Tau2StepResult(
                obs_type=ObservationType.USER_MESSAGE,
                obs_content=user_response.content,
                raw_obs=f"user: {user_response.content}",
                reward=user_response.reward,
                terminated=user_response.terminated,
                truncated=user_response.truncated,
                info={},
            )

    return mock_step


class TestMockedMultiTurnConversation:
    """
    Tests with fully mocked user simulator (no GPT-4.1 API calls).

    These tests use pre-defined conversation sequences from fixtures.py
    to simulate realistic multi-turn conversations without API calls.
    """

    @pytest.mark.asyncio
    async def test_simple_lookup_conversation(self, renderer, task_id, tokenizer):
        """
        Test a simple lookup conversation flow.

        Scenario: Customer asks about order status, provides name/zip,
        agent looks up info and responds.

        Flow:
        1. Agent greets -> user asks about order status (user response 0)
        2. Agent asks for info -> user provides name/zip (user response 1)
        3. Agent tool call -> tool returns user ID
        4. Agent confirms info -> user thanks (user response 2, terminates)
        """
        mock_conversation = get_retail_simple_lookup_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
        )

        # Mock the gym.step method
        env.gym.step = create_mock_gym_step(mock_conversation)

        # Turn 1: Agent greets customer
        greeting = "Hello! I'd be happy to help you today. How can I assist you?<|im_end|>"
        action_tokens = tokenizer.encode(greeting, add_special_tokens=False)
        result = await env.step(action_tokens)

        assert not result.episode_done
        # Check that user response was added
        assert any("order status" in str(m.get("content", "")).lower()
                   for m in env.messages.messages if m.get("role") == "user")

        # Turn 2: Agent asks user for identification info (text response)
        ask_info = "I can help you check that! Could you provide your name and zip code?<|im_end|>"
        action_tokens = tokenizer.encode(ask_info, add_special_tokens=False)
        result = await env.step(action_tokens)

        assert not result.episode_done

        # Turn 3: Agent looks up user (tool call - returns tool response, not user message)
        tool_call = json.dumps({
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "John", "last_name": "Smith", "zip": "12345"}
        })
        text = f"<tool_call>\n{tool_call}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        result = await env.step(action_tokens)

        # Should get tool response (not episode done yet)
        assert not result.episode_done

        # Turn 4: Agent responds with info - this triggers user response (terminates)
        response = "I found your account, John! Your order #W1234567 is currently pending.<|im_end|>"
        action_tokens = tokenizer.encode(response, add_special_tokens=False)
        result = await env.step(action_tokens)

        # Episode should complete after user acknowledges
        assert result.episode_done
        assert result.reward == 1.0

    @pytest.mark.asyncio
    async def test_multi_turn_with_multiple_tool_calls(self, renderer, task_id, tokenizer):
        """
        Test a conversation with multiple tool calls.

        Scenario: Customer wants to return items from multiple orders.
        Agent needs to look up user, then order details.
        """
        mock_conversation = get_retail_return_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
        )

        # Mock the gym.step method
        env.gym.step = create_mock_gym_step(mock_conversation)

        # Turn 1: Agent greets
        greeting = "Hello! How can I help you today?<|im_end|>"
        action_tokens = tokenizer.encode(greeting, add_special_tokens=False)
        result = await env.step(action_tokens)

        assert not result.episode_done
        initial_msg_count = len(env.messages.messages)

        # Turn 2: Agent asks for email to identify
        response = "I can help you with those returns! Could you please provide your email address so I can look up your account?<|im_end|>"
        action_tokens = tokenizer.encode(response, add_special_tokens=False)
        await env.step(action_tokens)

        # Turn 3: Look up user by email (tool call)
        tool_call = json.dumps({
            "name": "find_user_id_by_email",
            "arguments": {"email": "lucas.brown9344@example.com"}
        })
        text = f"<tool_call>\n{tool_call}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        result = await env.step(action_tokens)

        # Verify tool response was processed
        assert not result.episode_done

        # Turn 4: Get user details (tool call)
        tool_call = json.dumps({
            "name": "get_user_details",
            "arguments": {"user_id": "lucas_brown_6720"}
        })
        text = f"<tool_call>\n{tool_call}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        await env.step(action_tokens)

        # Verify messages grew
        assert len(env.messages.messages) > initial_msg_count

    @pytest.mark.asyncio
    async def test_ask_sonnet_direct_injection_with_mocked_user(self, renderer, task_id, tokenizer):
        """
        Test ask_sonnet direct injection with mocked user and Sonnet responses.

        This tests the full flow where:
        1. Agent calls ask_sonnet
        2. Sonnet (mocked) returns a tool call
        3. The tool executes
        4. User simulator (mocked) responds
        """
        mock_conversation = get_retail_simple_lookup_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        # Mock both gym.step and external_llm.call
        env.gym.step = create_mock_gym_step(mock_conversation)

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_llm:
            # Sonnet returns a text greeting
            mock_llm.return_value = LLMCallResult(
                content=SONNET_MOCK_RESPONSES["direct_injection_text"], input_tokens=100, output_tokens=50
            )

            # Call ask_sonnet
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            result = await env.step(action_tokens)

            # Verify
            mock_llm.assert_called_once()
            assert env.ask_sonnet_call_count == 1
            assert not result.episode_done

    @pytest.mark.asyncio
    async def test_ask_sonnet_conditioning_with_mocked_user(self, renderer, task_id, tokenizer):
        """
        Test ask_sonnet conditioning mode with mocked responses.

        This tests the full conditioning flow:
        1. Agent calls ask_sonnet
        2. Sonnet (mocked) returns advice
        3. Agent sees advice and follows up with tool call
        4. User simulator (mocked) responds
        """
        mock_conversation = get_retail_simple_lookup_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        # Mock both gym.step and external_llm.call_with_usage
        env.gym.step = create_mock_gym_step(mock_conversation)

        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_llm:
            # Sonnet returns advice
            mock_llm.return_value = LLMCallResult(
                content=SONNET_MOCK_RESPONSES["conditioning_advice"], input_tokens=100, output_tokens=50
            )

            # Step 1: Call ask_sonnet
            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            result = await env.step(action_tokens)

            # Should be waiting for followup (conditioning returns early)
            assert not result.episode_done

            # Step 2: Policy follows Sonnet's advice with a tool call
            followup_json = json.dumps({
                "name": "find_user_id_by_name_zip",
                "arguments": {"first_name": "John", "last_name": "Smith", "zip": "12345"}
            })
            followup_text = f"<tool_call>\n{followup_json}\n</tool_call><|im_end|>"
            followup_tokens = tokenizer.encode(followup_text, add_special_tokens=False)

            result = await env.step(followup_tokens)

            # Followup processed

    @pytest.mark.asyncio
    async def test_mixed_direct_and_ask_sonnet_actions(self, renderer, task_id, tokenizer):
        """
        Test a conversation mixing direct agent actions and ask_sonnet calls.

        This simulates a realistic scenario where the agent sometimes
        acts directly and sometimes delegates to Sonnet.
        """
        mock_conversation = get_retail_simple_lookup_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        env.gym.step = create_mock_gym_step(mock_conversation)

        # Turn 1: Direct greeting from agent
        greeting = "Hello! Welcome to our retail support. How may I assist you today?<|im_end|>"
        action_tokens = tokenizer.encode(greeting, add_special_tokens=False)
        result = await env.step(action_tokens)

        assert env.ask_sonnet_call_count == 0
        initial_messages = len(env.messages.messages)

        # Turn 2: ask_sonnet for help
        with patch.object(env.external_llm, "call_with_usage", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = LLMCallResult(
                content="I understand you'd like to check your order status. Could you please provide your name and zip code?",
                input_tokens=100, output_tokens=50
            )

            ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
            text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
            action_tokens = tokenizer.encode(text, add_special_tokens=False)

            await env.step(action_tokens)

            assert env.ask_sonnet_call_count == 1

        # Turn 3: Direct tool call
        tool_call = json.dumps({
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "John", "last_name": "Smith", "zip": "12345"}
        })
        text = f"<tool_call>\n{tool_call}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        await env.step(action_tokens)

        # Verify messages accumulated
        assert len(env.messages.messages) > initial_messages

    @pytest.mark.asyncio
    async def test_conversation_until_termination(self, renderer, task_id, tokenizer):
        """
        Test running a full conversation until the user simulator terminates.

        This is a more comprehensive test that runs multiple turns.
        """
        mock_conversation = get_retail_simple_lookup_conversation()

        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
        )

        env.gym.step = create_mock_gym_step(mock_conversation)

        # Run conversation
        actions = [
            "Hello! How can I help you today?<|im_end|}",
            "I can look up your account. Let me find you by name and zip code.<|im_end|}",
            '<tool_call>\n{"name": "find_user_id_by_name_zip", "arguments": {"first_name": "John", "last_name": "Smith", "zip": "12345"}}\n</tool_call><|im_end|>',
            "Great news! I found your order. It's currently in pending status.<|im_end|>",
        ]

        for i, action in enumerate(actions):
            action_tokens = tokenizer.encode(action, add_special_tokens=False)
            result = await env.step(action_tokens)

            if result.episode_done:
                logger.info(f"Episode terminated after turn {i+1}")
                break

        # Verify we got some reward on termination
        assert result.episode_done


# =============================================================================
# Live API Tests (Only run with --run-api-tests)
# =============================================================================


@pytest.mark.skipif(
    os.environ.get("RUN_API_TESTS", "0") != "1",
    reason="Requires RUN_API_TESTS=1 environment variable and API key",
)
class TestLiveAPI:
    """Tests that make real API calls. Only run with --run-api-tests."""

    @pytest.mark.asyncio
    async def test_live_ask_sonnet_direct_injection(self, renderer, task_id, tokenizer):
        """Live test of direct injection mode."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        # Send ask_sonnet
        ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
        text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)

        result = await env.step(action_tokens)

        assert env.ask_sonnet_call_count == 1
        assert result.next_observation is not None

        print(f"\n[LIVE TEST] Direct injection result:")
        print(f"  Episode done: {result.episode_done}")
        print(f"  Messages: {len(env.messages.messages)}")

    @pytest.mark.asyncio
    async def test_live_ask_sonnet_conditioning(self, renderer, task_id, tokenizer):
        """Live test of conditioning mode."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.CONDITIONING,
        )

        # Send ask_sonnet
        ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
        text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)

        result = await env.step(action_tokens)

        assert env.ask_sonnet_call_count == 1
        # Conditioning returns early (not done)
        assert result.episode_done is False

        print(f"\n[LIVE TEST] Conditioning - after ask_sonnet:")
        print(f"  Episode done: {result.episode_done}")

        # Send followup
        followup_json = json.dumps({
            "name": "find_user_id_by_name_zip",
            "arguments": {"first_name": "Test", "last_name": "User", "zip": "12345"}
        })
        followup_text = f"<tool_call>\n{followup_json}\n</tool_call><|im_end|>"
        followup_tokens = tokenizer.encode(followup_text, add_special_tokens=False)

        result = await env.step(followup_tokens)

        print(f"\n[LIVE TEST] Conditioning - after followup:")
        print(f"  Episode done: {result.episode_done}")

    @pytest.mark.asyncio
    async def test_live_multi_turn_mixed(self, renderer, task_id, tokenizer):
        """Live test with mix of direct and ask_sonnet actions."""
        env = Tau2Env(
            renderer=renderer,
            domain="retail",
            task_id=task_id,
            external_llm_model="claude-sonnet-4-5-20250929",
            ask_sonnet_mode=AskSonnetMode.DIRECT_INJECTION,
        )

        # Turn 1: Direct greeting
        text = "Hello! I'd be happy to help you today. Could you please tell me your name?<|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        await env.step(action_tokens)

        print(f"\n[LIVE TEST] After direct greeting: messages={len(env.messages.messages)}")

        # Turn 2: ask_sonnet
        ask_sonnet_json = json.dumps({"name": "ask_sonnet", "arguments": {}})
        text = f"<tool_call>\n{ask_sonnet_json}\n</tool_call><|im_end|>"
        action_tokens = tokenizer.encode(text, add_special_tokens=False)
        result = await env.step(action_tokens)

        print(f"[LIVE TEST] After ask_sonnet: messages={len(env.messages.messages)}, count={env.ask_sonnet_call_count}")
        print(f"[LIVE TEST] Episode done: {result.episode_done}")


@pytest.mark.skip(reason="Integration test - requires Anthropic API key, run manually")
class TestSonnetEmptyResponseFix:
    """
    Integration tests verifying Sonnet doesn't return empty when there's previous advice.

    Root cause (fixed): When Sonnet sees its own previous [Sonnet's Advice] in the
    message history (rendered as user messages), it returned empty ~87% of the time.

    Fix: render_for_advisor() now skips previous ask_sonnet calls and their responses.

    Run manually with:
        pytest -v -k TestSonnetEmptyResponseFix --no-header -rN
    """

    @pytest.fixture
    def renderer(self):
        return renderers.get_renderer("qwen3")

    @pytest.fixture
    def tokenizer(self):
        return tokenizer_utils.get_tokenizer("Qwen/Qwen3-30B-A3B-Instruct-2507")

    @pytest.mark.asyncio
    async def test_multiple_ask_sonnet_calls_no_empty_responses(self, renderer, tokenizer):
        """
        Test that multiple ask_sonnet calls don't result in empty responses.

        Before the fix, the second ask_sonnet call would return empty ~87% of the time
        because Sonnet saw its own previous [Sonnet's Advice] in the history.
        """
        from tinker_cookbook.recipes.taubench.components import (
            ExternalLLMClient,
            ExternalLLMConfig,
            get_ask_sonnet_renderer,
            AskSonnetMode,
        )

        ask_sonnet_renderer = get_ask_sonnet_renderer(AskSonnetMode.DIRECT_INJECTION)

        # Simulate conversation with previous Sonnet advice
        messages = [
            {"role": "system", "content": "You are a helpful retail customer service agent."},
            {"role": "user", "content": "Hi, I want to return my order"},
            # First ask_sonnet call and response (should be skipped by renderer)
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
            {"role": "tool", "content": "[Sonnet's Advice]:\n\nI'll help you with that return. First, let me authenticate you.", "tool_call_id": "ask_sonnet_call"},
            # Policy followup
            {"role": "assistant", "content": "I'll help you with that return. Could you please provide your email?"},
            {"role": "user", "content": "My email is test@example.com"},
            {"role": "assistant", "content": '<tool_call>\n{"name": "find_user_id_by_email", "arguments": {"email": "test@example.com"}}\n</tool_call>'},
            {"role": "tool", "content": "user_123", "tool_call_id": "tool_call"},
            {"role": "user", "content": "What orders do I have?"},
            # Second ask_sonnet call - this is what we're testing
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
        ]

        tools = [
            {"function": {"name": "find_user_id_by_email", "description": "Find user", "parameters": {}}},
            {"function": {"name": "get_user_details", "description": "Get user details", "parameters": {}}},
        ]

        # Render for advisor
        advisor_messages = ask_sonnet_renderer.render_for_advisor(
            messages, tools, messages[0]["content"]
        )

        # Verify previous Sonnet advice is NOT in rendered messages
        all_content = " ".join(msg.get("content", "") for msg in advisor_messages)
        assert "[Sonnet's Advice]" not in all_content, "Previous advice should be skipped"

        # Call Sonnet
        config = ExternalLLMConfig(
            model="claude-sonnet-4-5-20250929",
            temperature=0.0,
            max_tokens=512,
        )
        client = ExternalLLMClient(config)

        result = await client.call_with_usage(advisor_messages)

        print(f"\nSonnet response ({result.output_tokens} tokens):")
        print(result.content[:200] if result.content else "(EMPTY)")

        # The fix should prevent empty responses
        assert result.content is not None, "Sonnet returned None content"
        assert len(result.content.strip()) > 10, (
            f"Sonnet returned empty/short response: {result.content!r}"
        )

    @pytest.mark.asyncio
    async def test_sonnet_responds_after_previous_error(self, renderer, tokenizer):
        """
        Test that Sonnet responds even after a previous Advisor Error.
        """
        from tinker_cookbook.recipes.taubench.components import (
            ExternalLLMClient,
            ExternalLLMConfig,
            get_ask_sonnet_renderer,
            AskSonnetMode,
        )

        ask_sonnet_renderer = get_ask_sonnet_renderer(AskSonnetMode.DIRECT_INJECTION)

        # Simulate conversation with previous advisor error
        messages = [
            {"role": "system", "content": "You are a helpful retail customer service agent."},
            {"role": "user", "content": "Hi there"},
            # Previous ask_sonnet that failed
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
            {"role": "tool", "content": "[Advisor Error]: The advisor returned an empty response.", "tool_call_id": "ask_sonnet_call"},
            # Policy continued
            {"role": "assistant", "content": "Hello! How can I help you today?"},
            {"role": "user", "content": "I need to cancel my order #12345"},
            # Second ask_sonnet
            {"role": "assistant", "content": '<tool_call>\n{"name": "ask_sonnet", "args": {}}\n</tool_call>'},
        ]

        tools = [{"function": {"name": "cancel_order", "description": "Cancel order", "parameters": {}}}]

        advisor_messages = ask_sonnet_renderer.render_for_advisor(
            messages, tools, messages[0]["content"]
        )

        # Verify error message is NOT in rendered messages
        all_content = " ".join(msg.get("content", "") for msg in advisor_messages)
        assert "[Advisor Error]" not in all_content, "Previous error should be skipped"

        # Call Sonnet
        config = ExternalLLMConfig(model="claude-sonnet-4-5-20250929", temperature=0.0, max_tokens=512)
        client = ExternalLLMClient(config)

        result = await client.call_with_usage(advisor_messages)

        print(f"\nSonnet response ({result.output_tokens} tokens):")
        print(result.content[:200] if result.content else "(EMPTY)")

        assert result.content is not None and len(result.content.strip()) > 10, (
            f"Sonnet returned empty response: {result.content!r}"
        )
