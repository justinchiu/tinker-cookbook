"""Tests for Tau2GymWrapper — observation parsing, tool conversion, error handling.

Key properties tested:
1. Observation parsing: "user: ..." → USER_MESSAGE, "tool: ..." → TOOL_RESULT, other → OTHER
2. Tool conversion: tau2 tool format → OpenAI function format
3. JSON decode error handling: malformed actions don't crash, produce error observation
4. Empty observation on termination doesn't warn
5. Step returns structured Tau2StepResult
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tinker_cookbook.recipes.taubench.components.types import ObservationType, Tau2StepResult
from tinker_cookbook.recipes.taubench.components.tau2_gym_wrapper import Tau2GymWrapper


# ---------------------------------------------------------------------------
# Observation parsing (_parse_observation)
# ---------------------------------------------------------------------------


class TestParseObservation:
    """Test observation string parsing into typed (ObservationType, content) pairs."""

    def test_user_message(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, content = wrapper._parse_observation("user: Hello, I need help")
        assert obs_type == ObservationType.USER_MESSAGE
        assert content == "Hello, I need help"

    def test_tool_result(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, content = wrapper._parse_observation("tool: Order #123 found")
        assert obs_type == ObservationType.TOOL_RESULT
        assert content == "Order #123 found"

    def test_tool_result_json(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        json_str = json.dumps({"status": "shipped", "tracking": "XYZ"})
        obs_type, content = wrapper._parse_observation(f"tool: {json_str}")
        assert obs_type == ObservationType.TOOL_RESULT
        assert json.loads(content)["status"] == "shipped"

    def test_other_format(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, content = wrapper._parse_observation("some unexpected format")
        assert obs_type == ObservationType.OTHER
        assert content == "some unexpected format"

    def test_empty_string_is_other(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, content = wrapper._parse_observation("")
        assert obs_type == ObservationType.OTHER
        assert content == ""

    def test_empty_on_termination_no_warning(self, caplog):
        """Empty observation when terminated=True should NOT produce a warning."""
        import logging

        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        with caplog.at_level(logging.WARNING):
            obs_type, content = wrapper._parse_observation("", terminated=True)
        assert obs_type == ObservationType.OTHER
        assert "Unexpected obs format" not in caplog.text

    def test_empty_not_terminated_warns(self, caplog):
        """Empty observation when NOT terminated should warn."""
        import logging

        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        with caplog.at_level(logging.WARNING):
            obs_type, _ = wrapper._parse_observation("", terminated=False)
        assert obs_type == ObservationType.OTHER
        assert "Unexpected obs format" in caplog.text

    def test_user_prefix_exact(self):
        """'user:' without space is OTHER, not USER_MESSAGE."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, _ = wrapper._parse_observation("user:no space")
        assert obs_type == ObservationType.OTHER

    def test_tool_prefix_exact(self):
        """'tool:' without space is OTHER, not TOOL_RESULT."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        obs_type, _ = wrapper._parse_observation("tool:no space")
        assert obs_type == ObservationType.OTHER


# ---------------------------------------------------------------------------
# Tool format conversion
# ---------------------------------------------------------------------------


class TestGetTools:
    def test_converts_to_openai_format(self):
        """Tau2 tools should be converted to OpenAI function calling format."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        # Mock the gym environment and its tools
        mock_tool = MagicMock()
        mock_tool.model_dump_json.return_value = json.dumps(
            {
                "name": "get_order_details",
                "short_desc": "Look up order by ID",
                "long_desc": "Look up order details by order ID",
                "params": {
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"],
                },
            }
        )

        mock_env = MagicMock()
        mock_env._get_tools.return_value = [mock_tool]
        wrapper.env = mock_env

        tools = wrapper.get_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_order_details"
        assert tools[0]["function"]["description"] == "Look up order by ID"
        assert "order_id" in tools[0]["function"]["parameters"]["properties"]

    def test_uses_long_desc_when_short_desc_empty(self):
        """Falls back to long_desc when short_desc is empty."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        mock_tool = MagicMock()
        mock_tool.model_dump_json.return_value = json.dumps(
            {
                "name": "check_status",
                "short_desc": "",
                "long_desc": "Check the status of an item",
                "params": {"type": "object", "properties": {}},
            }
        )

        mock_env = MagicMock()
        mock_env._get_tools.return_value = [mock_tool]
        wrapper.env = mock_env

        tools = wrapper.get_tools()
        assert tools[0]["function"]["description"] == "Check the status of an item"


# ---------------------------------------------------------------------------
# Step — async stepping with error handling
# ---------------------------------------------------------------------------


class TestStep:
    @pytest.mark.asyncio
    async def test_step_returns_structured_result(self):
        """Step should return Tau2StepResult with correct fields."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        mock_env = MagicMock()
        mock_env.step.return_value = (
            "user: Thanks for your help!",
            0.0,
            False,
            False,
            {},
        )
        wrapper.env = mock_env

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_env.step.return_value
            result = await wrapper.step('{"name": "greet", "arguments": {}}')

        assert isinstance(result, Tau2StepResult)
        assert result.obs_type == ObservationType.USER_MESSAGE
        assert result.obs_content == "Thanks for your help!"
        assert result.reward == 0.0
        assert result.terminated is False

    @pytest.mark.asyncio
    async def test_step_with_termination(self):
        """Terminated episode returns correct result."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        mock_env = MagicMock()
        wrapper.env = mock_env

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = ("", 1.0, True, False, {"success": True})
            result = await wrapper.step("goodbye")

        assert result.terminated is True
        assert result.reward == 1.0
        assert result.obs_type == ObservationType.OTHER

    @pytest.mark.asyncio
    async def test_step_json_decode_error_handled(self):
        """JSONDecodeError during step should not crash, returns error observation."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        mock_env = MagicMock()
        wrapper.env = mock_env

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.side_effect = json.JSONDecodeError("test", "doc", 0)
            result = await wrapper.step("{bad json}")

        assert result.obs_type == ObservationType.TOOL_RESULT
        assert "Error" in result.obs_content or "Invalid JSON" in result.raw_obs
        assert result.terminated is False
        assert result.reward == 0.0

    @pytest.mark.asyncio
    async def test_step_tool_result_observation(self):
        """Tool result observation is correctly typed."""
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)

        mock_env = MagicMock()
        wrapper.env = mock_env

        with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = (
                'tool: {"result": "order shipped"}',
                0.0,
                False,
                False,
                {},
            )
            result = await wrapper.step('{"name": "get_order", "arguments": {"id": "1"}}')

        assert result.obs_type == ObservationType.TOOL_RESULT
        assert '"result": "order shipped"' in result.obs_content


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_stores_domain_and_task_id(self):
        """Constructor should store domain and task_id."""
        with patch(
            "tinker_cookbook.recipes.taubench.components.tau2_gym_wrapper.AgentGymEnv"
        ) as MockGym:
            wrapper = Tau2GymWrapper(domain="retail", task_id="task_001")
            assert wrapper.domain == "retail"
            assert wrapper.task_id == "task_001"
            MockGym.assert_called_once_with(domain="retail", task_id="task_001", user_llm=None)

    def test_custom_user_llm(self):
        """Constructor passes user_llm to gym."""
        with patch(
            "tinker_cookbook.recipes.taubench.components.tau2_gym_wrapper.AgentGymEnv"
        ) as MockGym:
            Tau2GymWrapper(domain="airline", task_id="t1", user_llm="gpt-4o")
            MockGym.assert_called_once_with(domain="airline", task_id="t1", user_llm="gpt-4o")


# ---------------------------------------------------------------------------
# System prompt and initial observation
# ---------------------------------------------------------------------------


class TestSystemPromptAndObs:
    def test_get_system_prompt(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        mock_env = MagicMock()
        mock_env._get_system_prompt.return_value = "You are a helpful agent."
        wrapper.env = mock_env
        assert wrapper.get_system_prompt() == "You are a helpful agent."

    def test_get_initial_observation(self):
        wrapper = Tau2GymWrapper.__new__(Tau2GymWrapper)
        mock_env = MagicMock()
        mock_env.reset.return_value = ("user: Hello!", {})
        wrapper.env = mock_env
        obs = wrapper.get_initial_observation()
        assert obs == "user: Hello!"
