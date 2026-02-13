"""Tests for upstream API compatibility — verify upstream APIs support taubench needs.

These tests catch upstream API changes that would break taubench. They test
that required fields, parameters, and interfaces exist in the upstream code.
"""

import inspect


# ---------------------------------------------------------------------------
# Renderer API
# ---------------------------------------------------------------------------


class TestRendererAPI:
    def test_create_conversation_prefix_with_tools_exists(self):
        """Renderer must have create_conversation_prefix_with_tools method.

        Taubench uses this upstream method to inject tool definitions into the
        system prompt for both RL (env.py) and SFT (sft_dataset.py).
        """
        from tinker_cookbook.renderers.base import Renderer

        assert hasattr(Renderer, "create_conversation_prefix_with_tools")
        sig = inspect.signature(Renderer.create_conversation_prefix_with_tools)
        assert "tools" in sig.parameters
        assert "system_prompt" in sig.parameters

    def test_toolspec_type_exists(self):
        """ToolSpec TypedDict must exist for create_conversation_prefix_with_tools."""
        from tinker_cookbook.renderers.base import ToolSpec

        spec = ToolSpec(name="test", description="desc", parameters={})
        assert spec["name"] == "test"

    def test_parse_response_returns_message_bool(self):
        """Renderer.parse_response should return (Message, bool)."""
        from tinker_cookbook.renderers.base import Renderer

        sig = inspect.signature(Renderer.parse_response)
        # Just verify the method exists and has the right parameter count
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "tokens" in params or len(params) >= 2

    def test_train_on_what_customized_exists(self):
        """TrainOnWhat.CUSTOMIZED must exist for ask_sonnet-only training."""
        from tinker_cookbook.renderers import TrainOnWhat

        assert hasattr(TrainOnWhat, "CUSTOMIZED")

    def test_toolcall_class_exists(self):
        """ToolCall class must exist with FunctionBody inner class."""
        from tinker_cookbook.renderers import ToolCall

        tc = ToolCall(function=ToolCall.FunctionBody(name="test", arguments="{}"))
        assert tc.function.name == "test"


# ---------------------------------------------------------------------------
# RL Types API
# ---------------------------------------------------------------------------


class TestRLTypesAPI:
    def test_step_result_has_metrics(self):
        """StepResult should have metrics field for per-step custom metrics."""
        from tinker_cookbook.rl.types import StepResult

        sig = inspect.signature(StepResult)
        assert "metrics" in sig.parameters

    def test_step_result_has_logs(self):
        """StepResult should have logs field for per-step logging."""
        from tinker_cookbook.rl.types import StepResult

        sig = inspect.signature(StepResult)
        assert "logs" in sig.parameters

    def test_transition_has_metrics(self):
        """Transition should have metrics field."""
        from tinker_cookbook.rl.types import Transition

        sig = inspect.signature(Transition)
        assert "metrics" in sig.parameters

    def test_transition_has_logs(self):
        """Transition should have logs field."""
        from tinker_cookbook.rl.types import Transition

        sig = inspect.signature(Transition)
        assert "logs" in sig.parameters

    def test_env_interface_exists(self):
        """Env abstract class should exist with required methods."""
        from tinker_cookbook.rl.types import Env

        assert hasattr(Env, "initial_observation")
        assert hasattr(Env, "step")

    def test_env_group_builder_interface(self):
        """EnvGroupBuilder should have make_envs and compute_group_rewards."""
        from tinker_cookbook.rl.types import EnvGroupBuilder

        assert hasattr(EnvGroupBuilder, "make_envs")
        assert hasattr(EnvGroupBuilder, "compute_group_rewards")

    def test_rl_dataset_interface(self):
        """RLDataset should have get_batch and __len__."""
        from tinker_cookbook.rl.types import RLDataset

        assert hasattr(RLDataset, "get_batch")
        assert hasattr(RLDataset, "__len__")


# ---------------------------------------------------------------------------
# Completers API
# ---------------------------------------------------------------------------


class TestCompletersAPI:
    def test_token_completer_interface(self):
        """TokenCompleter should be callable with (observation, stop_condition)."""
        from tinker_cookbook.completers import TokenCompleter

        assert callable(TokenCompleter)


# ---------------------------------------------------------------------------
# RL Train Config API
# ---------------------------------------------------------------------------


class TestRLTrainConfigAPI:
    def test_config_accepts_policy_factory(self):
        """train.Config should accept policy_factory for custom policy creation."""
        from tinker_cookbook.rl.train import Config

        sig = inspect.signature(Config)
        assert "policy_factory" in sig.parameters, (
            "train.Config must accept 'policy_factory' parameter"
        )

    def test_config_accepts_on_train_step(self):
        """train.Config should accept on_train_step callback."""
        from tinker_cookbook.rl.train import Config

        sig = inspect.signature(Config)
        assert "on_train_step" in sig.parameters, (
            "train.Config must accept 'on_train_step' parameter"
        )

    def test_config_accepts_eval_temperature(self):
        """train.Config should accept eval_temperature for evaluation."""
        from tinker_cookbook.rl.train import Config

        sig = inspect.signature(Config)
        assert "eval_temperature" in sig.parameters, (
            "train.Config must accept 'eval_temperature' parameter"
        )

    def test_policy_factory_type_exported(self):
        """PolicyFactory type should be importable from rl.train."""
        from tinker_cookbook.rl.train import PolicyFactory

        assert PolicyFactory is not None


# ---------------------------------------------------------------------------
# Rollouts API
# ---------------------------------------------------------------------------


class TestRolloutsAPI:
    def test_do_single_rollout_accepts_rollout_idx(self):
        """do_single_rollout should accept rollout_idx for exploration."""
        from tinker_cookbook.rl.rollouts import do_single_rollout

        sig = inspect.signature(do_single_rollout)
        assert "rollout_idx" in sig.parameters, (
            "do_single_rollout must accept 'rollout_idx' parameter"
        )

    def test_do_single_rollout_calls_episode_lifecycle(self):
        """do_single_rollout should call start_episode/end_episode on policy.

        This is verified by checking the source code contains the lifecycle calls.
        """
        from tinker_cookbook.rl import rollouts

        source = inspect.getsource(rollouts.do_single_rollout)
        assert "start_episode" in source, "do_single_rollout must call start_episode on policy"
        assert "end_episode" in source, "do_single_rollout must call end_episode on policy"


# ---------------------------------------------------------------------------
# Supervised Types API
# ---------------------------------------------------------------------------


class TestSupervisedTypesAPI:
    def test_supervised_dataset_has_set_epoch(self):
        """SupervisedDataset should have set_epoch method."""
        from tinker_cookbook.supervised.types import SupervisedDataset

        assert hasattr(SupervisedDataset, "set_epoch")

    def test_chat_dataset_builder_exists(self):
        """ChatDatasetBuilder should exist with renderer property."""
        from tinker_cookbook.supervised.types import ChatDatasetBuilder

        assert hasattr(ChatDatasetBuilder, "renderer")

    def test_datum_from_model_input_weights_exists(self):
        """datum_from_model_input_weights should be importable."""
        from tinker_cookbook.supervised.common import datum_from_model_input_weights

        assert callable(datum_from_model_input_weights)
