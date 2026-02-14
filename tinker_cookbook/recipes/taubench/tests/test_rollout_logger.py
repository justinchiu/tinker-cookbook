"""Tests for RolloutLogger — episode logging with sampling limits."""

import json

from tinker_cookbook.recipes.taubench.components.rollout_logger import RolloutLogger


class TestBasicLogging:
    def test_creates_file(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        path = rl.log_episode(
            domain="retail",
            task_id="task_001",
            reward=1.0,
            messages=[{"role": "user", "content": "hi"}],
        )
        assert path is not None
        assert tmp_path.joinpath("rollouts").exists()

    def test_json_structure(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        path = rl.log_episode(
            domain="airline",
            task_id="task_002",
            reward=0.0,
            messages=[{"role": "user", "content": "help"}],
            metadata={"ask_sonnet_count": 2},
        )
        assert path is not None
        data = json.loads(open(path).read())
        assert "timestamp" in data
        assert "messages" in data
        assert data["domain"] == "airline"
        assert data["task_id"] == "task_002"
        assert data["reward"] == 0.0
        assert data["metadata"]["ask_sonnet_count"] == 2

    def test_filename_contains_status_and_domain(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        path_success = rl.log_episode("retail", "t1", reward=1.0, messages=[])
        assert path_success is not None
        assert "success" in path_success
        assert "retail" in path_success

        path_fail = rl.log_episode("airline", "t2", reward=0.0, messages=[])
        assert path_fail is not None
        assert "failure" in path_fail
        assert "airline" in path_fail


class TestSamplingLimits:
    def test_max_success_blocks_excess(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path), max_success_per_iter=2, max_failure_per_iter=0)
        rl.start_iteration(0)
        p1 = rl.log_episode("retail", "t1", reward=1.0, messages=[])
        p2 = rl.log_episode("retail", "t2", reward=1.0, messages=[])
        p3 = rl.log_episode("retail", "t3", reward=1.0, messages=[])
        assert p1 is not None
        assert p2 is not None
        assert p3 is None  # Blocked by limit

    def test_max_failure_blocks_excess(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path), max_success_per_iter=0, max_failure_per_iter=1)
        rl.start_iteration(0)
        p1 = rl.log_episode("retail", "t1", reward=0.0, messages=[])
        p2 = rl.log_episode("retail", "t2", reward=0.0, messages=[])
        assert p1 is not None
        assert p2 is None

    def test_zero_means_unlimited(self, tmp_path):
        """max_*_per_iter=0 means no limit. Verify with a limit=1 control to prove limits work."""
        rl_limited = RolloutLogger(
            log_dir=str(tmp_path / "limited"), max_success_per_iter=1, max_failure_per_iter=1
        )
        rl_limited.start_iteration(0)
        assert rl_limited.log_episode("retail", "t1", reward=1.0, messages=[]) is not None
        assert rl_limited.log_episode("retail", "t2", reward=1.0, messages=[]) is None  # blocked

        rl_unlimited = RolloutLogger(
            log_dir=str(tmp_path / "unlimited"), max_success_per_iter=0, max_failure_per_iter=0
        )
        rl_unlimited.start_iteration(0)
        paths = [
            rl_unlimited.log_episode("retail", f"t{i}", reward=1.0, messages=[]) for i in range(10)
        ]
        assert all(p is not None for p in paths)

    def test_start_iteration_resets_counters(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path), max_success_per_iter=1, max_failure_per_iter=1)
        rl.start_iteration(0)
        rl.log_episode("retail", "t1", reward=1.0, messages=[])
        assert rl.log_episode("retail", "t2", reward=1.0, messages=[]) is None

        rl.start_iteration(1)
        p = rl.log_episode("retail", "t3", reward=1.0, messages=[])
        assert p is not None  # Counter reset


class TestSerialization:
    def test_tool_call_objects_serialized(self, tmp_path):
        """ToolCall pydantic objects should serialize to dicts."""
        from tinker_cookbook.renderers.base import ToolCall

        tc_obj = ToolCall(function=ToolCall.FunctionBody(name="get_order", arguments='{"id": "1"}'))
        rl = RolloutLogger(log_dir=str(tmp_path))
        messages = [
            {"role": "assistant", "content": "", "tool_calls": [tc_obj]},
        ]
        path = rl.log_episode("retail", "t1", reward=1.0, messages=messages)
        assert path is not None
        data = json.loads(open(path).read())
        tc = data["messages"][0]["tool_calls"][0]
        assert tc["function"]["name"] == "get_order"
        assert tc["function"]["arguments"] == '{"id": "1"}'

    def test_unparsed_tool_call_objects_serialized(self, tmp_path):
        """UnparsedToolCall pydantic objects should serialize to dicts."""
        from tinker_cookbook.renderers.base import UnparsedToolCall

        utc = UnparsedToolCall(raw_text="bad json {", error="Invalid JSON")
        rl = RolloutLogger(log_dir=str(tmp_path))
        messages = [
            {"role": "assistant", "content": "text", "unparsed_tool_calls": [utc]},
        ]
        path = rl.log_episode("retail", "t1", reward=1.0, messages=messages)
        assert path is not None
        data = json.loads(open(path).read())
        utc_data = data["messages"][0]["unparsed_tool_calls"][0]
        assert utc_data["raw_text"] == "bad json {"
        assert utc_data["error"] == "Invalid JSON"


class TestLoadEpisode:
    def test_roundtrip(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        msgs = [{"role": "user", "content": "test"}]
        path = rl.log_episode("retail", "t1", reward=0.5, messages=msgs)
        assert path is not None
        loaded = RolloutLogger.load_episode(path)
        assert loaded["domain"] == "retail"
        assert loaded["reward"] == 0.5
        assert loaded["messages"] == msgs


class TestBoundaryReward:
    def test_reward_exactly_half_is_failure(self, tmp_path):
        """reward=0.5 is NOT success (threshold is > 0.5, not >=)."""
        rl = RolloutLogger(log_dir=str(tmp_path))
        path = rl.log_episode("retail", "t1", reward=0.5, messages=[])
        assert path is not None
        assert "failure" in path

    def test_reward_just_above_half_is_success(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        path = rl.log_episode("retail", "t1", reward=0.51, messages=[])
        assert path is not None
        assert "success" in path


class TestFilenameFormat:
    def test_iteration_number_in_filename(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        rl.start_iteration(7)
        path = rl.log_episode("retail", "t1", reward=1.0, messages=[])
        assert path is not None
        assert "iter0007" in path


class TestListEpisodes:
    def test_list_episodes_returns_logged_files(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        rl.log_episode("retail", "t1", reward=1.0, messages=[])
        rl.log_episode("retail", "t2", reward=0.0, messages=[])
        episodes = rl.list_episodes()
        assert len(episodes) == 2

    def test_list_episodes_disabled_returns_empty(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path), enabled=False)
        assert rl.list_episodes() == []


class TestDisabled:
    def test_disabled_returns_none(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path), enabled=False)
        path = rl.log_episode("retail", "t1", reward=1.0, messages=[])
        assert path is None

    def test_episode_count(self, tmp_path):
        rl = RolloutLogger(log_dir=str(tmp_path))
        assert rl.get_episode_count() == 0
        rl.log_episode("retail", "t1", reward=1.0, messages=[])
        assert rl.get_episode_count() == 1
