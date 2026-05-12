"""Acceptance test: Trajectory roundtrip (Section J).

Write + read trajectory; tool mode has tool_call_id on observations.
"""

import json

from mini_swe_agent.trajectory.schema import Observation, Step, Trajectory
from mini_swe_agent.trajectory.reader import load_trajectory
from mini_swe_agent.trajectory.writer import save_trajectory


class TestTrajectoryRoundtrip:
    """Trajectory JSON can be written and read back with all fields intact."""

    def test_write_and_read(self, temp_trajectory_path):
        traj = Trajectory(
            task="Fix the bug",
            model_name="claude-sonnet-4-6",
            terminal_state="submitted",
            total_steps=3,
            total_cost=0.005,
            messages=[{"role": "system", "content": "You are an agent."}],
            steps=[
                Step(
                    step_index=0,
                    action="ls",
                    action_family="text",
                    observation=Observation(
                        returncode=0,
                        stdout="file.txt",
                        tool_call_id="tc_001",
                    ),
                )
            ],
        )

        save_trajectory(traj, temp_trajectory_path)

        loaded = load_trajectory(temp_trajectory_path)
        assert loaded.task == "Fix the bug"
        assert loaded.model_name == "claude-sonnet-4-6"
        assert loaded.terminal_state == "submitted"
        assert loaded.total_steps == 3
        assert loaded.total_cost == 0.005
        assert len(loaded.steps) == 1

    def test_tool_call_id_present_in_tool_mode(self, temp_trajectory_path):
        """Observation in tool-call mode must carry tool_call_id."""
        traj = Trajectory(
            task="test",
            model_name="test-model",
            terminal_state="submitted",
            steps=[
                Step(
                    step_index=0,
                    action="echo hello",
                    action_family="tool_call",
                    observation=Observation(
                        returncode=0,
                        stdout="hello",
                        tool_call_id="tc_abc123",
                    ),
                )
            ],
        )

        save_trajectory(traj, temp_trajectory_path)

        with open(temp_trajectory_path) as f:
            raw = json.load(f)

        assert raw["steps"][0]["observation"]["tool_call_id"] == "tc_abc123"
        assert raw["steps"][0]["action_family"] == "tool_call"

    def test_messages_with_tool_calls(self, temp_trajectory_path):
        """Messages field preserves tool_call_id on tool messages."""
        traj = Trajectory(
            task="test",
            model_name="test-model",
            terminal_state="limit_step",
            messages=[
                {"role": "user", "content": "run ls"},
                {"role": "assistant", "content": "ok", "tool_calls": [
                    {"id": "tc_1", "type": "function", "function_name": "bash", "arguments": {"command": "ls"}}
                ]},
                {"role": "tool", "content": "file.txt", "tool_call_id": "tc_1", "name": "bash"},
            ],
        )

        save_trajectory(traj, temp_trajectory_path)

        with open(temp_trajectory_path) as f:
            raw = json.load(f)

        msgs = raw["messages"]
        assert msgs[1]["tool_calls"][0]["id"] == "tc_1"
        assert msgs[2]["tool_call_id"] == "tc_1"

    def test_bare_messages_array(self, temp_trajectory_path):
        """A file with just a messages array is loaded as a trajectory."""
        messages = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        with open(temp_trajectory_path, "w") as f:
            json.dump(messages, f)

        loaded = load_trajectory(temp_trajectory_path)
        assert len(loaded.messages) == 2
