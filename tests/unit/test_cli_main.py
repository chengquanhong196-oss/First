"""Tests for CLI commands via Click CliRunner."""
import json
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from click.testing import CliRunner

from mini_swe_agent.cli.main import main


@pytest.fixture
def runner():
    return CliRunner()


class TestCLIRun:
    def test_run_help_shows_warning(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "WARNING" in result.output

    def test_run_requires_task(self, runner):
        result = runner.invoke(main, ["run"])
        assert result.exit_code != 0

    def test_run_basic_invocation(self, runner):
        mock_state = MagicMock()
        mock_state.value = "submitted"
        mock_traj = MagicMock()
        mock_traj.total_steps = 1
        mock_traj.total_cost = 0.001

        with patch("mini_swe_agent.cli.main.asyncio.run", return_value=(mock_state, mock_traj)), \
             patch("mini_swe_agent.trajectory.writer.save_trajectory"), \
             patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop:
            mock_loop.return_value.run = AsyncMock(return_value=(mock_state, mock_traj))

            result = runner.invoke(main, [
                "run",
                "--task", "echo hello",
                "--model", "anthropic:claude-sonnet-4-6",
                "--yolo",
            ])
            assert result.exit_code == 0

    def test_run_with_confirm_mode(self, runner):
        mock_state = MagicMock()
        mock_state.value = "submitted"
        mock_traj = MagicMock()
        mock_traj.total_steps = 1
        mock_traj.total_cost = 0.001

        with patch("mini_swe_agent.cli.main.asyncio.run", return_value=(mock_state, mock_traj)), \
             patch("mini_swe_agent.trajectory.writer.save_trajectory"), \
             patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop:
            mock_loop.return_value.run = AsyncMock(return_value=(mock_state, mock_traj))

            result = runner.invoke(main, [
                "run",
                "--task", "fix bug",
                "--confirm",
                "--step-limit", "20",
            ])
            assert result.exit_code == 0


class TestCLIBatch:
    def test_batch_help(self, runner):
        result = runner.invoke(main, ["batch", "--help"])
        assert result.exit_code == 0
        assert "regex-filter" in result.output
        assert "shuffle-seed" in result.output
        assert "slice" in result.output
        assert "redo-existing" in result.output

    def test_batch_requires_tasks_file(self, runner):
        result = runner.invoke(main, ["batch"])
        assert result.exit_code != 0

    def test_batch_with_options(self, runner, tmp_path):
        tasks_file = tmp_path / "tasks.json"
        tasks_file.write_text(json.dumps([
            {"instance_id": "1", "task": "fix bug"},
        ]))

        with patch("mini_swe_agent.cli.main.asyncio.run"), \
             patch("mini_swe_agent.batch.runner.run_batch") as mock_batch:

            result = runner.invoke(main, [
                "batch",
                "-f", str(tasks_file),
                "-o", str(tmp_path / "preds.json"),
                "--regex-filter", "fix",
                "--shuffle-seed", "42",
                "--slice", "0:5",
                "--redo-existing",
                "--parallel", "2",
            ])
            assert result.exit_code == 0
            kwargs = mock_batch.call_args.kwargs
            assert kwargs["regex_filter"] == "fix"
            assert kwargs["shuffle_seed"] == 42
            assert kwargs["slice_start"] == 0
            assert kwargs["slice_end"] == 5
            assert kwargs["redo_existing"] is True
            assert kwargs["workers"] == 2


class TestCLIConfig:
    def test_config_outputs_json(self, runner):
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "model" in parsed

    def test_config_with_override(self, runner):
        result = runner.invoke(main, ["config", "-c", "model.name=from-cli"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["model"]["name"] == "from-cli"


class TestCLIInspect:
    def test_inspect_requires_path(self, runner):
        result = runner.invoke(main, ["inspect"])
        assert result.exit_code != 0

    def test_inspect_help_shows_warning(self, runner):
        result = runner.invoke(main, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "WARNING" in result.output


class TestCreateExecutor:
    def test_local_executor_default(self):
        from mini_swe_agent.cli.main import _create_executor
        from mini_swe_agent.config.schema import Config

        config = Config(**{
            "model": {"provider": "anthropic", "name": "test", "max_tokens": 4096},
            "executor": {"backend": "local", "timeout": 30.0},
            "limits": {"max_steps": 10, "max_cost": 1.0, "max_consecutive_format_errors": 3},
        })
        executor = _create_executor(config)
        assert executor.backend_name == "local"

    def test_docker_fallback_when_unavailable(self):
        from mini_swe_agent.cli.main import _create_executor
        from mini_swe_agent.config.schema import Config
        from mini_swe_agent.executor.local import LocalExecutor

        with patch(
            "mini_swe_agent.executor.docker.DockerExecutor.is_available",
            new_callable=PropertyMock,
            return_value=False,
        ):
            config = Config(**{
                "model": {"provider": "anthropic", "name": "test", "max_tokens": 4096},
                "executor": {"backend": "docker", "timeout": 30.0, "docker_image": "sandbox"},
                "limits": {"max_steps": 10, "max_cost": 1.0, "max_consecutive_format_errors": 3},
            })
            executor = _create_executor(config)
            assert isinstance(executor, LocalExecutor)
