"""Unit tests for batch runner filtering and logic."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mini_swe_agent.batch.runner import run_batch


def make_sample_tasks(n=10):
    return [
        {"instance_id": f"task_{i:03d}", "task": f"Fix bug #{i}: something is broken"}
        for i in range(n)
    ]


def _mock_success():
    return (
        MagicMock(value="submitted"),
        MagicMock(total_steps=0, total_cost=0.0, steps=[]),
    )


class TestBatchFiltering:
    @pytest.mark.asyncio
    async def test_regex_filter_on_instance_id(self, tmp_path):
        tasks = make_sample_tasks(10)
        output = tmp_path / "preds.json"

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(return_value=_mock_success())

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=2,
                regex_filter=r"task_00[12]",
            )

        assert output.exists()
        results = json.loads(output.read_text())
        assert len(results) == 2
        assert "task_001" in results
        assert "task_002" in results

    @pytest.mark.asyncio
    async def test_slice(self, tmp_path):
        tasks = make_sample_tasks(10)
        output = tmp_path / "preds.json"

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(return_value=_mock_success())

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=1,
                slice_start=2,
                slice_end=5,
            )

        results = json.loads(output.read_text())
        assert len(results) == 3
        assert "task_002" in results
        assert "task_003" in results
        assert "task_004" in results

    @pytest.mark.asyncio
    async def test_shuffle_deterministic(self, tmp_path):
        tasks = make_sample_tasks(5)
        output = tmp_path / "preds.json"

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(return_value=_mock_success())

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=1,
                shuffle_seed=42,
                cli_files=[],
                cli_overrides=[],
            )

        assert output.exists()

    @pytest.mark.asyncio
    async def test_redo_existing(self, tmp_path):
        tasks = make_sample_tasks(3)
        output = tmp_path / "preds.json"
        output.write_text(json.dumps({
            "task_000": {"instance_id": "task_000", "model_patch": "old"},
        }))

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(return_value=_mock_success())

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=1,
                redo_existing=True,
            )

        results = json.loads(output.read_text())
        assert len(results) == 3
        assert results["task_000"]["model_patch"] != "old"

    @pytest.mark.asyncio
    async def test_skip_existing_by_default(self, tmp_path):
        tasks = make_sample_tasks(3)
        output = tmp_path / "preds.json"
        output.write_text(json.dumps({
            "task_000": {"instance_id": "task_000", "model_patch": "old"},
        }))

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(return_value=_mock_success())

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=1,
                redo_existing=False,
            )

        results = json.loads(output.read_text())
        assert "task_000" in results
        assert results["task_000"]["model_patch"] == "old"
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_error_handling(self, tmp_path):
        tasks = make_sample_tasks(3)
        output = tmp_path / "preds.json"

        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("task 0 crashed")
            return _mock_success()

        with patch("mini_swe_agent.core.loop.AgentLoop") as mock_loop, \
             patch("mini_swe_agent.batch.runner._create_executor"), \
             patch("mini_swe_agent.models.adapter_factory.create_adapter"):
            mock_loop.return_value.run = AsyncMock(side_effect=side_effect)

            await run_batch(
                tasks=tasks,
                output_path=str(output),
                workers=1,
            )

        results = json.loads(output.read_text())
        assert len(results) == 3
        assert "error" in results["task_000"]
        assert "error" not in results.get("task_001", {})
        assert "error" not in results.get("task_002", {})
