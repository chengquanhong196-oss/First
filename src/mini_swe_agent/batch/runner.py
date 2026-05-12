"""Batch runner: execute multiple tasks and produce preds.json."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from pathlib import Path
from typing import Any

import click

from mini_swe_agent.config.schema import Config
from mini_swe_agent.config.sources import resolve_config

logger = logging.getLogger(__name__)


def _create_executor(config: Config):
    """Create an executor based on configuration. Falls back to Local if Docker is unavailable."""
    if config.executor.backend == "docker":
        from mini_swe_agent.executor.docker import DockerExecutor
        docker_executor = DockerExecutor(
            image=config.executor.docker_image,
            timeout=config.executor.timeout,
        )
        if docker_executor.is_available:
            return docker_executor
        click.echo("WARNING: Docker unavailable, falling back to local executor.", err=True)
    from mini_swe_agent.executor.local import LocalExecutor
    return LocalExecutor()


async def run_batch(
    tasks: list[dict[str, Any]],
    output_path: str = "preds.json",
    model: str | None = None,
    cli_files: list[str] | None = None,
    cli_overrides: list[str] | None = None,
    yolo: bool = True,
    cost_limit: float | None = None,
    step_limit: int | None = None,
    workers: int = 1,
    regex_filter: str | None = None,
    shuffle_seed: int | None = None,
    slice_start: int | None = None,
    slice_end: int | None = None,
    redo_existing: bool | None = None,
) -> None:
    """Run a batch of tasks and write preds.json.

    Each task dict must have at least 'instance_id' and 'task' keys.

    preds.json format:
    {
      "instance_id": {
        "instance_id": "...",
        "model_name_or_path": "...",
        "model_patch": "..."
      }
    }
    """
    cli_files = cli_files or []
    cli_overrides = cli_overrides or []

    # Build base config
    cli_flags: dict[str, Any] = {}
    if model:
        cli_flags["model"] = {"name": model}
    if cost_limit is not None:
        cli_flags["limits"] = cli_flags.get("limits", {})
        cli_flags["limits"]["max_cost"] = cost_limit
    if step_limit is not None:
        cli_flags["limits"] = {**(cli_flags.get("limits") or {}), "max_steps": step_limit}
    # Pass the overrides via batch config for the runner to use
    if regex_filter is not None:
        cli_flags["batch"] = cli_flags.get("batch", {})
        cli_flags["batch"]["regex_filter"] = regex_filter
    if shuffle_seed is not None:
        cli_flags["batch"] = cli_flags.get("batch", {})
        cli_flags["batch"]["shuffle_seed"] = shuffle_seed
    if slice_start is not None:
        cli_flags["batch"] = cli_flags.get("batch", {})
        cli_flags["batch"]["slice_start"] = slice_start
    if slice_end is not None:
        cli_flags["batch"] = cli_flags.get("batch", {})
        cli_flags["batch"]["slice_end"] = slice_end
    if redo_existing is not None:
        cli_flags["batch"] = cli_flags.get("batch", {})
        cli_flags["batch"]["redo_existing"] = redo_existing

    merged = resolve_config(
        cli_files=cli_files,
        cli_overrides=cli_overrides,
        cli_flags=cli_flags,
    )
    config = Config(**merged)

    # Apply regex filter
    if regex_filter:
        pattern = re.compile(regex_filter)
        tasks = [t for t in tasks if pattern.search(t.get("instance_id", "")) or pattern.search(t.get("task", ""))]

    # Apply slice
    effective_slice_start = slice_start if slice_start is not None else config.batch.slice_start
    effective_slice_end = slice_end if slice_end is not None else config.batch.slice_end
    if effective_slice_start is not None or effective_slice_end is not None:
        tasks = tasks[effective_slice_start:effective_slice_end]

    # Shuffle with deterministic seed
    effective_shuffle_seed = shuffle_seed if shuffle_seed is not None else config.batch.shuffle_seed
    if effective_shuffle_seed is not None:
        rng = random.Random(effective_shuffle_seed)
        rng.shuffle(tasks)

    # Check existing and redo_existing
    effective_redo = redo_existing if redo_existing is not None else config.batch.redo_existing
    existing_ids = set()
    output_path_obj = Path(output_path)
    if output_path_obj.exists() and not effective_redo:
        try:
            with open(output_path_obj, "r") as f:
                existing = json.load(f)
            existing_ids = set(existing.keys())
        except Exception:
            pass

    tasks_to_run = [t for t in tasks if t.get("instance_id") not in existing_ids]

    logger.info(
        "Batch: %d total, %d to run, %d existing skipped, %d workers",
        len(tasks), len(tasks_to_run), len(existing_ids), workers,
    )

    # Load existing results for incremental save
    results: dict[str, dict[str, Any]] = {}
    if existing_ids:
        try:
            with open(output_path_obj, "r") as f:
                results = json.load(f)
        except Exception:
            pass

    # File lock for incremental writes
    _write_lock = asyncio.Lock()

    def _save_results() -> None:
        """Write results to output file (called under lock)."""
        with open(output_path_obj, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Run tasks
    semaphore = asyncio.Semaphore(workers)

    async def run_one(task: dict[str, Any]) -> None:
        async with semaphore:
            instance_id = task.get("instance_id", task.get("id", str(hash(task.get("task", "")))))
            task_text = task.get("task", task.get("instruction", ""))

            try:
                from mini_swe_agent.core.loop import AgentLoop
                from mini_swe_agent.models.adapter_factory import create_adapter
                from mini_swe_agent.types import RunMode

                mode = RunMode.YOLO if yolo else RunMode.CONFIRM
                adapter = create_adapter(config)
                executor = _create_executor(config)

                loop = AgentLoop(
                    task=task_text,
                    model=adapter,
                    executor=executor,
                    config=config,
                    mode=mode,
                )

                state, trajectory = await loop.run()

                results[instance_id] = {
                    "instance_id": instance_id,
                    "model_name_or_path": config.model.name,
                    "model_patch": "",
                }

                if state.value == "submitted":
                    from mini_swe_agent.core.submission import extract_submission_body
                    for step in trajectory.steps:
                        if step.observation and step.observation.returncode == 0:
                            body = extract_submission_body(step.observation.stdout)
                            if body:
                                results[instance_id]["model_patch"] = body
                                break

                logger.info("[%s] %s — %d steps, $%.6f", instance_id, state.value, trajectory.total_steps, trajectory.total_cost)

            except Exception as e:
                logger.error("[%s] Error: %s", instance_id, e)
                results[instance_id] = {
                    "instance_id": instance_id,
                    "model_name_or_path": config.model.name,
                    "model_patch": "",
                    "error": str(e),
                }

            # Incremental save after each task
            async with _write_lock:
                _save_results()

    await asyncio.gather(*(run_one(t) for t in tasks_to_run))

    # Final save
    async with _write_lock:
        _save_results()

    logger.info("Batch complete: %d results written to %s", len(results), output_path)
