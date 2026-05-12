"""CLI entry point: Click command group and `run` subcommand."""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

from mini_swe_agent.cli.help import WARNING_TEXT
from mini_swe_agent.config.schema import Config
from mini_swe_agent.config.sources import resolve_config
from mini_swe_agent.utils.logging import setup_logging

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


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=WARNING_TEXT,
)
@click.version_option(message="%(prog)s %(version)s")
def main() -> None:
    """Mini SWE Agent — closed-loop model-driven shell execution."""
    setup_logging()


@main.command(epilog=WARNING_TEXT)
@click.option("--task", "-t", required=True, help="Task description for the agent.")
@click.option("--model", "-m", default=None, help="Model identifier (provider:name).")
@click.option(
    "--config", "-c",
    "config_sources",
    multiple=True,
    help="Config file path or key=value pair (repeatable).",
)
@click.option("--yolo/--confirm", default=True, help="Auto-execute without confirmation (default: yolo).")
@click.option("--exit-immediately", is_flag=True, help="Exit process immediately after terminal state.")
@click.option("--output", "-o", default=None, help="Trajectory output path.")
@click.option("--cost-limit", type=float, default=None, help="Maximum USD cost.")
@click.option("--step-limit", type=int, default=None, help="Maximum agent steps.")
def run(
    task: str,
    model: str | None,
    config_sources: tuple[str, ...],
    yolo: bool,
    exit_immediately: bool,
    output: str | None,
    cost_limit: float | None,
    step_limit: int | None,
) -> None:
    """Run a single agent task.

    The agent will repeatedly call the model, parse shell commands from the
    response, execute them, and feed observations back until the task is
    complete, a limit is reached, or the user interrupts.
    """
    # Resolve configuration
    cli_files = []
    cli_overrides = []
    for src in config_sources:
        if "=" in src:
            cli_overrides.append(src)
        else:
            cli_files.append(src)

    cli_flags: dict[str, Any] = {}
    if model:
        cli_flags["model"] = {"name": model}
    if cost_limit is not None:
        cli_flags["limits"] = cli_flags.get("limits", {})
        cli_flags["limits"]["max_cost"] = cost_limit
    if step_limit is not None:
        cli_flags["limits"] = cli_flags.get("limits", {})
        cli_flags["limits"]["max_steps"] = step_limit
    if output:
        cli_flags["trajectory"] = cli_flags.get("trajectory", {})
        cli_flags["trajectory"]["output_path"] = output

    merged = resolve_config(
        cli_files=cli_files,
        cli_overrides=cli_overrides,
        cli_flags=cli_flags,
    )

    try:
        config = Config(**merged)
    except Exception as e:
        click.echo(f"FATAL_CONFIG: Invalid configuration: {e}", err=True)
        sys.exit(1)

    mode_str = "yolo" if yolo else "confirm"
    click.echo(f"Task: {task[:80]}{'...' if len(task) > 80 else ''}")
    click.echo(f"Model: {config.model.name} | Mode: {mode_str} | "
               f"Step limit: {config.limits.max_steps} | Cost limit: ${config.limits.max_cost:.2f}")

    # Run
    from mini_swe_agent.core.loop import AgentLoop
    from mini_swe_agent.models.adapter_factory import create_adapter
    from mini_swe_agent.types import RunMode
    from mini_swe_agent.trajectory.writer import save_trajectory

    mode = RunMode.YOLO if yolo else RunMode.CONFIRM
    adapter = create_adapter(config)
    executor = _create_executor(config)

    loop = AgentLoop(
        task=task,
        model=adapter,
        executor=executor,
        config=config,
        mode=mode,
    )

    state, trajectory = asyncio.run(loop.run())

    # Save trajectory
    output_path = config.trajectory.output_path
    save_trajectory(trajectory, output_path)

    click.echo(f"\nTerminal state: {state.value}")
    click.echo(f"Steps: {trajectory.total_steps} | Cost: ${trajectory.total_cost:.6f}")
    click.echo(f"Trajectory saved to: {output_path}")

    if exit_immediately:
        sys.exit(0 if state.value == "submitted" else 1)


@main.command("config", epilog=WARNING_TEXT)
@click.option(
    "--config", "-c",
    "config_sources",
    multiple=True,
    help="Config file path or key=value pair (repeatable).",
)
def show_config(config_sources: tuple[str, ...]) -> None:
    """Print the fully merged configuration as JSON and exit."""
    cli_files = []
    cli_overrides = []
    for src in config_sources:
        if "=" in src:
            cli_overrides.append(src)
        else:
            cli_files.append(src)

    merged = resolve_config(cli_files=cli_files, cli_overrides=cli_overrides)
    click.echo(json.dumps(merged, indent=2, default=str))


@main.command("inspect", epilog=WARNING_TEXT)
@click.argument("path", type=click.Path(exists=True))
def inspect(path: str) -> None:
    """Launch Textual TUI to inspect a .traj.json trajectory file."""
    from mini_swe_agent.inspector.app import run_inspector
    run_inspector(path)


@main.command("batch", epilog=WARNING_TEXT)
@click.option("--tasks-file", "-f", required=True, type=click.Path(exists=True), help="JSON file with tasks.")
@click.option("--output", "-o", default="preds.json", help="Output preds.json path.")
@click.option("--model", "-m", default=None, help="Model identifier.")
@click.option("--config", "-c", "config_sources", multiple=True, help="Config file or key=value.")
@click.option("--yolo/--confirm", default=True, help="Auto-execute mode.")
@click.option("--cost-limit", type=float, default=None, help="Per-task cost limit.")
@click.option("--step-limit", type=int, default=None, help="Per-task step limit.")
@click.option("--parallel", "-p", type=int, default=1, help="Number of concurrent tasks.")
@click.option("--regex-filter", default=None, help="Regex filter on instance_id or task.")
@click.option("--shuffle-seed", type=int, default=None, help="Deterministic shuffle seed.")
@click.option("--slice", default=None, help="Task slice as START:END (e.g., 0:10).")
@click.option("--redo-existing", is_flag=True, default=None, help="Re-run tasks already in preds.json.")
def batch(
    tasks_file: str,
    output: str,
    model: str | None,
    config_sources: tuple[str, ...],
    yolo: bool,
    cost_limit: float | None,
    step_limit: int | None,
    parallel: int,
    regex_filter: str | None,
    shuffle_seed: int | None,
    slice: str | None,
    redo_existing: bool | None,
) -> None:
    """Run multiple tasks from a JSON file and produce preds.json."""
    from mini_swe_agent.batch.runner import run_batch as batch_run

    # Load tasks
    with open(tasks_file, "r") as f:
        tasks_data = json.load(f)

    tasks = tasks_data if isinstance(tasks_data, list) else tasks_data.get("tasks", [])

    cli_files = []
    cli_overrides = []
    for src in config_sources:
        if "=" in src:
            cli_overrides.append(src)
        else:
            cli_files.append(src)

    # Parse slice
    slice_start = None
    slice_end = None
    if slice:
        parts = slice.split(":")
        if len(parts) == 2:
            slice_start = int(parts[0]) if parts[0] else None
            slice_end = int(parts[1]) if parts[1] else None

    asyncio.run(batch_run(
        tasks=tasks,
        output_path=output,
        model=model,
        cli_files=cli_files,
        cli_overrides=cli_overrides,
        yolo=yolo,
        cost_limit=cost_limit,
        step_limit=step_limit,
        workers=parallel,
        regex_filter=regex_filter,
        shuffle_seed=shuffle_seed,
        slice_start=slice_start,
        slice_end=slice_end,
        redo_existing=redo_existing,
    ))

    click.echo(f"Predictions saved to: {output}")


if __name__ == "__main__":
    main()
