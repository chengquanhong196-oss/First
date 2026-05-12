"""Configuration models validated with Pydantic v2."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model provider and connection settings."""

    provider: str = "anthropic"
    name: str = "claude-sonnet-4-6"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.0
    max_tokens: int = 4096
    thinking_budget: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class ExecutorConfig(BaseModel):
    """Shell executor settings."""

    backend: Literal["local", "docker"] = "local"
    timeout: float = 120.0
    docker_image: str = ""
    max_output_chars: int = 100000


class LimitsConfig(BaseModel):
    """Agent limits."""

    max_steps: int = 50
    max_cost: float = 10.0
    max_consecutive_format_errors: int = 3
    missing_cost_policy: Literal["ignore", "error"] = "ignore"


class TemplateConfig(BaseModel):
    """Observation and system prompt templates."""

    system_prompt: str = ""
    observation_template: str = (
        "Command: {{ command }}\n"
        "Return code: {{ returncode }}\n"
        "Stdout:\n{{ stdout }}\n"
        "Stderr:\n{{ stderr }}\n"
        "{% if exception %}Exception: {{ exception }}\n{% endif %}"
        "{% if timed_out %}TIMED OUT after {{ elapsed }}s\n{% endif %}"
    )
    observation_max_chars: int = 10000
    observation_truncate_head: int = 5000
    observation_truncate_tail: int = 5000


class TrajectoryConfig(BaseModel):
    """Trajectory output settings."""

    output_path: str = "./trajectory.traj.json"
    save_on_interrupt: bool = True


class BatchConfig(BaseModel):
    """Batch processing settings."""

    workers: int = 1
    redo_existing: bool = False
    shuffle_seed: int | None = None
    regex_filter: str = ""
    slice_start: int | None = None
    slice_end: int | None = None


class Config(BaseModel):
    """Root configuration model."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    limits: LimitsConfig = Field(default_factory=LimitsConfig)
    template: TemplateConfig = Field(default_factory=TemplateConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    batch: BatchConfig = Field(default_factory=BatchConfig)
    extra: dict[str, Any] = Field(default_factory=dict)
