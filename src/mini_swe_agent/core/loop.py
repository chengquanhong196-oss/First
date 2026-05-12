"""Main agent loop: the state machine that drives model → parse → execute → observe."""
from __future__ import annotations

import asyncio
import logging
import signal
import time
from datetime import datetime, timezone
from typing import Any

from mini_swe_agent.config.schema import Config
from mini_swe_agent.core.limits import CostLimiter, StepLimiter
from mini_swe_agent.core.submission import check_submission, extract_submission_body
from mini_swe_agent.executor.base import SandboxExecutor
from mini_swe_agent.models.base import ModelAdapter
from mini_swe_agent.models.messages import Message, ModelResponse
from mini_swe_agent.parser.detector import detect_action_family
from mini_swe_agent.parser.errors import FormatError
from mini_swe_agent.parser.extractor import extract_command
from mini_swe_agent.trajectory.schema import Observation, Step, Trajectory
from mini_swe_agent.types import ActionFamily, RunMode, TerminalState
from mini_swe_agent.utils.templates import render_observation

logger = logging.getLogger(__name__)


class AgentLoop:
    """Runs the model → parse → (confirm) → execute → observe loop.

    State machine:
        INIT → MODEL → PARSE → (CONFIRM) → EXECUTE → OBSERVE → check → MODEL
    Terminal states: SUBMITTED, LIMIT_STEP, LIMIT_COST, INTERRUPT, FATAL_CONFIG
    """

    def __init__(
        self,
        task: str,
        model: ModelAdapter,
        executor: SandboxExecutor,
        config: Config,
        mode: RunMode = RunMode.YOLO,
    ) -> None:
        self._task = task
        self._model = model
        self._executor = executor
        self._config = config
        self._mode = mode

        self._step_limiter = StepLimiter(config.limits.max_steps)
        self._cost_limiter = CostLimiter(config.limits.max_cost)

        self._messages: list[Message] = []
        self._steps: list[Step] = []
        self._interrupted = False
        self._total_cost = 0.0
        self._consecutive_format_errors = 0
        self._start_time = datetime.now(timezone.utc)

    # ── public API ──────────────────────────────────────────────

    async def run(self) -> tuple[TerminalState, Trajectory]:
        """Execute the agent loop until a terminal state is reached."""
        self._setup_signal_handlers()
        self._build_initial_messages()

        state = TerminalState.FATAL_CONFIG

        try:
            while True:
                # Check step limit
                if self._step_limiter.exhausted:
                    state = TerminalState.LIMIT_STEP
                    break

                # Check cost limit
                if self._cost_limiter.exhausted:
                    state = TerminalState.LIMIT_COST
                    break

                # ── MODEL ──
                step_messages_before = [
                    _message_to_dict(m) for m in self._messages
                ]
                try:
                    response = await self._model.send(self._messages)
                except Exception as e:
                    logger.error("Model call failed: %s", e)
                    state = TerminalState.FATAL_CONFIG
                    break

                # Append response messages
                for resp_msg in response.messages:
                    self._messages.append(resp_msg)

                self._total_cost += response.cost
                if self._cost_limiter.add(response.cost):
                    state = TerminalState.LIMIT_COST
                    break

                self._step_limiter.increment()

                # ── PARSE ──
                assistant_dicts = [
                    _message_to_dict(m) for m in response.messages
                ]
                step = Step(
                    step_index=len(self._steps),
                    messages_before=step_messages_before,
                    assistant_message=assistant_dicts[0] if assistant_dicts else None,
                )

                try:
                    family = detect_action_family(response)
                    command, tool_call_id = extract_command(response, family)
                    step.action = command
                    step.action_family = family.value
                    self._consecutive_format_errors = 0
                except FormatError as e:
                    step.format_error = e.feedback
                    self._steps.append(step)
                    self._messages.append(Message(
                        role="user",
                        content=e.feedback,
                    ))
                    self._consecutive_format_errors += 1
                    if (
                        self._consecutive_format_errors
                        >= self._config.limits.max_consecutive_format_errors
                    ):
                        logger.error(
                            "Too many consecutive format errors (%d). Aborting.",
                            self._consecutive_format_errors,
                        )
                        state = TerminalState.FATAL_CONFIG
                        break
                    continue

                # ── CONFIRM ──
                if self._mode == RunMode.CONFIRM:
                    approved = await self._confirm_action(command)
                    if not approved:
                        self._messages.append(Message(
                            role="user",
                            content="User rejected this action. Provide an alternative command.",
                        ))
                        step.observation = Observation(
                            returncode=-1,
                            exception="User rejected action",
                        )
                        self._steps.append(step)
                        continue

                # ── EXECUTE ──
                result = await self._executor.execute(
                    command, self._config.executor.timeout
                )

                # ── OBSERVE ──
                obs_text = self._render_observation(result)
                self._messages.append(Message(
                    role="tool",
                    content=obs_text,
                    tool_call_id=tool_call_id,
                    name="bash",
                ))

                step.observation = Observation(
                    returncode=result.returncode,
                    stdout=result.stdout,
                    stderr=result.stderr,
                    exception=result.exception,
                    timed_out=result.timed_out,
                    elapsed=result.elapsed,
                    tool_call_id=tool_call_id,
                )
                self._steps.append(step)

                # ── SUBMISSION CHECK ──
                if check_submission(result.stdout, result.returncode):
                    state = TerminalState.SUBMITTED
                    break

        except asyncio.CancelledError:
            state = TerminalState.INTERRUPT
        except KeyboardInterrupt:
            state = TerminalState.INTERRUPT
        except Exception as e:
            logger.error("Unexpected error in agent loop: %s", e, exc_info=True)
            state = TerminalState.FATAL_CONFIG

        if self._interrupted:
            state = TerminalState.INTERRUPT

        trajectory = self._build_trajectory(state)
        return state, trajectory

    # ── internal helpers ────────────────────────────────────────

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, self._handle_interrupt)
            loop.add_signal_handler(signal.SIGTERM, self._handle_interrupt)
        except NotImplementedError:
            pass

    def _handle_interrupt(self) -> None:
        self._interrupted = True

    def _build_initial_messages(self) -> None:
        system_prompt = self._config.template.system_prompt or (
            "You are a software engineering agent. Your task is to solve "
            "the problem described by the user. On each turn, you may "
            "provide exactly ONE shell command to execute. "
            "Use the bash tool (or wrap the command in "
            "```mswea_bash_command``` / <mswea_bash_command> blocks). "
            "When you have completed the task, run:\n"
            "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\n"
            "followed by your final answer."
        )
        self._messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=self._task),
        ]

    async def _confirm_action(self, command: str) -> bool:
        """Ask the user to approve a command. Returns True to proceed."""
        loop = asyncio.get_event_loop()
        print(f"\n{'─' * 60}")
        print(f"Command: {command}")
        print(f"{'─' * 60}")
        try:
            answer = (await loop.run_in_executor(
                None, input, "Execute? [y/N] "
            )).strip().lower()
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False

    def _render_observation(self, result) -> str:
        template = self._config.template.observation_template
        variables = {
            "command": result.command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exception": result.exception,
            "timed_out": result.timed_out,
            "elapsed": result.elapsed,
        }
        return render_observation(
            template,
            variables,
            max_chars=self._config.template.observation_max_chars,
            truncate_head=self._config.template.observation_truncate_head,
            truncate_tail=self._config.template.observation_truncate_tail,
        )

    def _build_trajectory(self, state: TerminalState) -> Trajectory:
        safe_config = self._config.model_dump()
        if "model" in safe_config and "api_key" in safe_config["model"]:
            del safe_config["model"]["api_key"]

        return Trajectory(
            task=self._task,
            model_name=self._model.model_name,
            terminal_state=state.value,
            steps=self._steps,
            messages=[_message_to_dict(m) for m in self._messages],
            total_cost=self._total_cost,
            total_steps=len(self._steps),
            start_time=self._start_time,
            end_time=datetime.now(timezone.utc),
            config=safe_config,
        )


def _message_to_dict(msg: Message) -> dict[str, Any]:
    """Convert a Message to a JSON-serializable dict for trajectory."""
    d = {
        "role": msg.role,
        "content": msg.content,
        "timestamp": msg.timestamp.isoformat(),
    }
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function_name": tc.function_name,
                "arguments": tc.arguments,
            }
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    if msg.name:
        d["name"] = msg.name
    return d
