"""Textual screens for trajectory inspection."""

from textual.containers import Container, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, Static

from mini_swe_agent.inspector.sanitize import sanitize


class StepListScreen(Screen):
    """Main screen showing all steps in the trajectory."""

    CSS = """
    StepListScreen {
        layout: vertical;
    }
    #step-list {
        height: 1fr;
    }
    .step-entry {
        padding: 1;
        border: solid $primary;
        margin: 1;
        height: auto;
    }
    .step-entry:focus {
        border: solid $accent;
        background: $surface;
    }
    """

    def __init__(self, trajectory_data: dict) -> None:
        super().__init__()
        self.trajectory = trajectory_data

    def compose(self):
        yield Header(show_clock=True)
        yield VerticalScroll(id="step-list")
        yield Footer()

    def on_mount(self) -> None:
        scroll = self.query_one("#step-list", VerticalScroll)
        messages = self.trajectory.get("messages", [])
        steps = self.trajectory.get("steps", [])

        # Display summary
        task = sanitize(self.trajectory.get("task", "No task"))
        model = sanitize(self.trajectory.get("model_name", "Unknown"))
        state = sanitize(self.trajectory.get("terminal_state", "Unknown"))
        cost = self.trajectory.get("total_cost", 0)
        scroll.mount(Static(
            f"[bold]Task:[/] {task}\n"
            f"[bold]Model:[/] {model}  [bold]State:[/] {state}  [bold]Cost:[/] ${cost:.6f}",
            classes="step-entry",
        ))

        if steps:
            for step in steps:
                action = sanitize(step.get("action", "(format error)"))
                obs = step.get("observation", {})
                rc = obs.get("returncode", "?")
                scroll.mount(Static(
                    f"[bold]Step {step.get('step_index', '?')}[/]\n"
                    f"  Action: {action}\n"
                    f"  Return code: {rc}",
                    classes="step-entry",
                ))
        else:
            # Fall back to aggregating by assistant messages
            step_num = 0
            for i, msg in enumerate(messages):
                if msg.get("role") == "assistant":
                    content = sanitize(str(msg.get("content", ""))[:120])
                    scroll.mount(Static(
                        f"[bold]Step {step_num}[/]\n"
                        f"  Content: {content}",
                        classes="step-entry",
                    ))
                    step_num += 1


class StepDetailScreen(Screen):
    """Detail screen for a single step."""

    CSS = """
    StepDetailScreen {
        layout: vertical;
    }
    #detail-view {
        height: 1fr;
        padding: 1;
    }
    """

    def __init__(self, step: dict) -> None:
        super().__init__()
        self.step = step

    def compose(self):
        yield Header(show_clock=True)
        yield VerticalScroll(id="detail-view")
        yield Footer()

    def on_mount(self) -> None:
        scroll = self.query_one("#detail-view", VerticalScroll)
        action = sanitize(self.step.get("action", "(none)"))
        obs = self.step.get("observation", {})
        stdout = sanitize(obs.get("stdout", ""))
        stderr = sanitize(obs.get("stderr", ""))
        exception = sanitize(str(obs.get("exception", "")))

        text = (
            f"[bold]Action:[/]\n{action}\n\n"
            f"[bold]Return code:[/] {obs.get('returncode', '?')}\n"
            f"[bold]Stdout:[/]\n{stdout}\n\n"
            f"[bold]Stderr:[/]\n{stderr}\n"
        )
        if exception:
            text += f"\n[bold red]Exception:[/] {exception}\n"

        scroll.mount(Static(text))
