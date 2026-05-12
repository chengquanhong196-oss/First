"""Textual TUI application for inspecting trajectory files."""

import sys

from mini_swe_agent.trajectory.reader import load_trajectory_raw


def run_inspector(path: str) -> None:
    """Launch the Textual inspector for a trajectory file."""
    try:
        data = load_trajectory_raw(path)
    except Exception as e:
        print(f"Error loading trajectory: {e}", file=sys.stderr)
        sys.exit(1)

    from mini_swe_agent.inspector.screens import StepListScreen

    from textual.app import App

    class InspectorApp(App):
        """Textual app for inspecting Mini SWE Agent trajectories."""

        TITLE = f"Mini SWE Agent — Trajectory Inspector"
        SUB_TITLE = path

        def on_mount(self) -> None:
            self.push_screen(StepListScreen(data))

    app = InspectorApp()
    app.run()
