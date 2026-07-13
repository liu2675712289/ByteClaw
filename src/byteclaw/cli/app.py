"""Typer application for the ``byteclaw`` command."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty

from byteclaw.core.agent import stream_agent_events

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _render_event(event: dict) -> None:
    """Render one agent event to the terminal."""

    event_type = event["type"]
    if event_type == "tool_call":
        console.print(f"[bold cyan]Tool call:[/] {event['name']}")
        console.print(Pretty(event["args"]))
    elif event_type == "tool_result":
        console.print(f"[bold green]Tool result:[/] {event['name']}")
        console.print(Pretty(event["result"]))
    elif event_type == "final_answer":
        console.print(
            Panel(str(event["content"]), title="ByteClaw", border_style="green")
        )


@app.command()
def main(
    task: Annotated[str, typer.Argument(help="Task for ByteClaw to complete.")],
    workspace: Annotated[
        Path,
        typer.Option(
            "--workspace",
            "-w",
            help="Workspace directory; it is created when missing.",
            file_okay=False,
            dir_okay=True,
        ),
    ] = Path("workspace"),
) -> None:
    """Run ByteClaw on TASK inside a workspace."""

    for event in stream_agent_events(task, workspace=workspace):
        _render_event(event)
