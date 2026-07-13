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

    node = event["node"]
    output = event["output"]

    if node == "planner":
        console.print("[bold blue]📋 Planner[/]")
        console.print(Pretty(output))
    elif node == "actor":
        console.print("[bold cyan]🔧 Actor[/]")
        console.print(Pretty(output))
    elif node == "verifier":
        passed = isinstance(output, dict) and output.get("passed") is True
        label = "✅ Verifier" if passed else "❌ Verifier"
        console.print(f"[bold]{label}[/]")
        console.print(Pretty(output))
    elif node == "final":
        content = (
            output.get("final_answer", output)
            if isinstance(output, dict)
            else output
        )
        console.print(Panel(str(content), title="📝 Final", border_style="green"))


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
    max_attempts: Annotated[
        int,
        typer.Option(
            "--max-attempts",
            help="Maximum planner-verifier attempts.",
            min=1,
        ),
    ] = 3,
) -> None:
    """Run ByteClaw on TASK inside a workspace."""

    for event in stream_agent_events(
        task, workspace=workspace, max_attempts=max_attempts
    ):
        _render_event(event)
