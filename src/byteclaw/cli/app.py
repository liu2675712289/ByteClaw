"""Typer application for the ``byteclaw`` command."""

import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from typer import Option

from byteclaw.core.agent import stream_agent_events

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    context_settings={"allow_extra_args": True},
    epilog="Launch the interactive interface with: byteclaw tui [OPTIONS]",
)
console = Console()


def _render_event(event: dict) -> None:
    """Render one agent event to the terminal."""

    event_type = event.get("type")
    if event_type == "custom_event":
        console.print(Pretty(event.get("event")))
        return
    if event_type == "graph_event":
        graph_event = event.get("event", {})
        if isinstance(graph_event, dict):
            for node, output in graph_event.items():
                _render_node_output(node, output)
        return
    _render_node_output(event["node"], event["output"])


def _render_node_output(node: str, output: object) -> None:
    """Render one graph node output."""

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
    ctx: typer.Context,
    workspace: Annotated[Path | None, Option("--workspace", "-w")] = None,
    max_attempts: Annotated[int, Option("--max-attempts")] = 3,
    approval_mode: Annotated[
        Literal["inline", "auto", "deny"], Option("--approval-mode")
    ] = "inline",
    checkpoint_mode: Annotated[
        Literal["light", "strict", "off"], Option("--checkpoint-mode")
    ] = "light",
    trace_mode: Annotated[
        Literal["on", "off"], Option("--trace-mode")
    ] = "on",
    resume: Annotated[Path | None, Option("--resume")] = None,
) -> None:
    """Run ByteClaw on TASK inside a workspace."""

    task = " ".join(ctx.args).strip()
    if not task and resume is None:
        raise typer.UsageError("Provide a task or use --resume.")

    for event in stream_agent_events(
        task,
        workspace=workspace or resume or Path("workspace"),
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
        resume_workspace=resume,
    ):
        _render_event(event)


def run() -> None:
    """Dispatch the installed command while preserving ``byteclaw TASK``."""

    if len(sys.argv) > 1 and sys.argv[1] == "tui":
        from byteclaw.cli.tui.app import tui_cli

        sys.argv = [f"{sys.argv[0]} tui", *sys.argv[2:]]
        tui_cli()
        return
    app()
