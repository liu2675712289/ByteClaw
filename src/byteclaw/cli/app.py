"""Typer application for the ``byteclaw`` command."""

from pathlib import Path
from typing import Annotated

import typer
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from byteclaw.core.state import RuntimeState
from byteclaw.providers.openai_provider import create_model
from byteclaw.tools.registry import build_tools

app = typer.Typer(add_completion=False, no_args_is_help=True)


def run_task(task: str, state: RuntimeState) -> str:
    """Run a small tool-calling loop and return the model's final response."""

    tools = build_tools(state)
    tools_by_name = {tool.name: tool for tool in tools}
    model = create_model().bind_tools(tools)
    messages = [
        SystemMessage(
            content=(
                "You are ByteClaw, a coding assistant. Work only inside the provided "
                "workspace. Use tools to inspect and modify files, then summarize the result."
            )
        ),
        HumanMessage(content=task),
    ]

    for _ in range(25):
        response = model.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            return str(response.content)

        for tool_call in response.tool_calls:
            tool = tools_by_name.get(tool_call["name"])
            if tool is None:
                result = f"Unknown tool: {tool_call['name']}"
            else:
                try:
                    result = tool.invoke(tool_call["args"])
                except Exception as exc:  # return tool errors so the model can recover
                    result = f"{type(exc).__name__}: {exc}"
            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    raise RuntimeError("The model exceeded the 25-step tool-call limit")


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

    state = RuntimeState(workspace)
    typer.echo(run_task(task, state))
