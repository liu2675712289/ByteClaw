"""Textual multi-turn interface for ByteClaw."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import Header, Input, RichLog, Static
from typer import Option

from byteclaw.cli.tui.approval import (
    ApprovalGate,
    ApprovalModal,
    ApprovalRequestedMessage,
)
from byteclaw.cli.tui.logo import ByteClawLogo
from byteclaw.core.agent import stream_session_events
from byteclaw.core.approval import ApprovalDecision, ApprovalRequest
from byteclaw.core.session import load_or_create_session


tui_cli = typer.Typer(
    add_completion=False,
    help="Launch ByteClaw's interactive multi-turn terminal interface.",
)


class AgentEventMessage(Message):
    """Deliver one agent event from a worker thread to the UI thread."""

    def __init__(self, event: dict[str, Any]) -> None:
        self.event = event
        super().__init__()


class AgentRunFinishedMessage(Message):
    """Notify the UI that the current session turn has finished."""


class ByteClawTuiApp(App[None]):
    """Interactive multi-turn ByteClaw terminal application."""

    TITLE = "🐾 ByteClaw"
    CSS = """
    Screen {
        layout: vertical;
    }

    Header {
        height: 1;
    }

    #status-bar {
        width: 100%;
        height: 1;
        padding: 0 1;
        content-align: right middle;
        background: $surface;
        color: $text-muted;
    }

    #plan-panel {
        width: 100%;
        min-height: 3;
        max-height: 9;
        padding: 1 2;
        border: round $primary;
        overflow-y: auto;
    }

    #event-log {
        width: 100%;
        height: 1fr;
        padding: 0 1;
        border: round $secondary;
    }

    #task-input {
        width: 100%;
        height: 3;
        margin-top: 1;
        border: tall $accent;
    }
    """

    def __init__(
        self,
        workspace: str | Path | None = None,
        *,
        max_attempts: int = 3,
        approval_mode: str = "inline",
        checkpoint_mode: str = "light",
        trace_mode: str = "on",
    ) -> None:
        super().__init__()
        self.workspace = Path(workspace or "workspace").expanduser().resolve()
        session = load_or_create_session(self.workspace)
        self.session_id = str(session.get("session_id", "unknown"))
        self.max_attempts = max_attempts
        self.approval_mode = approval_mode
        self.checkpoint_mode = checkpoint_mode
        self.trace_mode = trace_mode
        self.event_lines: list[str] = []
        self.plan_text = "[Plan] No active plan"
        self._busy = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield ByteClawLogo(id="byteclaw-logo")
        yield Static(self._status_text("ready"), id="status-bar", markup=False)
        yield Static(Text(self.plan_text), id="plan-panel")
        yield RichLog(
            id="event-log",
            wrap=True,
            markup=False,
            auto_scroll=True,
            max_lines=2000,
        )
        yield Input(
            placeholder="💬 Input: ask a question or describe a task",
            id="task-input",
        )

    def on_mount(self) -> None:
        self._log(f"🐾 Session ready in {self.workspace}")
        self.query_one("#task-input", Input).focus()

    @on(Input.Submitted, "#task-input")
    def submit_task(self, event: Input.Submitted) -> None:
        task = event.value.strip()
        if not task or self._busy:
            return
        event.input.value = ""
        event.input.disabled = True
        self._busy = True
        self._set_status("running")
        self._log(f"💬 You: {task}")
        self.run_session_turn(task)

    @work(thread=True)
    def run_session_turn(self, task: str) -> None:
        try:
            for event in stream_session_events(
                task,
                session_workspace=self.workspace,
                max_attempts=self.max_attempts,
                approval_mode=self.approval_mode,
                approval_handler=self._approval_handler,
                checkpoint_mode=self.checkpoint_mode,
                trace_mode=self.trace_mode,
            ):
                self.post_message(AgentEventMessage(event))
        except Exception as exc:
            self.post_message(
                AgentEventMessage(
                    {
                        "type": "tui_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            )
        finally:
            self.post_message(AgentRunFinishedMessage())

    def _approval_handler(
        self, request: ApprovalRequest
    ) -> ApprovalDecision:
        gate = ApprovalGate()
        self.post_message(
            ApprovalRequestedMessage(request, self.workspace, gate)
        )
        approved = gate.wait()
        return ApprovalDecision(
            approved=approved,
            reason=(
                "Approved in ByteClaw TUI"
                if approved
                else "Denied in ByteClaw TUI"
            ),
        )

    @on(AgentEventMessage)
    def handle_agent_event(self, message: AgentEventMessage) -> None:
        event = message.event
        event_type = event.get("type")
        if event_type == "custom_event":
            payload = event.get("event", {})
            if isinstance(payload, Mapping):
                self._handle_custom_event(payload)
            else:
                self._log(f"• {_brief(payload)}")
        elif event_type == "graph_event":
            graph_event = event.get("event", {})
            if isinstance(graph_event, Mapping):
                for node, output in graph_event.items():
                    self._handle_graph_output(str(node), output)
        elif event_type == "tui_error":
            self._log(f"❌ {event.get('error', 'Unknown error')}")
        else:
            self._handle_custom_event(event)

    @on(AgentRunFinishedMessage)
    def handle_run_finished(self) -> None:
        self._busy = False
        self._set_status("ready")
        task_input = self.query_one("#task-input", Input)
        task_input.disabled = False
        task_input.focus()

    @on(ApprovalRequestedMessage)
    def handle_approval_requested(
        self, message: ApprovalRequestedMessage
    ) -> None:
        self._set_status("approval required")
        self._log(
            f"⚠ Approval requested: {message.request.risk_reason}"
        )

        def resolve(decision: bool | None) -> None:
            approved = bool(decision)
            message.gate.resolve(approved)
            self._log("✅ Command approved" if approved else "⛔ Command denied")
            self._set_status("running")

        self.push_screen(
            ApprovalModal(message.request, message.workspace),
            resolve,
        )

    def _handle_custom_event(self, event: Mapping[str, Any]) -> None:
        event_type = str(event.get("type", "event"))
        if event_type == "plan_snapshot":
            self._update_plan(_event_todos(event))
        elif event_type == "memory":
            memory = event.get("memory", {})
            if isinstance(memory, Mapping):
                working = memory.get("working_memory", {})
                if isinstance(working, Mapping):
                    self._update_plan(_event_todos(working))
        elif event_type == "tool_call":
            name = str(event.get("name", "Tool"))
            args = event.get("args", {})
            if name == "TodoWriteTool" and isinstance(args, Mapping):
                self._update_plan(_event_todos(args))
            self._log(_format_tool_call(name, args))
        elif event_type == "tool_result":
            name = str(event.get("name", "Tool"))
            result = event.get("result")
            if name == "TodoWriteTool" and isinstance(result, Mapping):
                self._update_plan(_event_todos(result))
            self._log(_format_tool_result(name, result))
        elif event_type == "handoff":
            self._log(
                "🔄 Handoff: "
                f"{event.get('from', '?')} → {event.get('to', '?')}"
            )
        elif event_type == "checkpoint_saved":
            self._log(
                "💾 Checkpoint: "
                f"{event.get('status', 'saved')} @ "
                f"{event.get('latest_node', '?')}"
            )
        elif event_type == "approval_requested":
            self._log(f"⚠ Approval requested: {_brief(event)}")
        elif event_type == "final_answer":
            self._log(f"✨ {_brief(event.get('content', ''))}")
        elif event_type == "ai_message":
            self._log(f"🤖 {_brief(event.get('content', ''))}")

    def _handle_graph_output(self, node: str, output: Any) -> None:
        if not isinstance(output, Mapping):
            self._log(f"• {node}: {_brief(output)}")
            return
        if node == "planner":
            self._update_plan(_event_todos(output))
            summary = output.get("plan_summary")
            if summary:
                self._log(f"📋 Plan: {_brief(summary)}")
        elif node == "intent_router":
            self._log(
                "↪ Route: "
                f"{output.get('intent_route', 'workflow')} "
                f"({_brief(output.get('intent_reason', ''))})"
            )
        elif node == "verifier":
            icon = "✅" if output.get("passed") is True else "❌"
            self._log(f"{icon} Verifier: {_brief(output)}")
        elif node in {"final", "chat_responder"}:
            content = output.get("final_answer", output.get("chat_response", ""))
            self._log(f"✨ {_brief(content)}")

    def _update_plan(self, todos: list[dict[str, Any]]) -> None:
        if not todos:
            return
        icons = {
            "completed": "✅",
            "in_progress": "🔄",
            "blocked": "⛔",
            "pending": "⬜",
        }
        items = []
        for todo in todos:
            status = str(todo.get("status", "pending"))
            label = todo.get("content") or todo.get("id") or "todo"
            items.append(f"{icons.get(status, '⬜')} {label}")
        self.plan_text = "[Plan] " + "  ".join(items)
        self.query_one("#plan-panel", Static).update(Text(self.plan_text))

    def _log(self, line: str) -> None:
        self.event_lines.append(line)
        self.query_one("#event-log", RichLog).write(Text(line))

    def _status_text(self, state: str) -> str:
        return f"session: {self.session_id[:12]} • {state}"

    def _set_status(self, state: str) -> None:
        self.query_one("#status-bar", Static).update(self._status_text(state))


def _event_todos(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    todos = event.get("todos", [])
    if not isinstance(todos, list):
        return []
    return [dict(todo) for todo in todos if isinstance(todo, Mapping)]


def _format_tool_call(name: str, args: Any) -> str:
    lowered_name = name.lower()
    icon = "🔧"
    if "search" in lowered_name:
        icon = "🔍"
    elif "notepad" in lowered_name:
        icon = "📝"
    if isinstance(args, Mapping):
        for key in ("file_path", "path"):
            if args.get(key):
                return f"{icon} {name} → {_brief(args[key])}"
        for key in ("command", "query", "instruction", "content"):
            if args.get(key):
                return f"{icon} {name}: {_brief(args[key])}"
    return f"{icon} {name}: {_brief(args)}"


def _format_tool_result(name: str, result: Any) -> str:
    ok = result.get("ok") if isinstance(result, Mapping) else None
    icon = "❌" if ok is False else "📝"
    return f"{icon} {name}: {_brief(result)}"


def _brief(value: Any, limit: int = 700) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False, default=str)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


@tui_cli.command()
def main(
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
) -> None:
    """Start the ByteClaw TUI."""

    ByteClawTuiApp(
        workspace=workspace,
        max_attempts=max_attempts,
        approval_mode=approval_mode,
        checkpoint_mode=checkpoint_mode,
        trace_mode=trace_mode,
    ).run()


if __name__ == "__main__":
    tui_cli()
