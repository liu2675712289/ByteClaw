import tempfile
import threading
import unittest
from os import environ
from pathlib import Path
from unittest.mock import patch

from textual.containers import Horizontal
from textual.widgets import Input, Static
from typer.testing import CliRunner

from byteclaw.cli.tui.app import (
    AgentEventMessage,
    ByteClawTuiApp,
    _format_tool_call,
    tui_cli,
)
from byteclaw.cli.tui.approval import (
    ApprovalGate,
    ApprovalModal,
    ApprovalRequestedMessage,
)
from byteclaw.cli.tui.logo import (
    LOGO_TITLE,
    ByteClawLogo,
    render_logo,
)
from byteclaw.core.approval import ApprovalRequest


class ApprovalGateTests(unittest.TestCase):
    def test_gate_blocks_until_first_resolution(self) -> None:
        gate = ApprovalGate()
        decisions = []
        waiter = threading.Thread(target=lambda: decisions.append(gate.wait()))

        waiter.start()
        self.assertTrue(waiter.is_alive())
        gate.resolve(True)
        waiter.join(timeout=1)
        gate.resolve(False)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(decisions, [True])
        self.assertTrue(gate.wait())


class LogoTests(unittest.TestCase):
    def test_render_logo_contains_title_and_rich_styles(self) -> None:
        logo = render_logo()

        self.assertEqual(
            logo.plain,
            LOGO_TITLE,
        )
        self.assertEqual(len(LOGO_TITLE.splitlines()), 4)
        self.assertEqual(max(map(len, LOGO_TITLE.splitlines())), 11)
        self.assertNotIn("Self-evolving Agent Harness", logo.plain)
        self.assertGreater(len(logo.spans), 0)
        self.assertNotEqual(render_logo(0).spans, render_logo(1).spans)

class TuiTests(unittest.IsolatedAsyncioTestCase):
    def test_tui_cli_passes_runtime_options_to_app(self) -> None:
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            with patch("byteclaw.cli.tui.app.ByteClawTuiApp") as tui_app:
                result = runner.invoke(
                    tui_cli,
                    [
                        "--workspace",
                        str(workspace),
                        "--max-attempts",
                        "5",
                        "--approval-mode",
                        "auto",
                        "--checkpoint-mode",
                        "strict",
                        "--trace-mode",
                        "off",
                    ],
                )

        self.assertEqual(result.exit_code, 0, result.output)
        tui_app.assert_called_once_with(
            workspace=workspace,
            max_attempts=5,
            approval_mode="auto",
            checkpoint_mode="strict",
            trace_mode="off",
        )
        tui_app.return_value.run.assert_called_once_with()

    async def test_agent_messages_update_plan_and_event_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ByteClawTuiApp(Path(temp_dir) / "workspace")
            async with app.run_test(size=(100, 35)) as pilot:
                logo = app.query_one("#byteclaw-logo", ByteClawLogo)
                initial_frame = logo.animation_frame
                logo._advance_animation()
                self.assertNotEqual(logo.animation_frame, initial_frame)
                app.post_message(
                    AgentEventMessage(
                        {
                            "type": "graph_event",
                            "event": {
                                "planner": {
                                    "plan_summary": "Create the app",
                                    "todos": [
                                        {
                                            "id": "todo-1",
                                            "content": "Create app.py",
                                            "status": "completed",
                                        },
                                        {
                                            "id": "todo-2",
                                            "content": "Run tests",
                                            "status": "in_progress",
                                        },
                                    ],
                                }
                            },
                        }
                    )
                )
                app.post_message(
                    AgentEventMessage(
                        {
                            "type": "custom_event",
                            "event": {
                                "type": "tool_call",
                                "name": "FileWriteTool",
                                "args": {"file_path": "app.py"},
                            },
                        }
                    )
                )
                app.post_message(
                    AgentEventMessage(
                        {
                            "type": "custom_event",
                            "event": {
                                "type": "handoff",
                                "from": "planner",
                                "to": "codeAgent",
                            },
                        }
                    )
                )
                await pilot.pause()

                self.assertIn("✅ Create app.py", app.plan_text)
                self.assertIn("🔄 Run tests", app.plan_text)
                self.assertIn(
                    "🔧 FileWriteTool → app.py",
                    app.event_lines,
                )
                self.assertIn(
                    "🔄 Handoff: planner → codeAgent",
                    app.event_lines,
                )

    async def test_brand_panel_shows_version_model_and_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(environ, {"OPENAI_MODEL": "test-model"}):
                app = ByteClawTuiApp(Path(temp_dir) / "workspace")
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause()
                    panel = app.query_one("#brand-panel", Horizontal)
                    logo = app.query_one("#byteclaw-logo", ByteClawLogo)
                    status_bar = app.query_one("#status-bar", Static)

                    self.assertEqual(panel.size.height, 4)
                    self.assertEqual(logo.outer_size.width, 13)
                    self.assertFalse(status_bar.display)
                    self.assertEqual(
                        app._brand_text().plain,
                        f"ByteClaw v{app.app_version}\n"
                        f"test-model\n{app.workspace}",
                    )

    async def test_input_runs_session_turn_and_reenables_input(self) -> None:
        events = iter(
            [
                {
                    "type": "graph_event",
                    "event": {
                        "chat_responder": {
                            "final_answer": "Hello from ByteClaw"
                        }
                    },
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ByteClawTuiApp(Path(temp_dir) / "workspace")
            with patch(
                "byteclaw.cli.tui.app.stream_session_events",
                return_value=events,
            ) as stream:
                async with app.run_test(size=(100, 35)) as pilot:
                    await pilot.press("h", "e", "l", "l", "o", "enter")
                    await pilot.pause(0.2)

                    task_input = app.query_one("#task-input", Input)
                    self.assertFalse(task_input.disabled)
                    self.assertIn("💬 You: hello", app.event_lines)
                    self.assertIn("✨ Hello from ByteClaw", app.event_lines)

        stream.assert_called_once()
        self.assertEqual(stream.call_args.args, ("hello",))
        self.assertEqual(
            stream.call_args.kwargs["session_workspace"],
            app.workspace,
        )

    async def test_approval_message_resolves_from_button_and_key(self) -> None:
        request = ApprovalRequest(
            id="approval-test",
            command="python -m pip install requests",
            risk_reason="Python package installation",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            app = ByteClawTuiApp(Path(temp_dir) / "workspace")
            async with app.run_test(size=(100, 35)) as pilot:
                denied = ApprovalGate()
                app.post_message(
                    ApprovalRequestedMessage(request, app.workspace, denied)
                )
                await pilot.pause()
                self.assertIsInstance(app.screen, ApprovalModal)
                await pilot.click("#deny-button")
                await pilot.pause()
                self.assertFalse(denied.wait())

                approved = ApprovalGate()
                app.post_message(
                    ApprovalRequestedMessage(request, app.workspace, approved)
                )
                await pilot.pause()
                await pilot.press("y")
                await pilot.pause()
                self.assertTrue(approved.wait())

    def test_tool_call_formatter_prefers_workspace_path(self) -> None:
        self.assertEqual(
            _format_tool_call(
                "FileWriteTool",
                {"file_path": "src/byteclaw/app.py"},
            ),
            "🔧 FileWriteTool → src/byteclaw/app.py",
        )
        self.assertEqual(
            _format_tool_call("WebSearchTool", {"query": "Flask tutorial"}),
            "🔍 WebSearchTool: Flask tutorial",
        )
        self.assertEqual(
            _format_tool_call(
                "NotepadAppendTool",
                {"content": "Created Flask app"},
            ),
            "📝 NotepadAppendTool: Created Flask app",
        )


if __name__ == "__main__":
    unittest.main()
