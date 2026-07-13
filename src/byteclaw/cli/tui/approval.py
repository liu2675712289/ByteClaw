"""Interactive approval primitives for the Textual interface."""

from pathlib import Path
from threading import Event, Lock

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from byteclaw.core.approval import ApprovalRequest


class ApprovalGate:
    """Block an agent thread until the UI resolves an approval request."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._approved = False

    def resolve(self, approved: bool) -> None:
        """Resolve the gate once; subsequent decisions are ignored."""

        with self._lock:
            if self._event.is_set():
                return
            self._approved = approved
            self._event.set()

    def wait(self) -> bool:
        """Wait for and return the UI decision."""

        self._event.wait()
        with self._lock:
            return self._approved


class ApprovalRequestedMessage(Message):
    """Carry one approval request and its synchronization gate to the UI."""

    def __init__(
        self,
        request: ApprovalRequest,
        workspace: Path,
        gate: ApprovalGate,
    ) -> None:
        self.request = request
        self.workspace = workspace
        self.gate = gate
        super().__init__()


class ApprovalModal(ModalScreen[bool]):
    """Show command risk details and return an approve/deny decision."""

    BINDINGS = [
        Binding("y", "approve", "Approve", show=False),
        Binding("enter", "approve", "Approve", show=False),
        Binding("n", "deny", "Deny", show=False),
        Binding("escape", "deny", "Deny", show=False),
    ]

    CSS = """
    ApprovalModal {
        align: center middle;
        background: $background 70%;
    }

    #approval-dialog {
        width: 80%;
        max-width: 100;
        height: auto;
        padding: 1 2;
        border: thick $warning;
        background: $surface;
    }

    #approval-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    .approval-field {
        margin-bottom: 1;
    }

    #approval-command {
        width: 100%;
        max-height: 8;
        padding: 1;
        border: round $secondary;
        overflow-y: auto;
    }

    #approval-actions {
        width: 100%;
        height: auto;
        align-horizontal: center;
        margin-top: 1;
    }

    #approval-actions Button {
        margin: 0 1;
        min-width: 18;
    }
    """

    def __init__(self, request: ApprovalRequest, workspace: Path) -> None:
        self.request = request
        self.workspace = workspace
        super().__init__()

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label("⚠ Approval required", id="approval-title")
            yield Static(
                f"Tool: {self.request.tool_name}",
                classes="approval-field",
                markup=False,
            )
            yield Static(
                f"Risk: {self.request.risk_reason}",
                classes="approval-field",
                markup=False,
            )
            yield Static(
                f"Workspace: {self.workspace}",
                classes="approval-field",
                markup=False,
            )
            yield Static(
                self.request.command,
                id="approval-command",
                markup=False,
            )
            with Horizontal(id="approval-actions"):
                yield Button(
                    "[Y] Approve",
                    id="approve-button",
                    variant="success",
                )
                yield Button(
                    "[N] Deny",
                    id="deny-button",
                    variant="error",
                )

    @on(Button.Pressed, "#approve-button")
    def approve_button(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#deny-button")
    def deny_button(self) -> None:
        self.dismiss(False)

    def action_approve(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)
