"""Shared state schema for the ByteClaw LangGraph workflow."""

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from byteclaw.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class VerificationResult(TypedDict):
    command: str
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str


class VerificationCheck(TypedDict):
    name: str
    passed: bool
    detail: str


class ByteGraphState(TypedDict, total=False):
    task: str
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    plan_summary: str
    todos: list[TodoItem]
    acceptance_criteria: list[str]
    verification_commands: list[str]
    verification_results: list[VerificationResult]
    verification_checks: list[VerificationCheck]
    passed: bool
    attempts: int
    max_attempts: int
    last_actor_summary: str
    last_error: str
    final_answer: str
