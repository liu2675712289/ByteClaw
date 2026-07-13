"""Shared state schema for the ByteClaw LangGraph workflow."""

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from byteclaw.core.state import RuntimeState


class TodoItem(TypedDict):
    id: str
    content: str
    status: str
    note: str


class SourceItem(TypedDict, total=False):
    title: str
    url: str
    content: str
    score: float | None


class AgentHandoff(TypedDict, total=False):
    from_agent: str
    to_agent: str
    instruction: str
    result: str


class CompressionEvent(TypedDict, total=False):
    timestamp: str
    node: str
    token_count: int
    token_limit: int
    summary: str
    token_count_before: int
    token_count_after: int
    summary_chars: int
    next_node: str


class LayeredMemory(TypedDict):
    rules: dict[str, Any]
    working_memory: dict[str, Any]
    history_summary_store: dict[str, Any]


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
    session_context: str | dict[str, Any]
    session: dict[str, Any]
    runtime: RuntimeState
    messages: Annotated[list[BaseMessage], add_messages]
    intent_route: Literal["chat", "workflow"]
    intent_reason: str
    intent_confidence: float
    chat_response: str
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
    research_notes: str
    sources: list[SourceItem]
    agent_handoffs: list[AgentHandoff]
    code_agent_summary: str
    verifier_summary: str
    context_summary: str
    context_token_count: int
    context_token_limit: int
    context_should_compress: bool
    context_next_node: str
    compression_events: list[CompressionEvent]
    memory_snapshot: LayeredMemory
    history_summary: str
    last_error: str
    final_answer: str
