"""Run the ByteClaw workflow with approval, checkpoints, and tracing."""

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from byteclaw.core.checkpoint import CheckpointManager
from byteclaw.core.state import ApprovalHandler, create_runtime
from byteclaw.core.trace import TraceRecorder
from byteclaw.graph.workflow import build_workflow


_CHECKPOINT_CUSTOM_EVENT_TYPES = {
    "checkpoint_saved",
    "handoff",
    "tool_result",
}


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_attempts: int = 3,
    approval_mode: str = "inline",
    approval_handler: ApprovalHandler | None = None,
    checkpoint_mode: str = "light",
    resume_workspace: str | Path | None = None,
    trace_mode: str = "on",
) -> Iterator[dict[str, Any]]:
    """Run the workflow and persist every event needed for recovery."""

    runtime = create_runtime(
        workspace,
        approval_mode=approval_mode,
        approval_handler=approval_handler,
        checkpoint_mode=checkpoint_mode,
        resume_from=resume_workspace,
        trace_mode=trace_mode,
    )
    manager = CheckpointManager(runtime, task=task)
    trace = TraceRecorder(runtime, task=task)

    resumed = resume_workspace is not None
    resume_event = None
    if resumed:
        inputs, resume_event = CheckpointManager.load_resume_inputs(
            runtime,
            task=task or None,
            max_attempts=max_attempts,
        )
        inputs["runtime"] = runtime
        inputs["max_attempts"] = max_attempts
    else:
        inputs = {
            "task": task,
            "runtime": runtime,
            "max_attempts": max_attempts,
        }
    current_state = dict(inputs)
    latest_node = "start"
    workflow = build_workflow()

    trace.start(inputs, resumed=resumed, resume_event=resume_event)
    try:
        _save_checkpoint(
            manager,
            trace,
            current_state,
            status="started",
            latest_node=latest_node,
        )
        for stream_mode, event in workflow.stream(
            inputs, stream_mode=["updates", "custom"]
        ):
            if stream_mode == "custom":
                trace.record_custom_event(event)
                if isinstance(event, Mapping) and event.get("node"):
                    latest_node = str(event["node"])
                if (
                    manager.mode == "strict"
                    or _custom_event_needs_checkpoint(event)
                ):
                    _save_checkpoint(
                        manager,
                        trace,
                        current_state,
                        status="running",
                        latest_node=latest_node,
                        event=event,
                    )
                yield {"type": "custom_event", "event": event}
                continue

            trace.record_graph_update(event)
            latest_node = _merge_graph_update(
                current_state,
                event,
                latest_node,
            )
            _save_checkpoint(
                manager,
                trace,
                current_state,
                status="running",
                latest_node=latest_node,
                event=event,
            )
            yield {"type": "graph_event", "event": event}

        _save_checkpoint(
            manager,
            trace,
            current_state,
            status="finished",
            latest_node=latest_node,
        )
        trace.end(
            status="finished",
            latest_node=latest_node,
            final_state=current_state,
        )
    except KeyboardInterrupt:
        _save_checkpoint(
            manager,
            trace,
            current_state,
            status="interrupted",
            latest_node=latest_node,
        )
        trace.end(
            status="interrupted",
            latest_node=latest_node,
            final_state=current_state,
        )


def _custom_event_needs_checkpoint(event: Any) -> bool:
    return (
        isinstance(event, Mapping)
        and event.get("type") in _CHECKPOINT_CUSTOM_EVENT_TYPES
    )


def _merge_graph_update(
    current_state: dict[str, Any],
    event: Any,
    latest_node: str,
) -> str:
    if not isinstance(event, Mapping):
        return latest_node
    for node, update in event.items():
        latest_node = str(node)
        if not isinstance(update, Mapping):
            continue
        for key, value in update.items():
            if (
                key == "messages"
                and isinstance(current_state.get(key), list)
                and isinstance(value, list)
            ):
                current_state[key] = [*current_state[key], *value]
            else:
                current_state[key] = value
    return latest_node


def _save_checkpoint(
    manager: CheckpointManager,
    trace: TraceRecorder,
    current_state: Mapping[str, Any],
    *,
    status: str,
    latest_node: str,
    event: Any = None,
) -> None:
    saved_event = manager.save(
        current_state,
        status=status,
        latest_node=latest_node,
        event=event,
    )
    if saved_event is not None:
        trace.record_custom_event(saved_event)
