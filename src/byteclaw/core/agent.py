"""Stream normalized events from the ByteClaw workflow."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from byteclaw.core.state import RuntimeState
from byteclaw.graph.workflow import build_workflow


def stream_agent_events(
    task: str,
    *,
    workspace: str | Path,
    max_attempts: int = 3,
) -> Iterator[dict[str, Any]]:
    """Run the workflow and yield node outputs in a CLI-friendly format."""

    inputs = {
        "task": task,
        "runtime": RuntimeState(Path(workspace)),
        "max_attempts": max_attempts,
    }
    workflow = build_workflow()

    for stream_mode, payload in workflow.stream(
        inputs, stream_mode=["updates", "custom"]
    ):
        if stream_mode == "custom":
            yield {"type": "node_output", "node": "planner", "output": payload}
            continue

        for node, output in payload.items():
            yield {"type": "node_output", "node": node, "output": output}
