"""LangGraph workflow construction for ByteClaw."""

from langgraph.graph import END, START, StateGraph

from byteclaw.graph.nodes import (
    context_compressor_node,
    context_compressor_route,
    context_monitor_node,
    context_monitor_route,
    final_node,
    planner_node,
    verifier_node,
)
from byteclaw.graph.state import ByteGraphState


def build_complex_workflow():
    """Compile the monitored supervisor-verifier workflow."""

    graph = StateGraph(ByteGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("context_monitor", context_monitor_node)
    graph.add_node("context_compressor", context_compressor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "context_monitor")
    graph.add_conditional_edges(
        "context_monitor",
        context_monitor_route,
        {
            "context_compressor": "context_compressor",
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_conditional_edges(
        "context_compressor",
        context_compressor_route,
        {
            "verifier": "verifier",
            "planner": "planner",
            "final": "final",
        },
    )
    graph.add_edge("verifier", "context_monitor")
    graph.add_edge("final", END)
    return graph.compile()


def build_workflow():
    """Build the current ByteClaw workflow through its stable entry point."""

    return build_complex_workflow()
