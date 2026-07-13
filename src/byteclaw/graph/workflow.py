"""LangGraph workflow construction for ByteClaw."""

from langgraph.graph import END, START, StateGraph

from byteclaw.graph.nodes import (
    actor_node,
    final_node,
    planner_node,
    verifier_node,
    verifier_route,
)
from byteclaw.graph.state import ByteGraphState


def build_workflow():
    """Compile the ByteClaw planner-actor-verifier workflow."""

    graph = StateGraph(ByteGraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("actor", actor_node)
    graph.add_node("verifier", verifier_node)
    graph.add_node("final", final_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "actor")
    graph.add_edge("actor", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_route,
        {
            "final": "final",
            "planner": "planner",
        },
    )
    graph.add_edge("final", END)
    return graph.compile()

