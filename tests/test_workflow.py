import unittest
from unittest.mock import patch

from byteclaw.graph.nodes import final_node
from byteclaw.graph.workflow import build_complex_workflow, build_workflow
from byteclaw.prompts.stage2 import (
    ACTOR_PROMPT,
    FINAL_PROMPT,
)
from byteclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT


class WorkflowTests(unittest.TestCase):
    def test_final_node_formats_passed_and_failed_states(self) -> None:
        passed = final_node(
            {
                "passed": True,
                "attempts": 1,
                "max_attempts": 3,
                "code_agent_summary": "Created and tested index.html",
            }
        )
        failed = final_node(
            {
                "passed": False,
                "attempts": 3,
                "max_attempts": 3,
                "last_error": "Tests failed",
            }
        )

        self.assertIn("Status: PASSED", passed["final_answer"])
        self.assertIn("Created and tested index.html", passed["final_answer"])
        self.assertIn("Status: FAILED", failed["final_answer"])
        self.assertIn("Tests failed", failed["final_answer"])

    def test_build_workflow_executes_expected_path(self) -> None:
        calls: list[str] = []

        def planner(state):
            calls.append("planner")
            return {
                "plan_summary": "Plan",
                "todos": [],
                "acceptance_criteria": [],
                "verification_commands": [],
                "context_next_node": "verifier",
            }

        def context_monitor(state):
            calls.append("context_monitor")
            return {
                "context_should_compress": False,
                "context_next_node": state.get(
                    "context_next_node", "verifier"
                ),
            }

        def verifier(state):
            calls.append("verifier")
            return {"passed": True, "attempts": 1}

        with (
            patch("byteclaw.graph.workflow.planner_node", planner),
            patch(
                "byteclaw.graph.workflow.context_monitor_node",
                context_monitor,
            ),
            patch("byteclaw.graph.workflow.verifier_node", verifier),
        ):
            workflow = build_workflow()
            result = workflow.invoke({"task": "Build", "max_attempts": 3})

        self.assertEqual(
            calls,
            ["planner", "context_monitor", "verifier", "context_monitor"],
        )
        self.assertIn("Status: PASSED", result["final_answer"])
        self.assertEqual(
            set(workflow.get_graph().nodes),
            {
                "__start__",
                "planner",
                "context_monitor",
                "context_compressor",
                "verifier",
                "final",
                "__end__",
            },
        )

    def test_complex_workflow_routes_through_compressor(self) -> None:
        calls: list[str] = []

        def planner(state):
            calls.append("planner")
            return {"context_next_node": "verifier"}

        def context_monitor(state):
            calls.append("context_monitor")
            return {
                "context_should_compress": not bool(
                    state.get("history_summary")
                )
            }

        def context_compressor(state):
            calls.append("context_compressor")
            return {
                "history_summary": "compressed",
                "context_should_compress": False,
            }

        def verifier(state):
            calls.append("verifier")
            return {"passed": True, "attempts": 1}

        with (
            patch("byteclaw.graph.workflow.planner_node", planner),
            patch(
                "byteclaw.graph.workflow.context_monitor_node",
                context_monitor,
            ),
            patch(
                "byteclaw.graph.workflow.context_compressor_node",
                context_compressor,
            ),
            patch("byteclaw.graph.workflow.verifier_node", verifier),
        ):
            result = build_complex_workflow().invoke({"task": "Build"})

        self.assertEqual(
            calls,
            [
                "planner",
                "context_monitor",
                "context_compressor",
                "verifier",
                "context_monitor",
            ],
        )
        self.assertIn("Status: PASSED", result["final_answer"])

    def test_stage3_planner_prompt_defines_supervisor_tools(self) -> None:
        self.assertIn("planner/supervisor", PLANNER_PROMPT)
        self.assertIn("CallSearchAgentTool", PLANNER_PROMPT)
        self.assertIn("CallCodeAgentTool", PLANNER_PROMPT)
        self.assertIn("Always call TodoWriteTool", PLANNER_PROMPT)
        self.assertIn("TodoUpdateTool", ACTOR_PROMPT)
        self.assertIn("recommended_next_instruction", VERIFIER_PROMPT)
        self.assertIn("Check the actual workspace", VERIFIER_PROMPT)
        self.assertIn("passed or failed", FINAL_PROMPT)


if __name__ == "__main__":
    unittest.main()
