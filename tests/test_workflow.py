import unittest
from unittest.mock import patch

from byteclaw.graph.nodes import final_node
from byteclaw.graph.workflow import build_workflow
from byteclaw.prompts.stage2 import (
    ACTOR_PROMPT,
    FINAL_PROMPT,
    PLANNER_PROMPT,
    VERIFIER_PROMPT,
)


class WorkflowTests(unittest.TestCase):
    def test_final_node_formats_passed_and_failed_states(self) -> None:
        passed = final_node(
            {
                "passed": True,
                "attempts": 1,
                "max_attempts": 3,
                "last_actor_summary": "Created and tested index.html",
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
            }

        def actor(state):
            calls.append("actor")
            return {"last_actor_summary": "Done"}

        def verifier(state):
            calls.append("verifier")
            return {"passed": True, "attempts": 1}

        with (
            patch("byteclaw.graph.workflow.planner_node", planner),
            patch("byteclaw.graph.workflow.actor_node", actor),
            patch("byteclaw.graph.workflow.verifier_node", verifier),
        ):
            workflow = build_workflow()
            result = workflow.invoke({"task": "Build", "max_attempts": 3})

        self.assertEqual(calls, ["planner", "actor", "verifier"])
        self.assertIn("Status: PASSED", result["final_answer"])

    def test_stage2_prompts_define_each_role_and_json_contract(self) -> None:
        self.assertIn("plan_summary", PLANNER_PROMPT)
        self.assertIn("TodoUpdateTool", ACTOR_PROMPT)
        self.assertIn("recommended_next_instruction", VERIFIER_PROMPT)
        self.assertIn("passed or failed", FINAL_PROMPT)


if __name__ == "__main__":
    unittest.main()
