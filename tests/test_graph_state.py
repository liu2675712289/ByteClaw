import unittest
from typing import Annotated, get_args, get_origin, get_type_hints

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from byteclaw.graph.state import (
    AgentHandoff,
    ByteGraphState,
    CompressionEvent,
    LayeredMemory,
    SourceItem,
    TodoItem,
    VerificationResult,
)


class GraphStateTests(unittest.TestCase):
    def test_byte_graph_state_is_optional_typed_dict(self) -> None:
        self.assertFalse(ByteGraphState.__total__)
        self.assertEqual(ByteGraphState.__required_keys__, frozenset())

    def test_messages_field_uses_add_messages_reducer(self) -> None:
        annotation = get_type_hints(ByteGraphState, include_extras=True)["messages"]

        self.assertIs(get_origin(annotation), Annotated)
        message_type, reducer = get_args(annotation)
        self.assertEqual(get_origin(message_type), list)
        self.assertEqual(get_args(message_type), (BaseMessage,))
        self.assertIs(reducer, add_messages)

    def test_nested_state_types_have_required_fields(self) -> None:
        self.assertEqual(
            TodoItem.__required_keys__,
            frozenset({"id", "content", "status", "note"}),
        )
        self.assertEqual(
            VerificationResult.__required_keys__,
            frozenset({"command", "ok", "exit_code", "stdout", "stderr"}),
        )
        self.assertFalse(AgentHandoff.__total__)
        self.assertEqual(
            AgentHandoff.__optional_keys__,
            frozenset({"from_agent", "to_agent", "instruction", "result"}),
        )
        self.assertFalse(SourceItem.__total__)

    def test_stage3_state_fields_are_declared(self) -> None:
        annotations = get_type_hints(ByteGraphState, include_extras=True)

        self.assertIs(annotations["research_notes"], str)
        self.assertEqual(get_args(annotations["sources"]), (SourceItem,))
        self.assertEqual(
            get_args(annotations["agent_handoffs"]), (AgentHandoff,)
        )
        self.assertIs(annotations["code_agent_summary"], str)
        self.assertIs(annotations["context_summary"], str)
        self.assertIs(annotations["context_token_count"], int)
        self.assertIs(annotations["context_token_limit"], int)
        self.assertIs(annotations["context_should_compress"], bool)
        self.assertIs(annotations["context_next_node"], str)
        self.assertEqual(
            get_args(annotations["compression_events"]), (CompressionEvent,)
        )
        self.assertIs(annotations["memory_snapshot"], LayeredMemory)
        self.assertIs(annotations["history_summary"], str)


if __name__ == "__main__":
    unittest.main()
