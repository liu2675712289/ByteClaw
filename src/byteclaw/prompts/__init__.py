"""Prompt definitions used by ByteClaw workflows."""

from byteclaw.prompts.stage2 import (
    ACTOR_PROMPT,
    FINAL_PROMPT,
)
from byteclaw.prompts.stage3 import PLANNER_PROMPT, VERIFIER_PROMPT
from byteclaw.prompts.stage4 import CONTEXT_COMPRESSION_PROMPT

__all__ = [
    "ACTOR_PROMPT",
    "CONTEXT_COMPRESSION_PROMPT",
    "FINAL_PROMPT",
    "PLANNER_PROMPT",
    "VERIFIER_PROMPT",
]
