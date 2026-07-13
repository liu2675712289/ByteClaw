"""LangChain tool registration."""

from langchain_core.tools import StructuredTool

from byteclaw.core.state import RuntimeState
from byteclaw.tools.bash_tool import BashTool
from byteclaw.tools.file_tools import FileEditTool, FileReadTool, FileWriteTool
from byteclaw.tools.grep_tool import GrepTool


def build_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build the tools passed to ``ChatOpenAI.bind_tools``."""

    return [
        StructuredTool.from_function(
            func=FileReadTool(state).__call__,
            name="file_read",
            description="Read a range of lines from a UTF-8 file in the workspace.",
        ),
        StructuredTool.from_function(
            func=FileWriteTool(state).__call__,
            name="file_write",
            description="Create or overwrite a UTF-8 file in the workspace.",
        ),
        StructuredTool.from_function(
            func=FileEditTool(state).__call__,
            name="file_edit",
            description="Replace a text fragment that occurs exactly once in a workspace file.",
        ),
        StructuredTool.from_function(
            func=GrepTool(state).__call__,
            name="grep",
            description="Search workspace text files using a regular expression.",
        ),
        StructuredTool.from_function(
            func=BashTool(state).__call__,
            name="bash",
            description="Run a shell command with the workspace as its working directory.",
        ),
    ]


def build_read_only_tools(state: RuntimeState) -> list[StructuredTool]:
    """Build workspace tools that cannot modify files or run commands."""

    return [
        StructuredTool.from_function(
            func=FileReadTool(state).__call__,
            name="file_read",
            description="Read a range of lines from a UTF-8 file in the workspace.",
        ),
        StructuredTool.from_function(
            func=GrepTool(state).__call__,
            name="grep",
            description="Search workspace text files using a regular expression.",
        ),
    ]
