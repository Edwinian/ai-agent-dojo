from typing import Annotated, Any

from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId
from langgraph.types import Command


@tool
def write_file(
    file_path: str,
    content: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command[Any]:
    """Write content to a file in the virtual filesystem."""
    return Command(
        update={
            "files": {file_path: content},
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


write_file_tool = write_file

__all__ = ["write_file", "write_file_tool"]
