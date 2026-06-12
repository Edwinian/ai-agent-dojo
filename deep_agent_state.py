from typing import Annotated

from langchain.agents import AgentState
from langchain.agents.middleware.todo import Todo
from typing_extensions import NotRequired


def file_reducer(
    left: dict[str, str] | None,
    right: dict[str, str] | None,
) -> dict[str, str]:
    """Merge two file dictionaries, with right side taking precedence."""
    if left is None:
        return right or {}
    if right is None:
        return left
    return {**left, **right}


class DeepAgentState(AgentState):
    """Extended agent state that includes task tracking and virtual file system.

    Inherits from LangGraph's AgentState and adds:
    - todos: List of Todo items for task planning and progress tracking
    - files: Virtual file system stored as dict mapping filenames to content
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]


__all__ = ["DeepAgentState", "Todo", "file_reducer"]
