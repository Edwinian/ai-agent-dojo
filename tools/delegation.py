"""Sub-agent delegation tool for context-isolated task execution."""

from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from constants import ModelName
from type import SubAgent
from tools import tools_by_name

TASK_DESCRIPTION_PREFIX = """Launch a new agent to handle complex, multi-step tasks autonomously.

Available agent types:
{other_agents}

When using this tool, you must specify a subagent_type parameter to select which agent type to use."""


def _create_delegation_tool(
    subagents: list[SubAgent],
    state_schema: type[AgentState],
):
    """Create a task delegation tool that enables context isolation through sub-agents.

    This function implements the core pattern for spawning specialized sub-agents with
    isolated contexts, preventing context clash and confusion in complex multi-step tasks.

    Args:
        subagents: List of specialized sub-agent configurations
        state_schema: The state schema (typically AgentState or a subclass)

    Returns:
        A 'task' tool that can delegate work to specialized sub-agents
    """

    agents = {}
    init_model: dict[ModelName, BaseChatModel | None] = {}

    for _agent in subagents:
        _tools = []
        if _agent.tool_names is not None:
            _tools = [tools_by_name[t] for t in _agent.tool_names]

        if init_model.get(_agent.model_name) is None:
            init_model[_agent.model_name] = init_chat_model(
                model=_agent.model_name.value
            )

        agents[_agent.name] = create_agent(
            init_model[_agent.model_name],
            tools=_tools,
            system_prompt=_agent.prompt,
            state_schema=state_schema,
        )

    other_agents_string = "\n".join(
        f"- {_agent.name}: {_agent.description}" for _agent in subagents
    )

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=other_agents_string))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[AgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ):
        """Delegate a task to a specialized sub-agent with isolated context.

        This creates a fresh context for the sub-agent containing only the task description,
        preventing context pollution from the parent agent's conversation history.
        """
        if subagent_type not in agents:
            return f"Error: invoked agent of type {subagent_type}, the only allowed types are {[f'`{k}`' for k in agents]}"

        sub_agent = agents[subagent_type]

        state["messages"] = [{"role": "user", "content": description}]

        result = sub_agent.invoke(state)

        return Command(
            update={
                "files": result.get("files", {}),
                "messages": [
                    ToolMessage(
                        result["messages"][-1].content, tool_call_id=tool_call_id
                    )
                ],
            }
        )

    return task


__all__ = ["_create_delegation_tool", "TASK_DESCRIPTION_PREFIX"]
