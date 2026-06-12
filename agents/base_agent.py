from abc import ABC
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from dotenv import load_dotenv
from langchain.agents import create_agent
from langfuse.langchain import CallbackHandler
from langgraph.graph.state import CompiledStateGraph

from constants import ModelName
from huggingface_service import HuggingFaceService

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

GRAPH_STATE = TypeVar("GRAPH_STATE")


class BaseAgent(ABC, Generic[GRAPH_STATE]):
    def __init__(
        self,
        model_name: ModelName = ModelName.DEEPSEEK_V4_PRO,
        state: type[GRAPH_STATE] | None = None,
        tools: list[Callable[..., Any]] | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.state = state
        self.tools = tools
        self.system_prompt = system_prompt
        self.compiled_agent = self.compile()

    def compile(self) -> CompiledStateGraph:
        if self.state is None:
            raise ValueError("state schema is required to compile an agent")

        llm = HuggingFaceService[self.state](model_name=self.model_name).chat_llm
        return create_agent(
            model=llm,
            tools=self.tools,
            state_schema=self.state,
            system_prompt=self.system_prompt,
        )

    def invoke(self, input: GRAPH_STATE):  # pylint: disable=redefined-builtin
        langfuse_handler = CallbackHandler()
        return self.compiled_agent.invoke(
            input=input,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "ls_model_name": self.model_name,
                },
            },
        )

    def draw_mermaid_png(self, xray: bool = False):
        return self.compiled_agent.get_graph(xray=xray).draw_mermaid_png()
