from abc import ABC
from typing import Generic, TypeVar

from langfuse.langchain import CallbackHandler
from langgraph.graph import StateGraph

from type import GraphEdge, GraphNode
from constants import ModelName

NODE_NAME = TypeVar("NODE_NAME")
GRAPH_STATE = TypeVar("GRAPH_STATE")


class BaseGraph(ABC, Generic[NODE_NAME, GRAPH_STATE]):
    def __init__(
        self,
        model_name: ModelName = ModelName.DEEPSEEK_V4_PRO,
        state: type[GRAPH_STATE] | None = None,
        nodes: list[GraphNode[NODE_NAME, GRAPH_STATE]] | None = None,
        edges: list[GraphEdge[NODE_NAME, GRAPH_STATE]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.state = state
        self.nodes = nodes or []
        self.edges = edges or []

    def compile(self):
        graph = StateGraph(self.state)
        for node in self.nodes:
            graph.add_node(node.name, node.function)

        for edge in self.edges:
            has_direct_edge = edge.target_node is not None
            has_conditional_edge = (
                edge.route_function is not None and edge.route_map is not None
            )

            if has_direct_edge:
                graph.add_edge(edge.source_node, edge.target_node)

            if has_conditional_edge:
                graph.add_conditional_edges(
                    edge.source_node, edge.route_function, edge.route_map
                )

            if has_direct_edge and has_conditional_edge:
                raise ValueError(
                    f"Invalid edge config for '{edge.source_node}': cannot define both target_node and conditional route."
                )

            if not has_direct_edge and not has_conditional_edge:
                raise ValueError(
                    f"Invalid edge config for '{edge.source_node}': must define target_node or conditional route."
                )

        return graph.compile()

    def invoke(self, input: GRAPH_STATE):  # pylint: disable=redefined-builtin
        compiled_graph = self.compile()
        langfuse_handler = CallbackHandler()
        return compiled_graph.invoke(
            input=input,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {
                    "ls_model_name":  self.model_name ,
                },
            },
        )
