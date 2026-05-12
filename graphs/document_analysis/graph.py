from enum import Enum

from langgraph.graph import START
from langgraph.prebuilt import ToolNode, tools_condition

from constants import ModelName
from graphs.base_graph import BaseGraph
from type import GraphEdge, GraphNode

from .nodes import assistant
from .state import DocumentState
from .tools import tools


class DocumentAnalysisNode(str, Enum):
    ASSISTANT = "assistant"

class DocumentAnalysisGraph(BaseGraph[DocumentAnalysisNode, DocumentState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        self.model_name = model_name
        nodes: list[GraphNode[DocumentAnalysisNode, DocumentState]] = [
            GraphNode[DocumentAnalysisNode, DocumentState](
                name=DocumentAnalysisNode.ASSISTANT, function=assistant
            ),
            GraphNode[DocumentAnalysisNode, DocumentState](
                name='tools', function=ToolNode(tools)
            ),
        ]
        edges: list[GraphEdge[DocumentAnalysisNode, DocumentState]] = [
            GraphEdge[DocumentAnalysisNode, DocumentState](
                source_node=START, target_node=DocumentAnalysisNode.ASSISTANT
            ),
            GraphEdge[DocumentAnalysisNode, DocumentState](
                source_node=DocumentAnalysisNode.ASSISTANT,
                # If the latest message requires a tool, route to tools
                # Otherwise, provide a direct response
                route_function=tools_condition,
            ),
            GraphEdge[DocumentAnalysisNode, DocumentState](
                source_node='tools',
                target_node=DocumentAnalysisNode.ASSISTANT,
            ),
        ]
        super().__init__(
            model_name=model_name,
            state=DocumentState,
            nodes=nodes,
            edges=edges,
        )


__all__ = ["DocumentAnalysisGraph"]
