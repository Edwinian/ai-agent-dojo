from langgraph.graph import END, START

from base_graph import BaseGraph
from constants import ModelName, NodeName
from llm_service import LLMService
from email_processsing_nodes import (
    classify_email,
    draft_response,
    handle_spam,
    notify_mr_hugg,
    read_email,
)
from email_processing_routes import route_email
from email_processing_state import EmailState
from type import GraphEdge, GraphNode


class EmailProcessingGraph(BaseGraph[NodeName, EmailState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        self.model_name = model_name
        self.llm_service = LLMService(model_name=model_name)
        nodes: list[GraphNode[NodeName, EmailState]] = [
            GraphNode(name=NodeName.READ_EMAIL, function=read_email),
            GraphNode(
                name=NodeName.CLASSIFY_EMAIL,
                function=lambda state: classify_email(state, self.llm_service),
            ),
            GraphNode(name=NodeName.HANDLE_SPAM, function=handle_spam),
            GraphNode(
                name=NodeName.DRAFT_RESPONSE,
                function=lambda state: draft_response(state, self.llm_service),
            ),
            GraphNode(name=NodeName.NOTIFY_MR_HUGG, function=notify_mr_hugg),
        ]
        edges: list[GraphEdge[NodeName, EmailState]] = [
            GraphEdge(source_node=START, target_node=NodeName.READ_EMAIL),
            GraphEdge(source_node=NodeName.READ_EMAIL, target_node=NodeName.CLASSIFY_EMAIL),
            GraphEdge(
                source_node=NodeName.CLASSIFY_EMAIL,
                route_function=route_email,
                route_map={
                    NodeName.HANDLE_SPAM: NodeName.HANDLE_SPAM,
                    NodeName.DRAFT_RESPONSE: NodeName.DRAFT_RESPONSE,
                },
            ),
            GraphEdge(source_node=NodeName.HANDLE_SPAM, target_node=END),
            GraphEdge(
                source_node=NodeName.DRAFT_RESPONSE, target_node=NodeName.NOTIFY_MR_HUGG
            ),
            GraphEdge(source_node=NodeName.NOTIFY_MR_HUGG, target_node=END),
        ]
        super().__init__(model_name=model_name, state=EmailState, nodes=nodes, edges=edges)

__all__ = ["EmailProcessingGraph"]
