from langgraph.graph import END, START

from constants import ModelName
from graphs.base_graph import BaseGraph
from type import GraphEdge, GraphNode

from .nodes import (
    EmailNode,
    classify_email,
    draft_response,
    handle_spam,
    notify_mr_hugg,
    read_email,
)
from .routes import route_email
from .state import EmailState


class EmailProcessingGraph(BaseGraph[EmailNode, EmailState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        self.model_name = model_name
        nodes: list[GraphNode[EmailNode, EmailState]] = [
            GraphNode[EmailNode, EmailState](name=EmailNode.READ_EMAIL, function=read_email),
            GraphNode[EmailNode, EmailState](name=EmailNode.CLASSIFY_EMAIL, function=classify_email),
            GraphNode[EmailNode, EmailState](name=EmailNode.HANDLE_SPAM, function=handle_spam),
            GraphNode[EmailNode, EmailState](name=EmailNode.DRAFT_RESPONSE, function=draft_response),
            GraphNode[EmailNode, EmailState](name=EmailNode.NOTIFY_MR_HUGG, function=notify_mr_hugg),
        ]
        edges: list[GraphEdge[EmailNode, EmailState]] = [
            GraphEdge[EmailNode, EmailState](source_node=START, target_node=EmailNode.READ_EMAIL),
            GraphEdge[EmailNode, EmailState](
                source_node=EmailNode.READ_EMAIL, target_node=EmailNode.CLASSIFY_EMAIL
            ),
            GraphEdge[EmailNode, EmailState](
                source_node=EmailNode.CLASSIFY_EMAIL,
                route_function=route_email,
                route_map={
                    EmailNode.HANDLE_SPAM: EmailNode.HANDLE_SPAM,
                    EmailNode.DRAFT_RESPONSE: EmailNode.DRAFT_RESPONSE,
                },
            ),
            GraphEdge[EmailNode, EmailState](source_node=EmailNode.HANDLE_SPAM, target_node=END),
            GraphEdge[EmailNode, EmailState](
                source_node=EmailNode.DRAFT_RESPONSE, target_node=EmailNode.NOTIFY_MR_HUGG
            ),
            GraphEdge[EmailNode, EmailState](source_node=EmailNode.NOTIFY_MR_HUGG, target_node=END),
        ]
        super().__init__(model_name=model_name, state=EmailState, nodes=nodes, edges=edges)


__all__ = ["EmailProcessingGraph"]
