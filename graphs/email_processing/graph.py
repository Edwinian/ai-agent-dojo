from langgraph.graph import END, START

from constants import ModelName, NodeName
from graphs.base_graph import BaseGraph
from type import GraphEdge, GraphNode

from .nodes import (
    classify_email,
    draft_response,
    handle_spam,
    notify_mr_hugg,
    read_email,
)
from .routes import route_email
from .state import EmailState


class EmailProcessingGraph(BaseGraph[NodeName, EmailState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        self.model_name = model_name
        nodes: list[GraphNode[NodeName, EmailState]] = [
            GraphNode[NodeName, EmailState](name=NodeName.READ_EMAIL, function=read_email),
            GraphNode[NodeName, EmailState](name=NodeName.CLASSIFY_EMAIL, function=classify_email),
            GraphNode[NodeName, EmailState](name=NodeName.HANDLE_SPAM, function=handle_spam),
            GraphNode[NodeName, EmailState](name=NodeName.DRAFT_RESPONSE, function=draft_response),
            GraphNode[NodeName, EmailState](name=NodeName.NOTIFY_MR_HUGG, function=notify_mr_hugg),
        ]
        edges: list[GraphEdge[NodeName, EmailState]] = [
            GraphEdge[NodeName, EmailState](source_node=START, target_node=NodeName.READ_EMAIL),
            GraphEdge[NodeName, EmailState](
                source_node=NodeName.READ_EMAIL, target_node=NodeName.CLASSIFY_EMAIL
            ),
            GraphEdge[NodeName, EmailState](
                source_node=NodeName.CLASSIFY_EMAIL,
                route_function=route_email,
                route_map={
                    NodeName.HANDLE_SPAM: NodeName.HANDLE_SPAM,
                    NodeName.DRAFT_RESPONSE: NodeName.DRAFT_RESPONSE,
                },
            ),
            GraphEdge[NodeName, EmailState](source_node=NodeName.HANDLE_SPAM, target_node=END),
            GraphEdge[NodeName, EmailState](
                source_node=NodeName.DRAFT_RESPONSE, target_node=NodeName.NOTIFY_MR_HUGG
            ),
            GraphEdge[NodeName, EmailState](source_node=NodeName.NOTIFY_MR_HUGG, target_node=END),
        ]
        super().__init__(model_name=model_name, state=EmailState, nodes=nodes, edges=edges)


__all__ = ["EmailProcessingGraph"]
