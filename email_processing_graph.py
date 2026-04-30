from langgraph.graph import END, START, StateGraph
from langfuse.langchain import CallbackHandler

from constants import ModelName, NodeName
from llm_service import LLMService
from nodes import (
    classify_email,
    draft_response,
    handle_spam,
    notify_mr_hugg,
    read_email,
    set_llm_service,
)
from routes import route_email
from state import EmailState
from type import GraphEdge, GraphNode


class EmailProcessingGraph:
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        self.model_name = model_name
        self.llm_service = LLMService(model_name=model_name)
        set_llm_service(self.llm_service)
        self.state = EmailState
        self.nodes: list[GraphNode[NodeName, EmailState]] = [
            GraphNode(name=NodeName.READ_EMAIL, function=read_email),
            GraphNode(name=NodeName.CLASSIFY_EMAIL, function=classify_email),
            GraphNode(name=NodeName.HANDLE_SPAM, function=handle_spam),
            GraphNode(name=NodeName.DRAFT_RESPONSE, function=draft_response),
            GraphNode(name=NodeName.NOTIFY_MR_HUGG, function=notify_mr_hugg),
        ]
        self.edges: list[GraphEdge[NodeName, EmailState]] = [
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

    def compile(self):
        email_graph = StateGraph(self.state)
        for node in self.nodes:
            email_graph.add_node(node.name, node.function)

        for edge in self.edges:
            has_direct_edge = edge.target_node is not None
            has_conditional_edge = (
                edge.route_function is not None and edge.route_map is not None
            )

            if has_direct_edge:
                email_graph.add_edge(edge.source_node, edge.target_node)

            if has_conditional_edge:
                email_graph.add_conditional_edges(
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

        return email_graph.compile()

    def invoke(self, input: EmailState):  # pylint: disable=redefined-builtin
        compiled_graph = self.compile()
        langfuse_handler = CallbackHandler()
        return compiled_graph.invoke(
            input=input,
            config={
                "callbacks": [langfuse_handler],
                "metadata": {"ls_model_name": self.model_name.value},
            },
        )

__all__ = ["EmailProcessingGraph"]
