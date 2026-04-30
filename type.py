from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

NODE_NAME = TypeVar("NODE")
GRAPH_STATE = TypeVar("GRAPH_STATE")

@dataclass(frozen=True)
class GraphNode(Generic[NODE_NAME, GRAPH_STATE]):
    name: NODE_NAME
    function: Callable[[GRAPH_STATE], GRAPH_STATE]


@dataclass(frozen=True)
class GraphEdge(Generic[NODE_NAME, GRAPH_STATE]):
    source_node: NODE_NAME
    target_node: Optional[NODE_NAME] = None
    route_function: Optional[Callable[[GRAPH_STATE], NODE_NAME]] = None
    route_map: Optional[dict[NODE_NAME, NODE_NAME]] = None
