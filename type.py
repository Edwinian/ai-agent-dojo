from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar

from constants import ModelName
from tools import ToolName

NODE_NAME = TypeVar("NODE")
GRAPH_STATE = TypeVar("GRAPH_STATE")


@dataclass(frozen=True)
class SubAgent:
    """Configuration for a specialized sub-agent."""

    name: str
    description: str
    prompt: str
    model_name: ModelName = ModelName.DEEPSEEK_V4_FLASH
    tool_names: list[ToolName] | None = None


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
