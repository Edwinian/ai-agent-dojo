from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable


class LlmService(ABC):
    def __init__(
        self,
        model_name: Any,
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.tools = tools

    @abstractmethod
    def init_model(self) -> Any:
        """Initialize and return the underlying model client."""

    @abstractmethod
    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        """Invoke the model with implementation-specific arguments."""
