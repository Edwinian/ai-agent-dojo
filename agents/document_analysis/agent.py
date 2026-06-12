from constants import ModelName
from agents.base_agent import BaseAgent

from .state import DocumentState
from .tools import tools


class DocumentAnalysisAgent(BaseAgent[DocumentState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        super().__init__(
            model_name=model_name,
            state=DocumentState,
            tools=tools,
        )


__all__ = ["DocumentAnalysisAgent"]
