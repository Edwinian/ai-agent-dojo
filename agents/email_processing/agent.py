from constants import ModelName
from agents.base_agent import BaseAgent

from .state import EmailState
from .tools import tools


class EmailProcessingAgent(BaseAgent[EmailState]):
    def __init__(self, model_name: ModelName = ModelName.DEEPSEEK_V4_PRO) -> None:
        super().__init__(
            model_name=model_name,
            state=EmailState,
            tools=tools,
        )


__all__ = ["EmailProcessingAgent"]
