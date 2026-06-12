from typing import Optional

from typing_extensions import NotRequired

from deep_agent_state import DeepAgentState


class DocumentState(DeepAgentState):
    # The document provided
    input_file: NotRequired[Optional[str]]  # Contains file path (PDF/PNG)


__all__ = ["DocumentState"]
