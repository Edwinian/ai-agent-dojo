from typing import Any, Dict, Optional

from deep_agent_state import DeepAgentState


class EmailState(DeepAgentState):
    email: Dict[str, Any]
    email_category: Optional[str]
    spam_reason: Optional[str]
    is_spam: Optional[bool]
    email_draft: Optional[str]


__all__ = ["EmailState"]
