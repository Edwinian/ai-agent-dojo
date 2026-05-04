from .nodes import EmailNode
from .state import EmailState


def route_email(state: EmailState) -> EmailNode:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return EmailNode.HANDLE_SPAM
    return EmailNode.DRAFT_RESPONSE
