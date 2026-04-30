from constants import NodeName
from email_processing_state import EmailState


def route_email(state: EmailState) -> NodeName:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return NodeName.HANDLE_SPAM
    return NodeName.DRAFT_RESPONSE
