import time

import requests
from langchain_core.messages import AnyMessage
from langchain_core.messages import SystemMessage

from llm_service import LLMService

from .state import DocumentState as AgentState
from .tools import tools

llm_with_tools = LLMService[AgentState](tools=tools).chat_llm
MAX_LLM_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2


def invoke_with_retry(messages: list[AnyMessage]) -> object:
    for attempt in range(1, MAX_LLM_RETRIES + 1):
        try:
            return llm_with_tools.invoke(messages)
        except (
            requests.exceptions.ReadTimeout,
            requests.exceptions.ConnectionError,
            requests.exceptions.ProxyError,
        ):
            if attempt == MAX_LLM_RETRIES:
                raise
            # Basic exponential backoff for transient Hugging Face router latency.
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))


def assistant(state: AgentState):
    # System message
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image = state["input_file"]
    sys_msg = SystemMessage(
        content=(
            "You are a helpful butler named Alfred that serves Edwin. "
            "You can analyse documents and run computations with provided tools:\n"
            f"{textual_description_of_tool} \n "
            "You have access to some optional images. "
            f"Currently the loaded image is: {image}"
        )
    )

    return {
        "messages": [invoke_with_retry([sys_msg] + state["messages"])],
        "input_file": state["input_file"],
    }
