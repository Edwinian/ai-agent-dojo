from langchain_core.messages import SystemMessage

from llm_service import LLMService

from .state import DocumentState as AgentState
from .tools import tools

llm_with_tools = LLMService[AgentState](tools=tools).chat_llm


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
            "You are a helpful butler named Alfred that serves Mr. Wayne and Batman. "
            "You can analyse documents and run computations with provided tools:\n"
            f"{textual_description_of_tool} \n "
            "You have access to some optional images. "
            f"Currently the loaded image is: {image}"
        )
    )

    return {
        "messages": [llm_with_tools.invoke([sys_msg] + state["messages"])],
        "input_file": state["input_file"],
    }
