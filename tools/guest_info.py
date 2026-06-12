from langchain.tools import tool

from utils import load_docs, retrieve_info


@tool("guest_info_retriever")
def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation to other guests."""
    docs = load_docs(document_name="guest")
    return retrieve_info(query=query, docs=docs)


guest_info_tool = extract_text

__all__ = ["extract_text", "guest_info_tool"]
