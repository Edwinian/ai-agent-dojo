from langchain_core.tools import Tool

from utils import load_docs, retrieve_info

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    docs = load_docs(document_name="guest")
    return retrieve_info(query=query, docs=docs)


guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.",
)
