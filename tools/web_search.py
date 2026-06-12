from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun


@tool
def web_search(query: str) -> str:
    """Run a DuckDuckGo web search and return the textual summary of results."""
    search = DuckDuckGoSearchRun()
    results = search.invoke(query)
    return str(results)


web_search_tool = web_search

__all__ = ["web_search", "web_search_tool"]
