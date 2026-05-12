from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool


def web_search(query: str) -> str:
    """Run a DuckDuckGo web search and return the textual result."""
    search_tool = DuckDuckGoSearchRun()
    results = search_tool.invoke(query)
    return str(results)


web_search_tool = Tool(
    name="web_search",
    func=web_search,
    description="Run a DuckDuckGo web search and return the textual summary of results.",
)

__all__ = ["web_search", "web_search_tool"]
