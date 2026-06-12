from langchain.tools import tool

SEARCH_RESULT = (
    "Web search results would appear here. Replace SEARCH_RESULT with a live search "
    "integration (e.g. DuckDuckGo or Tavily) when ready."
)


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the web for information on a specific topic.

    This tool performs web searches and returns relevant results
    for the given query. Use this when you need to gather information from
    the internet about any topic.

    Args:
        query: The search query string. Be specific and clear about what
               information you're looking for.

    Returns:
        Search results from search engine.

    Example:
        web_search("machine learning applications in healthcare")
    """
    del query
    return SEARCH_RESULT


web_search_tool = web_search

__all__ = ["SEARCH_RESULT", "web_search", "web_search_tool"]
