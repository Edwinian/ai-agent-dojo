from huggingface_hub import list_models
from huggingface_hub.hf_api import ModelInfo
from langchain_core.tools import Tool


def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        models = list[ModelInfo](
            list_models(author=author, sort="downloads", direction=-1, limit=1)
        )

        if models:
            model = models[0]
            return (
                f"The most downloaded model by {author} is {model.id} "
                f"with {model.downloads:,} downloads."
            )
        return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"


get_hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description=(
        "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
    ),
)

__all__ = ["get_hub_stats", "get_hub_stats_tool"]
