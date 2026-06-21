from enum import Enum

from .divide import divide
from .extract_text_from_image import extract_text_from_image
from .get_hub_stats import get_hub_stats
from .get_weather_info import get_weather_info
from .guest_info import extract_text
from .to_do import read_todos, write_todos
from .web_search import web_search
from .write_file import write_file


class ToolName(str, Enum):
    DIVIDE = "divide"
    EXTRACT_TEXT = "extract_text"
    EXTRACT_TEXT_FROM_IMAGE = "extract_text_from_image"
    GET_HUB_STATS = "get_hub_stats"
    GET_WEATHER_INFO = "get_weather_info"
    READ_TODOS = "read_todos"
    WEB_SEARCH = "web_search"
    WRITE_FILE = "write_file"
    WRITE_TODOS = "write_todos"


tools_by_name = {
    ToolName.DIVIDE: divide,
    ToolName.EXTRACT_TEXT: extract_text,
    ToolName.EXTRACT_TEXT_FROM_IMAGE: extract_text_from_image,
    ToolName.GET_HUB_STATS: get_hub_stats,
    ToolName.GET_WEATHER_INFO: get_weather_info,
    ToolName.READ_TODOS: read_todos,
    ToolName.WEB_SEARCH: web_search,
    ToolName.WRITE_FILE: write_file,
    ToolName.WRITE_TODOS: write_todos,
}

__all__ = ["ToolName", "tools_by_name"]
