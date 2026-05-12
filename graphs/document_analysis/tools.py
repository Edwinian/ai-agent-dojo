from tools.get_hub_stats import get_hub_stats_tool as hub_stats_tool
from tools.get_weather_info import get_weather_info_tool as weather_info_tool
from tools.guest_info import guest_info_tool
from tools.web_search import web_search_tool

tools = [
    guest_info_tool,
    web_search_tool,
    weather_info_tool,
    hub_stats_tool,
]
