from tools import ToolName
from type import SubAgent
from utils import get_today_str

RESEARCHER_INSTRUCTIONS = """You are a research assistant conducting research on the user's input topic. For context, today's date is {date}.

"""


research_sub_agent = SubAgent(
    name="research-agent",
    description="Delegate research to the sub-agent researcher. Only give this researcher one topic at a time.",
    prompt=RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
    tool_names=[ToolName.WEB_SEARCH],
)

__all__ = ["RESEARCHER_INSTRUCTIONS", "research_sub_agent"]
