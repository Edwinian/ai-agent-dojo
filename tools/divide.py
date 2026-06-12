from langchain.tools import tool


@tool
def divide(a: int, b: int) -> float:
    """Divide the integer a by the integer b and return the quotient as a float."""
    return a / b


divide_tool = divide

__all__ = ["divide", "divide_tool"]
