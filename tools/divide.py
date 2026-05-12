from langchain_core.tools import Tool


def divide(a: int, b: int) -> float:
    """Divide a by b."""
    return a / b


divide_tool = Tool(
    name="divide",
    func=divide,
    description="Divide the integer a by the integer b and return the quotient as a float.",
)

__all__ = ["divide", "divide_tool"]
