from pathlib import Path

from langchain_core.messages import HumanMessage

from .graph import DocumentAnalysisGraph


def main() -> None:
    compiled_graph = DocumentAnalysisGraph()
    invoice_path = Path(__file__).with_name("invoice.png")

    # Simple Calculations
    # messages = [HumanMessage(content="Divide 6790 by 5")]
    # messages = compiled_graph.invoke({"messages": messages, "input_file": None})

    # # Show the messages
    # for m in messages["messages"]:
    #     m.pretty_print()

    # Document Analysis
    messages = [HumanMessage(content="What content is in the invoice?")]
    result = compiled_graph.invoke(
        {"messages": messages, "input_file": str(invoice_path)}
    )
    for message in result["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
