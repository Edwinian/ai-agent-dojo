from langchain_core.messages import HumanMessage

from .graph import DocumentAnalysisGraph


def main() -> None:
    compiled_graph = DocumentAnalysisGraph().compiled_graph
    messages = [HumanMessage(content="Divide 6790 by 5")]
    messages = compiled_graph.invoke({"messages": messages, "input_file": None})

    # Show the messages
    for m in messages["messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
