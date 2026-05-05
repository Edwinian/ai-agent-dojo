from langchain_core.messages import HumanMessage

from .graph import DocumentAnalysisGraph


def main() -> None:
    compiled_graph = DocumentAnalysisGraph()

    # # Simple Calculations
    # messages = [HumanMessage(content="Divide 6790 by 5")]
    # messages = compiled_graph.invoke({"messages": messages, "input_file": None})

    # # Show the messages
    # for m in messages["messages"]:
    #     m.pretty_print()

    # Document Analysis
    messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
    result = compiled_graph.invoke({"messages": messages})

    print("🎩 Edwin's Response:")
    print(result['messages'][-1].content)

    # for message in result["messages"]:
    #     message.pretty_print()


if __name__ == "__main__":
    main()
