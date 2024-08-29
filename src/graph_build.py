from langgraph.graph import END, StateGraph
from langgraph.checkpoint import MemorySaver
from langgraph_process import *
from pprint import pprint


def main():
    workflow = StateGraph(GraphState)

    # create all the node
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_document", grade_documents)
    workflow.add_node("generate", generate)

    # build graph
    workflow.set_conditional_entry_point(
        route_question, {
            "websearch": "websearch",
            "vectorstore": "retrieve"
        }
    )
    workflow.add_edge("retrieve", "grade_document")
    workflow.add_conditional_edges(
        "grade_document",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        check_hallucinating,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    app = workflow.compile()

    # test without interface
    input = {"question": "What are the types of agent memory?"}

    for output in app.stream(input):
        for key, value in output.items():
            pprint(f"Finished processing {key}")

    pprint(value["generation"])


if __name__ == "__main__":
    main()
