import streamlit as st
from pprint import pprint

from langgraph.graph import StateGraph, END

from langgraph_process import (
    GraphState,
    web_search,
    retrieve,
    grade_documents,
    generate,
    route_question,
    decide_to_generate,
    check_hallucinating,
)


def build_app():
    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_document", grade_documents)
    workflow.add_node("generate", generate)

    # Edges
    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "retrieve",
        },
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
            "supported_and_useful": END,
            "supported_but_not_useful": "websearch",
            "unsupported": "websearch",
        },
    )

    return workflow.compile()


def render_documents(docs):
    if not docs:
        st.info("No HOA documents were retrieved for this question.")
        return

    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("path") or f"Document {i}"
        doc_type = meta.get("doc_type", "unknown")
        year = meta.get("year")

        with st.expander(f"{i}. {source} ({doc_type}{f', {year}' if year else ''})"):
            st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))
            st.json(meta)


def main():
    st.set_page_config(page_title="DocDigger HOA RAG", layout="wide")
    st.title("DocDigger – HOA / Strata Assistant")

    if "graph_app" not in st.session_state:
        st.session_state["graph_app"] = build_app()

    with st.sidebar:
        st.header("Configuration")
        st.markdown(
            """
            - Ensure your HOA documents are indexed using `src/index_hoa_docs.py`.
            - The app reads vector store locations from `conf/local/hoa_index.yml`
              (or `HOA_INDEX_CONFIG` environment variable).
            """
        )

    user_question = st.text_input("Ask a question about your HOA documents:")

    if user_question:
        app = st.session_state["graph_app"]
        with st.spinner("Thinking..."):
            last_state = None
            debug_events = []
            for event in app.stream({"question": user_question}):
                debug_events.append(event)
                for _, state in event.items():
                    last_state = state

        if last_state is None:
            st.error("The workflow did not return any state.")
            return

        generation = last_state.get("generation")
        documents = last_state.get("documents", [])

        st.subheader("Answer")
        if isinstance(generation, dict) and "answer" in generation:
            st.markdown(generation["answer"])
        else:
            st.markdown(str(generation))

        st.subheader("Document context")
        render_documents(documents)

        with st.expander("Advanced – routing & grading details"):
            st.write("Raw node outputs from the LangGraph run:")
            for event in debug_events:
                pprint(event, stream=st.text)


if __name__ == "__main__":
    main()

