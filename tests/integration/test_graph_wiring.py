from __future__ import annotations

import importlib

import pytest


@pytest.mark.integration
def test_compiled_graph_streams_with_stubbed_nodes(monkeypatch, sample_documents):
    # Import the module and monkeypatch its imported node fns (app.py imports them into its namespace)
    import app as app_mod

    def route_question(_state):
        return "vectorstore"

    def retrieve(state):
        return {"question": state["question"], "documents": sample_documents}

    def grade_documents(state):
        return {"question": state["question"], "documents": state["documents"], "web_search": "No"}

    def decide_to_generate(_state):
        return "generate"

    def generate(state):
        return {"question": state["question"], "documents": state["documents"], "generation": {"answer": "ok"}}

    def check_hallucinating(_state):
        return "supported_and_useful"

    monkeypatch.setattr(app_mod, "route_question", route_question)
    monkeypatch.setattr(app_mod, "retrieve", retrieve)
    monkeypatch.setattr(app_mod, "grade_documents", grade_documents)
    monkeypatch.setattr(app_mod, "decide_to_generate", decide_to_generate)
    monkeypatch.setattr(app_mod, "generate", generate)
    monkeypatch.setattr(app_mod, "check_hallucinating", check_hallucinating)

    graph = app_mod.build_app()

    last_state = None
    for event in graph.stream({"question": "q"}):
        for _node, state in event.items():
            last_state = state

    assert last_state is not None
    assert last_state["generation"] == {"answer": "ok"}

