from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
from langchain_core.documents import Document


def reload_langgraph_process(monkeypatch, *, hoa_index_config: str | None = None):
    if hoa_index_config is None:
        monkeypatch.delenv("HOA_INDEX_CONFIG", raising=False)
    else:
        monkeypatch.setenv("HOA_INDEX_CONFIG", hoa_index_config)
    import langgraph_process as lgp

    return importlib.reload(lgp)


def test_load_retriever_returns_none_when_config_missing(monkeypatch, tmp_path):
    # Point to a config file that does not exist so _load_retriever returns None.
    missing_cfg = str(tmp_path / "missing.yml")
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=missing_cfg)
    assert lgp._load_retriever() is None


def test_retrieve_returns_empty_docs_when_no_retriever(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    lgp.RETRIEVER = None
    out = lgp.retrieve({"question": "q"})
    assert out["question"] == "q"
    assert out["documents"] == []


def test_retrieve_uses_retriever_when_present(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    expected = [Document(page_content="x")]

    class FakeRetriever:
        def get_relevant_documents(self, question: str):
            assert question == "q"
            return expected

    lgp.RETRIEVER = FakeRetriever()
    out = lgp.retrieve({"question": "q"})
    assert out["documents"] == expected


def test_grade_documents_filters_and_sets_web_search(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    docs = [Document(page_content="relevant"), Document(page_content="irrelevant")]

    class FakeGrader:
        def invoke(self, payload):
            if payload["documents"] == "relevant":
                return {"score": "yes"}
            return {"score": "no"}

    monkeypatch.setattr(lgp, "initiate_chat_ollama", lambda: object())
    monkeypatch.setattr(lgp, "create_retrieval_grader_agent", lambda _llm: FakeGrader())

    out = lgp.grade_documents({"question": "q", "documents": docs})
    assert [d.page_content for d in out["documents"]] == ["relevant"]
    assert out["web_search"] == "Yes"


def test_route_question_routes_websearch_and_vectorstore(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    monkeypatch.setattr(lgp, "initiate_chat_ollama", lambda: object())

    class FakeRouter:
        def __init__(self, datasource: str | None):
            self._datasource = datasource

        def invoke(self, _payload):
            return {"datasource": self._datasource}

    monkeypatch.setattr(lgp, "create_question_router_agent", lambda _llm: FakeRouter("web_search"))
    assert lgp.route_question({"question": "q"}) == "websearch"

    monkeypatch.setattr(lgp, "create_question_router_agent", lambda _llm: FakeRouter("vectorstore"))
    assert lgp.route_question({"question": "q"}) == "vectorstore"

    monkeypatch.setattr(lgp, "create_question_router_agent", lambda _llm: FakeRouter("unknown"))
    assert lgp.route_question({"question": "q"}) == "vectorstore"


def test_decide_to_generate_branches(tmp_path, monkeypatch):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    assert lgp.decide_to_generate({"web_search": "Yes"}) == "websearch"
    assert lgp.decide_to_generate({"web_search": "No"}) == "generate"


def test_web_search_appends_results(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))

    class FakeTavily:
        def __init__(self, k: int):
            assert k == 3

        def invoke(self, payload):
            assert payload["query"] == "q"
            return [{"content": "r1"}, {"content": "r2"}]

    monkeypatch.setattr(lgp, "TavilySearchResults", FakeTavily)
    out = lgp.web_search({"question": "q", "documents": []})
    assert out["question"] == "q"
    assert len(out["documents"]) == 1
    assert "r1" in out["documents"][0].page_content


@pytest.mark.parametrize(
    ("grounded", "useful", "expected"),
    [
        ("no", "no", "unsupported"),
        ("yes", "no", "supported_but_not_useful"),
        ("yes", "yes", "supported_and_useful"),
    ],
)
def test_check_hallucinating_routes(monkeypatch, tmp_path, grounded, useful, expected):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    monkeypatch.setattr(lgp, "initiate_chat_ollama", lambda: object())

    class FakeHallucinationGrader:
        def invoke(self, _payload):
            return {"score": grounded}

    class FakeAnswerGrader:
        def invoke(self, _payload):
            return {"score": useful}

    monkeypatch.setattr(lgp, "create_hallucination_grader_agent", lambda _llm: FakeHallucinationGrader())
    monkeypatch.setattr(lgp, "create_answer_grader_agent", lambda _llm: FakeAnswerGrader())

    out = lgp.check_hallucinating(
        {
            "question": "q",
            "documents": [Document(page_content="ctx")],
            "generation": {"answer": "a"},
        }
    )
    assert out == expected


def test_generate_invokes_generate_agent(monkeypatch, tmp_path):
    lgp = reload_langgraph_process(monkeypatch, hoa_index_config=str(tmp_path / "missing.yml"))
    monkeypatch.setattr(lgp, "initiate_chat_ollama", lambda: object())

    class FakeGenerateAgent:
        def invoke(self, payload):
            assert payload["question"] == "q"
            assert payload["context"][0].page_content == "ctx"
            return {"answer": "ok"}

    monkeypatch.setattr(lgp, "create_generate_agent", lambda _llm: FakeGenerateAgent())
    out = lgp.generate({"question": "q", "documents": [Document(page_content="ctx")]})
    assert out["generation"] == {"answer": "ok"}

