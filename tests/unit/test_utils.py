import types

import pytest

import utils as utils


def test_text_splitter_empty_returns_empty_list():
    assert utils.text_splitter("") == []
    assert utils.text_splitter(None) == []  # type: ignore[arg-type]


def test_text_splitter_splits_non_empty_text():
    text = "A" * 2500
    chunks = utils.text_splitter(text)
    assert isinstance(chunks, list)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) and c for c in chunks)


def test_create_text_embedding_respects_env_var(monkeypatch):
    created = {}

    def fake_ollama_embeddings(*, model: str):
        created["model"] = model
        return types.SimpleNamespace(model=model)

    monkeypatch.setenv("DOC_DIGGER_EMBEDDING_MODEL", "my-embed-model")
    monkeypatch.setattr(utils, "OllamaEmbeddings", fake_ollama_embeddings)

    emb = utils.create_text_embedding()
    assert created["model"] == "my-embed-model"
    assert getattr(emb, "model") == "my-embed-model"

