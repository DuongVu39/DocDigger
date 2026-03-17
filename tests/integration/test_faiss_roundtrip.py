from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data_manipulation as dm


@pytest.mark.integration
def test_create_and_load_vector_store_roundtrip(monkeypatch, tmp_path: Path):
    # Arrange a tiny dataframe with pre-chunked text
    df = pd.DataFrame(
        [
            {
                "Title": "bylaws_2022.pdf",
                "path": str(tmp_path / "bylaws_2022.pdf"),
                "doc_type": "bylaw",
                "year": "2022",
                "chunked_text": ["pets allowed", "common areas leash"],
            }
        ]
    )

    store_name = str(tmp_path / "hoa_store")
    index_name = str(tmp_path / "hoa.index")

    # Deterministic, offline embedding stub
    from conftest import DeterministicEmbeddings

    monkeypatch.setattr(dm, "create_text_embedding", lambda: DeterministicEmbeddings(dim=8))

    # Act: create store and reload
    dm.create_vector_store(df, store_name=store_name, index_name=index_name)
    assert (tmp_path / "hoa_store.pkl").exists()
    assert (tmp_path / "hoa.index").exists()

    store = dm.load_vector_store(store_name=store_name, index_name=index_name)
    retriever = store.as_retriever()
    docs = retriever.invoke("pets")

    # Assert: retrieval returns something and includes metadata we provided
    assert isinstance(docs, list)
    assert len(docs) >= 1
    assert docs[0].metadata.get("source") == "bylaws_2022.pdf"


@pytest.mark.integration
def test_create_vector_store_reuses_existing_store(monkeypatch, tmp_path: Path):
    df = pd.DataFrame(
        [
            {
                "Title": "minutes_2023_agm.txt",
                "path": str(tmp_path / "minutes_2023_agm.txt"),
                "doc_type": "minutes",
                "year": "2023",
                "chunked_text": ["fees reviewed annually"],
            }
        ]
    )
    store_name = str(tmp_path / "hoa_store")
    index_name = str(tmp_path / "hoa.index")

    from conftest import DeterministicEmbeddings

    monkeypatch.setattr(dm, "create_text_embedding", lambda: DeterministicEmbeddings(dim=8))

    dm.create_vector_store(df, store_name=store_name, index_name=index_name)

    # If the reuse path is taken, FAISS.from_texts should not be called.
    called = {"from_texts": 0}
    from langchain_community.vectorstores import FAISS
    real_from_texts = FAISS.from_texts

    def counting_from_texts(*args, **kwargs):
        called["from_texts"] += 1
        return real_from_texts(*args, **kwargs)

    monkeypatch.setattr(FAISS, "from_texts", counting_from_texts)

    dm.create_vector_store(df, store_name=store_name, index_name=index_name)
    assert called["from_texts"] == 0

