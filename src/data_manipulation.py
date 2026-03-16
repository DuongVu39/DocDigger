import itertools
import logging
import os
import pickle
from pathlib import Path
from typing import List

import pandas as pd
from langchain.vectorstores import faiss
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS

from utils import text_splitter, create_text_embedding


logger = logging.getLogger(__name__)


ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


def _infer_doc_type(path: Path) -> str:
    name = path.name.lower()
    if "bylaw" in name or "by-law" in name:
        return "bylaw"
    if "minute" in name or "agm" in name or "sgm" in name:
        return "minutes"
    if "notice" in name or "rule" in name:
        return "notice"
    if "financial" in name or "fee" in name or "budget" in name:
        return "financial"
    return "other"


def _infer_year(path: Path) -> str | None:
    digits = [token for token in path.stem.split("_") if token.isdigit()]
    for token in digits:
        if len(token) == 4:
            return token
    return None


def get_data_list(base_folder: str) -> pd.DataFrame:
    """
    Walk a folder of HOA/strata documents and return basic metadata.

    The resulting DataFrame includes at least:
        - Title: human‑readable name
        - path: absolute file path
        - doc_type: coarse document category
        - year: year inferred from filename where possible
    """
    root = Path(base_folder)
    if not root.exists():
        raise FileNotFoundError(f"HOA docs folder not found: {base_folder}")

    records: List[dict] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        records.append(
            {
                "Title": path.name,
                "path": str(path.resolve()),
                "doc_type": _infer_doc_type(path),
                "year": _infer_year(path),
            }
        )

    df = pd.DataFrame(records)
    logger.info("Discovered %d HOA documents under %s", len(df), base_folder)
    return df


def clean_document(path: str) -> str:
    """
    Load and lightly normalise a single document.

    Currently supports PDFs and plain‑text/Markdown files. Scanned PDFs that
    require OCR will need an external OCR pass before ingestion.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)
    elif suffix in {".txt", ".md"}:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        text = "\n".join(d.page_content for d in docs)
    else:
        raise ValueError(f"Unsupported file type for HOA ingestion: {suffix}")

    # Normalise whitespace and strip obvious noise
    text = " ".join(text.split())
    return text


def create_text_splitter(book_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the project‑wide splitter to a DataFrame containing a `Content` column.

    Produces a `chunked_text` column with lists of chunk strings.
    """
    book_df["chunked_text"] = book_df["Content"].apply(text_splitter)
    return book_df


def create_vector_store(df: pd.DataFrame, store_name: str, index_name: str) -> None:
    """
    Build (or reload) a FAISS vector store and persist it to disk.

    Writes:
        - `{index_name}`: FAISS index
        - `{store_name}.pkl`: pickled FAISS store object without the index
    """
    metadatas: List[dict] = []
    for _, row in df.iterrows():
        base_meta = {
            "source": row.get("Title"),
            "path": row.get("path"),
            "doc_type": row.get("doc_type", "other"),
            "year": row.get("year"),
        }
        metadatas.extend([base_meta] * len(row["chunked_text"]))

    hf = create_text_embedding()

    if os.path.exists(f"{store_name}.pkl") and os.path.exists(index_name):
        logger.info("Existing vector store found. Reloading from disk.")
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store: FAISS = pickle.load(f)
        vector_store.index = faiss.read_index(index_name)
    else:
        logger.info("Creating new FAISS vector store.")
        chunked_text = list(itertools.chain.from_iterable(df["chunked_text"].tolist()))
        vector_store = FAISS.from_texts(
            chunked_text,
            embedding=hf,
            metadatas=metadatas,
        )

    # Persist index + store
    faiss.write_index(vector_store.index, index_name)
    vector_store.index = None
    with open(f"{store_name}.pkl", "wb") as f:
        pickle.dump(vector_store, f)
    logger.info(
        "Persisted FAISS index to %s and store to %s.pkl", index_name, store_name
    )


def load_vector_store(store_name: str, index_name: str) -> FAISS:
    """
    Load a persisted FAISS vector store from disk.
    """
    if not os.path.exists(f"{store_name}.pkl"):
        raise FileNotFoundError(f"Vector store pickle not found: {store_name}.pkl")
    if not os.path.exists(index_name):
        raise FileNotFoundError(f"FAISS index not found: {index_name}")

    with open(f"{store_name}.pkl", "rb") as f:
        vector_store: FAISS = pickle.load(f)
    vector_store.index = faiss.read_index(index_name)
    logger.info(
        "Loaded FAISS vector store from %s.pkl and %s", store_name, index_name
    )
    return vector_store

