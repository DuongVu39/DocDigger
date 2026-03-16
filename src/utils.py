import logging
from typing import List

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def text_splitter(text: str) -> List[str]:
    """
    Split raw text into smaller, semantically coherent chunks.

    Uses a token-aware recursive splitter with a small overlap so that
    related clauses stay together while keeping chunks retrieval‑friendly.
    """
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=100,
    )
    docs = splitter.create_documents([text])
    chunks = [d.page_content for d in docs]
    logger.debug("Split text into %d chunks", len(chunks))
    return chunks


def create_text_embedding():
    """
    Return the embedding model used across the project.

    Uses an Ollama embedding model; override via DOC_DIGGER_EMBEDDING_MODEL.
    """
    import os

    # e.g. "nomic-embed-text" or any embedding-capable Ollama model you have pulled
    model_name = os.getenv("DOC_DIGGER_EMBEDDING_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model_name)
