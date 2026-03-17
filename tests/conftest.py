import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@dataclass
class DeterministicEmbeddings(Embeddings):
    """Offline embedding stub compatible with LangChain vector stores."""

    dim: int = 8

    def _embed(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        ints = np.frombuffer(digest[: self.dim], dtype=np.uint8).astype(np.float32)
        vec = (ints / 255.0).tolist()
        return vec

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def sample_documents() -> list[Document]:
    return [
        Document(
            page_content="Pets are allowed but must be leashed in common areas.",
            metadata={"source": "bylaws_2022.pdf", "doc_type": "bylaw", "year": "2022"},
        ),
        Document(
            page_content="The strata fee schedule is reviewed annually.",
            metadata={"source": "minutes_2023_agm.pdf", "doc_type": "minutes", "year": "2023"},
        ),
    ]

