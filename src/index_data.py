import logging
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


def pre_processing_data(source: list[str]):
    """
    Pre-processes the data from the source
    Args:
        source (): List of website links with data to index

    Returns:

    """
    docs = [WebBaseLoader(url).load() for url in source]
    docs_list = [item for sublist in docs for item in sublist]
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )
    doc_splits = text_splitter.split_documents(docs_list)
    logger.info("Data pre-processed successfully")
    return doc_splits


def index_data(doc_splits: list[Document]):
    """
    Indexes the data from the source into vector store

    Args:


    Returns:

    """
    logger.info(f"Indexing data")
    vector_store = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embeddings=GPT4AllEmbeddings(),
    )
    retriever = vector_store.as_retriever()
    logger.info("Data indexed successfully")
    return retriever
