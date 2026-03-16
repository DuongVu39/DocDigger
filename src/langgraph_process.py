from typing_extensions import TypedDict
import logging
import os

import yaml
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from create_agents import (
    initiate_chat_ollama,
    create_retrieval_grader_agent,
    create_generate_agent,
    create_hallucination_grader_agent,
    create_answer_grader_agent,
    create_question_router_agent,
)
from data_manipulation import load_vector_store

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """
    GraphState represents the state of a graph.

    Attributes:
        question: Question asked the LLM
        generation: Generated answer from the LLM
        web_search: Boolean indicating if the answer is needed through web search
        documents: list of documents that are used to generate the answer
    """

    question: str
    generation: str
    web_search: str
    documents: list[Document]


def _load_retriever():
    """
    Load the persisted HOA vector store as a retriever, if available.
    """
    config_path = os.getenv("HOA_INDEX_CONFIG", "conf/local/hoa_index.yml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        store_name = cfg["vector_store"]["store_name"]
        index_name = cfg["vector_store"]["index_name"]
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Could not load HOA index config from %s: %s", config_path, exc)
        return None

    try:
        vector_store = load_vector_store(store_name, index_name)
        logger.info("Loaded HOA retriever from %s", config_path)
        return vector_store.as_retriever()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("Failed to load HOA vector store: %s", exc)
        return None


RETRIEVER = _load_retriever()


def retrieve(state):
    """
    Retrieve the answer from the vector store.

    Args:
        state: GraphState object

    Returns:
        Updated state containing retrieved `documents`.
    """
    logger.info("Retrieving answer from the vector store")
    question = state["question"]

    if RETRIEVER is None:
        logger.warning("No retriever is available; returning empty document list.")
        documents: list[Document] = []
    else:
        documents = RETRIEVER.get_relevant_documents(question)

    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate an answer to the question using RAG on retrieved documents.
    Args:
        state: GraphState object
        rag_chain: RAG chain object (prompt | llm | JsonOutputParser)

    Returns:

    """
    base_llm = initiate_chat_ollama()
    generate_agent = create_generate_agent(base_llm)

    logger.info("Generating answer")
    question = state["question"]
    documents = state["documents"]

    # Generate answer using RAG
    generation = generate_agent.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines the quality of the documents for the question if it's relevant to the question.
    If irrelevant, flag the question for web search.

    Args:
        state ():

    Returns:

    """
    logger.info("Checking the relevance of the retrieved documents to the question")
    question = state["question"]
    documents = state["documents"]

    # initiate grader agent
    base_llm = initiate_chat_ollama()
    retrieval_grader = create_retrieval_grader_agent(base_llm)

    # Score each retrieved document
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "documents": doc.page_content}
        )
        grade = score["score"]  # Score the document

        # Check relevance == yes or no
        if grade.lower() == "yes":
            logger.info("GRADE: Document Relevant")
            filtered_docs.append(doc)
        else:
            logger.info("GRADE: Document Irrelevant")
            web_search = "Yes"
            continue

    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Perform a web search to retrieve the answer to the question.

    Args:
        state: GraphState object

    Returns:

    """
    web_search_engine = TavilySearchResults(k=3)
    logger.info("Performing web search")
    question = state["question"]
    documents = state["documents"]

    # Perform web search
    docs = web_search_engine.invoke({"query": question})
    web_results = "\n".join([doc["content"] for doc in docs])
    web_results = Document(page_content=web_results)

    if documents is None:
        documents = [web_results]
    else:
        documents.append(web_results)

    return {"documents": documents, "question": question}


def route_question(state):
    """
    Route the question to the appropriate function based on the state of the graph.

    Args:
        state: GraphState object
        question_router: QuestionRouter object

    Returns:

    """
    logger.info("Routing the question")
    question = state["question"]

    base_llm = initiate_chat_ollama()
    question_router = create_question_router_agent(base_llm)
    source = question_router.invoke({"question": question})

    datasource = source.get("datasource")
    if datasource == "web_search":
        logger.info("Routing to web search")
        return "websearch"

    if datasource == "vectorstore":
        logger.info("Routing to retrieve from vector store")
        return "vectorstore"

    logger.info("Unknown datasource '%s', defaulting to vectorstore", datasource)
    return "vectorstore"


def decide_to_generate(state):
    """
    Determine if the answer should be generated or web search.

    Args:
        state: GraphState object

    Returns:

    """
    logger.info("Assess the graded documents")
    web_search = state["web_search"]

    if web_search == "Yes":
        logger.info("DECISION: Web search \n All documents are irrelevant to the question")
        return "websearch"
    else:
        logger.info("DECISION: Generate answer \n Relevant documents found")
        return "generate"


def check_hallucinating(state):
    """
    Check if the LLM is hallucinating (if the generation is not relevant to the question).

    Args:
        state:
        hallucination_checker:

    Returns:

    """
    logger.info("Checking for hallucination and usefulness")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    base_llm = initiate_chat_ollama()
    hallucination_checker = create_hallucination_grader_agent(base_llm)
    answer_grader = create_answer_grader_agent(base_llm)

    # 1) Check grounding
    score = hallucination_checker.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"].lower()

    if grade != "yes":
        logger.info(
            "HALLUCINATION: Generation is not grounded in the documents, falling back."
        )
        return "unsupported"

    logger.info("GROUNDED IN DOCUMENT: Generation is based on the retrieved documents")

    # 2) Check usefulness
    logger.info("Checking if the generation is useful for the question")
    score = answer_grader.invoke({"question": question, "generation": generation})
    grade = score["score"].lower()

    if grade == "yes":
        logger.info("Generation is grounded and useful")
        return "supported_and_useful"

    logger.info("Generation is grounded but not useful enough")
    return "supported_but_not_useful"

