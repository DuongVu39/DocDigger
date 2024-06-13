from typing_extensions import TypedDict
import logging
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from create_agents import *

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
    documents: list[str]


def retrieve(state):
    """
    Retrieve the answer from the vector store.

    Args:
        state: GraphState object
        retriever: Vector store retriever

    Returns:
        str: The answer to the question
    """
    # initiate retriever agent
    base_llm = initiate_chat_ollama()
    retriever = create_retrieval_grader_agent(base_llm)

    # retrieving answer
    logger.info("Retrieving answer from the vector store")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate an answer to the question using RAG on retrieved documents.
    Args:
        state: GraphState object
        rag_chain: RAG chain object (prompt | llm | JsonOutputParser)

    Returns:

    """
    # initiate generate agent
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
    # initiate grader agent
    base_llm = initiate_chat_ollama()
    retrieval_grader = create_retrieval_grader_agent(base_llm)

    logger.info("Checking the relevance of the retrieved documents to the question")
    question = state["question"]
    documents = state["documents"]

    # Score each retrieved document
    filtered_docs = []
    web_search = "No"
    for doc in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
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
    # initiate grader agent
    base_llm = initiate_chat_ollama()
    question_router = create_question_router_agent(base_llm)

    logger.info("Routing the question")
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source["datasource"] == "websearch":
        logger.info("Routing to web search")
        return "websearch"
    else:
        logger.info("Routing to retrieve from vector store")
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
    # initiate grader agent
    base_llm = initiate_chat_ollama()
    hallucination_checker = create_hallucination_grader_agent(base_llm)

    logger.info("Checking for hallucination")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_checker.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    if grade.lower() == "yes":
        logger.info("GROUNDED IN DOCUMENT: Generation is based on the retrieved documents")
        # check if the generation is relevant to the question
        logger.info("Checking if the generation is relevant to the question")
        score = hallucination_checker.invoke(
            {"question": question, "generation": generation}
        )
        grade = score["score"]
        if grade.lower() == "yes":
            logger.info("NOT HALLUCINATING: Generation is relevant to the question")
            return "useful"
        else:
            logger.info("HALLUCINATION: Generation is not relevant to the question")
            return "not useful"
    else:
        logger.info("HALLUCINATION: Generation is not grounded in the documents, RE-TRY")
        return "not-supported"
