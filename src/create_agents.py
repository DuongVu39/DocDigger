from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate


def initiate_chat_ollama(local_llm="llama3"):
    """
    Initiates the ChatOllama object

    Returns:
        ChatOllama: ChatOllama object
    """
    chat_ollama = ChatOllama(model=local_llm, format="json", temperature=0)
    return chat_ollama


def create_retrieval_grader_agent(base_llm: ChatOllama):
    """
    Creates the retrieval grader agent

    Args:
        base_llm (): ChatOllama object

    Returns:
        ChatOllama: ChatOllama object
    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
        of a retrieved document to a user question. If the document contains keywords related to the user question, 
        grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide the binary score as a JSON with a single key 'score' and no premable or explanation.
         <|eot_id|><|start_header_id|>user<|end_header_id|>
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    # pipe prompt into base llm and then parse the output
    retrieval_grader = prompt | base_llm | JsonOutputParser()
    return retrieval_grader


def create_generate_agent(base_llm: ChatOllama):
    """
    Creates the generate agent
    Args:
        base_llm ():

    Returns:

    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "document"],
    )

    # pipe prompt into base llm and then parse the output
    generate = prompt | base_llm | JsonOutputParser()
    return generate


def create_hallucination_grader_agent(base_llm):
    """
    Creates the hallucination grader agent

    Args:
        base_llm ():

    Returns:

    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
        input_variables=["document", "generation"],
    )

    # pipe prompt into base llm and then parse the output
    hallucination_grader = prompt | base_llm | JsonOutputParser()
    return hallucination_grader


def create_answer_grader_agent(base_llm):
    """
    Creates the answer grader agent

    Args:
        base_llm ():

    Returns:

    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
        answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
        useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
        \n ------- \n
        {generation} 
        \n ------- \n
        Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["generation", "question"],
    )

    # pipe prompt into base llm and then parse the output
    answer_grader = prompt | base_llm | JsonOutputParser()

    return answer_grader


def create_question_router_agent(base_llm: ChatOllama):
    """
    Creates the question router agent
    Current implementation include three topics to route to vectorstore:
        - LLM agents,
        - Prompt engineering, and
        - Adversarial attacks
    Otherwise it will route to web search

    Args:
        base_llm ():

    Returns:

    """
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
    user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
    prompt engineering, and adversarial attacks. You do not need to be stringent with the keywords 
    in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' 
    or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and 
    no premable or explanation. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
        input_variables=["question"],
    )

    # pipe prompt into base llm and then parse the output
    question_router = prompt | base_llm | JsonOutputParser()

    return question_router
