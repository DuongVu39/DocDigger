# DocDigger
Chatbot using rag and llama 3 to dig through strata doc for my questions

## General architecture

![](img/architecture.png)

### Components
- Vector Store: A database of all the documents in the strata doc
- Retriever Agent: retrieves the most relevant documents from the vector store
- Document Grader Agent: ranks the documents based on relevance to the questions
- Generate Answer Agent: generates the answer to the question based on the retrieved documents
- Web Search Agent: if the Document Grader Agent is not confident in the answer, it will use the Web Search Agent to find the answer
- Hallucination Checker: checks if the answer is hallucinated (not relevant to the question)
- Chatbot: the interface for the user to ask questions

## Current process:
- Set up all rag agents
- Tested locally

## TODO
- Switch out data from the internet with scannded documents from the strata doc
- Set up data pipeline to process the documents
- Save out vector store for all documents
- Need to setup Streamlit for interface