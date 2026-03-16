## DocDigger

Chatbot using Retrieval-Augmented Generation (RAG) and a local `llama3.1` model (via Ollama) to answer questions about strata / HOA documents.

The goal is to:

- **Ingest and index HOA documents** into a vector store
- **Route questions** either to the HOA corpus or to web search
- **Grade and filter retrieved documents**
- **Generate answers** grounded in the underlying docs
- **Detect hallucinations** and fall back to safer behaviour

This README describes the current components in `src/`, how they fit together, and how to run and extend the system.

---

## High-level architecture

The architecture follows a LangGraph-style workflow with multiple LLM “agents” that each focus on a specific sub-task.



The main pieces are:

- **Vector store**: persistent index of document chunks from HOA/strata docs or web pages
- **Retriever**: fetches potentially relevant chunks from the vector store
- **Document grader**: scores retrieved chunks for relevance to the question
- **Answer generator**: synthesises an answer from the filtered chunks
- **Web search agent**: backs off to web search when the internal corpus is insufficient
- **Hallucination checker / answer grader**: checks whether an answer is grounded and useful
- **LangGraph workflow**: orchestrates all these components into a single question-answering pipeline

---

## Code structure (`src/`)

### `create_agents.py`

Defines all LLM-based “agents”, built on top of `ChatOllama` and `PromptTemplate` with `JsonOutputParser`:

- `**initiate_chat_ollama(local_llm="llama3.1")`**  
Creates a `ChatOllama` instance configured to return JSON, with deterministic (`temperature=0`) behaviour.
- `**create_retrieval_grader_agent(base_llm)**`  
LLM grader that takes a `question` and `documents` and returns `{"score": "yes" | "no"}` indicating whether the document is relevant.
- `**create_generate_agent(base_llm)**`  
Question-answering agent that takes `question` and `context` (retrieved chunks) and returns an answer.  
Prompt instructs it to keep answers concise and to admit when it does not know.
- `**create_hallucination_grader_agent(base_llm)**`  
Compares `generation` against `documents` and returns `{"score": "yes" | "no"}` to indicate whether the answer is supported by the facts.
- `**create_answer_grader_agent(base_llm)**`  
(Currently defined but not yet wired into the graph.)  
Grades whether a `generation` is actually useful in resolving the `question`.
- `**create_question_router_agent(base_llm)**`  
Routes a `question` to either `"vectorstore"` (internal RAG) or `"web_search"` as the `datasource`. The graph uses this to decide whether to start from the HOA corpus or the web.

These functions are intentionally small and composable so they can be swapped or fine-tuned later.

---

### `langgraph_process.py`

Contains the **state definition** and the **node functions** used by the LangGraph workflow.

- `**GraphState` (`TypedDict`)**  
Represents the evolving state passed between nodes. Fields:
  - `question: str`
  - `generation: str`
  - `web_search: str` (e.g. `"Yes"` / `"No"`)
  - `documents: list[...]` (intended to be a list of `Document` objects)
- `**retrieve(state)`**  
Currently a placeholder that needs to be wired to a real vector-store retriever. Conceptually responsible for:
  - Using the user `question` to query the vector store
  - Returning updated state with `documents` and `question`
- `**generate(state)**`  
Uses `create_generate_agent` to produce an answer from:
  - `question`
  - `documents` (retrieved / web-augmented context)
- `**grade_documents(state)**`  
Uses `create_retrieval_grader_agent` to:
  - Score each retrieved document for relevance to `question`
  - Filter out low-relevance docs
  - Set `web_search` to `"Yes"` if no relevant docs remain
- `**web_search(state)**`  
Uses `TavilySearchResults` to:
  - Run a web search when internal docs are insufficient
  - Append the web results as a `Document` to `documents`
- `**route_question(state)**`  
Uses `create_question_router_agent` to choose between:
  - `"vectorstore"` → use internal HOA/strata corpus
  - `"web_search"` → go directly to external web search
- `**decide_to_generate(state)**`  
If `web_search == "Yes"`, route to `websearch`, else proceed to `generate`.
- `**check_hallucinating(state)**`  
Uses `create_hallucination_grader_agent` to:
  - Check if `generation` is grounded in `documents`
  - Optionally also grade whether the answer is relevant/useful to the question
  - Return one of several route labels (e.g. `"useful"`, `"not useful"`, `"not-supported"`) that the graph uses to either finish, re-generate, or fall back to web search.

There are a few naming and wiring inconsistencies here today (see the design doc for planned clean-up), but this file is the core of the RAG reasoning flow.

---

### `graph_build.py`

Defines and runs the LangGraph workflow:

- **Graph construction**
  - Creates a `StateGraph(GraphState)` workflow
  - Adds nodes:
    - `"websearch"` → `web_search`
    - `"retrieve"` → `retrieve`
    - `"grade_document"` → `grade_documents`
    - `"generate"` → `generate`
  - Sets **conditional entry point** based on `route_question`
  - Adds edges:
    - `"retrieve"` → `"grade_document"`
    - Conditional from `"grade_document"` via `decide_to_generate` → `"websearch"` or `"generate"`
    - `"websearch"` → `"generate"`
    - Conditional from `"generate"` via `check_hallucinating`:
      - If the answer is good → `END`
      - Otherwise → re-generate or fall back to web search
- **Testing / CLI**
  - `main()` creates the compiled graph (`app = workflow.compile()`)
  - Sends a sample question:
    - `{"question": "What are the types of agent memory?"}`
  - Streams the outputs and prints the final `generation`.

You can run this end-to-end test with:

```bash
poetry run python src/graph_build.py
```

*(Assumes you have set up Ollama, models, and env vars; see below.)*

---

### `index_data.py`

Utilities to ingest data from the web and build a Chroma-based vector store (for now; HOA docs will eventually replace these web sources):

- `**pre_processing_data(source: list[str])*`*
  - Loads each URL in `source` with `WebBaseLoader`
  - Flattens all pages into a single list of `Document` objects
  - Uses `RecursiveCharacterTextSplitter.from_tiktoken_encoder` with:
    - `chunk_size=250`
    - `chunk_overlap=0`
  - Returns a list of split `Document` chunks (`doc_splits`)
- `**index_data(doc_splits: list[Document])**`
  - Builds a `Chroma` vector store:
    - `collection_name="rag-chroma"`
    - `embeddings=GPT4AllEmbeddings()`
  - Returns a `retriever` (`vector_store.as_retriever()`)

In the HOA setting, this pattern will be reused, but `source` will come from local PDFs / scanned docs instead of public URLs.

---

### `data_manipulation.py`

Early-stage utilities for working with HOA/strata documents in tabular form and persisting a FAISS-based store:

- `**create_text_splitter(book_df: pandas.DataFrame)**`
  - Applies a `text_splitter` function (to be implemented in `utils.py`) to the `"Content"` column
  - Stores list-of-chunks in a `"chunked_text"` column
  - Returns the updated DataFrame
- `**create_vector_store(df: pandas.DataFrame, store_name: str, index_name: str)**`
  - Builds metadata per chunk (e.g. `{"source": Title}`)
  - Uses a `create_text_embedding()` function from `utils.py` (to be implemented) to get an embedding model
  - Either:
    - Loads an existing FAISS-backed vector store from `{store_name}.pkl`, or
    - Creates a new `FAISS` store from the chunked texts
  - Writes the FAISS index to disk via `faiss.write_index(index_name)`
  - Persists the store object itself as `{store_name}.pkl` with `pickle`

The `get_data_list()` and `clean_document()` functions are placeholders that will become the HOA-specific ingestion and cleaning steps.

---

### `utils.py`

Currently a placeholder that will host:

- Text splitting utilities (e.g. `text_splitter(text: str) -> list[str]`)
- Embedding factory functions (e.g. `create_text_embedding()`)
- Any shared helpers used across indexing and retrieval

---

## Running the project

### 1. Prerequisites

- **Python**: Managed via `poetry` (see `pyproject.toml` / `poetry.lock`)
- **Ollama**:
  - Install from `https://ollama.com`
  - Pull the model used here (default: `llama3.1`):
    ```bash
    ollama pull llama3.1
    ```
- **Environment / API keys**:
  - `TAVILY_API_KEY` for `TavilySearchResults` (web search)
  - Any additional keys/config for GPT4All embeddings if required by your setup

### 2. Install dependencies

From the repo root:

```bash
poetry install
```

### 3. (Current) demo run

The current `graph_build.py` script runs an end-to-end example using:

- Web data (via `index_data.py`) for the vector store (to be replaced by HOA docs)
- A simple question about agent memory

Run:

```bash
poetry run python src/graph_build.py
```

As you build out HOA ingestion and indexing, the same workflow will serve HOA questions instead of generic tutorial content.

---

## **How to run the system**

**1. Prepare documents & config**

- Place HOA/strata docs (PDFs, .txt, .md) under data/hoa_docs or adjust docs_path in conf/local/hoa_index.yml.

- Optionally set HOA_INDEX_CONFIG to point to a different YAML config.

**2. Build the FAISS index**

- From the project root:
      poetry run python src/index_hoa_[docs.py](http://docs.py) --config conf/local/hoa_index.yml

**3. Run the LangGraph workflow without UI (existing)**

- python src/graph_[build.py](http://build.py) still runs the console example, now backed by the HOA retriever.

**4. Run the Streamlit HOA chat app**

- From the project root:
      streamlit run src/[app.py](http://app.py)

- Ask questions like “What are the pet restrictions in my building?”; you’ll see the grounded answer plus the underlying retrieved chunks and routing/grading trace.

