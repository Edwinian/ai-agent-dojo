# RAG Tool Setup (LangGraph + Hugging Face)

This guide documents how to create a guest-information RAG tool in this project, based on:
- your current code structure (`documents/`, `tools/`, `graphs/document_analysis/`)
- Hugging Face Agents Course Unit 3 (LangGraph track)

Reference:
- [Creating a RAG Tool for Guest Stories](https://huggingface.co/learn/agents-course/en/unit3/agentic-rag/invitees?agents-frameworks=langgraph)

---

## 1) Prepare the guest dataset as `Document` objects

Create or maintain a document module (in this repo: `documents/guest.py`) that:

1. Loads the dataset:
   - `datasets.load_dataset("agents-course/unit3-invitees", split="train")`
2. Converts each row into a `langchain_core.documents.Document`
3. Stores all entries in a list named `docs`

Why: BM25Retriever operates on `Document` objects, not raw dataset rows.

---

## 2) Build reusable retrieval utilities

In this repo, retrieval logic is centralized in `utils.py`:

- `load_docs(document_name: str) -> list[Document]`
  - Dynamically loads a module under `documents/<name>.py`
  - Returns its `docs` list
- `retrieve_info(query: str, docs: list[Document]) -> str`
  - Builds `BM25Retriever.from_documents(docs)`
  - Executes `.invoke(query)`
  - Returns top 3 results joined as text

Why: keeps retriever mechanics separate from tool wiring.

---

## 3) Wrap retrieval logic as a callable tool function

In `tools/guest_info.py`, expose a plain callable used by the graph:

- `extract_text(query: str) -> str`
  - `docs = load_docs("guest")`
  - `return retrieve_info(query, docs)`

This function is what the LLM can call through LangGraph’s `ToolNode`.

---

## 4) Register tool(s) for document analysis graph

In `graphs/document_analysis/tools.py`, define:

- `tools = [extract_text]`

Where `extract_text` is imported from `tools.guest_info`.

In the graph and nodes:
- `ToolNode(tools)` is used in `graphs/document_analysis/graph.py`
- `HuggingFaceService(..., tools=tools)` binds tools on the chat model in `graphs/document_analysis/nodes.py`

---

## 5) Build LangGraph flow with tool routing

The LangGraph flow should be:

1. `assistant` node invokes tool-bound chat model
2. If model emits tool call(s), route to `tools` node
3. `tools` node executes callable(s)
4. Return to `assistant` to produce final answer

This is handled using:
- `ToolNode(tools)`
- `tools_condition`
- loop edge `tools -> assistant`

---

## 6) Invoke the graph with a retrieval-style query

In `graphs/document_analysis/invoke_graph.py`, use a message like:

- `"Tell me about our guest named 'Lady Ada Lovelace'."`

Then run:

- `python invoke_document_analysis_graph.py`

Expected behavior:
- assistant calls `extract_text`
- retriever returns matching guest details
- assistant formats a final response

---

## 7) Dependencies checklist

Ensure `requirements.txt` includes:

- `langgraph`
- `langchain-core`
- `langchain-community` (BM25Retriever)
- `datasets`
- `langchain-huggingface`

(plus existing project dependencies like `python-dotenv`, `xai-sdk`)

---

## 8) Common issues and fixes

### A) Provider/model mismatch on HF Inference
- Error examples:
  - `Model not supported by provider ...`
  - `Payment Required` (credits exhausted)
- Fix:
  - choose a model/provider pair that supports chat + tool calling
  - configure `HF_PROVIDER` and model defaults accordingly

### B) Tool-calling not supported by chosen model
- Error example:
  - `"auto" tool choice requires ...`
- Fix:
  - use a tool-capable model/provider route
  - or disable tool binding for that model

### C) Dataset module import conflicts
- Avoid naming conflicts between local package names and external packages.
- In this repo, guest docs are loaded from `documents/guest.py` via `load_docs`.

---

## 9) Minimal implementation map (this repo)

- Dataset formatting: `documents/guest.py`
- Retrieval utility: `utils.py`
- Guest retrieval tool function: `tools/guest_info.py`
- Tool registry for document analysis: `graphs/document_analysis/tools.py`
- Graph structure: `graphs/document_analysis/graph.py`
- Assistant node/tool binding: `graphs/document_analysis/nodes.py`
- Local runner: `graphs/document_analysis/invoke_graph.py`

---

## 10) Extension ideas

From here, you can improve quality by:

1. Replacing BM25 with embedding retrievers for semantic recall
2. Adding tool result post-processing / summarization rules
3. Caching loaded docs/retriever to avoid rebuilding on every call
4. Adding unit tests for:
   - `load_docs`
   - `retrieve_info`
   - `extract_text` tool behavior
