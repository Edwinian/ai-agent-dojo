from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document


def load_docs(document_name: str) -> list[Document]:
    """Load a document module from `documents/<document_name>.py` and return its docs."""
    document_path = (
        Path(__file__).resolve().parent / "documents" / f"{document_name}.py"
    )
    spec = spec_from_file_location(f"{document_name}_document_module", document_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load document module from {document_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return list[Document](module.docs)


def retrieve_info(query: str, docs: list[Document]) -> str:
    """Retrieve top matching guest info snippets for a query using BM25."""
    bm25_retriever = BM25Retriever.from_documents(docs)
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."
