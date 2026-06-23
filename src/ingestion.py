import os
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

from src.config import (
    EMBEDDING_MODEL,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TOP_K_RESULTS,
)


def get_chroma_collection(session_id: str):
    """
    Returns a persistent ChromaDB collection scoped to a single browser session.

    Each session gets its own isolated collection (named groundwork_{session_id})
    so that documents uploaded by one user can never be retrieved when answering
    another user's question. session_id is generated once per browser session
    in app.py via st.session_state and passed down to every call site.

    Known v1 limitation (tracked in docs/groundwork_v2_backlog.md): nothing
    currently deletes old session collections — they persist on disk until
    manually cleaned up.
    """
    embedding_fn = SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection_name = f"groundwork_{session_id}"
    return client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts all text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits text into overlapping chunks by approximate token count.
    Using word count as a proxy for tokens (roughly 0.75 words per token,
    close enough for chunking purposes — exact token counting isn't needed here).
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_document_hash(file_path: str) -> str:
    """
    Generates a hash of the file contents — used to check if a document
    has already been embedded, so we never re-embed the same file twice.
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def is_document_indexed(collection, doc_hash: str) -> bool:
    """Checks whether this exact document (by content hash) is already in ChromaDB."""
    results = collection.get(where={"doc_hash": doc_hash}, limit=1)
    return len(results["ids"]) > 0


def ingest_document(file_path: str, collection) -> dict:
    """
    Full ingestion pipeline: extract → chunk → embed → store.
    Skips re-embedding if this exact document was already indexed.
    Returns a status dict the UI can use to show progress/feedback.
    """
    doc_hash = get_document_hash(file_path)

    if is_document_indexed(collection, doc_hash):
        return {"status": "already_indexed", "chunks_added": 0, "doc_hash": doc_hash}

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    filename = Path(file_path).name
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "doc_hash": doc_hash, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return {"status": "indexed", "chunks_added": len(chunks), "doc_hash": doc_hash}


def retrieve_relevant_chunks(query: str, collection, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """
    Retrieves the most relevant chunks for a query, with their source metadata.
    Returns a list of dicts so the UI can show source citations alongside answers.
    """
    results = collection.query(query_texts=[query], n_results=n_results)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "distance": results["distances"][0][i] if "distances" in results else None,
        })
    return chunks