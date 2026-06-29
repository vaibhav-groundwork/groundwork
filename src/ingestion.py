import hashlib
import logging
import os
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

logger = logging.getLogger(__name__)


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


def ingest_document(file_path: str, collection, display_name: str | None = None) -> dict:
    """
    Full ingestion pipeline: extract → chunk → embed → store.
    Skips re-embedding if this exact document was already indexed.
    Returns a status dict the UI can use to show progress/feedback.

    display_name: the name to store and show as the source citation,
    decoupled from file_path. Needed because callers (e.g. app.py) often
    read from a temporary file with a randomly-generated name — without
    this, citations would show the meaningless temp filename instead of
    the user's actual uploaded filename. Defaults to the file_path's own
    name if not provided, preserving existing behavior for any caller
    that doesn't need this distinction (e.g. reading a real file directly
    from disk, where the path's name IS the real name).
    """
    doc_hash = get_document_hash(file_path)

    if is_document_indexed(collection, doc_hash):
        return {"status": "already_indexed", "chunks_added": 0, "doc_hash": doc_hash}

    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    filename = display_name if display_name else Path(file_path).name
    ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "doc_hash": doc_hash, "chunk_index": i} for i in range(len(chunks))]

    collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    return {"status": "indexed", "chunks_added": len(chunks), "doc_hash": doc_hash}

def retrieve_relevant_chunks(query: str, collection, n_results: int = TOP_K_RESULTS) -> list[dict]:
    """
    Retrieves the most relevant chunks for a query across all indexed documents,
    with fair per-document allocation when multiple documents are present.

    Why fair multi-document retrieval:
      A single pooled query ranks all chunks by semantic similarity to the query
      regardless of which document they come from. If the query phrasing happens
      to match one document's vocabulary better, that document can dominate all
      n_results slots, silently excluding another document the user uploaded and
      expected to be considered. Per-document sub-queries guarantee every
      uploaded document contributes at least one chunk to the answer context.

    Document-count cap:
      When more distinct documents exist than n_results, only the first n_results
      documents (by source name, alphabetically) are queried. This bounds total
      retrieval cost as upload count grows — each additional document would
      otherwise add a ChromaDB round-trip that scales linearly.

    Per-document failure isolation:
      Each sub-query is wrapped in its own try/except. A single document's query
      failure (e.g. a corrupt metadata filter) does not block retrieval from the
      remaining documents — the failing source is logged and skipped, and
      whatever results were already gathered are still returned.

    n_results budget distribution:
      The budget is divided evenly (integer division) across queried documents,
      with any remainder given to the first document processed, so the total
      chunk count returned stays at or near n_results.

    Returns a list of dicts with keys: text, source, distance — same shape as
    the original single-query implementation, so no caller needs to change.
    """
    # ── Step 1: determine distinct source documents ───────────────────────────
    distinct_sources: set[str] = set()
    try:
        all_meta = collection.get(include=["metadatas"])
        for meta in (all_meta.get("metadatas") or []):
            if meta and "source" in meta:
                distinct_sources.add(meta["source"])
    except Exception as exc:
        logger.warning(
            "retrieve_relevant_chunks: failed to determine distinct sources — "
            "%s: %s. Falling back to pooled query.",
            type(exc).__name__,
            exc,
        )
        # distinct_sources stays empty → treated as 0 sources → pooled fallback

    num_sources = len(distinct_sources)
    logger.info(
        "retrieve_relevant_chunks: found %d distinct source(s) in collection.",
        num_sources,
    )

    # ── Step 2: single pooled query for 0 or 1 sources (unchanged behavior) ──
    if num_sources <= 1:
        results = collection.query(query_texts=[query], n_results=n_results)
        chunks = []
        for i in range(len(results["documents"][0])):
            chunks.append({
                "text": results["documents"][0][i],
                "source": results["metadatas"][0][i].get("source", "unknown"),
                "distance": results["distances"][0][i] if "distances" in results else None,
            })
        return chunks

    # ── Step 3: fair per-document retrieval for 2+ sources ───────────────────
    sorted_sources = sorted(distinct_sources)

    if len(sorted_sources) > n_results:
        excluded = len(sorted_sources) - n_results
        logger.warning(
            "retrieve_relevant_chunks: %d document(s) excluded from this query "
            "due to the n_results=%d cap (querying first %d by name order).",
            excluded,
            n_results,
            n_results,
        )
        sorted_sources = sorted_sources[:n_results]

    num_queried = len(sorted_sources)
    base = n_results // num_queried
    remainder = n_results % num_queried

    logger.info(
        "retrieve_relevant_chunks: distributing n_results=%d across %d source(s) "
        "— base=%d per source, first source gets +%d extra.",
        n_results,
        num_queried,
        base,
        remainder,
    )

    chunks: list[dict] = []
    for i, source_name in enumerate(sorted_sources):
        per_doc_n = base + (remainder if i == 0 else 0)
        try:
            results = collection.query(
                query_texts=[query],
                n_results=per_doc_n,
                where={"source": source_name},
            )
            for j in range(len(results["documents"][0])):
                chunks.append({
                    "text": results["documents"][0][j],
                    "source": results["metadatas"][0][j].get("source", "unknown"),
                    "distance": results["distances"][0][j] if "distances" in results else None,
                })
        except Exception as exc:
            logger.warning(
                "retrieve_relevant_chunks: query failed for source=%r — "
                "%s: %s. Skipping this source.",
                source_name,
                type(exc).__name__,
                exc,
            )

    return chunks