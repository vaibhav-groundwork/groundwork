"""
rag_agent.py — LangGraph node for document-grounded question answering.

Pipeline position:
  graph.py routes document-mode runs directly to rag_node, bypassing the
  web-research loop. The caller (app.py) has already ingested the document
  into a ChromaDB collection and passes it via RAGState.

Scope boundaries:
  - This file never creates, populates, or validates collections — ingestion
    lives entirely in src.ingestion.
  - Input validation (empty question, excessive length) is the responsibility
    of the UI layer in app.py — this node assumes a non-empty, reasonable
    question string arrives in state.

Exports:
  RAGState  — TypedDict shared with graph.py
  rag_node  — single LangGraph node for retrieval + generation
"""

import logging
from src.utils import strip_em_dashes
from typing import TypedDict

from src.config import MAX_TOKENS_SONNET, SONNET_MODEL
from src.ingestion import retrieve_relevant_chunks
from src.tracing import call_llm

logger = logging.getLogger(__name__)

# Character ceiling applied to the combined chunk text before sending to
# Claude. Keeps token cost predictable regardless of future changes to
# CHUNK_SIZE or TOP_K_RESULTS in config.
_CONTEXT_CHAR_LIMIT = 6000


# ── State schema ──────────────────────────────────────────────────────────────

class RAGState(TypedDict):
    question: str
    collection: object          # ChromaDB collection, created/populated upstream
    answer: str
    sources: list[dict]
    status_message: str


# ── Node: rag_node ────────────────────────────────────────────────────────────

def rag_node(state: RAGState) -> dict:
    """
    LangGraph node — retrieves relevant document chunks from ChromaDB and uses
    Claude Sonnet to answer the user's question grounded strictly in those chunks.

    Two meaningful outcomes differ from a plain "answer not found":

    1. Zero chunks returned — the collection is empty or the query matched
       nothing; Claude is NOT called because there is no context to ground an
       answer in. A friendly, actionable message is returned instead.

    2. Chunks present but insufficient — Claude is called and explicitly
       instructed to say so honestly rather than hallucinate.  The system
       prompt requires Claude to be specific about what the document *does*
       cover, if that's apparent from the chunks, giving the user useful
       signal rather than a generic "I don't know."

    Prompt caching is deliberately omitted: a RAG question-answer is a
    one-shot call per run, so the system prompt is never repeated within the
    same session in a way that would benefit from caching.

    Returns a partial-state dict; LangGraph merges it into the full state.
    """
    question: str = state["question"]
    collection = state["collection"]

    logger.info("rag_node: incoming question=%r", question)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    chunks: list[dict] = retrieve_relevant_chunks(query=question, collection=collection)

    if not chunks:
        logger.warning(
            "rag_node: retrieve_relevant_chunks returned 0 results — "
            "collection appears empty or no document has been uploaded."
        )
        no_doc_message = (
            "No document content is available to search. "
            "Please upload a document first so I can answer questions about it."
        )
        return {
            "answer": no_doc_message,
            "sources": [],
            "status_message": "⚠️ No document content found — please upload a document first.",
        }

    # ── Build context, capped at _CONTEXT_CHAR_LIMIT chars ───────────────────
    context_parts: list[str] = []
    total_chars = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        entry = f"[Source: {source}]\n{text}"
        if total_chars + len(entry) > _CONTEXT_CHAR_LIMIT:
            remaining = _CONTEXT_CHAR_LIMIT - total_chars
            if remaining > 0:
                context_parts.append(entry[:remaining])
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context_text = "\n\n---\n\n".join(context_parts)

    # ── Prompt construction ───────────────────────────────────────────────────
    system_prompt = (
        "You are a research assistant answering questions strictly from the document "
        "excerpts provided below. Do not use any outside knowledge or make inferences "
        "beyond what the excerpts explicitly state.\n\n"
        "If the provided excerpts do not contain enough relevant information to answer "
        "the question well, say so clearly and honestly — do not guess. Where possible, "
        "mention specifically what the document *does* cover based on the excerpts, so "
        "the user understands the scope of available content.\n\n"
        "Always cite the source filename when referencing information from a specific excerpt."
        "Do not use em dashes (—) anywhere in your response. Use commas, periods, or parentheses instead.\n\n"
    )

    user_message = (
        f"Question: {question}\n\n"
        f"Document excerpts:\n\n{context_text}"
    )

    messages = [{"role": "user", "content": user_message}]

    # ── LLM call (with graceful error handling) ───────────────────────────────
    try:
        response = call_llm(
            model=SONNET_MODEL,
            messages=messages,
            system=system_prompt,
            max_tokens=MAX_TOKENS_SONNET,
            cache_system_prompt=False,
        )
    except Exception as exc:
        logger.warning(
            "rag_node: call_llm failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return {
            "answer": (
                "Something went wrong while answering your question — please try again."
            ),
            "sources": [],
            "status_message": "⚠️ Error generating answer — please try again.",
        }

    answer: str = response.choices[0].message.content.strip()
    logger.info("rag_node: answer generated, length=%d characters.", len(answer))

    # ── Build sources list for UI citation display ────────────────────────────
    sources = [
        {"source": c.get("source", "unknown"), "text": c.get("text", "")}
        for c in chunks
    ]

    # Distinguish "found a real answer" from "honestly declined" so the status
    # message doesn't misleadingly claim success on a correct refusal. This is
    # a lightweight heuristic, not a guarantee — it checks for common phrasing
    # Claude uses when the system prompt's honesty instruction kicks in.
    declined_phrases = [
        "cannot answer",
        "can't answer",
        "does not contain",
        "doesn't contain",
        "not contain any information",
        "do not have enough information",
        "don't have enough information",
    ]
    answer_lower = answer.lower()
    declined = any(phrase in answer_lower for phrase in declined_phrases)

    status_message = (
        "📄 The document doesn't appear to cover this — see the explanation below."
        if declined
        else "✅ Found an answer in your document."
    )

    return {
        "answer": strip_em_dashes(answer),
        "sources": sources,
        "status_message": status_message,
    }
