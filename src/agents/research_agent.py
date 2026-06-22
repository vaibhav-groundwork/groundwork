"""
research_agent.py — LangGraph nodes for the autonomous web-research loop.

Pipeline position:
  graph.py calls search_node → analyse_node in a conditional loop until
  analyse_node signals needs_more_search=False or the hard search cap is hit.

Exports:
  ResearchState  — TypedDict shared across all research-loop nodes
  search_node    — fetches real web results via SerpAPI
  analyse_node   — uses Claude Haiku to review gaps and synthesise final notes
"""

import json
import logging
import re
from typing import TypedDict

import requests

from src.config import (
    HAIKU_MODEL,
    MAX_TOKENS_HAIKU,
    MAX_SEARCH_CALLS,
    SERPAPI_KEY,
)
from src.tracing import call_llm

logger = logging.getLogger(__name__)


# ── State schema ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    topic: str
    search_results: list[dict]
    search_count: int
    next_query: str
    needs_more_search: bool
    research_notes: str
    status_message: str


# ── Node 1: search_node ───────────────────────────────────────────────────────

def search_node(state: ResearchState) -> dict:
    """
    LangGraph node — executes a single SerpAPI web search and appends results
    to the shared state.

    On the first pass (search_count == 0) the original topic is used as the
    query.  On subsequent passes the refined next_query produced by
    analyse_node is used instead.

    Results are accumulated (never overwritten) so analyse_node always has the
    full picture.  Each snippet is truncated to 200 characters to keep
    downstream token costs predictable.

    Returns a partial-state dict; LangGraph merges it into the full state.
    """
    topic = state["topic"]
    existing_results: list[dict] = state.get("search_results", [])
    search_count: int = state.get("search_count", 0)
    next_query: str = state.get("next_query", "")

    query = topic if search_count == 0 else (next_query or topic)

    status_message = f"🔍 Searching for: {query}"
    logger.info("search_node: query=%r (pass %d)", query, search_count + 1)

    try:
        response = requests.get(
            "https://serpapi.com/search",
            params={
                "q": query,
                "api_key": SERPAPI_KEY,
                "num": 3,
                "engine": "google",
            },
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()

        raw_results = data.get("organic_results", [])[:3]
        new_results = [
            {
                "title": r.get("title", ""),
                "snippet": (r.get("snippet", ""))[:200],
                "link": r.get("link", ""),
            }
            for r in raw_results
        ]

        logger.info(
            "search_node: retrieved %d result(s) for query=%r",
            len(new_results),
            query,
        )

        return {
            "search_results": existing_results + new_results,
            "search_count": search_count + 1,
            "status_message": status_message,
        }

    except Exception as exc:
        logger.warning(
            "search_node: search failed for query=%r — %s: %s",
            query,
            type(exc).__name__,
            exc,
        )
        return {
            "search_results": existing_results,
            "search_count": search_count + 1,
            "status_message": "⚠️ Search failed, continuing with available information.",
        }


# ── Node 2: analyse_node ──────────────────────────────────────────────────────

def analyse_node(state: ResearchState) -> dict:
    """
    LangGraph node — reviews accumulated search results, decides whether
    another search pass is needed, and on the final pass synthesises clean
    research notes for the downstream synthesis/writer agent.

    Three early-exit paths (no Claude call):
      1. Zero results across all searches → notes flagged as empty, loop ends.
      2. search_count >= MAX_SEARCH_CALLS → hard cap enforced, loop ends.

    On intermediate passes, Claude responds with JSON only:
      { "needs_more_search": bool, "next_query": str }

    On the final pass (needs_more_search=False, by any path), a second Claude
    call synthesises structured research_notes from all gathered snippets.

    Returns a partial-state dict; LangGraph merges it into the full state.
    """
    search_results: list[dict] = state.get("search_results", [])
    search_count: int = state.get("search_count", 0)

    # ── Path 1: no results at all ────────────────────────────────────────────
    if not search_results:
        logger.warning(
            "analyse_node: no search results available after %d search(es) — "
            "skipping Claude call; research_notes will be empty.",
            search_count,
        )
        return {
            "needs_more_search": False,
            "next_query": "",
            "research_notes": (
                "No information was found for this topic across all search attempts. "
                "The brief will be limited — please verify the topic or try a more "
                "specific query."
            ),
            "status_message": "⚠️ No results found — brief will have limited information.",
        }

    # ── Path 2: hard search cap ──────────────────────────────────────────────
    if search_count >= MAX_SEARCH_CALLS:
        logger.info(
            "analyse_node: search limit reached (%d/%d) — forcing final pass.",
            search_count,
            MAX_SEARCH_CALLS,
        )
        research_notes = _synthesise_notes(state)
        return {
            "needs_more_search": False,
            "next_query": "",
            "research_notes": research_notes,
            "status_message": "✅ Research complete — synthesising notes.",
        }

    # ── Path 3: ask Claude whether another search is needed ──────────────────
    snippets_text = _format_snippets(search_results)

    gap_system = (
        "You are a research analyst reviewing web search results. "
        "Your job is to identify gaps in the information gathered so far and decide "
        "whether one additional targeted search would meaningfully improve coverage. "
        "Respond in strict JSON only — no prose, no markdown fences:\n"
        '{"needs_more_search": <bool>, "next_query": "<specific refined query, or empty string>"}'
    )
    gap_messages = [
        {
            "role": "user",
            "content": (
                f"Research topic: {state['topic']}\n\n"
                f"Results gathered so far ({len(search_results)} sources):\n\n"
                f"{snippets_text}\n\n"
                "Should we run one more search to fill a meaningful gap? "
                "If yes, provide a specific refined query. If the coverage is adequate, say no."
            ),
        }
    ]

    response = call_llm(
        model=HAIKU_MODEL,
        messages=gap_messages,
        system=gap_system,
        max_tokens=MAX_TOKENS_HAIKU,
        cache_system_prompt=True,
    )
    raw_text = response.choices[0].message.content.strip()

    decision = _parse_json_response(raw_text)
    needs_more = bool(decision.get("needs_more_search", False))
    next_query = decision.get("next_query", "") or ""

    logger.info(
        "analyse_node: decision — needs_more_search=%s, next_query=%r",
        needs_more,
        next_query,
    )

    if needs_more:
        return {
            "needs_more_search": True,
            "next_query": next_query,
            "research_notes": "",
            "status_message": (
                f"🧠 Reviewing findings — found {len(search_results)} source(s) so far"
            ),
        }

    # ── Final pass: synthesise notes ─────────────────────────────────────────
    research_notes = _synthesise_notes(state)
    return {
        "needs_more_search": False,
        "next_query": "",
        "research_notes": research_notes,
        "status_message": "✅ Research complete — synthesising notes.",
    }


# ── Private helpers ───────────────────────────────────────────────────────────

def _format_snippets(search_results: list[dict]) -> str:
    """Formats accumulated search results into a compact, numbered block."""
    lines = []
    for i, r in enumerate(search_results, 1):
        lines.append(
            f"[{i}] {r.get('title', 'No title')}\n"
            f"    {r.get('snippet', '')}\n"
            f"    Source: {r.get('link', 'unknown')}"
        )
    return "\n\n".join(lines)


def _parse_json_response(text: str) -> dict:
    """
    Parses a JSON response from Claude, tolerating markdown code fences.
    Falls back to safe defaults on any parse error.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning(
            "analyse_node: failed to parse JSON from model response — "
            "raw text: %r",
            text[:200],
        )
        return {"needs_more_search": False, "next_query": ""}


def _synthesise_notes(state: ResearchState) -> str:
    """
    Calls Claude Haiku once to produce clean, structured research notes from
    all accumulated search results.  Called only on the final analysis pass.
    """
    search_results: list[dict] = state.get("search_results", [])
    snippets_text = _format_snippets(search_results)

    synth_system = (
        "You are a research analyst preparing a structured briefing document. "
        "Synthesise the search results below into clean, well-organised research notes. "
        "Group findings by theme. Include specific facts and cite sources by their URL. "
        "Write for a downstream writer agent — this is the input they will work from, "
        "not a final deliverable. Be thorough and factual; do not pad or summarise vaguely."
    )
    synth_messages = [
        {
            "role": "user",
            "content": (
                f"Research topic: {state['topic']}\n\n"
                f"All gathered sources ({len(search_results)} total):\n\n"
                f"{snippets_text}\n\n"
                "Please produce structured research notes organised by theme."
            ),
        }
    ]

    logger.info(
        "analyse_node: running final synthesis over %d source(s).",
        len(search_results),
    )

    response = call_llm(
        model=HAIKU_MODEL,
        messages=synth_messages,
        system=synth_system,
        max_tokens=MAX_TOKENS_HAIKU,
        cache_system_prompt=True,
    )
    return response.choices[0].message.content.strip()
