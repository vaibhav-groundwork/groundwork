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
from src.utils import strip_em_dashes
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


# Tool schema for the gap-analysis decision (analyse_node's intermediate passes).
# Replaces prompt-based JSON parsing — see judge_agent.py for the same pattern.
# Fixes a bug: Claude occasionally appends explanatory prose after
# the JSON block (e.g. "**Rationale:** ..."), which broke regex-based fence
# stripping and silently fell back to needs_more_search=False, terminating the
# loop early even when Claude's actual intent was to continue searching.
_GAP_ANALYSIS_TOOL = {
    "type": "function",
    "function": {
        "name": "report_gap_analysis",
        "description": "Report whether another search is needed and what to search for.",
        "parameters": {
            "type": "object",
            "properties": {
                "needs_more_search": {
                    "type": "boolean",
                    "description": "True if another search would meaningfully improve coverage.",
                },
                "next_query": {
                    "type": "string",
                    "description": "A specific refined search query. Empty string if no more search is needed.",
                },
            },
            "required": ["needs_more_search", "next_query"],
        },
    },
}

_FOLLOWUP_ROUTING_TOOL = {
    "type": "function",
    "function": {
        "name": "report_followup_routing",
        "description": "Decide whether a follow-up question can be answered from existing research notes or requires new web search.",
        "parameters": {
            "type": "object",
            "properties": {
                "needs_search": {
                    "type": "boolean",
                    "description": "True if the question asks about something not covered in the prior notes and requires new web research.",
                },
            },
            "required": ["needs_search"],
        },
    },
}


# ── State schema ──────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    topic: str
    prior_context: str     # accumulated research notes from earlier questions
                            # this session, empty string if this is the first
                            # question. Set by app.py before invoking the graph.
    route_decision: str    # "answer_from_context" or "needs_search", set by
                            # route_followup_node before the graph branches.
    search_results: list[dict]
    search_count: int
    next_query: str
    needs_more_search: bool
    research_notes: str
    sources: list[dict]    # deduplicated {title, link} dicts, kept
                            # separate from research_notes so prose stays
                            # clean, mirrors rag_agent.py's sources field
    status_message: str

# ── Node 0: route_followup_node ───────────────────────────────────────────────

def route_followup_node(state: ResearchState) -> dict:
    """
    LangGraph node — only meaningful when prior_context is non-empty (i.e. this
    is a follow-up question within an existing research session, not the first
    question). Decides whether the new question can be answered directly from
    already-gathered research notes, or genuinely requires new web search.

    On the very first question of a session (prior_context == ""), short-circuits
    straight to needs_search=True with no Claude call, since there is nothing
    to answer from yet.

    Defaults to needs_search=True on any tool-call failure — the safer failure
    mode here is running an unnecessary search, not silently answering from
    possibly-insufficient context.
    """
    prior_context = state.get("prior_context", "")

    if not prior_context:
        logger.info("route_followup_node: no prior_context — routing to needs_search.")
        return {"route_decision": "needs_search"}

    routing_system = (
        "You are deciding how to handle a follow-up question in an ongoing research "
        "session. Call the report_followup_routing tool with your decision. "
        "Do not use em dashes (—) anywhere in your response."
    )
    routing_messages = [
        {
            "role": "user",
            "content": (
                f"Prior research notes from this session:\n\n{prior_context}\n\n"
                f"New question: {state['topic']}\n\n"
                "Can this question be fully and accurately answered using ONLY the "
                "prior research notes above, with no new information needed? Or does "
                "it require new web research because it asks about something not "
                "covered in the prior notes?"
            ),
        }
    ]

    try:
        response = call_llm(
            model=HAIKU_MODEL,
            messages=routing_messages,
            system=routing_system,
            max_tokens=MAX_TOKENS_HAIKU,
            cache_system_prompt=True,
            tools=[_FOLLOWUP_ROUTING_TOOL],
            tool_choice={"type": "function", "function": {"name": "report_followup_routing"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("call_llm returned no tool_calls for report_followup_routing.")
        decision = json.loads(tool_calls[0].function.arguments)
        needs_search = bool(decision.get("needs_search", True))
    except Exception as exc:
        logger.warning(
            "route_followup_node: routing tool call failed — %s: %s. "
            "Defaulting to needs_search=True to avoid silently giving an unsupported answer.",
            type(exc).__name__,
            exc,
        )
        needs_search = True

    route_decision = "needs_search" if needs_search else "answer_from_context"
    logger.info("route_followup_node: route_decision=%r", route_decision)
    return {"route_decision": route_decision}


# ── Node 0b: answer_from_notes_node ───────────────────────────────────────────

def answer_from_notes_node(state: ResearchState) -> dict:
    """
    LangGraph node — answers a follow-up question directly from existing
    research notes, with no new web search. Reached only when
    route_followup_node decides route_decision == "answer_from_context".

    No new sources are produced here (sources field is left empty for this
    pass), since the answer is grounded entirely in already-accumulated
    context whose sources are already tracked by the calling application.
    """
    answer_system = (
        "You are answering a follow-up question using only previously gathered "
        "research notes. Do not invent information not present in the notes — "
        "if something genuinely isn't covered, say so honestly rather than guessing. "
        "Do not use em dashes (—) anywhere in your response. Use commas, periods, or parentheses instead."
    )
    answer_messages = [
        {
            "role": "user",
            "content": (
                f"Prior research notes:\n\n{state['prior_context']}\n\n"
                f"Question: {state['topic']}\n\n"
                "Please answer the question using only the notes above."
            ),
        }
    ]

    logger.info("answer_from_notes_node: answering follow-up from prior_context, no new search.")

    response = call_llm(
        model=HAIKU_MODEL,
        messages=answer_messages,
        system=answer_system,
        max_tokens=MAX_TOKENS_HAIKU,
        cache_system_prompt=True,
    )

    return {
        "research_notes": strip_em_dashes(response.choices[0].message.content.strip()),
        "sources": [],
        "needs_more_search": False,
        "status_message": "✅ Answered from existing research — no new search needed.",
    }


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
        
        existing_sources: list[dict] = state.get("sources", [])
        new_sources = [
            {"title": r["title"], "link": r["link"]}
            for r in new_results
            if r.get("link")
        ]

        return {
            "search_results": existing_results + new_results,
            "sources": existing_sources + new_sources,
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
            "sources": state.get("sources", []),
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
            "research_notes": strip_em_dashes(
                "No information was found for this topic across all search attempts. "
                "The brief will be limited — please verify the topic or try a more "
                "specific query."
            ),
            "sources": [],
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
            "sources": _dedupe_sources(state.get("sources", [])),
            "status_message": "✅ Research complete — synthesising notes.",
        }

    # ── Path 3: ask Claude whether another search is needed ──────────────────
    snippets_text = _format_snippets(search_results)

    gap_system = (
        "You are a research analyst reviewing web search results. "
        "Your job is to identify gaps in the information gathered so far and decide "
        "whether one additional targeted search would meaningfully improve coverage. "
        "Call the report_gap_analysis tool with your decision. "
        "Do not use em dashes (—) anywhere in your response. Use commas, periods, or parentheses instead."
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

    try:
        response = call_llm(
            model=HAIKU_MODEL,
            messages=gap_messages,
            system=gap_system,
            max_tokens=MAX_TOKENS_HAIKU,
            cache_system_prompt=True,
            tools=[_GAP_ANALYSIS_TOOL],
            tool_choice={"type": "function", "function": {"name": "report_gap_analysis"}},
        )
        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError("call_llm returned no tool_calls for report_gap_analysis.")
        decision = json.loads(tool_calls[0].function.arguments)
    except Exception as exc:
        logger.warning(
            "analyse_node: gap-analysis tool call failed — %s: %s. "
            "Defaulting to needs_more_search=False to avoid an unbounded loop.",
            type(exc).__name__,
            exc,
        )
        decision = {"needs_more_search": False, "next_query": ""}

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
            "sources": _dedupe_sources(state.get("sources", [])),
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
        "sources": _dedupe_sources(state.get("sources", [])),
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

def _dedupe_sources(sources: list[dict]) -> list[dict]:
    """
    Deduplicates a sources list by link, preserving first-seen order.
    Called once, on the final research pass, since intermediate passes
    don't need a clean list — only the version actually returned to
    downstream consumers (eventually synthesis_node) needs to be clean.
    """
    seen = set()
    deduped = []
    for s in sources:
        link = s.get("link", "")
        if link and link not in seen:
            seen.add(link)
            deduped.append(s)
    return deduped

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
        "Group findings by theme. Include specific facts. "
        "Do NOT include URLs or any inline citation markers in your prose — sources are "
        "tracked and displayed separately by the application, so your writing should read "
        "as clean narrative text with no parenthetical links. "
        "Write for a downstream writer agent — this is the input they will work from, "
        "not a final deliverable. Be thorough and factual; do not pad or summarise vaguely. "
        "Do not use em dashes (—) anywhere in your response. Use commas, periods, or parentheses instead."
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
    return strip_em_dashes(response.choices[0].message.content.strip())
