"""
graph.py — the single LangGraph StateGraph in the Groundwork codebase.

Why only the research loop uses a graph:
  LangGraph's StateGraph machinery exists to solve one specific problem:
  autonomous branching and looping that runs without a human in the middle.
  The research loop is the only part of this pipeline that genuinely has that
  property — it runs search_node → analyse_node repeatedly until the model
  decides it has enough information (or the hard cap is hit), with no user
  interaction between iterations.

  rag_node, synthesis_node, and judge_node are plain function calls
  orchestrated directly by app.py. They are deliberately NOT wrapped in a graph
  because:
    - Each runs exactly once per user action (no looping or branching).
    - The human-in-the-loop regeneration flow (user clicks "Regenerate") is
      handled natively by Streamlit's re-run execution model. There is nothing
      LangGraph could add here that Streamlit doesn't already provide for free,
      and adding a graph would make the pause-for-human-input behaviour harder
      to reason about, not easier.
  Keeping non-looping nodes as plain function calls also makes them easier to
  test, easier to step through in a debugger, and more readable to anyone who
  hasn't used LangGraph before.

Public interface:
  run_research(topic) is the intended call site in app.py — NOT
  RESEARCH_GRAPH.invoke() directly. run_research() adds input validation and
  graceful error handling that the raw compiled graph object does not have.
  Calling RESEARCH_GRAPH.invoke() directly from app.py would bypass both.

Exports:
  RESEARCH_GRAPH      — compiled StateGraph (available for inspection/testing)
  build_research_graph — factory function that constructs the graph
  run_research         — validated public entry point for app.py
"""

import logging

from langgraph.graph import END, StateGraph

from src.agents.research_agent import ResearchState, analyse_node, search_node

logger = logging.getLogger(__name__)


# ── Conditional edge function ─────────────────────────────────────────────────

def _should_continue_research(state: ResearchState) -> str:
    """
    Conditional edge function called after each analyse_node pass.
    Returns 'search' to run another search iteration, or 'end' to exit the loop.
    """
    continue_search: bool = state["needs_more_search"]
    decision = "search" if continue_search else "end"
    logger.info(
        "_should_continue_research: needs_more_search=%s → routing to '%s'.",
        continue_search,
        decision,
    )
    return decision


# ── Graph factory ─────────────────────────────────────────────────────────────

def build_research_graph():
    """
    Constructs and returns a compiled StateGraph for the autonomous research loop.

    Graph topology:
      search ──► analyse ──┬──► search  (if needs_more_search is True)
                           └──► END     (if needs_more_search is False)
    """
    graph = StateGraph(ResearchState)

    graph.add_node("search", search_node)
    graph.add_node("analyse", analyse_node)

    graph.set_entry_point("search")

    graph.add_edge("search", "analyse")
    graph.add_conditional_edges(
        "analyse",
        _should_continue_research,
        {"search": "search", "end": END},
    )

    return graph.compile()


# ── Module-level compiled graph ───────────────────────────────────────────────

RESEARCH_GRAPH = build_research_graph()


# ── Public entry point ────────────────────────────────────────────────────────

def run_research(topic: str) -> dict:
    """
    Validated public entry point for the research pipeline.

    app.py should call this function, not RESEARCH_GRAPH.invoke() directly.
    Adds two layers the raw graph does not provide:
      1. Input validation — rejects empty or whitespace-only topics before
         invoking the graph, returning a structured error dict instead.
      2. Error handling — catches any exception raised during graph execution
         and returns a structured error dict rather than propagating to app.py.

    Args:
        topic: The research question or subject. Must be a non-empty,
               non-whitespace string.

    Returns:
        The final ResearchState dict produced by the graph on success, or a
        partial dict with research_notes and status_message populated with
        error information on failure.
    """
    if not isinstance(topic, str) or not topic.strip():
        logger.warning(
            "run_research: received invalid topic=%r — must be a non-empty string.",
            topic,
        )
        return {
            "research_notes": (
                "A valid topic is required to run research. "
                "Please provide a non-empty search topic."
            ),
            "status_message": "⚠️ Please enter a valid topic before starting research.",
        }

    initial_state: ResearchState = {
        "topic": topic.strip(),
        "search_results": [],
        "search_count": 0,
        "next_query": "",
        "needs_more_search": True,
        "research_notes": "",
        "status_message": "",
    }

    logger.info("run_research: starting research loop for topic=%r.", topic.strip())

    try:
        final_state = RESEARCH_GRAPH.invoke(initial_state)
    except Exception as exc:
        logger.warning(
            "run_research: graph execution failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return {
            "research_notes": (
                "Something went wrong during research — please try again."
            ),
            "status_message": "⚠️ Research failed — please try again.",
        }

    logger.info(
        "run_research: completed. search_count=%d, research_notes length=%d characters.",
        final_state.get("search_count", 0),
        len(final_state.get("research_notes", "")),
    )

    return final_state
