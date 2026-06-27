"""
judge_agent.py — LangGraph node that evaluates a generated report against its
source material, producing structured per-dimension scores and an overall rating.

Pipeline position:
  graph.py routes to judge_node after synthesis_node, giving the user a
  quality signal before they decide whether to accept or regenerate the brief.
  Regeneration-limit tracking (MAX_REWRITE_ATTEMPTS) is app.py's responsibility
  — this node always scores whatever report it receives, with no memory of or
  opinion about prior attempts.

Cross-provider judging design:
  The writer (synthesis_node) uses Claude Sonnet; the judge uses JUDGE_MODEL
  (gpt-4o-mini via LiteLLM). Using a different provider avoids self-grading
  bias — a model scoring its own outputs has a measurable tendency to inflate ratings.
  Swapping the judge to a different provider is a one-line change in config.py; 
  this file is provider-agnostic.

Structured output approach — Level 3 (tool-based):
  Other nodes in this codebase use prompt-based JSON (Level 1/2): they ask the
  model to "respond in strict JSON only" and parse the result, with a regex
  fallback to strip markdown fences. That approach works well for simple
  decisions (needs_more_search, next_query) but is fragile for a richer
  multi-field scoring object where every field matters.

  This node uses OpenAI-style function calling (tools + tool_choice) to
  constrain the model to return arguments matching a declared schema — the
  model physically cannot respond with freeform text when tool_choice forces a
  specific function. LiteLLM passes tools/tool_choice uniformly across
  providers. This eliminates the "JSON wrapped in markdown fences" failure mode
  entirely for the most structurally important output in the pipeline.

Weighted scoring rationale:
  Accuracy and groundedness are weighted 0.3 each (total 0.6) because factual
  errors and invented detail directly undermine user trust and are the hardest
  to fix after the fact. Helpfulness and conciseness are weighted 0.2 each —
  important for UX but recoverable through user feedback or regeneration.

Score clamping:
  _clamp_score() defends against the rare case where a returned integer falls
  outside 1-5 despite the schema constraint — e.g. a model that misreads
  "minimum: 1" as 0-indexed. Clamping is applied before the weighted average so
  a rogue value cannot skew overall_score outside the displayable range.

Exports:
  JudgeState   — TypedDict shared with graph.py
  judge_node   — single LangGraph node for report evaluation
  _JUDGE_TOOL  — tool schema, exported so graph.py can inspect it if needed
"""

import json
import logging
from typing import TypedDict
from src.utils import strip_em_dashes

from src.config import JUDGE_MODEL
from src.tracing import call_llm

logger = logging.getLogger(__name__)


# ── Tool schema ───────────────────────────────────────────────────────────────

_JUDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "score_report",
        "description": (
            "Score a report against the source material it was generated from, "
            "across four dimensions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "accuracy_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Is the report factually consistent with the source material?",
                },
                "accuracy_reasoning": {"type": "string"},
                "groundedness_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": (
                        "Is every claim traceable to the source material, "
                        "with no invented detail?"
                    ),
                },
                "groundedness_reasoning": {"type": "string"},
                "helpfulness_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": "Does the report usefully address the stated topic?",
                },
                "helpfulness_reasoning": {"type": "string"},
                "conciseness_score": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "description": (
                        "Is the length and structure proportionate to the substance? "
                        "Does it avoid restating the same fact across multiple sections?"
                    ),
                },
                "conciseness_reasoning": {"type": "string"},
            },
            "required": [
                "accuracy_score",
                "accuracy_reasoning",
                "groundedness_score",
                "groundedness_reasoning",
                "helpfulness_score",
                "helpfulness_reasoning",
                "conciseness_score",
                "conciseness_reasoning",
            ],
        },
    },
}

# Weights must sum to 1.0.
# Accuracy and groundedness are weighted higher than helpfulness and conciseness
# because factual errors and invented detail directly undermine user trust and
# are the hardest to recover from after the fact.
_ACCURACY_WEIGHT = 0.3
_GROUNDEDNESS_WEIGHT = 0.3
_HELPFULNESS_WEIGHT = 0.2
_CONCISENESS_WEIGHT = 0.2


# ── State schema ──────────────────────────────────────────────────────────────

class JudgeState(TypedDict):
    topic: str
    source_material: str
    report: str             # synthesis_node output being evaluated
    scores: dict
    overall_score: float
    score_explanation: str
    status_message: str


# ── Private helpers ───────────────────────────────────────────────────────────

def _clamp_score(value) -> int:
    """
    Converts value to int and clamps to [1, 5] inclusive.
    Defends against the rare case where a returned score falls outside the
    schema's stated bounds despite the tool constraint.
    """
    return max(1, min(5, int(value)))


# ── Node: judge_node ──────────────────────────────────────────────────────────

def judge_node(state: JudgeState) -> dict:
    """
    LangGraph node — scores a generated report against its source material
    using tool-based structured output (Level 3), producing per-dimension
    scores and a weighted overall rating.

    See module docstring for the full design rationale covering: cross-provider
    judging, tool-based vs prompt-based JSON, weighted scoring, and the
    deliberate boundary that regeneration-limit tracking lives in app.py.

    The entire block from the API call through parsing tool arguments is wrapped
    in a single try/except — covering the network call, accessing tool_calls,
    the non-empty check (missing or empty tool_calls raises an explicit
    ValueError), and JSON parsing. One except clause handles all failure modes
    rather than silently propagating an IndexError from a different code path.

    Returns a partial-state dict; LangGraph merges it into the full state.
    """
    topic: str = state["topic"]
    source_material: str = state.get("source_material", "")
    report: str = state.get("report", "")

    logger.info(
        "judge_node: topic=%r, report length=%d characters.",
        topic,
        len(report),
    )

    # ── Prompts ───────────────────────────────────────────────────────────────
    system_prompt = (
        "You are an impartial report evaluator. Your task is to score a generated "
        "report strictly against the source material it was produced from, across "
        "four dimensions:\n\n"
        "- ACCURACY: Is every factual claim in the report consistent with the source material?\n"
        "- GROUNDEDNESS: Is every claim traceable to the source material, with nothing invented?\n"
        "- HELPFULNESS: Does the report usefully address the stated topic for a reader?\n"
        "- CONCISENESS: This is the dimension most evaluators are too lenient on. Specifically check: "
        "does the report restate the SAME fact or figure across multiple sections using different "
        "wording? A report can be well-formatted and still score low on conciseness if multiple "
        "sections substantively repeat the same one or two underlying facts. Compare the number of "
        "genuinely distinct facts in the source material to the number of sections in the report — "
        "if the report has more sections than the source material has distinct facts, this is a "
        "strong signal of padding and should be scored 3 or below.\n\n"
        "Evaluate only what is present in the source material — do not reward or penalise the "
        "report for omitting information that was never in the source. "
        "You must call the score_report tool with your scores and reasoning. "
        "Do not respond with plain text."
        "Do not use em dashes (—) anywhere in your response. Use commas, periods, or parentheses instead.\n\n"
    )
    user_message = (
        f"Topic: {topic}\n\n"
        f"--- SOURCE MATERIAL ---\n{source_material}\n\n"
        f"--- REPORT TO EVALUATE ---\n{report}"
    )

    messages = [{"role": "user", "content": user_message}]

    # ── API call + parsing (single try/except covers the full block) ──────────
    _failure_return = {
        "scores": {},
        "overall_score": 0.0,
        "score_explanation": (
            "Evaluation could not be completed — the scoring model did not return "
            "a valid structured response. Please try again."
        ),
        "status_message": "⚠️ Could not evaluate this report — please try again.",
    }

    try:
        response = call_llm(
            model=JUDGE_MODEL,
            messages=messages,
            system=system_prompt,
            max_tokens=1024,
            cache_system_prompt=False,
            tools=[_JUDGE_TOOL],
            tool_choice={"type": "function", "function": {"name": "score_report"}},
        )

        tool_calls = response.choices[0].message.tool_calls
        if not tool_calls:
            raise ValueError(
                "call_llm returned a response with no tool_calls — "
                "expected score_report to be called."
            )

        args: dict = json.loads(tool_calls[0].function.arguments)

    except Exception as exc:
        logger.warning(
            "judge_node: evaluation failed — %s: %s",
            type(exc).__name__,
            exc,
        )
        return _failure_return

    # ── Clamp scores ──────────────────────────────────────────────────────────
    accuracy = _clamp_score(args["accuracy_score"])
    groundedness = _clamp_score(args["groundedness_score"])
    helpfulness = _clamp_score(args["helpfulness_score"])
    conciseness = _clamp_score(args["conciseness_score"])

    # ── Weighted overall score ────────────────────────────────────────────────
    overall_score = round(
        accuracy * _ACCURACY_WEIGHT
        + groundedness * _GROUNDEDNESS_WEIGHT
        + helpfulness * _HELPFULNESS_WEIGHT
        + conciseness * _CONCISENESS_WEIGHT,
        1,
    )

    logger.info("judge_node: overall_score=%.1f", overall_score)

    # ── Score explanation (plain language, suitable for direct UI display) ────
    score_explanation = strip_em_dashes(
        f"Overall score: {overall_score}/5.0. "
        f"Accuracy ({accuracy}/5) and groundedness ({groundedness}/5) are weighted "
        f"more heavily (30% each) because factual consistency and traceability to "
        f"the source material matter most to report quality. "
        f"Helpfulness ({helpfulness}/5) and conciseness ({conciseness}/5) each "
        f"contribute 20%."
    )

    # ── Structured scores dict ────────────────────────────────────────────────
    scores = {
        "accuracy": {
            "score": accuracy,
            "reasoning": strip_em_dashes(args.get("accuracy_reasoning", "")),
        },
        "groundedness": {
            "score": groundedness,
            "reasoning": strip_em_dashes(args.get("groundedness_reasoning", "")),
        },
        "helpfulness": {
            "score": helpfulness,
            "reasoning": strip_em_dashes(args.get("helpfulness_reasoning", "")),
        },
        "conciseness": {
            "score": conciseness,
            "reasoning": strip_em_dashes(args.get("conciseness_reasoning", "")),
        },
    }

    return {
        "scores": scores,
        "overall_score": overall_score,
        "score_explanation": score_explanation,
        "status_message": "✅ Report evaluated.",
    }
