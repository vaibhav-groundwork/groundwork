import os
from src.config import (
    LANGCHAIN_TRACING,
    LANGCHAIN_PROJECT,
    ANTHROPIC_API_KEY,
)


def setup_tracing() -> None:
    """
    Configures LangSmith tracing for LiteLLM.
    Call once at app startup before any agent runs.
    Tracing is off by default — enabled via LANGCHAIN_TRACING_V2=true in .env
    """
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY or ""

    if LANGCHAIN_TRACING == "true":
        import litellm
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]
        print(f"LangSmith tracing enabled via LiteLLM → project: {LANGCHAIN_PROJECT}")
    else:
        print("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")


def call_llm(
    model: str,
    messages: list,
    system: str = None,
    max_tokens: int = 2048,
    cache_system_prompt: bool = False,
):
    """
    Single entry point for every LLM call in Groundwork — Claude, GPT, or
    any future provider — all routed through LiteLLM for consistent tracing.

    Every agent imports this function instead of calling Anthropic() or
    OpenAI() directly. This guarantees:
      - one consistent tracing path regardless of which model is used
      - one place to add caching, retries, or fallbacks later
      - swapping models is a one-line change at the call site, not a rewrite

    cache_system_prompt: marks the system prompt as cacheable (Anthropic
    prompt caching) — worth enabling for agents that run multiple sequential
    calls with the same system prompt, like the research loop.
    """
    from litellm import completion

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }

    if system:
        if cache_system_prompt:
            # NOTE: this cache_control block is Anthropic-specific syntax.
            # OpenAI auto-caches prompts >1024 tokens with no explicit block needed.
            # When judge_agent.py calls a non-Anthropic model, branch here based
            # on provider rather than always emitting Anthropic's cache format.
            kwargs["messages"] = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ] + messages
        else:
            kwargs["messages"] = [{"role": "system", "content": system}] + messages

    response = completion(**kwargs)
    return response