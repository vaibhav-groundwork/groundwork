import subprocess
import os
import logging
from src.config import (
    LANGCHAIN_TRACING,
    LANGCHAIN_PROJECT,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
)
logger = logging.getLogger(__name__)

def _ensure_npm_packages() -> None:
    """
    Installs npm packages on first run if node_modules does not exist.
    Required for Streamlit Cloud which does not execute setup.sh automatically.
    """
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    node_modules = os.path.join(app_dir, "node_modules")
    if not os.path.exists(node_modules):
        logger.info("node_modules not found — running npm install...")
        result = subprocess.run(
            ["npm", "install"],
            cwd=app_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("npm install completed successfully.")
        else:
            logger.warning("npm install failed: %s", result.stderr)

def setup_tracing() -> None:
    """
    Configures LangSmith tracing for LiteLLM.
    Call once at app startup before any agent runs.
    Tracing is off by default — enabled via LANGCHAIN_TRACING_V2=true in .env
    """
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING
    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY or ""
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY or ""
    if LANGCHAIN_TRACING == "true":
        import litellm
        litellm.success_callback = ["langsmith"]
        litellm.failure_callback = ["langsmith"]
        print(f"LangSmith tracing enabled via LiteLLM → project: {LANGCHAIN_PROJECT}")
    else:
        print("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")
    _ensure_npm_packages()    


def call_llm(
    model: str,
    messages: list,
    system: str = None,
    max_tokens: int = 2048,
    cache_system_prompt: bool = False,
    tools: list = None,
    tool_choice: dict = None,
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

    tools / tool_choice: optional structured-output mode. When tools is
    provided, the model is constrained to return arguments matching the
    given schema rather than free-form text — used by judge_agent.py for
    reliable, guaranteed-shape scoring output (Level 3 structured output,
    vs the prompt-based JSON approach used elsewhere in this codebase).
    LiteLLM passes these through uniformly regardless of provider.
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
            # Branch here based on provider when a non-Anthropic model needs caching.
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

    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    response = completion(**kwargs)
    return response