# Groundwork — v2 Feature Backlog

A running list of ideas discussed during v1 build, deliberately deferred for later.

---

## 1. Specialist / persona agents (post-synthesis enrichment)
- Strategic Consultant, Marketing Expert, Financial Analyst / Risk Assessor lenses
- Triggered on-demand by the user after the core brief is generated, NOT a mandatory pipeline step
- Must be genuinely specialized, not just "Claude Sonnet + a persona system prompt" —
  needs real differentiation via: domain-specific RAG knowledge base, domain-specific tools,
  domain-specific eval criteria, and possibly a different/fine-tuned model where justified
- Decide architecture + UX once v1 has shipped and has real user feedback

## 2. Multi-user authentication & persistent user IDs
- v1 has no login — session memory + run history only, via ChromaDB, no user_id filtering
- v2 upgrade path is a one-line change: add `user_id` to the existing ChromaDB `where` filter
- Requires real auth (OAuth/JWT/session management) — deliberately out of scope for v1

## 3. Persistent memory beyond a single deployment
- Streamlit Community Cloud free tier does not persist ChromaDB across redeploys
- v2: migrate to Pinecone (or Supabase) for true persistent memory across redeploys
- v2: migrate hosting to Railway for more control as usage grows

## 4. Provider-aware prompt caching in `call_llm()`
- Current `cache_system_prompt` flag always emits Anthropic's `cache_control` block
- OpenAI auto-caches >1024 token prompts with no explicit block — different mechanism entirely
- v2: branch caching logic based on which provider/model is being called

## 5. LLM call retry logic
- `call_llm()` currently has no retry/backoff if a call times out or fails transiently
- v2: add retry with backoff, decide which error types are retryable, likely at the
  `call_llm()` level so every agent benefits without individual changes

## 6. Judge agent as a true second opinion (cross-provider)
- v1: judge uses Haiku (same provider as the writer — known self-grading bias risk, documented)
- v2: route judge through GPT-4o-mini via LiteLLM for genuine cross-model evaluation

## 7. Closed-loop automatic rewrite (Pattern A, reconsidered)
- v1 ships human-in-the-loop: user sees judge feedback, clicks "Regenerate" (capped retries)
- v2 (maybe): explore a fully automatic judge→rewrite loop for specific use cases,
  with safeguards against the self-enhancement bias / infinite loop risks already discussed

## 8. ChromaDB session collection cleanup
- v1 fix: each browser session gets its own isolated ChromaDB collection via
  session_id (st.session_state), preventing cross-user data bleed — this IS in v1
- v2 gap: nothing currently deletes old/abandoned session collections — every
  session ever created leaves a permanent collection on the server (slow storage leak)
- v2: scheduled cleanup job (e.g. delete collections older than N hours), or
  manual admin cleanup endpoint

## 9. Structured "found answer" signal in rag_node
- v1 fix: keyword-matching heuristic (declined_phrases list) distinguishes
  "Claude gave a real answer" from "Claude honestly declined" for the
  status_message shown to the user — works for common phrasing but not
  a structural guarantee
- v2: have Claude return structured output (e.g. a found_answer: bool field
  alongside the prose answer) so this distinction is guaranteed correct
  rather than inferred from string matching — same structured-output pattern
  already used in analyse_node's needs_more_search decision

*Add new ideas to this file as they come up during the v1 build — review when v1 ships.*
