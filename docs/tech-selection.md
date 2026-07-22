# Tech Selection

> Every choice below optimizes for four criteria, in order:
> **(1) implementation simplicity, (2) interview explainability, (3) extension headroom, (4) fit for LLM-application job requirements.**

## Summary table

| Concern | Choice | Rejected alternatives | Why |
|---|---|---|---|
| Language / runtime | Python ≥ 3.12 + `uv` | poetry, pip-tools | `uv` is the current de-facto standard: lockfile, fast sync, single tool for venv + deps. |
| Agent framework | **Native tool-calling, hand-rolled Planner–Executor loop** | LangGraph, LangChain agents | See detailed rationale below — this is the core interview asset. |
| LLM access | Provider-agnostic `LLMClient` — **DeepSeek via OpenAI-compatible adapter (default)**; Anthropic adapter first-class | hard-coding one vendor | Domestic access + low cost for a tool-loop-heavy agent; provider swap stays a `.env` change. |
| Default model | `deepseek-v4-pro` (config: `REPOPILOT_MODEL`) | `claude-opus-4-8` (kept for eval comparison) | Tuned for agentic coding; parallel/multi-turn tool calls; 1M context; ~$0.435/$0.87 per M tokens (≈10× cheaper than Opus). |
| API service | FastAPI + uvicorn | Flask, Django | Async, Pydantic-native, OpenAPI for free; industry default for LLM services. |
| Schemas | Pydantic v2 everywhere | dataclasses, attrs | One validation story for tool args, API bodies, and trace records. |
| Storage | SQLite via SQLAlchemy 2.0 | PostgreSQL, raw sqlite3, Mongo | Zero-ops start. Storage currently accepts a SQLite filesystem path rather than a configurable database URL, so PostgreSQL is not a config-only swap. |
| Trace log | JSONL per run + DB index | plain text logs | Machine-readable traces power the eval harness and the frontend timeline. |
| Frontend | Streamlit (Phase 8) | Next.js | Dev-speed favored per project goals; Next.js listed as a stretch upgrade. |
| Tests | pytest (unit / integration markers) | unittest | Standard. |
| Lint/format | ruff (lint **and** format) | ruff + black | One tool, zero config conflicts. |
| Types | mypy on `app/` | pyright | CI-friendly, widely known. |
| Packaging/deploy | Docker + Docker Compose | separate images, unmanaged host processes | One runtime image serves the API and console; Compose gates console startup on API health. |
| Diff handling | `unidiff` + `git apply --check` | hand-rolled patcher | Correctness matters; git validates before mutation. |
| MCP | Optional read-only stdio adapter | mutation tools, making MCP a core dependency | Exposes only the three read-only registry tools and remains an optional dependency. |

## Why a hand-rolled agent loop instead of LangGraph

The rationale is deliberately recorded here because it is the most-asked interview question:

1. **Interview depth.** The project's purpose is demonstrating agent engineering. Owning the loop
   means every design question — replan policy, budget enforcement, approval interception,
   trace format — has an answer *I wrote*, not a framework default I inherited.
2. **Safety enforcement must live in my code anyway.** The approval gate has to intercept tool
   dispatch deterministically. Wrapping a framework's executor to guarantee that is harder than
   writing a ~200-line dispatch layer where the guarantee is structural.
3. **Debuggability.** Failure recovery (Phase 6) needs precise control over what re-enters the
   context window. A hand-rolled loop makes the state machine explicit and testable.
4. **Dependency risk.** LangGraph's API surface moves fast; a portfolio repo should still build
   in a year.

**Revisit trigger:** if the project ever needs parallel branches, durable interrupts/resume across
processes, or multi-agent topologies, port the orchestrator to LangGraph. The `AgentState` +
`ToolRegistry` interfaces are deliberately framework-shaped so the port is a contained refactor of
`app/agent/loop.py` only.

## Provider abstraction contract

```python
class LLMClient(Protocol):
    def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[ToolSchema] | None = None,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
    # LLMMessage is app/schemas/llm_io.py's provider-neutral message.
    # ToolSchema is the OpenAI function-tool dict emitted by registry.to_llm_schema().
    # LLMResponse normalizes: text blocks, tool_use blocks, stop_reason
    # ("tool_use" | "end_turn" | "max_tokens" | "refusal" | "pause_turn"),
    # token usage, estimated cost, and recovered_tool_calls for deterministic salvage.
```

Adapter notes:
- **OpenAI-compatible / DeepSeek (default):** base_url `https://api.deepseek.com/v1`;
  OpenAI-shaped `tools=[{"type": "function", ...}]` pass through unchanged; loop while
  `finish_reason == "tool_calls"`; results return as `role="tool"` messages. **Known quirk**
  (deepseek-ai/DeepSeek-V3#1244): `deepseek-v4-pro` can intermittently emit tool calls as plain
  text inside `content` instead of the `tool_calls` field. The adapter only performs deterministic
  salvage: strip the whole content, peel one layer of a JSON code fence opened with three
  backticks plus `json` or a `<tool_call>` tag, and accept only JSON shaped exactly as
  `{name, arguments}` or an array of that shape. A match synthesizes `ToolCall` objects and sets
  `recovered_tool_calls=True`; no second API call happens in the adapter. Failed salvage is handled
  by the loop's invalid-tool-call repair path (≤2), so re-asks remain a loop/budget concern. The
  eval invalid-tool-call rate still counts this path and P2 contract tests pin it.
- **Anthropic (alternative):** the loop stays provider-neutral because the `LLMClient` boundary
  always receives OpenAI-shaped tool schemas from `registry.to_llm_schema()`. OpenAI-compatible
  adapters pass that shape through; Anthropic translates it to `{name, description, input_schema}`.
  Anthropic Messages also needs five structural translations: system messages become the top-level
  `system=` parameter; `role=tool` becomes a `role=user` message with a `tool_result` block and
  matching `tool_use_id`; returned `tool_use.input` is already a dict and is not JSON-decoded;
  usage reads `input_tokens`/`output_tokens`; `max_tokens` is required, so the adapter defaults to
  4096 when unset. The loop continues while `stop_reason == "tool_use"`. Fable/Claude specifics:
  omit `thinking`, do not use assistant prefill, and leave `temperature` unset in P2 because some
  modern Claude variants reject it with 400.
