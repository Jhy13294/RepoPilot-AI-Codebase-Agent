# Agent Design

## 1. Why Planner–Executor (with Critic), not a single ReAct loop

| Property | Single ReAct loop | Planner–Executor + Critic (chosen) |
|---|---|---|
| Explainability | Reasoning buried in one long transcript | Explicit `Plan` object; UI can render progress |
| Recovery | Retry = "try again harder" | Critic verdicts drive targeted `retry` / `replan` |
| Evaluability | Hard to score partial progress | Per-step `success_check` → step-level metrics |
| Cost control | Context grows monotonically | Steps run as bounded micro-loops over a summarized state |

The Executor still behaves ReAct-style *within a step* (think → tool → observe), but the step is
bounded, verified, and disposable.

## 2. Run state machine

```mermaid
stateDiagram-v2
    [*] --> PLANNING
    PLANNING --> EXECUTING: plan produced
    EXECUTING --> AWAITING_APPROVAL: high-risk tool call
    AWAITING_APPROVAL --> EXECUTING: approved
    AWAITING_APPROVAL --> REPLANNING: denied / timeout
    EXECUTING --> VERIFYING: step finished
    VERIFYING --> EXECUTING: verdict=proceed (next step)
    VERIFYING --> EXECUTING: verdict=retry (same step, bounded)
    VERIFYING --> REPLANNING: verdict=replan
    REPLANNING --> EXECUTING: new plan, replan budget ok
    REPLANNING --> REPORTING: replan budget exhausted
    VERIFYING --> REPORTING: all steps done
    EXECUTING --> REPORTING: fatal error / budget exhausted
    REPORTING --> [*]: DONE / FAILED / CANCELLED
```

Terminal statuses: `DONE` (task achieved), `FAILED` (budgets/denials exhausted — with report),
`CANCELLED` (user abort). **Every terminal path emits a report**; there is no silent death.

## 3. Budgets (all from config, all enforced in the loop)

| Budget | Env var | Default | On exhaustion |
|---|---|---|---|
| Total steps | `REPOPILOT_MAX_STEPS` | 20 | → REPORTING |
| Replans per run | `REPOPILOT_MAX_REPLANS` | 3 | → REPORTING |
| Fix cycles (patch→test→fail→re-patch) | `REPOPILOT_MAX_FIX_CYCLES` | 2 | → REPORTING with best attempt |
| Per-tool timeout | `REPOPILOT_TOOL_TIMEOUT_S` | 60 | ToolError → Critic |
| Retries of an invalid tool call | (constant) | 2 | ToolError → Critic |

## 4. Prompt architecture

One system prompt template per role, assembled from shared sections:

1. **Role & mission** — "You are RepoPilot, a codebase task agent. You are not a chatbot. You
   ground every claim in tool evidence (file paths + line numbers)."
2. **Hard rules** — never invent paths; prefer reading before writing; high-risk tools will pause
   for human approval — plan around it; stop when `success_check` is met.
3. **Tool documentation** — generated from the registry (single source of truth), including risk
   levels so the model can plan approvals.
4. **Output contract** — role-specific structured output (see below), validated with Pydantic;
   on validation failure the error is fed back for up to 2 repair attempts.

| Role | Input | Structured output |
|---|---|---|
| Planner | task + repo overview + (on replan) failure summary | `Plan` (list of `PlanStep`) |
| Executor | current step + scratchpad + recent tool results | tool calls, then `StepResult` (findings + evidence) |
| Critic | step intent + `success_check` + `StepResult` + raw evidence | `Verdict{proceed|retry|replan, reason, hint}` |
| Reporter | full state | `FixReport` / `AnalysisReport` (markdown + structured fields) |

## 5. Context management

- **Tool result caps**: each tool truncates its payload (e.g. `read_file` window ≤ 400 lines,
  `search_code` ≤ 50 hits) and sets `meta.truncated` so the model knows to narrow its query.
- **Scratchpad summarization**: after each step, the Executor's findings are compressed into
  `AgentState.scratchpad`; old raw tool outputs are dropped from the prompt (still in the trace).
- **Evidence pinning**: file:line citations survive summarization so reports stay grounded.

## 6. Failure recovery (summary — full doc: `docs/failure-recovery.md`)

Recovery is a *routing decision on typed errors*, not a generic retry:
schema-invalid call → repair with validator message; empty search → broaden/兜底 strategies in the
hint; patch conflict → re-read + regenerate; test failure → Critic distills failing assertions into
the replan prompt; approval denied → the same call is **never retried**; the Planner must produce an
alternative or report. Every recovery event links to what it recovers from (`recovery_of` in trace).

## 7. LLM client behavior

- Agentic loop follows `stop_reason`: `tool_use` → dispatch + append `tool_result` (matching
  `tool_use_id`); `end_turn` → parse structured output; `max_tokens` → one continuation attempt;
  `refusal` → surface to user, mark run FAILED (never auto-retry a refusal); `pause_turn` → resume.
- Usage/cost recorded per call into the trace (`tokens_in/out`, `cost_usd`).
- Model quirks handled in the adapter, not in agent logic. E.g. `deepseek-v4-pro` (default):
  tool calls may intermittently arrive as plain text in `content` — adapter detects, re-parses,
  else re-asks (D-008); `claude-fable-5`: thinking always-on — omit the `thinking` param; no
  assistant prefill.
