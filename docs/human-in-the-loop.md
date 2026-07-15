# Human-in-the-Loop Safety

> Core claim: **the model is never trusted to police itself.** Risk policy is declared as data on
> each `ToolSpec`; enforcement happens in code, inside the registry dispatch path, where it cannot
> be prompted away.

## 1. Risk levels

| Level | Definition | Examples | Gate behavior |
|---|---|---|---|
| `low` | Read-only inside the path jail; no side effects | `get_file_tree`, `read_file`, `search_code` | Auto-execute, traced |
| `medium` | Produces artifacts but mutates nothing outside the run's own record | `propose_patch` (diff generation) | Auto-execute, traced **and** surfaced in the run timeline |
| `high` | Mutates workspace files, git state, or executes code | `apply_patch`, `run_tests`, `git_create_branch`, `git_commit` | **Blocked until explicit human approval** |

Classification rules of thumb: writes bytes → high; spawns a process → high; git mutation → high
(per project hard rules); anything ambiguous → the higher level.

## 2. Approval flow

```mermaid
sequenceDiagram
    participant EX as Executor
    participant RG as Registry.dispatch
    participant AG as ApprovalGate
    participant H as Human (CLI / API / UI)

    EX->>RG: apply_patch(args)
    RG->>AG: check(spec.risk_level == high)
    opt P7 planned: dedicated audit record
        AG->>AG: create ApprovalRequest(id, tool, rendered_args, rationale, risk)
    end
    AG-->>H: present: diff preview + agent rationale + risk badge
    Note over RG,H: P5 CLI blocks synchronously inside dispatch; no timeout or suspended run
    alt approved
        H-->>AG: approve(note?)
        AG-->>RG: ApprovalOutcome(approved=true)
        RG->>RG: execute tool
        RG-->>EX: ToolResult(ok=true)
    else denied
        H-->>AG: deny(reason?)
        AG-->>RG: ApprovalOutcome(approved=false, reason)
        RG-->>EX: ToolResult(ok=false, error=ApprovalDeniedError(reason))
    else timeout (P8 planned async path)
        Note over EX,H: run suspended in AWAITING_APPROVAL
        AG-->>AG: no answer before REPOPILOT_APPROVAL_TIMEOUT_S
        AG-->>RG: timed-out denial
        RG-->>EX: ToolResult(ok=false, error=ApprovalDeniedError("approval timed out"))
    end
    Note over AG: P7 planned: request + decision + actor + timestamps persisted (DB + trace)
```

Presentation requirements per request: tool name, risk badge, **human-readable rendering of args**
(for `apply_patch`: the actual diff), the agent's `rationale`, and run context (task, step intent).

## 3. Decision semantics

- **Approve** — optionally with a note; executes exactly the request presented (args are frozen at
  request time; any change requires a new request).
- **Deny (P5 synchronous CLI)** — dispatch returns a structured `ApprovalDeniedError`. On the
  first denied dispatch in a step, the Executor immediately terminates that step as incomplete;
  its terminal `tool_result` has `reason=approval_denied`, and its findings contain the gate's
  error message. The loop detects the denied `tool_call` in the trace and replans with the rejected
  diff (or the denied call args when no diff is present) plus an instruction to choose a different
  approach and not resubmit the identical call or diff. The human's free-text reason remains in the
  step findings but is not threaded into the Planner; dedicated reason auditing is P7-planned. Two
  denials in one run → forced REPORTING.
- **Timeout (P8 planned, async surfaces)** — equals deny with reason `"approval timed out"`.
  Default `REPOPILOT_APPROVAL_TIMEOUT_S=600`. The P5 CLI gate instead blocks synchronously for an
  interactive decision and has no approval timeout.

## 4. Surfaces

| Surface | Phase | Mechanism |
|---|---|---|
| CLI | P5 | Interactive prompt with colored diff (`rich`), y/n/note |
| REST API | P8 | `GET /approvals?status=pending`, `POST /approvals/{id}` `{decision, note}` — run is suspended (`AWAITING_APPROVAL`) meanwhile |
| Streamlit UI | P8 | Pending-approval panel with diff viewer and approve/deny buttons |

## 5. Non-bypassability (tested property, not a promise)

1. The gate lives **inside `registry.dispatch`** — there is no second code path to a tool impl.
   Tool impl functions are private to their modules; only the registry imports them.
2. Config cannot disable the gate for `high` (no such flag exists). P6 shipped `run_tests` as an
   always-gated high-risk tool; `auto_approve_tests_in_sandbox` is a **P9 planned** policy flag and
   did not ship in P6. If that sandbox policy is added, its automatic decision must still produce
   the dedicated `approval_decision` audit record planned for P7, with `actor="policy:sandbox"`.
3. **Tests must mock the gate, never bypass it** (project hard rule): unit tests patch
   `ApprovalGate.check` with an auto-approve fake and *assert it was called* for every high-risk
   dispatch. A dedicated test registers a dummy high-risk tool and asserts dispatch without
   approval is impossible.

## 6. Audit trail (P7 planned)

P5 has no `approval_requests` table and emits no dedicated `approval_request` or
`approval_decision` events. A denied dispatch is durably identifiable in the existing run trace by
its `tool_call` event (`error_type=ApprovalDeniedError`, with the presented `apply_patch` diff in
`args`); when the denial budget is exhausted, the final `report` event also records the run's
denial-specific `failure_summary`. The free-text denial reason in the Executor's step findings is
not yet structured as approval audit metadata.

P7 will store every request/decision twice: rows in `approval_requests` (queryable) and
`approval_request` / `approval_decision` events in the run trace (replayable). Fields: request id,
run id, step, tool, args hash + rendered form, rationale, risk, decision, actor, note, latencies.
The eval harness will compute **Human Approval Trigger Rate** from these events.
