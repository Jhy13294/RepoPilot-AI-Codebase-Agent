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
    opt P8 planned: request event + queryable request storage
        AG->>AG: create approval_request event + approval_requests row
    end
    AG-->>H: present: diff preview + agent rationale + risk badge
    Note over RG,H: P5 CLI blocks synchronously inside dispatch; no timeout or suspended run
    alt approved
        H-->>AG: approve(note?)
        AG-->>RG: ApprovalOutcome(approved=true)
        RG->>RG: emit approval_decision before tool_call
        RG->>RG: execute tool
        RG-->>EX: ToolResult(ok=true)
    else denied
        H-->>AG: deny(reason?)
        AG-->>RG: ApprovalOutcome(approved=false, reason)
        RG->>RG: emit approval_decision before tool_call
        RG-->>EX: ToolResult(ok=false, error=ApprovalDeniedError(reason))
    else timeout (P8 planned async path)
        Note over EX,H: run suspended in AWAITING_APPROVAL
        AG-->>AG: no answer before REPOPILOT_APPROVAL_TIMEOUT_S
        AG-->>RG: timed-out denial
        RG->>RG: emit approval_decision before tool_call
        RG-->>EX: ToolResult(ok=false, error=ApprovalDeniedError("approval timed out"))
    end
    Note over RG: P7 shipped: decision + actor + reason + timestamp in trace
    Note over AG: P8 planned: persist request-side fields as an event + DB row
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
  step findings and is also persisted as `approval_decision.reason`; it is not threaded into the
  Planner. Two denials in one run → forced REPORTING.
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
   the P7-shipped `approval_decision` audit record, with `actor="policy:sandbox"`.
3. **Tests must mock the gate, never bypass it** (project hard rule): unit tests patch
   `ApprovalGate.check` with an auto-approve fake and *assert it was called* for every high-risk
   dispatch. A dedicated test registers a dummy high-risk tool and asserts dispatch without
   approval is impossible.

## 6. Audit trail

P7 shipped one dedicated `approval_decision` trace event for every high-risk gate outcome. The
registry emits it before the corresponding `tool_call`, with run id and timestamp plus the tool,
risk, decision, gate-owned actor, and reason. A missing gate fails closed and emits
`decision="denied"`, `actor="system"`; CLI free-text denial notes are carried in `reason`. The eval
layer derives **Human Approval Trigger Rate** from these decision events.

The request half remains P8 work. There is still no emitted `approval_request` event, no queryable
`approval_requests` table, and no persisted request-side record containing request id, step,
rendered args, rationale, or latency. The original two-copy design therefore remains incomplete:
P7 delivered decision-level trace auditing, while P8 is responsible for the request event and the
DB-backed request/decision projection needed by async API and UI surfaces.
