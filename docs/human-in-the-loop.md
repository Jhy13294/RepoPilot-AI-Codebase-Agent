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
    AG->>AG: under one coordinator lock: create approval_requests row
    AG->>AG: emit approval_request metadata event
    AG->>AG: overlay AWAITING_APPROVAL in SQLite and register waiter
    AG-->>H: present: diff preview + agent rationale + risk badge
    Note over AG,H: The event exposes only request_id, tool_name, and risk_level; full args stay in the DB row
    Note over EX,H: P5 CLI blocks in dispatch; P8 parks this worker in check() while API/UI readers see the DB overlay
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
    end
    Note over RG: P7 shipped: decision + actor + reason + timestamp in trace
    Note over EX,H: P8 has no approval timeout or cross-restart automatic resume
```

The coordinator lock serializes creation of the durable row, append of the safe metadata event,
the temporary status overlay, and waiter registration so an immediate HTTP decision cannot become
a lost wakeup. This is an in-process concurrency boundary, not an atomic transaction spanning
SQLite and JSONL. `AWAITING_APPROVAL` is a read-side SQLite projection: the loop's in-memory
`AgentState` remains `EXECUTING`, and the same parked worker continues after a decision. Its next
normal state save supersedes the projection.

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
- **Timeout (deferred)** — the design intent is to treat expiry as a denial with reason
  `"approval timed out"`, but no timeout scheduler or `REPOPILOT_APPROVAL_TIMEOUT_S` setting exists
  in `app/`. Both the P5 CLI gate and P8 async gate currently wait for an explicit decision.

## 4. Surfaces

| Surface | Phase | Mechanism |
|---|---|---|
| CLI | Shipped in P5 | Interactive prompt with colored diff (`rich`), y/n/note |
| REST API | Shipped in P8 | `GET /approvals?run_id=<id>` (optional filter; pending requests only) and `POST /approvals/{id}` `{decision, note}`. The calling worker remains parked while `AWAITING_APPROVAL` is exposed as a read-side DB projection. |
| Streamlit UI | Shipped in P8 | HTTP-only pending-approval panel with full diff viewer and approve/deny buttons |

## 5. Non-bypassability (tested property, not a promise)

1. The gate lives **inside `registry.dispatch`** — there is no second code path to a tool impl.
   Tool impl functions are private to their modules; only the registry imports them.
2. Config cannot disable the gate for `high` (no such flag exists). P6 shipped `run_tests` as an
   always-gated high-risk tool; `auto_approve_tests_in_sandbox` did not ship in P9 and is now
   **unscheduled**. If that sandbox policy is added, its automatic decision must still produce
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

P8 completed the request half. `ApprovalCoordinator.request()` creates a durable
`approval_requests` row containing the validated args, then emits an `approval_request` event with
only the safe `request_id`, `tool_name`, and `risk_level` fields. `GET /approvals` queries pending
rows, optionally filtered by `run_id`. A decision uses a first-write-only conditional update
(`status = 'pending'`); a concurrent or repeated second decision raises
`ApprovalRequestAlreadyDecidedError` and the API returns HTTP 409. The registry remains the sole
emitter of the corresponding `approval_decision` event.

The remaining boundaries are explicit. There is no `GET /approvals/{id}` read endpoint, so a resolved
request's full diff is returned by the decision response but cannot be fetched again through HTTP;
the console's `decision_records` tuple is session memory and disappears on reload. There is no
approval timeout, no automatic continuation of a parked run after service restart, and no live
waiter to wake once the original worker process is gone. SQLite and JSONL preserve the request and
decision records, but durable resume remains separate deferred work.

## 7. Trust boundary & known limitations

The HTTP API and Streamlit console have no authentication or authorization. Both can be used to
approve high-risk actions, so exposing either to an untrusted network gives approval authority to
anyone who can reach it. They assume one operator on a trusted local network. For local container
deployments, publish the API as `127.0.0.1:8000:8000` and the console as
`127.0.0.1:8501:8501`; the processes inside the containers must still listen on `0.0.0.0`.
Multi-tenant deployments require a token or another real authentication and authorization layer.

Approval waits have no timeout. A worker remains blocked until a decision arrives, so a forgotten
run can occupy its worker thread indefinitely.

`LoopGuard` blocks identical adjacent calls and, after a successful call, an adjacent effective
repeat that differs only in rationale. Its window is only the most recent allowed call, so an
`A-B-A` alternation is not blocked. Guard state is not shared across runs.

The target repository must be trusted. `run_tests` spawns the configured test command with the
repository as its working directory, so that repository's own `conftest.py`, pytest plugins, and
test modules execute with the operator's privileges. The absence of a generic shell-execution tool
limits what the agent can choose to run; it does not sandbox what the repository's test suite does
once a human has approved running it. Point RepoPilot only at repositories whose test code you
would already run yourself.

There is no dedicated mitigation for prompt injection. Issue text, file contents, and test output
all enter the model's context, and any of them can carry instructions aimed at the agent. The
approval gate is the only barrier: every high-risk action is shown to a human with its concrete
arguments before it runs, so an injected instruction still has to survive that review. Read tools
are not gated, so an injected instruction can still influence which files the agent reads inside
the path jail without producing any prompt.
