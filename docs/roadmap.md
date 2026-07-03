# Roadmap — Phases 0–9

> Each phase ships something demonstrable. IDs: `RP-P<phase>-<TYPE>-<nnn>` (see
> `docs/project-management.md`). A phase is done only when its acceptance criteria all pass.

| Phase | Theme | Headline deliverable |
|---|---|---|
| P0 | Project bootstrap | Docs-first skeleton, this roadmap, toolchain config |
| P1 | Read-only tool layer | 3 tools + registry + path jail, fully tested |
| P2 | LLM client + basic loop | CLI Q&A over a repo via native tool calling |
| P3 | Agent core | Planner–Executor–Critic state machine + trace logger |
| P4 | Issue analysis | Issue → localization + explanation report |
| P5 | Patch + approval | Diff proposal, human gate, safe apply |
| P6 | Test & recovery | run_tests + fix-cycle recovery loop |
| P7 | Evaluation | Harness, fixtures, metrics report |
| P8 | Service + UI | FastAPI endpoints + Streamlit console |
| P9 | Polish & release | Docker, README demo, optional MCP adapter |

---

## Phase 0 — Bootstrap (docs before code)
- **Goal:** repo skeleton where every later PR has a home and a standard.
- **Tasks:** onboarding notes, code style, .gitignore, .env.example, all design docs, tasks/ + memory/
  templates, bilingual READMEs, pyproject/Docker/LICENSE, `git init` + first commit.
- **Acceptance:** `uv sync` succeeds; `ruff check` clean on empty skeleton; every doc listed in
  README Learning Notes exists; tasks/memory not tracked by git; first commit follows convention.
- **Risks:** over-documenting before validating with code — mitigated by keeping P1 small.
- **IDs:** RP-P0-DOCS-001…004, RP-P0-FEAT-001…002, RP-P0-TEST-001.

## Phase 1 — Read-only tools
- **Goal:** trustworthy hands before a brain.
- **Tasks:** config loader; `ToolResult/ToolError/ToolMeta`; path jail; registry with risk levels;
  `get_file_tree`, `read_file`, `search_code`; unit tests incl. jail escapes.
- **Acceptance:** all tools return valid envelopes on fixture repo; path traversal
  (`../`, absolute, symlink) rejected with `PathJailError` and traced; truncation flags verified;
  `ruff` + `mypy` + `pytest` green.
- **Risks:** Windows path edge cases (drive letters, symlink perms) — test on win32 explicitly.
- **IDs:** RP-P1-FEAT-001…006, RP-P1-SAFE-001, RP-P1-TEST-001…, RP-P1-DOCS-001.

## Phase 2 — LLM client + single-loop tool calling
- **Goal:** first end-to-end intelligence: repo Q&A from the CLI.
- **Tasks:** provider-agnostic `LLMClient` (DeepSeek/OpenAI-compatible default + Anthropic
  adapter); agentic loop on normalized stop reasons; plain-text tool-call repair
  (`deepseek-v4-pro` quirk, D-008) with contract test; arg-repair on validation failure;
  usage/cost accounting; `repopilot ask` CLI.
- **Acceptance:** `repopilot ask "where is date parsing?"` answers with ≥1 tool call and real
  citations on the fixture repo; invalid tool args repaired ≤2 attempts; per-call usage in trace;
  switching provider via `.env` only.
- **Risks:** provider schema drift — contract tests per adapter.
- **IDs:** RP-P2-FEAT-00x, RP-P2-TEST-00x.

## Phase 3 — Agent core (Planner/Executor/Critic)
- **Goal:** the state machine from `docs/agent-design.md`, running.
- **Tasks:** `AgentState` + persistence; Planner/Executor/Critic prompts + structured outputs;
  loop with budgets; JSONL trace logger + SQLite runs/steps tables; `repopilot run` CLI.
- **Acceptance:** plan visible in trace; forced low budget (`MAX_STEPS=3`) terminates with report;
  Critic `retry`/`replan` transitions observable in a scripted scenario; replay tool prints a
  readable timeline from JSONL.
- **Risks:** over-engineered state — keep `AgentState` flat, no framework.
- **IDs:** RP-P3-FEAT-00x, RP-P3-TEST-00x, RP-P3-REFACTOR-00x.

## Phase 4 — Issue analysis pipeline
- **Goal:** the first real product capability: triage.
- **Tasks:** `get_repo_overview`; `AnalysisReport` schema (suspected files, root cause, confidence,
  evidence citations, suggested fix direction); analysis prompt strategy; 3 seeded fixture issues.
- **Acceptance:** on 3 seeded issues, gold file in top-3 suspects ≥2/3; every claim carries a
  file:line citation that actually exists (citation validator).
- **IDs:** RP-P4-FEAT-00x, RP-P4-EVAL-001.

## Phase 5 — Patch generation + approval gate
- **Goal:** mutations, made safe.
- **Tasks:** `propose_patch`, `apply_patch` (git-validated, work branch), `git_create_branch`;
  **ApprovalGate** + CLI approval UX (rich diff); denial semantics; gate tests (mocked, per hard
  rule).
- **Acceptance:** applying without approval is impossible (dedicated test); denial → alternative
  plan or report, never identical resubmission; diff renders correctly in CLI; patch lands on
  `repopilot/fix-<run_id>` branch only.
- **Risks:** diff drift → `git apply --check` + re-read-regenerate path (P6 exercises it).
- **IDs:** RP-P5-FEAT-00x, RP-P5-SAFE-001…, RP-P5-TEST-00x.

## Phase 6 — Test execution + failure recovery
- **Goal:** close the loop: patch → test → learn → re-patch.
- **Tasks:** `run_tests` (structured result parse), sandbox policy flag; fix-cycle loop with
  `MAX_FIX_CYCLES`; LoopGuard (identical-call block); failure-summary prompts; recovery traces.
- **Acceptance:** fixture with seeded regression: agent recovers to green within 2 cycles;
  injected patch-conflict scenario recovers via re-read+regenerate; exhausted budgets produce the
  full graceful-failure report.
- **IDs:** RP-P6-FEAT-00x, RP-P6-BUG-00x, RP-P6-TEST-00x.

## Phase 7 — Evaluation harness
- **Goal:** numbers (see `docs/evaluation.md`).
- **Tasks:** ≥2 fixture repos, ≥8 tasks across 5 types in `eval/tasks.json`; `run_eval.py`;
  scorers; markdown report generator; first curated results into docs.
- **Acceptance:** one command produces the full metrics report; approval trigger rate = 100% on
  high-risk dispatches; repeat-3 variance reported.
- **IDs:** RP-P7-EVAL-00x, RP-P7-FEAT-00x.

## Phase 8 — FastAPI service + Streamlit console
- **Goal:** demo-able product surface.
- **Tasks:** `POST /runs`, `GET /runs/{id}` (status + trace), `GET/POST /approvals`; SSE or polling
  for live trace; Streamlit: new-task form, live timeline, pending-approval panel with diff viewer,
  report view.
- **Acceptance:** full fix flow driven from the browser incl. approve/deny; API documented via
  OpenAPI; run survives service restart (state from DB).
- **IDs:** RP-P8-FEAT-00x, RP-P8-TEST-00x.

## Phase 9 — Packaging, demo, MCP (optional)
- **Goal:** ship it as a portfolio piece.
- **Tasks:** Dockerfile + compose polish; demo GIF/script (`examples/`); README final pass (EN+ZH);
  `git_commit` tool; optional MCP server exposing the read-only tools; postgres compose profile.
- **Acceptance:** `docker compose up` → working UI from scratch on a clean machine; README
  quickstart verified end-to-end; (if MCP) Claude Desktop can call `search_code`.
- **IDs:** RP-P9-FEAT-00x, RP-P9-DOCS-00x.
