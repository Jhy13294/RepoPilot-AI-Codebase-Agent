# Project Management Methodology

RepoPilot is built with a lightweight, auditable solo-dev process. The live working files
(`tasks/`, `memory/`) are **gitignored** — they contain raw in-progress notes. This document plus
the templates in `docs/internal-templates/` are the public, stable description of the method.

## 1. Construction ID system

Every unit of work gets an ID before code is written:

```
RP-P<phase>-<TYPE>-<nnn>
│    │       │      └─ 3-digit sequence within (phase, type), never reused
│    │       └─ FEAT | BUG | REFACTOR | DOCS | TEST | EVAL | SAFE
│    └─ 0–9, matching docs/roadmap.md
└─ project prefix
```

- `SAFE` is reserved for safety-critical work (path jail, approval gate) so those changes are
  greppable in history: `git log --grep=SAFE`.
- IDs appear in: the task board entry, the branch name, and the commit trailer —
  `feat(tools): implement search_code with regex support [RP-P1-FEAT-003]`.
- One ID per commit where practical. An ID is closed by moving its entry to `done` with date +
  commit hash.

## 2. Task lifecycle

```
tasks/todo.md ──(start; at most 1–2 active)──▶ tasks/in-progress.md ──(acceptance met)──▶ tasks/done.md
                                                     │
                                                     └─ blocked? note the blocker inline, pick nothing new
```

Rules:
1. New work may only enter via `todo.md` with an ID and acceptance criteria ("done when …").
2. `in-progress.md` holds the *active* task with a running work log — resumed first every session
   (see onboarding notes session checklist).
3. `done.md` is append-only: ID, title, date, commit hash, and a one-line outcome.
4. Scope discovered mid-task becomes a **new** todo entry, not silent scope creep.

## 3. Memory files (`memory/`)

| File | Purpose | Update trigger |
|---|---|---|
| `project-context.md` | Current phase, active constraints, next focus — the "resume from here" snapshot | End of each working session |
| `decisions.md` | Numbered ADR-style log: context → decision → rationale → revisit trigger | Any choice a future session (or interviewer) might question |
| `lessons-learned.md` | Numbered lessons: symptom → root cause → rule adopted | Any bug/dead-end that cost > 30 minutes |

Nothing secret goes into memory files; they are gitignored for signal-to-noise, not secrecy —
polished versions of decisions and lessons are promoted into `docs/`.

## 4. Definition of Done (per task)

1. Acceptance criteria in the todo entry pass.
2. `ruff check` + `ruff format --check` + `mypy` + `pytest` green.
3. New tool/behavior covered by tests (happy + ≥1 failure path; jail test if paths involved).
4. Docs updated if a public contract changed.
5. Board updated (todo → done) and, if applicable, `memory/decisions.md` entry written.
6. Conventional commit with construction ID trailer.

## 5. Why this is public

The method itself is a portfolio artifact: it demonstrates that AI-assisted development still
needs — and rewards — explicit scoping, auditable decisions, and safety-first review. Templates:
see `docs/internal-templates/`.
