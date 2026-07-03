# Memory Templates

Live copies (gitignored): `memory/project-context.md`, `memory/decisions.md`,
`memory/lessons-learned.md`. Update triggers: `docs/project-management.md` §3.
No secrets ever go into memory files.

---

## Template: `memory/project-context.md`

```markdown
# Project Context (session-resume snapshot)

- **Last updated:** 2026-07-03
- **Current phase:** P1 — read-only tool layer (docs/roadmap.md)
- **Active task:** RP-P1-FEAT-004 (see tasks/in-progress.md)
- **Next focus after that:** RP-P1-FEAT-005 read_file

## Standing constraints
- English-only code; approval gate never bypassed (even in tests — mock it);
  all paths through the jail; every tool has schemas + risk_level.

## Current state of the world
- What works: …
- What is stubbed: …
- Known debt: …

## Open questions
- …
```

## Template: `memory/decisions.md`

```markdown
# Decision Log (ADR-lite)

## D-003 · 2026-07-03 · SQLite + SQLAlchemy first, PostgreSQL later
- **Context:** need persistence for runs/approvals; zero-ops start preferred.
- **Decision:** SQLAlchemy 2.0 models on SQLite; connection URL from env.
- **Rationale:** ORM boundary makes the PG swap a config change; no server to babysit in demos.
- **Alternatives rejected:** raw sqlite3 (rewrite cost at swap), Postgres-now (ops drag).
- **Revisit when:** concurrent approvals from the web UI need real row locking.
```

## Template: `memory/lessons-learned.md`

```markdown
# Lessons Learned

## L-001 · 2026-07-05 · Windows symlink behavior breaks naive jail checks
- **Symptom:** path-jail test passed on paper, failed on win32.
- **Root cause:** `Path.resolve()` semantics differ for dangling symlinks on Windows.
- **Rule adopted:** jail tests must run in CI on both POSIX and Windows semantics; never rely on
  string-prefix path checks.
- **Cost:** ~2 h.
```
