# Task Board Templates

Live copies (gitignored): `tasks/todo.md`, `tasks/in-progress.md`, `tasks/done.md`.
Lifecycle and rules: `docs/project-management.md` §2.

---

## Template: `tasks/todo.md`

```markdown
# TODO

> Pull from the top of the current phase. Every entry needs: ID, title, acceptance criteria.
> New scope discovered mid-task lands here, never silently into the active task.

## Phase 1 — Read-only tools

### RP-P1-FEAT-004 · Implement get_file_tree
- **What:** directory tree tool per docs/tool-calling-design.md §3.1
- **Done when:** returns TreePayload on fixture repo; depth/entry caps + truncated flag verified;
  jail escape attempt returns PathJailError; unit tests green.
- **Depends on:** RP-P1-SAFE-001, RP-P1-FEAT-003
- **Est:** 0.5 d

### RP-P1-… · …
```

## Template: `tasks/in-progress.md`

```markdown
# IN PROGRESS

> At most 1–2 entries. Resume this file first every session, before picking new work.

## RP-P1-FEAT-004 · Implement get_file_tree
- **Started:** 2026-07-05
- **Branch:** feat/rp-p1-feat-004-get-file-tree
- **Acceptance:** (copied from todo)
- **Work log:**
  - 2026-07-05: args schema + happy path done; symlink case on Windows behaves oddly — investigating.
- **Blockers:** none
- **Next concrete step:** add symlink-escape test, then truncation cap test.
```

## Template: `tasks/done.md`

```markdown
# DONE

> Append-only. Newest on top. One line of outcome; details live in the commit.

| ID | Title | Date | Commit | Outcome |
|---|---|---|---|---|
| RP-P1-FEAT-004 | get_file_tree | 2026-07-06 | abc1234 | Tree tool with caps + jail tests (7 tests) |
| RP-P0-DOCS-001 | Project docs skeleton | 2026-07-03 | def5678 | 12 design docs + templates |
```
