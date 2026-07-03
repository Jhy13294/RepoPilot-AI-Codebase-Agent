# eval/

Evaluation harness (lands in Phase 7 — `RP-P7-*`). Design: [docs/evaluation.md](../docs/evaluation.md).

- `tasks.json` — task suite (5 types: repo_qa, bug_localization, bug_explanation, patch, recovery)
- `fixtures/` — small self-contained repos with seeded bugs + pytest suites (Phase 7)
- `run_eval.py` — harness entry point (Phase 7)
- `reports/` — raw run outputs, gitignored; curated results go into docs/evaluation.md §4
