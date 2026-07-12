# eval/

The complete evaluation harness remains planned for Phase 7. Phase 4 ships a minimal issue-only
subset for measuring the issue-analysis pipeline. Design: [docs/evaluation.md](../docs/evaluation.md).

- `tasks.json` — task suite (5 types: repo_qa, bug_localization, bug_explanation, patch, recovery)
- `fixtures/` — includes the Phase 4 `buggy-calculator` fixture; broader coverage remains Phase 7
- `run_eval.py` — Phase 4 read-only runner for the issue-only subset; the full runner remains Phase 7
- `reports/` — raw run outputs, gitignored; curated results go into docs/evaluation.md §4
