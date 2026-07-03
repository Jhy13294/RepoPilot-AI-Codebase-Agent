"""RepoPilot — task-oriented codebase agent (issue triage + patch proposal).

Package layout (see docs/architecture.md):
    agent/    planner, executor, critic, state machine, agent loop (Phase 3+)
    tools/    tool implementations + registry (Phase 1+)
    schemas/  Pydantic models: tool I/O, plans, traces (Phase 1+)
    safety/   risk policy, approval gate, path jail (Phase 1 / 5)
    services/ LLM client, repo manager (Phase 2+)
    storage/  SQLite persistence, trace store (Phase 3+)

Top-level modules planned: config.py (RP-P1-FEAT-001), cli.py (RP-P2), main.py (RP-P8).
"""

__version__ = "0.1.0"
