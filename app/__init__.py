"""RepoPilot — task-oriented codebase agent (issue triage + patch proposal).

Package layout (see docs/architecture.md):
    agent/    planner, executor, critic, state machine, agent loop
    api/      FastAPI run service, event feed, approval endpoints
    console/  Streamlit HTTP-only run archive and control surface
    mcp/      read-only stdio Model Context Protocol server
    safety/   risk policy, approval gate, path jail, loop guard
    schemas/  Pydantic models: tool I/O, plans, traces
    services/ provider-agnostic LLM client
    storage/  SQLite persistence, trace store
    tools/    tool implementations + registry

Top-level modules: config.py (settings), cli.py (console entry points).
"""

__version__ = "0.1.0"
