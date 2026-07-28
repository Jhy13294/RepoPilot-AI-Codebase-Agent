"""Agent core: the Planner-Executor-Critic state machine (docs/agent-design.md).

Available modules:
    citations.py  Deterministic validation for workspace file-line citations
    critic.py     step verification -> proceed | retry | replan
    executor.py   per-step constrained tool-calling micro-loop
    loop.py       orchestrating state machine with budget enforcement
    planner.py    task -> Plan; replanning on Critic escalation
    reporter.py   Reporter role with constrained JSON output and no trace side effects
    state.py      AgentState, PlanStep, Budgets, RunStatus
    tool_loop.py  Single ReAct-style tool-calling loop for repository questions
    usage.py      Shared usage accounting helpers for agent roles
"""
