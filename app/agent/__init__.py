"""Agent core: the Planner-Executor-Critic state machine (docs/agent-design.md).

Planned modules (Phase 3, RP-P3-FEAT):
    state.py     AgentState, PlanStep, Budgets, RunStatus
    planner.py   task -> Plan; replanning on Critic escalation
    executor.py  per-step constrained tool-calling micro-loop
    critic.py    step verification -> proceed | retry | replan
    loop.py      orchestrating state machine with budget enforcement
    prompts.py   role prompt templates assembled from shared sections
"""
