"""Safety layer: risk policy, approval gate, path jail (docs/human-in-the-loop.md).

Available modules:
    approval.py        ApprovalGate + ApprovalRequest lifecycle
    async_approval.py  Durable approval coordination for worker-thread tool dispatch
    loop_guard.py      In-memory guard against unsafe consecutive tool calls
    path_jail.py       workspace-relative path resolution, escape rejection

Hard rule: high-risk dispatch blocks on this layer; no bypass flag exists.
Tests mock the gate and assert it was called - they never route around it.
"""
