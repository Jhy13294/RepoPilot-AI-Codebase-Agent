"""Safety layer: risk policy, approval gate, path jail (docs/human-in-the-loop.md).

Planned modules:
    path_jail.py  workspace-relative path resolution, escape rejection (RP-P1-SAFE-001)
    risk.py       risk-level policy helpers (Phase 5)
    approval.py   ApprovalGate + ApprovalRequest lifecycle (Phase 5, RP-P5-SAFE)

Hard rule: high-risk dispatch blocks on this layer; no bypass flag exists.
Tests mock the gate and assert it was called - they never route around it.
"""
