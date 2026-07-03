"""Persistence: SQLAlchemy models and trace store (docs/architecture.md section 1).

Planned modules (Phase 3, RP-P3-FEAT):
    db.py           engine/session setup; models: Run, Step, ToolCall, ApprovalRequest, Report
    trace_store.py  append-only JSONL trace writer + replay reader
"""
