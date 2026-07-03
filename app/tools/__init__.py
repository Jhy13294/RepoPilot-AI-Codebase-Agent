"""Tool implementations and the registry dispatch chokepoint (docs/tool-calling-design.md).

Planned modules:
    registry.py   ToolSpec, register/dispatch/to_llm_schema (RP-P1-FEAT-003)
    fs_read.py    get_file_tree, read_file (RP-P1-FEAT-004/005)
    fs_search.py  search_code (RP-P1-FEAT-006)
    repo_info.py  get_repo_overview (Phase 4)
    patch.py      propose_patch, apply_patch (Phase 5)
    tests_run.py  run_tests (Phase 6)
    git_ops.py    git_create_branch, git_commit (Phase 5 / 9)

Invariant: tool impl functions are module-private; the registry is the only caller.
"""
