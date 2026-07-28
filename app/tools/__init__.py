"""Tool implementations and the registry dispatch chokepoint (docs/tool-calling-design.md).

Available modules:
    apply_patch.py        apply_patch
    base.py               Base types shared by tool implementations and the registry
    get_file_tree.py      get_file_tree
    git_commit.py         git_commit
    git_create_branch.py  git_create_branch
    propose_patch.py      propose_patch
    read_file.py          read_file
    registry.py           ToolSpec, register/dispatch/to_llm_schema
    run_tests.py          run_tests
    search_code.py        search_code

Invariant: tool impl functions are module-private; the registry is the only caller.
"""
