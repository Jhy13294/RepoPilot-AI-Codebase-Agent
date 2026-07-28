# Code Style Guide

> Scope: all code under `app/`, `tests/`, `eval/`, `examples/`.

## 1. Language policy

| Artifact | Language |
|---|---|
| Code, comments, docstrings, identifiers, log messages | **English only** |
| Commit messages | English |
| `docs/` content | English (Chinese annotations allowed) |
| README | Two files: `README.md` (EN) + `README.zh-CN.md` (ZH) |

## 2. Toolchain

- **Python ≥ 3.12**, managed with `uv` (`uv sync`, `uv run`).
- **ruff** for both linting and formatting (`ruff check`, `ruff format`). No black — one tool, zero conflicts.
  - Enabled rule sets: `E, F, I, UP, B, SIM, RUF` (line length 100).
- **mypy** on `app/` (`disallow_untyped_defs = true`).
- **pytest** with `tests/unit` and `tests/integration` markers.

## 3. Typing & schemas

- Every function is fully type-annotated. No bare `dict`/`Any` at module boundaries.
- All tool inputs/outputs are **Pydantic v2 models** defined in `app/schemas/`.
- Every dispatched tool call returns the shared envelope:

```python
class ToolResult(BaseModel):
    """Uniform envelope returned by every tool execution."""

    ok: bool
    data: BaseModel | None = None          # tool-specific payload schema
    error: ToolError | None = None         # structured error (type + message)
    meta: ToolMeta                          # latency_ms, truncated, tool_name
```

## 4. Docstrings

Google style, English, focused on contract rather than narration:

```python
def read_file(args: ReadFileArgs, ctx: ToolContext) -> ReadFilePayload:
    """Read a text file inside the registered workspace.

    Args:
        args: Path (workspace-relative) and optional line range.
        ctx: Execution context carrying the workspace root and run id.

    Returns:
        ReadFilePayload with the selected text window; `truncated` is set
        when the returned content hits the line or size cap.

    Raises:
        ToolFailure for structured file failures such as missing or binary
        files. Path-jail failures may also propagate to dispatch; dispatch
        converts these boundary failures into a ToolError envelope.
    """
```

## 5. Errors & logging

- `ToolError` and `ToolFailure` have similar names but different roles. `ToolError` is
  structured data in `app/schemas/tool_io.py`: `type: ErrorType` (`InvalidArgsError`,
  `PathJailError`, `NotFoundError`, `BinaryFileError`, `ToolTimeoutError`,
  `PatchApplyError`, `TestExecutionError`, `ApprovalDeniedError`, `InternalToolError`),
  `message`, and optional `details`. It travels inside the `ToolResult` envelope returned by
  dispatch. `ToolFailure` is the exception type in `app/tools/base.py` that tool handlers raise
  for structured failures; `registry.dispatch` catches it at the tool boundary and converts it
  into a `ToolError`. The agent loop sees the envelope and decides what to do next.
- No `print()` in `app/`. Use the structured logger (JSON lines); every log record carries `run_id`.
- No bare `except:`; never swallow an exception without recording it in the trace.

## 6. Safety invariants (non-negotiable)

- Every tool declares `risk_level: low | medium | high` in its registration.
- High-risk tools are wired through `app/safety/approval.py`; the policy engine enforces
  this **in code** — the LLM is never trusted to self-police.
- All filesystem paths are resolved through the sandbox path-jail before use.

## 7. Commits & branches

- **Conventional Commits** + construction ID trailer:
  - `feat(tools): implement search_code with regex support [RP-P1-FEAT-003]`
  - `fix(agent): stop replan loop on repeated tool failure [RP-P6-BUG-001]`
  - Types: `feat, fix, refactor, docs, test, chore, perf, ci`
- Branch naming: `feat/rp-p1-feat-003-search-code`, `fix/rp-p6-bug-001-replan-loop`.
- One task ID per commit where practical; the ID makes `git log` cross-reference the task board.

## 8. Tests

- Test naming: `test_<unit>__<behavior>` (e.g. `test_read_file__rejects_path_outside_workspace`).
- Every tool ships with: happy path, at least one failure case, and (if applicable) a path-jail test.
- Approval gate has dedicated tests asserting high-risk calls are intercepted 100% of the time.
