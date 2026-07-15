# Tool Calling Design

## 1. Registry pattern

Every tool is registered with a complete spec — the registry is the **only** dispatch path:

```python
class ToolSpec(BaseModel):
    name: str                          # snake_case, stable API
    description: str                   # written for the LLM, includes when-NOT-to-use
    args_schema: type[BaseModel]
    returns_schema: type[BaseModel]
    risk_level: Literal["low", "medium", "high"]
    timeout_s: int = 60
    examples: list[ToolExample] = []   # few-shot material + doc generation

registry.register(spec, impl)          # impl: Callable[[ArgsT, ToolContext], PayloadT]
registry.to_llm_schema()               # JSON schema list for the LLM `tools` parameter
registry.dispatch(name, raw_args, ctx) # validate → optional loop guard → gate → execute → envelope → trace
```

`dispatch` responsibilities, in order: unknown-tool check → Pydantic validation of `raw_args`
(failure returns a structured hint, never crashes the run) → optional `LoopGuard` check →
**approval gate for high risk** → timeout-guarded execution → payload truncation → `ToolResult`
envelope → trace append. The P6 fix registry opts into `LoopGuard`; it hashes the tool name and
validated arguments and blocks only an identical consecutive call within the same run, before the
call can prompt for approval (D-041).

The handler returns a typed payload model; `dispatch` wraps it in `ToolResult`. `returns_schema`
is a declarative contract for documentation and future response-format hints. P1 dispatch does
not validate handler output against it at runtime; P1 handlers directly construct typed payloads,
so their shape is guaranteed by model construction. Runtime validation can be added later if the
tradeoff becomes worthwhile.

## 2. Tool roster by phase

| Tool | Phase | Risk | Approval | Purpose |
|---|---|---|---|---|
| `get_file_tree` | P1 | low | auto | Directory tree of the workspace (depth/limit caps) |
| `read_file` | P1 | low | auto | Read file content with optional line range |
| `search_code` | P1 | low | auto | Regex/literal search across the repo |
| `propose_patch` | P5 | medium | auto + logged | Produce a unified diff **without writing anything** |
| `apply_patch` | P5 | **high** | **required** | Validate (`git apply --check`) then apply a diff |
| `run_tests` | P6 | **high** | **required**¹ | Run the repo's test command, capture structured results |
| `git_create_branch` | P5 | **high** | **required** | Create/switch work branch before patching |
| `git_commit` | P9 | **high** | **required** | Commit applied changes with conventional message |

`get_repo_overview` was considered for Phase 4 but deferred and not adopted. The accepted baseline
localized all 6/6 runs at rank 1 with the existing read-only tools, so measure-first did not justify
adding it to the roster.

¹ As built in P6, `run_tests` always passes through the approval gate. A policy flag named
`auto_approve_tests_in_sandbox` is a **P9 planned** sandbox design and did not ship in P6.

**Deliberately excluded:** a generic `run_shell` tool. Arbitrary shell is the single biggest attack
surface and the least explainable capability; every needed action is a typed, auditable tool
instead. (Interview talking point.)

## 3. Phase-1 tool specs (full)

### 3.1 `get_file_tree` — risk: low

| | |
|---|---|
| Purpose | Give the agent repo topology without flooding context. |
| Args | `path: str = "."` (workspace-relative root) · `max_depth: int = 4 (1–8)` · `max_entries: int = 500` · `include_hidden: bool = False` |
| Returns | `TreePayload{root: str, entries: list[TreeEntry{path, kind: file|dir, size_bytes}], truncated: bool}` |
| Behavior | Traversal is deterministic breadth-first by directory level, with children sorted by name inside each directory. Hidden entries follow dotfile naming and are filtered unless `include_hidden=True`. Symlinks are always skipped: not listed and not descended into, so traversal cannot become a second path around the jail. Directory `size_bytes` is `0`; all paths are workspace-relative POSIX paths. |
| Failure cases | path outside jail → `PathJailError`; path missing → `NotFoundError`; entry cap hit → `ok=True, truncated=True` |
| Example | args `{"path": "src", "max_depth": 2}` → `{"ok": true, "data": {"root": "src", "entries": [{"path": "src/app.py", "kind": "file", "size_bytes": 2143}, ...], "truncated": false}}` |

### 3.2 `read_file` — risk: low

| | |
|---|---|
| Purpose | Ground analysis in real file content. |
| Args | `path: str` · `start_line: int = 1` · `end_line: int \| None` (window cap 400 lines, size cap 200 KB) |
| Returns | `ReadFilePayload{path, content, start_line, end_line, total_lines, truncated}` |
| Behavior | The line window is a 1-based inclusive range. `end_line` beyond EOF is silently clamped and does not set `truncated`; `truncated=True` only means real returned content hit the 400-line or 200 KiB cap. Lines are never split to fit the byte cap. Empty files succeed with `content=""`, `total_lines=0`, and `end_line=0`. Binary detection has two layers: a NUL-byte sample check and strict UTF-8 decode; either failure returns `BinaryFileError`. |
| Failure cases | jail violation; missing path or directory path; binary file → `BinaryFileError` (suggests `get_file_tree`); `start_line` beyond EOF → `InvalidArgsError`; content cap hit → `ok=True, truncated=True` |
| Example | `{"path": "app/utils/date.py", "start_line": 40, "end_line": 80}` → content with real line numbers for citation |

### 3.3 `search_code` — risk: low

| | |
|---|---|
| Purpose | Locate symbols/patterns; the primary navigation tool. |
| Args | `query: str` · `regex: bool = False` · `glob: str \| None` (e.g. `"**/*.py"`) · `max_results: int = 50` · `context_lines: int = 2` |
| Returns | `SearchPayload{matches: list[Match{path, line, text, context_before, context_after}], total_found, truncated}` |
| Behavior | Glob filters match workspace-relative POSIX paths and support recursive `**`: `*.md` is top-level only, while `**/*.py` is recursive. Each matching line produces one `Match`. Hidden paths, symlinked paths, and binary files are silently skipped; aggregate search keeps going instead of failing a whole query for one unreadable file, intentionally differing from `read_file`. `total_found` is the full true match count, and `truncated = total_found > len(matches)`. |
| Failure cases | invalid regex → `InvalidArgsError` with compiler message (model repairs the pattern); glob containing `..`, an absolute root, or a drive prefix → `InvalidArgsError`; 0 matches → `ok=True, matches=[]` (Critic hint: broaden query); result cap → truncated flag |
| Example | `{"query": "def parse_date", "glob": "**/*.py"}` → `{"ok": true, "data": {"matches": [{"path": "app/utils/date.py", "line": 41, "text": "def parse_date(raw: str) -> date:", ...}], "total_found": 1, "truncated": false}}` |

## 4. Phase-5 patch and Phase-6 test tool specs

### 4.1 `propose_patch` — risk: medium

| | |
|---|---|
| Purpose | Produce a unified diff for one existing workspace file without writing anything. |
| Args | `path: str` · `new_content: str` (the complete replacement content) |
| Returns | `ProposePatchPayload{path: str, diff: str, insertions: int, deletions: int, is_noop: bool}` |
| Approval | Medium risk; runs automatically through dispatch and is logged. |
| Behavior | Per D-035, the model supplies whole-file content and the tool owns diff syntax: it reads the current UTF-8 file through the jail and uses `difflib.unified_diff` to produce deterministic `a/`/`b/` headers with three context lines. Identical content succeeds with `is_noop=True` and an empty diff. |
| Failure cases | path outside jail → `PathJailError`; path missing or a directory → `NotFoundError`; non-UTF-8/binary current file → `BinaryFileError`; current or proposed content over 200 KiB → `InvalidArgsError` |
| Invariants | Never writes to disk or invokes Git; new-file creation and deletion are outside this tool's contract. |

### 4.2 `git_create_branch` — risk: high

| | |
|---|---|
| Purpose | Create or switch to the run's isolated work branch before patching. |
| Args | `rationale: str` (non-empty; the model cannot supply a branch name) |
| Returns | `GitCreateBranchPayload{branch: str, created: bool, switched: bool}` |
| Approval | Required before dispatch invokes the handler, including the no-op path. |
| Behavior | Per D-037, the target is derived as `repopilot/fix-<run_id>`. If it is already current, the tool succeeds as a no-op; otherwise, it creates the branch or switches to the existing branch only when the tracked worktree is clean. |
| Failure cases | dirty tracked worktree → `GitError{reason=dirty_worktree}`; repository inspection, branch creation/switching, process startup, or timeout failure → `GitError` with a typed reason |
| Invariants | Never stashes, cleans, discards changes, commits, deletes branches, or pushes. Untracked files do not block branch creation or switching. |

### 4.3 `apply_patch` — risk: high

| | |
|---|---|
| Purpose | The only way file content changes. Takes a unified diff (usually from `propose_patch`). |
| Args | `diff: str` (unified format) · `rationale: str` (shown to the human approver) |
| Returns | `ApplyPatchPayload{files_changed: list[str], insertions: int, deletions: int, applied: bool}` |
| Approval | Gate renders the diff + rationale to the approver. Denial returns `ApprovalDeniedError` to the agent — the identical diff may not be re-submitted. |
| Failure cases | off the run work branch → `PatchApplyError{reason=wrong_branch}`; `git apply --check` failure → `PatchApplyError{reason=check_failed, reject, stderr}` → agent re-reads the file and regenerates; touched path outside jail → `PathJailError`; empty diff → `InvalidArgsError` |
| Invariants | Applies on a work branch (`repopilot/fix-<run_id>`), never on the user's branch; whole-diff atomicity (all hunks or none). |

### 4.4 `run_tests` — risk: high

| | |
|---|---|
| Purpose | Execute the operator-configured pytest command and return an objective, structured test outcome after a patch. |
| Args | `rationale: str` (non-empty). The command and timeout are injected from `Settings.test_command` / `Settings.test_timeout_s`; they are never model arguments (D-040, preserving D-007). |
| Returns | `_RunTestsPayload{passed: int, failed: int, errors: int, skipped: int, total: int, exit_code: int, duration_ms: int, failures: list[_TestFailure{test_id, message}], failures_truncated: bool}`; at most 50 failure records are retained. |
| Approval | Required for every dispatch as built in P6. There is no host or sandbox bypass flag. |
| Behavior | Splits the operator-configured command into argv, appends the fixed `--junit-xml <temp path>` arguments, and runs with `cwd` at the jailed workspace, `stdin=DEVNULL`, captured output, `check=False`, and `shell=False`. JUnit XML is written to a system temporary directory and parsed with the standard library, so the tool itself writes no repository file (the invoked tests may). |
| Result boundary | A completed suite with failing tests is `ok=True` data (for example, `failed > 0`); `TestExecutionError` is reserved for `runner_not_started`, `timeout`, or `no_results`. |
| Critic evidence | `_RunTestsPayload.evidence_digest()` emits counts plus bounded failing `test_id`s, deliberately omitting messages. Registry dispatch stores that digest in the optional trace `outcome`; the loop renders the counts and at most 10 failing IDs into Critic raw evidence (D-043). |
| Invariants | Model-produced rationale never reaches argv; the subprocess uses `shell=False`; command execution remains a typed, approval-gated capability rather than a generic shell. |

## 5. Error taxonomy (shared by all tools)

`ToolError.type` ∈ `InvalidArgsError · PathJailError · NotFoundError · BinaryFileError ·
ToolTimeoutError · PatchApplyError · TestExecutionError · ApprovalDeniedError · InternalToolError ·
GitError · LoopBlockedError`
— each carries a `message` written **for the model** (actionable) and optional structured fields.
The mapping from error type → recovery strategy lives in `docs/failure-recovery.md`.
