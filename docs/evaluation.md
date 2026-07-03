# Evaluation Design

> Goal: replace "it feels smart" with numbers. The harness replays task specs against fixture
> repos, scores agent behavior from **traces** (not vibes), and produces a markdown report.

## 1. Task types

| Type | Description | Ground truth | Scorer |
|---|---|---|---|
| `repo_qa` | "Where is X handled? How does Y flow?" | Expected file paths + keyword rubric | path match + rubric keywords in answer |
| `bug_localization` | Issue text → find root-cause file/function | Gold file (+ line range) | top-3 file hit; line-range bonus |
| `bug_explanation` | Explain why the bug happens | Rubric keywords + gold citation | rubric + citation-validity check |
| `patch` | Produce a fix; apply; tests pass | Fixture repo with failing test | tests green after patch (in sandbox) |
| `recovery` | Same as patch, but with injected first-attempt failure (e.g. stale file version → patch conflict) | Recovery within budgets | recovered-and-green vs not |

Fixture repos live in `eval/fixtures/` (small, self-contained, with seeded bugs + pytest suites).
Task specs live in `eval/tasks.json`.

## 2. Metrics

| Metric | Definition |
|---|---|
| Task Success Rate | `# tasks meeting their scorer criterion / # tasks`, per type and overall |
| Tool Call Accuracy | `# schema-valid calls / # total calls` |
| Invalid Tool Call Rate | `1 − Tool Call Accuracy` (also tracked: hallucinated-path rate) |
| Average Steps | mean executed steps per successful task (efficiency) |
| Recovery Success Rate | among runs hitting ≥1 typed failure: `# reaching DONE / #` |
| Graceful Failure Rate | among FAILED runs: `# emitting complete report / #` (target: 100%) |
| Human Approval Trigger Rate | `# high-risk dispatches gated / # high-risk dispatches` (target: 100%, safety regression alarm) |
| Average Latency | wall-clock per task; also per-tool p50/p95 from `meta.latency_ms` |
| Estimated Cost | Σ `cost_usd` from trace usage records, per task |

All metrics are computable from `TraceEvent` streams alone — this is why trace-first design matters.

## 3. Harness (`eval/run_eval.py`, Phase 7)

1. Load `eval/tasks.json`; for each task: copy fixture repo to a temp workspace, run the agent with
   `approval_mode=auto_approve_recorded` (approvals auto-granted **but still traced**, so the
   trigger-rate metric stays honest).
2. Score with the task's checker; write raw results to `eval/reports/<timestamp>/` (gitignored).
3. Render `eval/reports/<timestamp>/report.md` — summary table + per-task drill-down + regressions
   vs the previous run. Curated results are copied into this doc §4 when milestones land.

CLI: `uv run python -m eval.run_eval --tasks eval/tasks.json --model deepseek-v4-pro --repeat 3`
(repeat ≥3 because agent runs are stochastic; report mean ± range).

## 4. Results log (curated)

| Date | Commit | Model | Overall success | Notes |
|---|---|---|---|---|
| _(pending Phase 7)_ | | | | |

## 5. Planned comparisons

- Model ladder: `deepseek-v4-flash` vs `deepseek-v4-pro` (default) vs `claude-opus-4-8`
  (cross-provider cost/quality curve — also exercises both adapters).
- Ablation: Critic on/off (does verification pay for its tokens?).
- Ablation: plan-first vs pure ReAct executor on `repo_qa` (steps + success).
