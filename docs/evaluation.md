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

## 3. Harness

The Phase 4 minimal runner in `eval/run_eval.py` evaluates only `bug_localization` and
`bug_explanation` tasks. It exposes only the read-only `get_file_tree`, `read_file`, and
`search_code` tools, so it has no approval gate. It writes raw JSON results under
`eval/reports/<timestamp>/` (gitignored).

Phase 4 CLI: `uv run python -m eval.run_eval --tasks eval/tasks.json --type issue --repeat 3 --out eval/reports`
The model is loaded from `.env`; the current runner has no `--model` option.

The complete Phase 7 harness is planned to cover all five task types, exercise approval tracing,
render a markdown report with regressions, and support model selection such as
`--model deepseek-v4-pro`. Repeated runs will report variance for stochastic behavior.

## 4. Results log (curated)

| Date | Commit | Model | Overall success | Notes |
|---|---|---|---|---|
| 2026-07-12 | 07e7793 | deepseek-v4-pro | Issue slice: top-3 localization 1.000 (6/6); citation validity 1.000 (3/3) | 3 tasks × 3 repeats (9 real loops); all 6 localization results were rank 1; mean 9.0 steps; $0.36. Passing the full `bug_explanation` rubric is a stricter secondary metric based on literal keyword matching. |

## 5. Planned comparisons

- Model ladder: `deepseek-v4-flash` vs `deepseek-v4-pro` (default) vs `claude-opus-4-8`
  (cross-provider cost/quality curve — also exercises both adapters).
- Ablation: Critic on/off (does verification pay for its tokens?).
- Ablation: plan-first vs pure ReAct executor on `repo_qa` (steps + success).
