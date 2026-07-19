# Evaluation Design

> Goal: replace "it feels smart" with numbers. The harness replays task specs against fixture
> repos, combines trace-derived operational metrics with task-specific ground truth, and preserves
> per-run scores and execution traces.

## 1. Task types

| Type | Description | Ground truth | Scorer |
|---|---|---|---|
| `repo_qa` | "Where is X handled? How does Y flow?" | Expected file paths + keyword rubric | path match + rubric keywords in answer |
| `bug_localization` | Issue text → find root-cause file/function | Gold file (+ validated line range) | top-3 file hit |
| `bug_explanation` | Explain why the bug happens | Rubric keywords + grounded-citation requirement | rubric + citation-grounding check |
| `patch` | Produce a fix; apply; tests pass | Fixture repo with failing test | independent final-workspace tests green |
| `recovery` | Same as patch, but with a configured first-attempt fault injection (e.g. stale file version → patch conflict) | Recovery within budgets | recovered-and-green vs not |

Fixture repos live in `eval/fixtures/` (small, self-contained, with seeded bugs + pytest suites).
Task specs live in `eval/tasks.json`.

## 2. Metrics

The shared `report.md` layer consumes `RunTrace` records: caller-owned task type and final loop
status paired with persisted `TraceEvent` streams. Task-specific ground-truth scorers are independent
and live in `results.json`; §3 defines that separation.

| Metric | Definition |
|---|---|
| Task Success Rate | `# runs whose agent loop reached DONE / # runs`, per type and overall (trace proxy) |
| Tool Call Accuracy | `# schema-valid calls / # total calls` |
| Invalid Tool Call Rate | `1 − Tool Call Accuracy` (also tracked: hallucinated-path rate) |
| Average Steps | mean `tool_call` events per DONE run (not the loop's `steps_used`) |
| Recovery Success Rate | among runs with ≥1 trace-observed failure: `# reaching DONE / #` (generic proxy) |
| Graceful Failure Rate | among FAILED runs: `# with a report trace event / #` (target: 100%) |
| Human Approval Trigger Rate | `# non-system approval decisions / # approval decisions`; actor `system` is the ungated safety alert (target: 100%) |
| Run Trace Latency | per-run sum of recorded event latency, with nearest-rank p50/p95; also per-tool p50/p95 |
| Estimated Cost | Σ recorded event `cost_usd` values across the suite |

These operational metrics remain reproducible from explicit run facts and traces. They do not
replace independent final-workspace tests, repository-QA rubrics, issue scorers, or the recovery
suite's expected-error-observed-plus-`DONE`-plus-green criterion.

## 3. Harness

`eval/run_eval.py` is the Phase 7 runner for all five task types. The CLI groups
`bug_localization` and `bug_explanation` under `--type issue`; it also accepts `repo_qa`, `patch`,
and `recovery`. Issue and repository-QA runs expose only `get_file_tree`, `read_file`, and
`search_code`. Each patch or recovery repetition starts from a fresh temporary Git copy and uses
the complete fix registry. High-risk dispatches still cross the approval gate; the evaluation gate
records approvals as actor `eval:auto` rather than bypassing enforcement. Human Approval Trigger
Rate therefore measures recorded gate-decision coverage, not manual owner clicks.

Each completed suite writes suite-specific scores to `results.json`, trace-derived metrics to
`report.md`, and per-run JSONL traces plus the SQLite projection under `state/` in a timestamped
`eval/reports/` directory (gitignored). Patch scoring reruns the frozen task test command after the
loop and keeps loop status separate from final-workspace test status. Repository-QA scoring checks
candidate paths and rubric keywords independently of loop status. Recovery scoring separately
records whether the expected typed error was observed, whether the loop reached `DONE`, whether
final tests passed, and whether all three conditions held.
`results.json` is authoritative for these suite-specific scores. `report.md` applies the shared
trace-derived contract: its Task Success Rate follows loop status, and its Recovery Success Rate is
the generic `DONE`-after-an-observed-failure proxy. Those trace rates can diverge from the suite
scorer and must not be substituted for it.

```bash
uv run python -m eval.run_eval --tasks eval/tasks.json --type patch --repeat 3 --out eval/reports
```

Replace `patch` with `issue`, `repo_qa`, or `recovery` to run the other suite. Provider and model
selection come from application settings (environment variables or `.env`); the runner has no
`--model` option. Repetitions are preserved individually in `results.json`, while each suite
records aggregate rates and means.

## 4. Results log (curated)

Selected suite results below come from `results.json`; they are not a single universal success
definition. Notes identify compound scorer outcomes and trace-derived proxies explicitly.

| Date | Commit | Model | Suite | Selected suite results | Notes |
|---|---|---|---|---|---|
| 2026-07-12 | 07e7793 | deepseek-v4-pro | issue | Top-3 localization 1.000 (6/6); `bug_explanation` citation validity 1.000 (3/3) | 3 tasks × 3 repeats (9 real loops); all 6 localization results were rank 1; mean 9.0 steps; $0.36. Passing the full `bug_explanation` rubric is a stricter secondary metric based on literal keyword matching. |
| 2026-07-18 | 706f9c1 | deepseek-v4-pro | patch | Final-workspace tests green 0.667 (4/6) | 2 tasks × 3 repeats. Loop `DONE`, and therefore the trace Task Success proxy, was 0.500 (3/6). EV-PATCH-001 was green 1/3; EV-PATCH-002 was green 3/3, including one `FAILED`-but-green run. Human Approval Trigger Rate was 1.000; Approval Ungated Count was 0. Mean 14.33 loop steps; $0.624612 suite-recorded cost. |
| 2026-07-19 | 706f9c1 | deepseek-v4-pro | recovery | Recovered-and-green 0.000 (0/6) | 2 tasks × 3 repeats. The expected error type was observed 0.500 (3/6), loop `DONE` was 0.167 (1/6), and final tests green was 0.000 (0/6). The expected error was observed for EV-REC-001 in 2/3 runs and EV-REC-002 in 1/3; neither recovered. The trace Recovery Success proxy was 0.167, while the independent scorer remained 0.000. The sole `DONE` run was tests-red and had not observed its configured error type. Human Approval Trigger Rate was 1.000; Approval Ungated Count was 0. Mean 14.67 loop steps; $0.611929 suite-recorded cost. |
| 2026-07-19 | 706f9c1 | deepseek-v4-pro | repo_qa | Independent scorer success 0.500 (3/6) | 2 tasks × 3 repeats. Path hit and full-rubric coverage were each 0.667 (4/6), but only three runs satisfied both. EV-QA-001 passed 3/3; EV-QA-002 passed 0/3. Loop `DONE`, and therefore the trace Task Success proxy, was 0.833 (5/6). Mean 7.17 loop steps; $0.248433 suite-recorded cost. |
| 2026-07-19 | 706f9c1 | deepseek-v4-pro | issue | Top-3 localization 1.000 (6/6); `bug_explanation` citation validity 1.000 (3/3) | 3 tasks × 3 repeats. All 6 localization results were rank 1. The full `bug_explanation` scorer (literal rubric plus required grounding) passed 1/3. Loop `DONE`, and therefore the trace Task Success proxy, was 0.333 (3/9). Mean 8.56 loop steps; $0.341161 suite-recorded cost. |

The four `706f9c1` rows form one curated baseline: 9 tasks × 3 repetitions, or 27 admitted live
loops. Suite-recorded costs total $1.826135. The three repetitions remain individually visible in
the raw results; the per-task splits above show the observed stochastic spread rather than reducing
it to a single mean.

Only complete repeat-3 suites enter this log. A patch launch whose independent pytest scorer could
not start and an operator-paused recovery launch remain in local audit traces but were excluded
wholesale; they incurred additional API cost outside the valid-suite total. Completed suites were
accepted as observed and were not rerun merely to turn a failed score into a pass.

## 5. Planned comparisons

- Model ladder: `deepseek-v4-flash` vs `deepseek-v4-pro` (default) vs `claude-opus-4-8`
  (cross-provider cost/quality curve — also exercises both adapters).
- Ablation: Critic on/off (does verification pay for its tokens?).
- Ablation: plan-first vs pure ReAct executor on `repo_qa` (steps + success).
