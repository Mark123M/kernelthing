### Reading the Nsight Compute report — `veloq ncu`

Every full score writes `{{REPORT}}` to the worktree. `{{VELOQ_BIN}}` parses and analyzes the report for performance optimization. Only read raw ncu artifacts when veloq is not available.

The report does not exist until your first full score. Each score overwrites it and prints
what it landed.

```bash
V={{VELOQ_BIN}}; REP={{REPORT}}
```

JSON on stdout is the contract — read `data.rows[]`, and on failure `error.message`.
The first call builds a cache next to the report (a few seconds); later calls are fast.
`--help` on any verb is authoritative.

```bash
{{NCU_VERBS}}
```

Two traps, equally true of the text view:

- **Profiled durations are not benchmark times.** Instrumentation inflates them. Use the
  profile for *ratios and bottlenecks*; only the scorer's metric counts.
- **Never paste profiler output into {{SUBMISSION_FILE}}**, not even in a comment. It
  contains the token the evaluator rejects on sight (the one spelled out in the profiling
  section above), and that rejects the whole submission.
