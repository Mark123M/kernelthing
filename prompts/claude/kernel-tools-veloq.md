
### Reading the profile properly — `veloq`

`ncu-details.txt` is the flattened text view. The `.ncu-rep` beside it holds the same
capture *structured*: per-launch metrics, the profiler's own rule findings with severities,
and per-source-line warp-stall histograms. `veloq` reads it — no GPU involved, it is a file
parser. Prefer it over grepping the text dump; fall back to the text only if `veloq` errors.

The capture lands at the **worktree root**, not under `profile/`: `popcorn` extracts
relative to its own cwd, so the report is `profile.<index>-<spec-slug>/profile.ncu-rep`.

```bash
REP=profile.*/profile.ncu-rep

# 1. WHICH launch is yours? Only the first ~10 kernels are captured, and a plain
#    PyTorch path fills them with copy/elementwise kernels. Always start here.
{{VELOQ_BIN}} ncu launches $REP --limit 20

# 2. Metrics + the profiler's rule findings for one launch (the highest-value verb:
#    each rule carries a severity and the focus_metrics that triggered it).
{{VELOQ_BIN}} ncu inspect $REP --row-id launch:<N>

# 3. Why warps stalled, ranked.
{{VELOQ_BIN}} ncu warp-stalls $REP --row-id launch:<N> --by reason

# 4. One counter family across every launch, for comparison.
{{VELOQ_BIN}} ncu metrics $REP --counter 'sm__throughput*,dram__throughput*'

# 5. The SASS the kernel actually compiled to (PTX too, when the cubin embeds it).
{{VELOQ_BIN}} ncu disasm $REP --row-id launch:<N>
```

JSON on stdout is the contract — read `data.rows[]`, and on failure `error.message`.
The first call builds a cache next to the report (a few seconds); later calls are fast.
{{PTX_NOTE}}
Two traps, both already true of the text dump and equally true here:

- **Profiled durations are not benchmark times.** Instrumentation inflates them. Use the
  profile for *ratios and bottlenecks*; only the scorer's metric counts.
- **Never paste profiler output into {{SUBMISSION_FILE}}**, not even in a comment. It
  contains the token the evaluator rejects on sight (the one spelled out in the profiling
  section above), and that rejects the whole submission.
{{VELOQ_REF_NOTE}}
