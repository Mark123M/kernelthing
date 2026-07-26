### Profiling with Nsight Compute

Golden rule: **Profile → Diagnose → Plan, in that order.** Profiling also runs remotely,
on the hosted Nsight Compute service:

Run it from the worktree root — one command, no `cd`, no pre-created directory:

```bash
POPCORN_BREV_PROFILER_URL={{PROFILER_URL}} {{POPCORN_BIN}} submit {{SUBMISSION_FILE}} \
    --leaderboard {{LEADERBOARD}} \
    --profile-brev \
    --benchmark-index {{BENCHMARK_INDEX}} \
    --no-tui \
    --output brev.json
```

Do not set the bash tool's `workdir` to a directory the same command creates. It is
checked before the command runs, so `mkdir -p profile` + `workdir: profile` fails with
`NotFound: FileSystem.access` every time.

It takes around 4–5 minutes. Artifacts extract **next to wherever you ran it** — the
`--output` path does not move them — one directory per profiled shape:

```
profile.<index>-<spec-slug>.zip
profile.<index>-<spec-slug>/
    profile.ncu-rep     <- the full capture, structured
    ncu-details.txt     <- flattened text view of the same data
    ncu-details.csv     <- same again, for grepping/sorting
```

```bash
cat profile.*/ncu-details.txt
```

`ncu-details.txt` is a plain `ncu --set full` dump — per-kernel sections (Speed Of Light,
Compute/Memory Workload Analysis, Occupancy, Launch Statistics, Warp State) followed by
`OPT`/`INF` rule findings that name the bottleneck outright. It needs no tooling, so it is
always available to you. But it is a rounded text rendering: `profile.ncu-rep` beside it
carries the same capture at full precision, per launch, and if a `veloq` section follows
below then that is the better way in.

Three things that will mislead you if you do not know them:

- **Profiled durations are not benchmark times.** Instrumentation inflates them —
  a kernel measured at 75.1 µs by the scorer reports ~168 µs under the profiler. Use the
  profile for *ratios and bottlenecks*, never as a timing measurement. Only the scorer's
  metric counts.
- **The profiler output contains the rejected token.** Kernel headers print
  `Context 1, S-t-r-e-a-m 7` (spelled out here so this file does not carry it). Never
  paste profiler text into {{SUBMISSION_FILE}}, even inside a comment — the evaluator
  rejects the whole submission on that substring.
- `--benchmark-index {{BENCHMARK_INDEX}}` is the shape this problem is scored on. Omitting
  it profiles every shape and takes far longer. Only the first ~10 kernels launched are
  captured, so if your submission dispatches more than that (a plain PyTorch path does),
  the capture may miss the one you care about — narrow the launch count first.
{{NCU_SKILL_NOTE}}
The target is a **B200 (sm_100)**. Cite specific metric values in your summary (compute vs
memory %-of-peak, achieved occupancy, dominant stall reason), not vague claims.

### Rules
- **NEVER** try to build, run, or time the kernel locally. Correctness and timings both come from the hosted evaluation service.
- Keep the `profile.*/` directories and the `.zip`/`.ncu-rep` files out of your commit. Prefer
  `git add {{SUBMISSION_FILE}}` over `git add -A`.
- Do not submit with `--mode leaderboard`. Ranked submissions are a human decision and
  are rate-limited separately; `--test-only`, the full score, and `--profile-brev` are
  yours to use freely.
- Only `{{SUBMISSION_FILE}}` is sent. It must be a single self-contained kernel.