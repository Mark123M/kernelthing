### There is no local GPU — everything runs on the competition hardware

This box has no GPU worth measuring on. Correctness and timings both come from the
hosted evaluation service, on the same hardware the leaderboard ranks. So: **never**
try to build, run, or time the kernel locally, and do not trust any number that did not
come back from a submission. `nvidia-smi`, `ncu`, `nsys` and a local `torch` run tell you
nothing here.

Two consequences worth internalising:

- Every measurement costs a remote round-trip of a minute or more. Think before you
  submit; batch your reasoning, not your submissions.
- Only `{{SUBMISSION_FILE}}` is sent. It must be a single self-contained Python file —
  native code goes through `torch.utils.cpp_extension.load_inline`, not a separate
  `.cu`.

### Checking your work

The cheap correctness pre-check — one submission, no timing:

```bash
{{SCORE_CMD}} --test-only
```

The authoritative score — correctness *and* the metric, which is what the search ranks
you on:

```bash
{{SCORE_CMD}}
```

Use `--test-only` while a change is still likely broken, and the full score once you
believe it works. Both print a JSON verdict; `"correct": true` is the bar, and `"metric"`
is the number to beat.

### Profiling with Nsight Compute (measure, don't guess)

Golden rule: **Profile → Diagnose → Plan, in that order.** Profiling also runs remotely,
on the hosted Nsight Compute service:

```bash
mkdir -p profile && cd profile   # scratch dir; keep .ncu-rep files out of the commit
POPCORN_BREV_PROFILER_URL={{PROFILER_URL}} {{POPCORN_BIN}} submit ../{{SUBMISSION_FILE}} \
    --leaderboard {{LEADERBOARD}} \
    --profile-brev \
    --benchmark-index {{BENCHMARK_INDEX}} \
    --no-tui \
    --output brev.json
```

It takes around 4–5 minutes and extracts one directory per profiled shape:

```
profile/
  profile.<index>-<spec-slug>.zip
  profile.<index>-<spec-slug>/
    ncu-details.txt     <- read this one
    ncu-details.csv     <- same data, for grepping/sorting
    profile.ncu-rep     <- ~80 MB GUI report; you do not need it
```

```bash
cat profile/profile.*/ncu-details.txt
```

`ncu-details.txt` is a plain `ncu --set full` dump — per-kernel sections (Speed Of Light,
Compute/Memory Workload Analysis, Occupancy, Launch Statistics, Warp State) followed by
`OPT`/`INF` rule findings that name the bottleneck outright. Read it directly; it needs no
tooling.

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

- Keep `profile/` and the `.zip`/`.ncu-rep` files out of your commit — you commit with
  `git add -A` and a single report is ~80 MB. They are already in `.gitignore`; leave it.
- Do not submit with `--mode leaderboard`. Ranked submissions are a human decision and
  are rate-limited separately; `--test-only`, the full score, and `--profile-brev` are
  yours to use freely.
