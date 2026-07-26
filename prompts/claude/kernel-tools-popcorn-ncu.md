### There is no local GPU — everything runs on the competition hardware

This box has no GPU worth measuring on. Correctness and timings both come from the
hosted evaluation service, on the same hardware the leaderboard ranks. So: **never**
try to build, run, or time the kernel locally, and do not trust any number that did not
come back from a submission.

`nvcc`, `ncu`, `nsys` and `nvidia-smi` **are installed here and will run** — that is a
trap, not a resource. They target a small consumer laptop GPU of a different
architecture, so they return plausible numbers that are wrong for the B200 you are
optimising for. Nothing blocks you from running them; do not. `torch` and `numpy` are
deliberately not installed, so a local correctness check is not available either — use
`--test-only` instead.

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

- Keep the `profile.*/` directories and the `.zip`/`.ncu-rep` files out of your commit —
  a single report is ~80 MB. They are already in `.gitignore`; leave it. Prefer
  `git add {{SUBMISSION_FILE}}` over `git add -A` regardless.
- Do not submit with `--mode leaderboard`. Ranked submissions are a human decision and
  are rate-limited separately; `--test-only`, the full score, and `--profile-brev` are
  yours to use freely.
