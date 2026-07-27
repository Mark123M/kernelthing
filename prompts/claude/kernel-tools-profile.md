### Measuring: `{{SCORE_CMD}}`

Two remote submissions on the hosted B200 — correctness, then timing. Only its metric
counts; nothing you run on this box measures the target hardware.

```bash
{{SCORE_CMD}} --test-only    # correctness only, ~10s, one submission
{{SCORE_CMD}}                # correctness + timing + ncu + nsys profiles, minutes
```

Use `--test-only` while a kernel still might be wrong. It is the cheap call and it is
meant to be run often.

### Profiling: you do not have to ask for it

A full score profiles the scored shape for you in parallel with the timing run:

- hosted Nsight Compute (`ncu`) prints Speed Of Light, Launch Statistics, Occupancy,
  Warp State, and every `OPT`/`INF` finding the profiler raised.
- Modal/B200 Nsight Systems (`nsys`) prints a compact timeline stats summary for launch
  order, CUDA API overhead, memcpy/kernel balance and gaps.

**Do not run `--profile-brev` yourself.** You already have the ncu capture, and a second
one only queues behind the one you were given. Do not run local `nsys` either; any local
GPU here is the wrong target.

The full dump and the structured `.ncu-rep` are written under `profile/latest/`; their
paths are in the printed block and in `bench.profile` of the verdict JSON. The nsys GUI
report, SQLite export and stats summary are under `profile/latest/nsys/`; their paths are
in `bench.nsys`.

State the bottleneck from those numbers — compute vs memory %-of-peak, achieved occupancy,
the dominant stall reason, the findings' own `Est. Speedup` — and cite the values you used.
Do not describe a bottleneck the profile does not show.

Two things that will mislead you:

- **Profiled durations are not benchmark times.** Instrumentation inflates them; a kernel
  the scorer measures at 75 µs reports ~170 µs under the profiler. The profile is for
  ratios and bottlenecks. Only the scorer's metric is a time.
- **Only the first ~10 kernels launched are captured.** If your submission dispatches more
  than that — a plain PyTorch path does — the capture may miss the one you care about.
  Narrow the launch count first.
- **The nsys input is a diagonal SPD tensor for the scored Cholesky shape.** Use it for
  timeline structure, not as a correctness oracle or a replacement benchmark.
{{NCU_SKILL_NOTE}}
The target is a **B200 (sm_100)**.

### Rules
- **NEVER** build, run, or time the kernel locally. `nvcc`, `ncu`, `nsys` and
  `compute-sanitizer` may be installed here; they would run against this box's consumer
  GPU at the wrong architecture and return numbers that look real. Correctness and timing
  both come from the hosted service.
- Keep `profile/` and any `.ncu-rep`/`.nsys-rep`/`.zip` out of your commits. Prefer
  `git add {{SUBMISSION_FILE}}` over `git add -A`.
- Do not submit with `--mode leaderboard`. Ranked submissions are a human decision and are
  rate-limited separately.
- Only `{{SUBMISSION_FILE}}` is sent. It must be a single self-contained kernel.
- Never paste profiler text into `{{SUBMISSION_FILE}}`, even in a comment. Nsight prints a
  token the evaluator rejects on sight; the printed block hyphenates it, which is a
  reminder rather than a licence.
