### Timing

Two remote submissions on the hosted B200. Always check correctness before timing.

```bash
{{SCORE_CMD}} --test-only    # correctness only, ~10s, one submission
{{SCORE_CMD}}                # correctness + timing + ncu + nsys profiles, minutes
```

### Profiling

A full score profiles the kernel with nsys and ncu in parallel.

- **Profiled durations are not benchmark times.** Instrumentation inflates them; a kernel
  the scorer measures at 75 µs reports ~170 µs under the profiler. The profile is for
  ratios and bottlenecks. Only the scorer's metric is a time.
- **Only the first ~10 kernels launched are captured with ncu**
- **The nsys input is a dense SPD batch at the scored Cholesky shape**, spectrum
  `linspace(1, 2)` so `cond` is exactly 2 like every benchmark row. Don't use it as a
  correctness oracle or a replacement benchmark.

The target is a **B200 (sm_100)**.

### Rules
- **NEVER** build, run, time, or profile the kernel locally. Correctness and performance
  both come from the hosted service.
- Keep `profile/` and any `.ncu-rep`/`.nsys-rep`/`.zip` out of your commits. Prefer
  `git add {{SUBMISSION_FILE}}` over `git add -A`.
- Do not submit with `--mode leaderboard`. Ranked submissions are a human decision and are
  rate-limited separately.
- Only `{{SUBMISSION_FILE}}` is sent. It must be a single self-contained kernel.
