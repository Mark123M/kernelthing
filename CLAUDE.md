# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

kernelthing runs an autonomous evolutionary search that optimizes a GPU kernel. It drives the
external `opencode` CLI agent (DeepSeek V4 by default) inside a bubblewrap sandbox, scores every
attempt on the hosted gpu-mode **popcorn** service (remote B200 — see `popcorn.py`), and keeps only
measured improvements. There is no local benchmark engine and no CUDA/torch dependency: every score
is a remote submission. See `README.md` for the user-facing story; this file covers what you need to
*change the code*.

## Commands

```bash
python -m pip install -e '.[dev]'     # dev deps: pytest, ruff, mypy
pytest                                 # full suite (testpaths=tests)
pytest tests/test_evolve.py::test_elites_and_best_maximize -q   # one test
ruff check .                           # line-length 100; T201/E501 intentionally ignored
mypy kernelthing                       # strict; tests/vendor/problems/sandbox excluded

kernelthing problems/<name> -j 8       # run the loop; web UI at http://127.0.0.1:8765
kernelthing score <problem-dir>        # authoritative scorer, prints {correct, metric, unit}
kernelthing score <dir> --test-only    # correctness only; halves the cost on popcorn problems
kernelthing web --root ~/.cache/kernelthing   # replay/serve runs with no loop process
python -m kernelthing ...              # equivalent to the `kernelthing` entry point
```

**Submodules:** `git submodule update --init` (the README's command) **fails** — `problems/zo_double_gemm_silu_layernorm`
is a gitlink with no `.gitmodules` entry. Scope it instead:
`git submodule update --init vendor/KernelWiki vendor/ncu-report-skill`.

**Pure-Python install.** There is no native build step and no compiled dependency — the package is
pure Python. `pip install -e '.[dev]'` needs no C compiler, CUDA, or torch. Scoring is remote, so the
only hard deps are `kernelguard` and `PyYAML`; the `popcorn` CLI is an external prerequisite.

### Test-environment realities

Tests never need a GPU, `opencode`, or the popcorn service (the network path is not exercised):

- `test_popcorn.py` runs the parsers/scorer against **verbatim captures** in `tests/fixtures/popcorn/`
  (both channels: `*.txt` `--output` files and `api-*.json` API responses) — no network.
- `test_bootstrap.py` fakes `opencode_client.run`; `validate_problem` is structural-only, so there is
  no runtime score to stub.
- `test_oc_guard.py` shells to `node tests/guard_driver.mjs` (skipped without node).
- `test_prompts.py::test_working_set_has_no_role_word_claude` scans **every** `prompts/**/*.md` for
  the literal string "Claude" and fails if found. Adding a prompt file that mentions it breaks CI.

## Architecture

### Layering — keep `evolve.py` pure

`evolve.py` is the search (Population, elites, MAP-Elites niches, UCB parent selection with virtual
loss) and is **side-effect-free on purpose** so it can be unit tested. All git, subprocess, worktree,
and IO work belongs in `orchestrator.py`. Do not reach for `subprocess` inside `evolve.py`.

`orchestrator.py` (~1000 lines) is the controller: it owns worktrees, agent turns, the remote scoring
submissions, and the budget. `RunContext` holds the mutable per-run state; the `_ev_*` methods
(`_ev_want_more` / `_ev_dispatch` / `_ev_collect`) are the dispatch loop. Scoring shells out to
`kernelthing score` per candidate (one process each), so candidates edit *and* score concurrently up
to `-j` — there is no serialized GPU stage anymore.

### The run directory is the *only* channel between the loop and the UI

Everything lands in `<problem-repo>/.humanize/rlcr/<timestamp>/` (see `state.py:LoopDirs` and
`journal.py`):

| file | role |
|---|---|
| `run.json` | immutable metadata written once at setup |
| `events.ndjson` | append-only journal; UI state is a pure fold over it |
| `control.json` | live knobs — UI writes (atomic replace), loop re-reads at dispatch boundaries |
| `live.lock` | flock held by the run process; liveness == "someone holds it" |
| `members/<id>/` | prompt.md, opencode.ndjson, stderr.log, summary.md, diff.patch, result.json |

`webui/` is a **pure reader** of those dirs, with exactly one write path (`POST /api/control`). Never
introduce shared in-memory state between the orchestrator and the web server — the decoupling is what
makes finished runs replay identically to live ones, and lets `kernelthing web` serve runs from
unrelated processes. NDJSON over SQLite is deliberate: runs are debuggable with `tail`/`grep`/`jq`.

### There is no local GPU path — scoring is remote

kernelthing does not run kernels locally. The local-benchmark stack was **removed**: there is no
`bench.py` (pygpubench scorer), no `gpupool.py` / `libktgpu.so` LD_PRELOAD exclusivity shim, no
`gpucontrol.py` clock/power locking, no `native/` C code, and no GPU pool anywhere. `torch` and
`pygpubench` are no longer dependencies. Every score is a remote popcorn submission (see below); the
sandbox binds **no** GPU device nodes. Do not reintroduce a local benchmark path — the target
hardware (B200) is only reachable through the hosted service, so a local number would be meaningless.

The oc_guard blocks a *ranked* submission — `popcorn ... --mode leaderboard`
(`checkBash` → `popcorn-leaderboard`): agents may test, benchmark and profile against the remote
service freely, but a ranked submission publishes to a public board and draws its own rate limit, so
it stays a human decision. (The older `gpu-tamper` rule — blocking `CUDA_VISIBLE_DEVICES`,
`LD_PRELOAD`, `KERNELTHING_*`, `libktgpu`, bare `env`/`printenv` — is retained as defensive env
hygiene even though the shim it once guarded is gone.)

### Scoring: the subprocess boundary

`Orchestrator._cli_score` shells out to `python -m kernelthing score` rather than calling
`popcorn.score` in-process. That is **not** incidental: it is the *same* code path agents run, and a
separate process per score keeps concurrent scorings isolated — they share a submission cache and
mutate the cwd. The verdict is one JSON line `{correct, metric, error, unit, bench}` that the
orchestrator, journal, `members/<id>/result.json`, and the web UI all consume unchanged, so scoring
stays swappable behind that boundary.

### The remote scoring backend (`popcorn.py`)

Some problems are graded on hardware this box does not have — the gpu-mode competitions measure on
B200 through the hosted popcorn service, so no local number is authoritative — this is the **only**
scoring backend. A problem sets `bench.backend = "popcorn"` in `problem.json` and `cli.score_command`
routes to `popcorn.score` right after `load_problem`; a problem without that backend key is a hard
error (there is nothing else to score it with).

`popcorn.score` returns a `(correct, metric, err, detail)` tuple and prints the
`{correct, metric, unit, error, bench}` line `Orchestrator._cli_score` parses, so `evolve`, the
journal, `members/<id>/result.json` and the web UI need no popcorn awareness at all — they only see
that JSON verdict. Keep it that way: the JSON line is the seam, so nothing downstream should learn how
a number was produced.

Consequences worth knowing before editing it:

- Scoring is **two** submissions — `--mode test`, then `--mode benchmark` only if test passed. A
  broken kernel never pays for a benchmark. `correct` is the conjunction: benchmark re-checks every
  shape, so a kernel that only breaks at scale still fails.
- **Nothing local touches a GPU.** `kernelthing score` takes only `[dir]` and `--test-only`; there is
  no `--gpu` / baseline plumbing. The metric is an absolute time, so there is no baseline to pin —
  `_evolve_seed` scores the seed once and moves on.
- The metric is a **time in microseconds, so these problems set `direction: minimize`.**
  `bench.popcorn.metric_mode` picks *which* time: `shape` (one `benchmark_index`, the default — a
  per-shape specialisation leaves the other entries constant and a whole-board geomean would bury
  it), `geomean`, or `leaderboard`.
- **Always set `bench.popcorn.benchmark_spec` alongside `benchmark_index`.** On the API path
  `benchmark-count` fixes the index space so a shape cannot move; on the text fallback it can —
  an unparsed failure row shortens the list and shifts every later index, silently repointing the
  metric at another shape. The pin catches that, and it costs nothing. Compared field-wise: the
  service prints `n; cond; seed; batch` in benchmark rows but `batch-n-cond-seed` in profiler
  artifact names, so string equality would false-alarm.
- Per-shape times exist **only in `test`/`benchmark` mode**. `format_submission_rows` returns
  `None` for `leaderboard`, so a ranked run yields the geomean line and nothing else.
- **Numbers come from the API, not the CLI's stdout.** The CLI submits; then `_attach_raw` does
  one authenticated `GET /user/submissions/<id>` (header `X-Popcorn-Cli-Id`, id from
  `~/.popcorn.yaml`) and reads `runs[].result` — a flat dict of
  `benchmark.N.{spec,mean,err,best,worst,std,runs}` in **nanoseconds**, plus `benchmark-count`
  and an authoritative `check`. The CLI itself parses that dict (`get_user_submission`) and then
  discards it while formatting, which is why `--output` only ever shows three significant figures.
  Concretely: `75141.44521620538` ns from the API vs `75.1 µs` from the text.
- **The text scrapers are the fallback, not the plan.** `parse_benchmark_output` /
  `parse_test_output` still exist and are validated against 111 real captures; they run only when
  the API is unreachable, so a network blip degrades a run to ~0.5% quantisation instead of
  killing it. `bench.source` in the verdict records which channel produced each number
  (`api` / `cli-text`) — check it before trusting a tight comparison.
  `tests/test_popcorn.py` runs against **verbatim captures** in `tests/fixtures/popcorn/` (both
  channels: `*.txt` are `--output` files, `api-*.json` are API responses), and asserts the two
  agree. Re-capture those files rather than editing them.
- In `leaderboard` mode the `Geomean score (public)` / `(secret)` lines come back in `details.runs`
  order, which is **not stable** — 26 of 54 real captures print secret first. `parse_leaderboard_score_s`
  matches on scope for that reason; taking the first line would silently score against the secret
  seed half the time.
- **Two timeout budgets** (`TEST_TIMEOUT_S` 300, `timeout_s` 1200). Measured: test 11s, benchmark
  275s over 15 shapes; the server caps a shape at 180s so a slow kernel can approach 2700s. Their
  sum must stay under `_cli_score`'s 1800s kill of the whole scoring process.
- Submissions are **cached on the submission file's sha256** (`_cache_path`). An agent self-tests
  and the orchestrator then scores the commit it produced, usually byte-identical — the cache
  roughly halves a run's remote traffic. `KERNELTHING_POPCORN_CACHE` relocates it,
  `KERNELTHING_POPCORN_BIN` overrides the binary (tests point it at a stub).
- The agent gets a different tool prompt (`prompts/claude/kernel-tools-popcorn-ncu.md` instead of
  `kernel-tools-ncu.md`, and no shared-GPU block) — see `Orchestrator._kernel_tools_block`.
  Profiling is `popcorn submit --profile-brev`, not local `ncu`.

### Problem contract (`problem.py`, `bench.py`, `prompts/claude/bootstrap-problem.md`)

A problem is a directory with `problem.json` inside a git repo:

- `edit_files` — the only files the agent may write (enforced by the guard, not by convention).
- `submission.py` (`bench.submission_qualname`) — the adapter, **protected**.
- `task.py` — `generate_test_case(*, seed, **args)`, the scoring objective, **protected**.
- `baseline.py` (`metric.baseline_qualname`) — the denominator, **protected**.
- `bootstrap.protected_files()` derives those basenames from the manifest; `validate_problem` rejects
  a manifest whose `edit_files` overlaps them (that overlap is a reward-hacking hole).
- `metric.kind` ∈ `latency_us` | `tflops` | `pct_baseline` | `speedup` (see `bench.derive_metric`).

`prepare_problem()` **copies** the problem dir into a standalone git repo at
`~/.cache/kernelthing/<name>/` and `git init`s it — the source repo is never touched, and worktrees
branch from that copy under `<problem_root>/wt/<ts>/evolve/`. `rewrite_plan_for_worktree` fixes up
plan paths because the problem sits at the repo root there, not under `problems/<name>/`.

### Anti-cheat is layered, and every gate fails open

1. **oc_guard** (`oc_guard/guard.js` + `guard_core.js`) — opencode `tool.execute.before` hook;
   rejects the tool call before it runs.
2. **kernelguard** (`gates.kernelguard_violations`) — static scan run *before* the expensive remote
   submission; only high-confidence hits (`should_filter` / `classification == "hacked"`) disqualify.
3. **the popcorn service** — the remote evaluator re-runs correctness in both `--mode test` and
   `--mode benchmark` (every shape re-checked at scale, on real B200 hardware) and returns an
   authoritative `check`; local timing tricks can't reach it because nothing runs locally.

Every gate returns "no violation" when its dependency is missing or errors. Preserve that: a gate
that raises can wedge an entire run.

**oc_guard trap:** opencode treats *every exported function* in a loaded plugin file as a plugin
factory. `guard.js` therefore exports exactly one symbol; all decision logic and its exports live in
`guard_core.js`, which opencode never loads directly. Do not merge them. `decide(cfg, tool, args)` is
pure and driven by `tests/guard_driver.mjs`.

### Sandbox (`sandbox.py`)

bwrap with `--ro-bind / /`, then specific paths rebound writable: the worktree, opencode's XDG state,
`/tmp`. Network stays up (the model API needs it) — the filesystem is the confinement boundary, which
is the only reason `opencode --auto` is safe here. User skill dirs (`~/.claude/skills`,
`~/.agents/skills`) are masked with tmpfs so no user-level skills leak into the agent's prompt.

### Prompts: most of `prompts/` is dead legacy

kernelthing began as a port of Humanize, and `prompts/` still carries that tree. Only these are live:

- `prompts/claude/bootstrap-problem.md`, `bootstrap-mode-{auto,interactive}.md` — loaded by
  `bootstrap.py`.
- `prompts/claude/kernel-tools-{wiki,ncu}.md` — loaded by `Orchestrator._kernel_tools_block`.
- `prompts/block/*.md` — rendered by `guard_core.js`, but only 20 of the 45 are referenced.
- Everything under `prompts/codex/`, `prompts/plan/`, `prompts/idea/`, and the rest of
  `prompts/claude/` is unreferenced (one exception: `test_prompts.py` asserts
  `codex/regular-review.md` exists).

The **operator prompts that actually drive the search are inline constants** in `orchestrator.py`
(`EVOLVE_EXPLORE_PROMPT`, `EVOLVE_EXPLOIT_PROMPT`, `EVOLVE_DESCRIPTOR_FOOTER`, `METHODOLOGY_PROMPT`).
That is where to edit agent instructions.

`prompts.render` is **single-pass by design**: a `{{VAR}}` appearing inside a substituted *value* is
not re-expanded (prevents placeholder injection), and unknown vars keep their `{{NAME}}` literal.
When composing nested templates, render the inner one first (see `bootstrap.render_prompt`).

## Conventions

- Module docstrings here are long and carry the real design rationale — read the header before
  editing a file, and update it when the rationale changes.
- `print(..., file=sys.stderr)` is the intended logger for this CLI (ruff's T201 is off for that).
- mypy is strict with `disallow_untyped_defs`; annotate new functions in `kernelthing/`.
- Long-lived file handles held across a `yield` (journal, flocks) carry `# noqa: SIM115` — the flock
  lives on the open fd, so closing early releases it.
- Exit reasons (`complete` / `maxiter` / `stopped` / `stalled_out`) are contract with `cli.run_loop`'s
  return code — changing the strings changes the process exit status.
