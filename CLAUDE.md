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
kernelthing score <dir> --brief        # verdict without the bench record (what agents run)
kernelthing score <dir> --dry-run      # submit nothing: replay what an agent reads, from fixtures
kernelthing score <dir> --dry-run cached|nsys-fail|test-fail   # ...the degraded blocks
kernelthing web --root ~/.cache/kernelthing   # replay/serve runs with no loop process
kernelthing web --root ~/.local/share/kernelthing/runs   # ...the same, over the durable archive
kernelthing archive --list                    # runs still sitting in the disposable managed root
kernelthing archive                           # export them (runs do this themselves on exit)
kernelthing transcripts --list                # runs visible to the transcript renderer
kernelthing transcripts <run-id|path> -o DIR  # agents' NDJSON logs -> markdown (archiving does this too)
python -m kernelthing ...              # equivalent to the `kernelthing` entry point
```

**Submodules:** `git submodule update --init` (the README's command) **fails** — `problems/zo_double_gemm_silu_layernorm`
is a gitlink with no `.gitmodules` entry. Scope it instead:
`git submodule update --init vendor/ncu-report-skill`. (`vendor/KernelWiki` is still declared in
`.gitmodules` but nothing loads it — the block that offered it to agents was removed 2026-07-27.)

**Pure-Python install.** There is no native build step and no compiled dependency — the package is
pure Python. `pip install -e '.[dev]'` needs no C compiler, CUDA, or torch. Scoring is remote, so the
only hard deps are `kernelguard` and `PyYAML`; the `popcorn` CLI is an external prerequisite.

**Never move or rename the checkout with its venv active.** Every agent is handed
`sys.executable`'s sibling `kernelthing` as its scoring command (`_score_cmd_str`). A stale `PATH`
entry still yields a working *parent* process — the entry-point shebang resolved once, at import —
while handing all N agents a path that no longer exists, so every candidate burns a full turn
discovering it cannot score and the run produces nothing. This actually happened
(`projects/kernelthing` → `projects/linalgkernelthing`). `Orchestrator._preflight`, called first
thing in `setup()`, now hard-fails on it; the fix is
`deactivate; source .venv/bin/activate; pip install -e '.[dev]'`.

### Test-environment realities

Tests never need a GPU, `opencode`, or the popcorn service (the network path is not exercised):

- `test_popcorn.py` runs the parsers/scorer against **verbatim captures** in `tests/fixtures/popcorn/`
  (both channels: `*.txt` `--output` files and `api-*.json` API responses) — no network.
- `test_bootstrap.py` fakes `opencode_client.run`; `validate_problem` is structural-only, so there is
  no runtime score to stub.
- `test_oc_guard.py` shells to a JS runtime via `tests/guard_driver.mjs` (`node`, `nodejs`, `bun`,
  `deno`, or `$KERNELTHING_NODE`). **Install one** — `sudo apt install nodejs`. At *runtime* the
  guard needs no system runtime at all (opencode loads the plugin with its own embedded one), so a
  box without node runs the guard correctly and silently loses all 49 of its tests. That asymmetry
  is how a broken guard ships; the skip message names the fix for the same reason. A full suite is
  **198 passed, 0 skipped**; any skip count at all means a prerequisite is missing, not that some
  tests are conditional — check the `-rs` reasons before trusting a green run.
- `test_prompts.py::test_working_set_has_no_role_word_claude` scans **every** `prompts/**/*.md` for
  the literal string "Claude" and fails if found. Adding a prompt file that mentions it breaks CI.
- `test_harness_deps.py` covers the prerequisites that fail *silently in a run* rather than loudly
  at startup — the interpreter preflight, `ncu_report` discovery, and whether each problem's
  `.gitignore` actually matches the capture layout popcorn produces. Every case in it is a real
  failure recovered from a run's artifacts, not a hypothetical.

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

### Run artifacts are durable — two mechanisms, and both matter (`archive.py`)

The run dir lives inside the managed repo under `~/.cache/kernelthing/<problem>/`, which is
XDG-disposable *and* rebuilt by `prepare_problem` on every run. That combination once meant starting a
second run silently destroyed the first one's entire record. Two independent guarantees now:

1. **`prepare_problem` preserves `.humanize/`** (`problem.NO_COPY` / `problem.PRESERVE`). It clears
   the managed dir entry-by-entry instead of `rmtree`-ing it, so the kept subtree is never moved and
   no crash can strand it. `.git/info/exclude` is written *before* the first `git add -A` — otherwise
   the preserved tree enters the initial commit and every worktree materialises every past run. This
   is what survives a hard power loss, since nothing gets to run an exit hook.
2. **Runs archive themselves on exit** (`Orchestrator._archive_run`, called from `run()`'s `finally`,
   after `journal.close()`). Artifacts are copied to `cfg.archive_root`
   (`~/.local/share/kernelthing/runs`, XDG *data*), covering clean exits, stalls, stops, exceptions
   and Ctrl-C. `kernelthing archive` does it by hand for runs whose process died first.

The archive layout **mirrors the managed root** (`<archive>/<problem>/.humanize/rlcr/<ts>/`) so an
archive root is a drop-in for `kernelthing web --root` — `discover_runs` globs `<root>/*/.humanize/
rlcr/*` and needs no special-casing. Beside it go the things the run dir lacks: `bundles/<ts>.bundle`
(all member commits, since the managed repo is rebuilt away), `best/<ts>/`, and `transcripts/<ts>/`
(see below).

### Transcripts (`transcript.py`)

`members/<id>/opencode.ndjson` is the *complete* record of one agent (assistant prose, reasoning,
every tool call with full input/output) but it is one JSON part per line, so reading a 750KB turn
means writing a parser first. `transcript.py` is that fold, written to files instead of a socket:
`export_transcripts(run_dir, out_dir)` writes `member-<id>.md` (prompt first, then the stream, nothing
clipped), an `index.md`, and copies of the run-level logs. `export_run` calls it, so **every archived
run — i.e. every run that ends — gets `transcripts/<ts>/` for free**; `kernelthing transcripts` covers
what that misses (a live run, a run dir elsewhere on disk, a re-render with `--jsonl`/`--max-output`).
It is a pure reader like `webui/`, and owns the NDJSON parsing helpers (`parts` / `tool_line` /
`is_tool` / `clip`) that `webui` imports back — the format is opencode's wire format, not HTTP.
Keep the transcript step inside its own `try` in `export_run`: a malformed log must not cost a run
its archive.

The best kernel is taken from **the journal, not HEAD** (`_best_scored_member` folds `events.ndjson`
exactly like the UI does). They agree only for a clean exit; a killed run never promoted its winner,
so HEAD is still the seed. Archiving is best-effort everywhere and swallows its own errors — a failed
copy must never change a run's exit status.

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

**`bench` is the archive; `--brief` drops it, and the agent's command carries that flag.**
Two callers read that line and they want opposite things. `_cli_score` wants everything: it
carries `bench` opaquely into `result.json` and the journal, and *no consumer anywhere reads a
field of it* — it is forensic, and worth keeping for exactly that reason. An agent wants the
verdict. Measured on one full score, `bench` is **3907 of 3982 chars (98%)**, of which `shapes`
alone is 63% — fourteen per-shape rows it cannot act on, since `metric_mode = "shape"` means one
index scores — and `profile`/`nsys` another 20%, repeating paths the banners above already
printed. So `Orchestrator._score_cmd_str` hands agents `kernelthing score . --brief` and
`_cli_score` runs its own invocation without it: **4715 → 806 chars per full score, nothing lost
from the archive.** Nothing is lost from the agent either — a failure's reason is in `error` (and
on stderr), never in `bench`.

The flag lives in `_score_cmd_str` rather than the prompt template because the template *appends*
to that string (`{{SCORE_CMD}} --test-only`), so one place covers both invocations; a problem
with its own `score_command` is returned untouched, since an arbitrary command need not know the
flag. `test_the_scoring_command_handed_to_agents_is_one_the_cli_accepts` runs the exact string
through the parser, both bare and with what the prompt appends — a flag that stopped parsing
would fail the way a dangling path does, silently and on every candidate (see `_preflight`).
`bootstrap-problem.md` deliberately keeps the full form: authoring a new problem is where the
`bench` record is the thing you are debugging, and it happens once, not 117 times.

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
- **...plus Nsight Compute and Nsight Systems captures, fired the moment the test passes and joined
  after the benchmark returns** (`profile_submission`, `profile_nsys_submission`,
  `PROFILE_TIMEOUT_S`). Agents no longer decide to profile; every full score arrives with both when
  configured. Details in "Automatic profiling" below.
- **Nothing local touches a GPU.** `kernelthing score` takes `[dir]`, `--test-only` and
  `--no-profile`; there is no `--gpu` / baseline plumbing. The metric is an absolute
  time, so there is no baseline to pin — `_evolve_seed` scores the seed once and moves on.
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
- **Three timeout budgets** (`TEST_TIMEOUT_S` 300, `timeout_s` 1200, `PROFILE_TIMEOUT_S` 1200).
  Measured: test 11s, benchmark 275s over 15 shapes; the server caps a shape at 180s so a slow
  kernel can approach 2700s. The profiles run *concurrently* with the benchmark, so the group costs
  `max(timeout_s, PROFILE_TIMEOUT_S)` after test — keep those two equal and the total stays
  300 + 1200, under `_cli_score`'s 1800s kill of the whole scoring process. Raise either alone and
  it starts killing scores mid-poll.
- Submissions are **cached on the submission file's sha256** (`_cache_path`; captures separately in
  `_profile_cache_path` / `_nsys_cache_path`). `KERNELTHING_POPCORN_CACHE` relocates it,
  `KERNELTHING_POPCORN_BIN`
  overrides the binary (tests point it at a stub). **The cache is read-only to agents**: bwrap
  ro-binds everything but the worktree, so an agent's own `kernelthing score` loads from it and
  silently fails to store (`_cache_store` swallows the `OSError` by design). Only the orchestrator's
  out-of-sandbox `_cli_score` populates it — in the 2026-07-24 run exactly 1 of 18 orchestrator
  scores was a hit, so do not count on it to halve anything.
- The agent's tool prompt is `prompts/claude/kernel-tools-profile.md` — see
  `Orchestrator._kernel_tools_block`, which returns `""` for a problem with no popcorn config
  (such a problem cannot be scored at all, so there is nothing to tell it). That block is
  **unconditional**, unlike the ones around it: it carries the scoring command and the submission
  rules, so gating it (as its `kernel-tools-popcorn-ncu.md` predecessor was gated on `cfg.ncu`)
  would leave `--no-ncu` agents unable to score. `cfg.ncu` now gates only
  the `ncu-report-skill` section (`_ncu_skill_note`).
- **`popcorn` extracts captures relative to its own cwd, not to `--output`.** `profile_submission`
  runs it in a `TemporaryDirectory` and moves the result, exactly as `submit` already did, which is
  why nothing lands at the worktree root any more and why the `workdir` trap is gone. The two ignore
  rules per problem (`profile/` **and** `profile.*/`) stay: `profile/` is where the scorer now puts
  captures, and `profile.*/` still catches anything run by hand.
- **`vendor/ncu-report-skill` is pointed at for its prose, never its `helpers/`.** The skill note
  names `reference/05-analysis-dimensions.md` and `reference/06-diagnosis-playbook.md`, tells agents
  to resolve the skill's relative links against the vendor dir, and says to ignore its local-GPU
  collection workflow — `SKILL.md` step 4 ("parse with `ncu_report`, not by eye-balling the CLI")
  contradicts this prompt. It then explicitly steers *away* from `helpers/`, which is cheaper than
  letting an agent find them: they are thin `ncu_report` wrappers for a local capture that read
  `range_by_idx(0).action_by_idx(0)` and stop, carry no SASS/PTX or per-line counter attribution,
  and whose `rule_speedups()` reads keys (`estimated_speedup_pct`, `rule_name`,
  `message_for_display`) that Nsight 2026.2's `rule_results_as_dicts()` does not emit — it ranks
  all 18 findings at `0.0`/`'?'` instead of erroring. `veloq ncu` covers every one of those axes
  (and `plot_timeline.py`'s PM-sampling series, its one unique trick, finds 0 `pmsampling:*`
  metrics in a hosted capture). Historically only 3 of 25 members ever opened `SKILL.md` and none
  ran a helper. `_ncu_report_pythonpath()` survives with **no production caller** — it is the
  diagnostic for the version trap below, not a prompt input; nothing in a run depends on the
  locally-installed Nsight any more, since veloq ships its own pinned reader.
- **Installing a newer Nsight does not make it the one in use** — both that glob and `ncu` itself
  can keep silently resolving the old build. The `.run` installer defaults to
  `/usr/local/NVIDIA-Nsight-Compute-<ver>/`, which matches neither glob shape. And
  `/usr/local/cuda-*/bin/ncu` is not the profiler: it is a CUDA-toolkit sh dispatcher that scans the
  same two shapes (`/opt/nvidia/nsight-compute/*`, `$CUDA_BIN/../nsight-compute-*`), picks the
  highest version found, and `exec`s it. Neither path errors — `ncu --version` just keeps reporting
  the old build and `_ncu_report_pythonpath()` keeps handing agents the old `ncu_report`.
  Symlinking the install into the scanned dir fixes both at once:
  `sudo ln -s /usr/local/NVIDIA-Nsight-Compute-2026.2 /opt/nvidia/nsight-compute/2026.2.1`.
  Name the link with **three** version fields — the dispatcher splits the basename on `.` and
  computes `$(( year*10000 + major*100 + minor ))`; a missing third field makes that arithmetic
  expansion an error. Verify with `ncu --version` *and*
  `python -c "from kernelthing.orchestrator import _ncu_report_pythonpath as p; print(p())"`,
  since the two resolve independently.
- **`nvcc`/`ncu`/`nsys`/`nvidia-smi` may be installed and will happily run** — on whatever consumer
  GPU the dev box has, at a different architecture from the B200 being optimised. Nothing in
  `guard_core.js` blocks them (`gpu-tamper` only catches `CUDA_VISIBLE_DEVICES`/`LD_PRELOAD`/
  `KERNELTHING_*`), so the prompt has to call them a trap explicitly rather than merely omitting
  them. `torch`/`numpy` are deliberately absent, which is why agents that try a local sanity check
  hit `ModuleNotFoundError` instead of a wrong answer — a good failure, worth keeping.

### Automatic profiling (`popcorn.profile_submission`, `profile_nsys_submission`)

Profiling used to be an option in the agent's prompt, and the 2026-07-24 run measured what that
cost: of 25 members, **15 tried it and 12 got a capture**. Four were killed by opencode's bash
tool — the *model* chose `timeout: 360000` on 8 of 23 calls and every one of the run's kills was
one of those, while all 15 calls at 420000+ finished. The service never killed a job. Worse, a
kill still pays: the brev job runs to completion server-side and the CLI downloads artifacts only
after it sees `succeeded`. Three more members lost turns to the `workdir` trap.

So the harness takes it. `bench.popcorn.profile` (default **true**) makes every full score capture
hosted Nsight Compute for the scored shape, and `bench.popcorn.nsys` (default **true**) also starts
a minimal Modal/B200 Nsight Systems timeline for Cholesky problems:

- **Fired at test-pass, joined after the benchmark** — ncu, nsys and benchmark overlap. The profile
  work is the long pole (hosted ncu is 245–270s alone, 515s observed behind a queue) and the
  benchmark is ~172s, so overlapping costs the difference rather than the sum. A broken kernel
  returns before the captures start and never takes a profiler slot.
- **`--test-only` never profiles by default.** It is the ~10s call an agent makes most while
  iterating on correctness; a capture would make it ~25× slower and there is no timing to reason
  about yet. `--no-profile` suppresses both profilers on a full score.
- **Neither does the seed** (`_evolve_seed` passes `no_profile=True`), and that is the one place
  where *not* capturing is the feature. `_evolve_seed` removes the seed worktree in its `finally`
  seconds after the score returns, so a capture taken there is deleted before anything reads it —
  `members/0/result.json` recorded `profile.ok=true` beside six artifact paths that all pointed at
  the deleted tree. Its only lasting effect was harmful: captures are cached on the submission's
  sha256, and the first candidates score the *unmodified* seed file, so they hit that entry. A hit
  carries the flat text and no report (`_profile_cache_load` never memoises the tens-of-MB
  `.ncu-rep`), so the agent gets 190KB of `ncu-details.txt` to read by eye on its first look at the
  problem, with every `veloq` verb in its prompt and both analysis skills inapplicable. Skipping the
  seed capture makes that first score a deliberate cache *miss*. Test and benchmark still cache —
  those hits are pure win, and the incumbent metric comes from them.
- **It is evidence, not a verdict.** Every failure path returns a `Profile` / `NsysProfile` with
  `ok=False` and an `error`; none raise. The join is in a `finally`, so no exit path leaks a
  thread. Losing a capture must never change whether a kernel scored.
- **A score prints state; the prompt carries the reference material.** This is the split to
  preserve. `format_profile_block` / `format_nsys_block` emit ~600 bytes total: per capture, a
  banner saying whether it landed, **one** path bound to the `REP=` / `NSYS=` variable, and a line
  pointing at the verbs. The verbs themselves (`VELOQ_NCU_HINTS` / `VELOQ_NSYS_HINTS`, ~4.4KB)
  render **once** into `kernel-tools-veloq.md` via `veloq_verb_block`, because they are static —
  the same 30 lines regardless of what was measured. The 2026-07-24 run took a **median of 5 full
  scores per member, 12 at the tail**, so printing them per score meant ~22KB (up to ~53KB) of
  byte-identical text permanently resident in one member's context, against 4.4KB once in a cached
  prompt prefix. Do not move static text back into a score; do not move per-score state into the
  prompt. Earlier still, the block inlined a 12KB digest plus up to 12KB of nsys stats, which put
  the same pre-chosen dozen numbers in front of the agent every time and invited it to stop there —
  one report holds ~22.5k metrics and ~108 rule findings a digest cannot reach. Every listed `ncu`
  verb is smoke-tested against a real capture; the `nsys` verbs are flag-checked against `--help`
  only, since no `.nsys-rep` exists on this box.
- **`format_analysis_directive` opens every score with the marching order** — one bottleneck, one
  optimization, the analysis skills by name, the CUDA-docs MCP and ptx-skill for API/hardware
  questions. Instruction before data: it leads, the capture blocks that say what landed follow.
  It prints nothing when neither capture landed, and *that* is the load-bearing case now that it
  is first — a score that captured nothing must not open by sending the agent to read reports that
  are not there. Since it no longer sits in the recency slot, the pointer back to the prompt's verb
  list rides on the capture blocks instead, so keep both of those naming `veloq ncu` / `veloq nsys`
  explicitly (front-loading reference material is how the vendored ncu skill ended up opened by
  3 of 25 members).

  **It is not wrapped, deliberately.** It used to go through `textwrap.fill(width=84)` — the only
  `textwrap` call in the codebase, wrapping one of five blocks in an output whose `REP=`/`NSYS=`
  lines run to 111 chars unwrapped. The reader is a model, not a terminal, and every candidate
  width breaks one of the hyphenated names the directive exists to hand over verbatim: 70 splits
  `ncu-report-skill`, 84 `nvidia-cuda-docs`, 100 `nsys-profile-analysis`. Keep it one line.

  One thing about the current wording is worth knowing before editing it: the per-report narrowing
  is **inert**. `format_analysis_directive` replaces `" and Nsight Systems"` / `"Nsight Compute
  and "`, and the text contains neither, so a score where only one capture landed still names both.
- **The flat text view is the fallback, and only that** (`_capture_block`). Two ways to reach it:
  no `veloq` on PATH, or a cached score (the 80MB report is deliberately not re-downloaded). Both
  print `ncu-details.txt` / `stats.txt` instead of a path the agent cannot query — and since the
  verb list is unconditional in the prompt now, this per-score line is the *only* thing that tells
  an agent the report named up there is absent this time. `profile_digest` still writes `digest.txt`
  beside them, but nothing the agent sees names it any more; it is a human artifact now.
- **Nsight Systems is v1 Cholesky-only.** It parses `batch`, `n` and `seed` from
  `benchmark_spec`, sends the submission source to `kernelthing/nsys_modal.py`, warms once, and
  captures one CUDA-profiler-range call. Missing Modal auth, missing CLI, unsupported
  leaderboards, malformed specs, timeouts and empty artifacts all become `bench.nsys.ok=false`.
- **Its input is a dense SPD batch, built from one Householder reflector.** `A = QDQ` for
  `Q = I - 2vvᵀ` and `D = linspace(1, 2)`, so the spectrum — and therefore `cond` — is exactly 2,
  matching every benchmark row on the board (they are all `cond: 2` with no `case:` field). It was
  `diag_embed(linspace(...))` alone, which had the right spectrum but exact-zero off-diagonals:
  `case: diagonal` is one of the service's *correctness* cases, not a timed one, so the timeline
  was of a code path the board never benchmarks. Any kernel branching on a value — early exit on a
  small pivot, an `isfinite` guard, a non-convergence fallback — profiled wrong. Nothing else was
  affected: ptxas never sees the data, zeros are normal FP32 values at full throughput, and the
  storage is dense either way. **Keep the construction O(batch·n²)** — the rank-2 update is why
  this is milliseconds instead of the minutes `torch.linalg.qr` would cost at n=32768. It all runs
  before `cudaProfilerStart`, so it never enters the capture, but it does spend the job's wall
  clock against `PROFILE_TIMEOUT_S`.
- **`brev_defuse` hyphenates the evaluator's rejected substring** before any of this reaches an
  agent. Nsight prints it once per kernel header, and quoting a capture verbatim would hand the
  agent a string that silently poisons any file it lands in.
- **The verdict JSON carries paths, never capture text** (`Profile.record` / `NsysProfile.record`). The JSON line is the
  seam; a 12KB digest or nsys stats body crossing it would land in every `result.json` and every
  journal event on every score. `score_command` prints the hint blocks to stdout **above** the
  verdict instead, because `_cli_score` finds the JSON by scanning stdout in reverse for the last
  line starting with `{`. `_one_line` flattens profiler error text into its banner for the same
  reason — it is the only part of a block we do not author, and a newline plus a brace in it would
  shadow the verdict.

`tests/fixtures/popcorn/ncu-details.txt` is a verbatim capture (index 2 of the cholesky board,
recovered from member 10's `opencode.ndjson`). Re-capture rather than hand-edit, like the
`--output` fixtures beside it.

### Seeing what a score prints without paying for one (`dryrun.py`)

The text above — two capture banners, the directive, the verdict — is the loop's tightest
feedback channel and was invisible until a real submission had been paid for.
`kernelthing score <dir> --dry-run [SCENARIO]` replays it from `tests/fixtures/popcorn/`.
Three properties are the whole point, and are what `tests/test_dryrun.py` holds:

- **It drives the real `popcorn.score_command`.** Only the three network boundaries are
  swapped (`_BOUNDARIES` = `submit`, `profile_submission`, `profile_nsys_submission`), so
  block order, wording and the verdict dict all come from production code — edit a banner
  and the replay follows without `dryrun.py` being touched. `test_stdout_is_the_production_fold_and_carries_no_dry_run_marker`
  re-renders the blocks from the verdict's own `bench` dict and demands equality.
- **stdout is byte-exact; the "this is a replay" notice goes to stderr.** A marker on
  stdout would defeat the comparison the command exists for. Nothing downstream ever
  passes `--dry-run`, so the fake verdict has no path into a journal.
- **It submits nothing, writes nothing, and does not touch the submission cache** — the
  swap is above `_cache_load`/`_cache_store` and above the code that creates
  `profile/latest/`. `resolve_config` and `_precheck` *do* run against the problem's real
  `submission.py`, so a dry run still catches a bad manifest or a rejected substring.

- **A replay is always brief, and there is no flag to opt out.** It replays what an
  *agent* reads and agents run `--brief`, so a bare `--dry-run` must too — faithfully
  rendering output no candidate ever sees would be worse than no replay at all.
  `test_the_replay_is_exactly_what_an_agent_runs` pins the bare command to
  `_score_cmd_str`'s flags. A **real** score is untouched and still defaults to full.
  Nothing needs an escape hatch for `bench`: a test that wants it takes `popcorn.score`
  under `_replayed` directly (a better check than a flag existing to be looked at), and
  every archived run already has 20-odd `result.json` files showing its shape.

`--test-only` / `--no-profile` are not re-implemented; they reach `score` unchanged.
The scenarios exist because the degraded blocks are the ones nobody sees until they happen
in a run: `cached` (the 80MB report is not re-downloaded, so there is no path to query),
`nsys-fail` (a reason instead of a path, and the directive drops to naming ncu alone),
`test-fail` (exit 1, no blocks at all — profilers fire at test-pass, so a broken kernel
never spends a slot). Fixtures ship with the source tree, so `--dry-run` needs an editable
install; it says so rather than raising when they are absent.

### Reading the capture: `veloq`, and the version trap (`config.veloq_python`)

The scorer writes `profile/latest/{ncu-details.txt,digest.txt,profile.ncu-rep}` and
`profile/latest/nsys/{profile.nsys-rep,profile.sqlite,stats.txt}`. The ncu text file is
a rounded rendering; the ncu report holds the same capture structured — per-launch metrics, the
profiler's rule findings with severities and `focus_metrics`, and per-source-line warp-stall
histograms. `veloq ncu <verb>` reads it, **on this box, with no GPU** — it is a file parser.
`veloq nsys <verb>` does the same for the `.nsys-rep` (it also accepts a pre-exported
`_pqtdir/`, not the `.sqlite`). Measured on a real capture: 9 launches, 108 rule findings,
22509 metrics.

**One verb list, in `popcorn.py`.** `VELOQ_NCU_HINTS` / `VELOQ_NSYS_HINTS` are the only copy;
`kernel-tools-veloq.md` carries `{{NCU_VERBS}}` and `kernel-tools-nsys.md` carries
`{{NSYS_VERBS}}`, which `_kernel_tools_block` and `dump_prompts` both fill from
`popcorn.veloq_verb_block(group)`. The constants are where the list is checked against a real
capture, so a hand-written copy inside a `.md` is how the prompt and reality drift apart. The
rendered commands are written against `$V`/`$REP`/`$NSYS`, assigned in a fence just above them —
the same variable names a score's block prints, so a path copied out of a score drops into a
command copied out of the prompt.

**One section per report, each self-contained** (its own `V=`/`REP=` or `V=`/`NSYS=` fence and
verbs). That is not cosmetic: the nsys section is dropped entirely when a problem sets
`bench.popcorn.nsys` false, and a combined section would either leak a timeline that never lands or
need a conditional the `.md` cannot express. `test_each_reports_prompt_section_carries_its_own_verbs`
asserts neither section leaks the other's surface.

**A report section is commands and constraints only.** It does *not* enumerate what the capture
holds — no "per-launch metrics, rule findings, warp-stall histograms, SASS/PTX". That is the digest
problem in miniature: a prose list pre-picks the dimensions and invites the agent to look no further
than the ones named, when the verb list beside it already says what each verb answers and the report
holds ~22.5k metrics. Keep the framing to where the file is, how to query it, and what not to
believe.

Two things make this fragile enough to be worth the note:

- **The bundled reader is mandatory, not preferred.** The hosted profiler emits Nsight
  Compute **2026.2** captures, and the reader must match. A too-old reader fails *every* verb,
  not just the warp-stall one — it dies building the sidecar with
  `'IAction' object has no attribute 'timed_warp_samples'`, which is what a 2025.4.1 `ncu_report`
  did here before 2026.2.1 was installed alongside it. So `config.veloq_python()` resolves veloq's
  own venv (`<xdg-data>/veloq/ncu-report-*/`) and the block is dropped when that is absent.
  **There is no fallback to `_ncu_report_pythonpath()`; do not add one** — and the fact that the
  system Nsight currently *does* match is not a reason to add one. The two versions float
  independently: the hosted profiler upgrades on gpu-mode's schedule and the local Nsight is a
  hand-installed tarball, so they agree only by coincidence and only until either moves. veloq's
  venv is version-pinned to the reader it needs; a fallback would silently swap in a mismatched
  one and turn a dropped prompt block into 108 rule findings of garbage.
- **The pin exists because of the XDG repoint.** `build_opencode_env` gives each candidate an
  isolated `XDG_DATA_HOME`, which is exactly where veloq looks for that venv — so under the
  sandbox it would vanish. `VELOQ_PYTHON` is pinned to the absolute path (bwrap ro-binds `/`,
  so it resolves from any worktree). It is a `setdefault`: an operator's exported value wins.
  `test_veloq_env_pin_survives_the_per_candidate_xdg_repoint` clears the var first for exactly
  that reason — a developer with it exported would otherwise be testing their own shell.

`<report>.veloq/` sidecars are written next to the report, so `profile.*/` already ignores them.

### Skills are vendored, never loaded (`sandbox.SKILL_HOMES`)

`~/.claude/skills` and `~/.agents/skills` are tmpfs-masked, and that is where the useful ones
actually live. So they are copied into `vendor/` and surfaced as **paths in the prompt**:

| tree | source skill | note builder | section heading |
|---|---|---|---|
| `vendor/veloq-ncu-skill` | `ncu-profile-analysis` (veloq's) | `_veloq_ref_note` | Diagnosing an ncu report |
| `vendor/veloq-nsys-skill` | `nsys-profile-analysis` (veloq's) | `_veloq_ref_note` | Reading an nsys timeline |
| `vendor/ncu-report-skill` | `b200-ncu-report-skill` (`DongyunZou/ncu-report-skill`) | `_ncu_skill_note` | B200 profiling reference |
| `vendor/ptx-skill` | `ptx-skill` | `_ptx_note` | PTX / CUDA ISA reference |

All are plain copies, not submodules — `git submodule update --init` is broken repo-wide. Refresh
one with `cp -r ~/.claude/skills/<name> vendor/<tree>`; `test_every_pointed_at_skill_file_exists`
is what catches a copy that lost its `references/`, since `_vendored()` only tests the directory.

**Each gets its own `###` section, all four through `_skill_part` + `kernel-tools-skill.md`.** They
used to be trailing `{{*_NOTE}}` placeholders glued onto the end of whatever section happened to be
nearby, which meant four different framings for four documents of the same kind. One template makes
them un-driftable, and it is why the note builders take no per-skill wording: what distinguishes the
sections is the heading. `_skill_part` returns `''` for an empty note, so an absent tree drops the
whole section — that check is what makes it safe to put the heading in the `.md`.

Point at **named files, never a bare `SKILL.md`**: of `vendor/ncu-report-skill`, only 3 of 25
members ever opened it and none ran a helper. Every note above follows that rule — description and
reference index are read *verbatim from the vendored SKILL.md* (`_skill_description` /
`_skill_section`), never transcribed, because a hand-written summary of someone else's document is
worse than theirs by construction and goes stale silently the moment the copy is refreshed.

The two veloq skills share `_veloq_ref_note` with **no per-skill parameter at all**: same document
shape, so any differing text would be paraphrase creeping back in. Neither contradicts this setup —
both are pure readers of a report the loop already downloaded — so neither needs a caveat.

**The other two do, and theirs are load-bearing:**

- `_ptx_note` leads with the caveat rather than the paths: that skill opens with
  `compute-sanitizer`, `cuda-gdb` and `nvcc -g -G`, and those binaries **are installed here and
  will run** — on a consumer GPU at the wrong architecture, returning numbers that look real.
- `_ncu_skill_note` is prose-only by construction. Its SKILL.md quickstart and
  `reference/03-collection.md` build a harness and profile locally (same trap), and its
  `helpers/*.py` are dead — they read `range_by_idx(0).action_by_idx(0)` and stop, carry no
  SASS/PTX or per-line attribution, and `rule_speedups()` reads keys Nsight 2026.2 no longer emits,
  so it ranks all 18 findings at `0.0`/`'?'` rather than erroring. That is why the index comes from
  the **`### Reference docs` subsection**, not the whole `## File index` — pasting that table would
  name all seven helpers one line above the sentence telling the agent to skip them. Hence
  `_skill_section`'s `level` parameter. It also excises the description's trailing Chinese trigger
  phrases, the same way `_ptx_note` drops its "Triggers on …" tail: router dispatch metadata means
  nothing to an agent already holding the path.

### CUDA-docs MCP (`opencode_client.CUDA_DOCS_MCP_URL`)

Declared in the generated opencode config, on by default (`cfg.mcp_cuda_docs`, `--no-cuda-docs`).
opencode's schema is `McpRemoteConfig` — `type: "remote"`, **not** the `"http"` that the
`.claude.json` form uses; copying that shape across silently produces no tools.

The server is OAuth-gated (401 + RFC 7591 dynamic registration), so it needs **a one-time
interactive auth**: a sandboxed candidate has no browser and cannot do it. The token then rides
in opencode's `auth.json`, which `_seed_auth_for_isolated_data` already copies per candidate.
Until someone completes that flow, the declaration is inert — verified that a dead MCP endpoint
leaves opencode's startup and exit path byte-identical to no MCP at all, so it fails open.

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
plan paths because the problem sits at the repo root there, not under `problems/<name>/`. It rebuilds
that repo on every run but **preserves `.humanize/`** and skips `runs/` in the copy (see the artifact
section above) — a `runs/` archive kept next to a problem must not end up committed into the managed
repo and materialised in every worktree.

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

### Prompts: the tree is now exactly the working set

kernelthing began as a port of Humanize and `prompts/` used to carry that whole tree — 50 of its 75
files were unreferenced (all of `codex/`, `plan/`, `idea/`, 16 of 21 `claude/`, 25 of 45 `block/`).
They are deleted. What remains is loaded, and `prompts/codex/`, `prompts/plan/` and `prompts/idea/`
no longer exist:

- `prompts/claude/bootstrap-problem.md`, `bootstrap-mode-{auto,interactive}.md` — loaded by
  `bootstrap.py`.
- `prompts/claude/kernel-tools-{profile,veloq,nsys,skill,cuda-docs}.md` — loaded by
  `Orchestrator._kernel_tools_block`. One section per tool surface; `veloq` is the Nsight Compute
  half and `nsys` the Nsight Systems half, split so the latter can be dropped whole for a problem
  with `bench.popcorn.nsys` off. `skill.md` is a two-line template (`### {{SKILL_TITLE}}` +
  `{{SKILL_NOTE}}`) rendered **once per vendored skill** by `_skill_part`, so all four skill
  sections have identical shape and none can drift.

  **Section order is three bands**, and `test_the_tools_block_is_ordered_turn_loop_then_reference_then_commands`
  pins the exact list — appending a `parts.append` at the bottom of the method is otherwise
  invisible until someone reads a rendered prompt:

  1. *the turn loop* — Timing / Profiling / Rules (`profile.md`, one file, three `###`s)
  2. *reference* — ncu-profile-analysis, ncu-report-skill, nsys-profile-analysis, CUDA-docs MCP, ptx-skill
  3. *commands* — `veloq ncu`, `veloq nsys`

  Commands last is the recency slot: this block sits immediately above the task. That the
  skills interpreting a report precede the verbs producing it is fine — every section is
  self-contained (`_veloq_ref_note` opens with its own vendored path), so none reads as a
  forward reference. The two *backward* citations are the constraint to preserve:
  `kernel-tools-veloq.md` cites "the profiling section above" for the evaluator-rejected token,
  and `kernel-tools-nsys.md` defers to the ncu section on durations-as-ratios. `dump_prompts.render_kernel_tools`
  mirrors the order, asserted by `test_the_dump_renders_the_sections_in_the_order_candidates_get_them` —
  a dump in a different order would document a prompt no candidate receives.
- `prompts/block/*.md` (20) — rendered by `guard_core.js`, one per `block(cfg, "<name>", ...)` call.
  `render()` falls back to the inline message when a file is missing, so a stale name degrades
  quietly; that is also why an unreferenced template is invisible until you go looking.

When adding a prompt file, wire it in the same commit — an unreferenced one is indistinguishable
from legacy a month later.

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
