# Orchestration flow: how kernelthing searches for faster kernels

How one `kernelthing <problem>` invocation turns a starting kernel into a better one. The search
policy is pure and lives in `kernelthing/evolve.py`; everything side-effecting (git worktrees, agent
turns, remote scoring submissions, journaling) lives in `kernelthing/orchestrator.py`.

The shape in one sentence: **a single controller thread keeps N agent workers busy editing kernels in
isolated git worktrees, each result is scored by an authoritative remote benchmark (the hosted
popcorn service) in its own subprocess, and the measurement — nothing else — decides what becomes a
parent for the next generation.**

---

## 1. The moving parts

```
                    ┌─────────────────────────────────────────────────────────┐
                    │  kernelthing process                                    │
  CLI args ───────► │                                                         │
                    │  cli.run_loop ──► Orchestrator.run ──► _run             │
                    │                                        │                │
                    │   ┌────────────────────────────────────┴─────────────┐  │
                    │   │ controller thread (the "pump")                   │  │
                    │   │   • owns Population (evolve.py)                  │  │
                    │   │   • picks operator + parent                      │  │
                    │   │   • dispatches / collects futures                │  │
                    │   └───────┬──────────────────────────────────────────┘  │
                    │           │ ThreadPoolExecutor(max_workers=64)          │
                    │   ┌───────┴────────┬────────────────┬───────────────┐   │
                    │   │ worker 0       │ worker 1       │ worker k ...  │   │
                    │   └───┬────────────┴───┬────────────┴───┬───────────┘   │
                    └───────┼────────────────┼────────────────┼───────────────┘
                            │ subprocess     │                │
                    ┌───────▼──────────┐ ┌───▼──────────┐ ┌───▼──────────┐
                    │ bwrap + opencode │ │  same, per   │ │  same, per   │
                    │  (edits kernel)  │ │    worker    │ │    worker    │
                    └───────┬──────────┘ └──────────────┘ └──────────────┘
                            │ subprocess
                    ┌───────▼───────────────────────────┐
                    │ python -m kernelthing score       │  ← its own process on purpose
                    │   └─► popcorn.score               │  ← submits to the hosted service
                    └───────┬───────────────────────────┘
                            │ HTTPS  popcorn submit + GET /user/submissions/<id>
                    ┌───────▼───────────────────────────┐
                    │ hosted popcorn service (B200)     │  ← the only place kernels run
                    └───────────────────────────────────┘

  side channels (files, not memory):
     run dir  .humanize/rlcr/<ts>/{run.json, events.ndjson, control.json, live.lock, members/}
              ▲ written by the loop            │ read by webui  │ control.json written by webui
```

Two facts that explain most of the design:

- **Nothing runs a kernel locally.** There is no local GPU, no benchmark engine, and no GPU lock;
  every score is a remote submission to the popcorn service, which grades on a real B200.
- **The web UI shares no memory with the loop.** It reads the run directory. `control.json` is the
  only path back in.

---

## 2. Startup: from CLI to a runnable problem

```
  kernelthing [problem|objective] ...
        │
        ├─► resolve_problem()
        │     ├─ dir has problem.json ──► load_problem()
        │     │                           └─► prepare_problem()   COPY into managed repo
        │     └─ else ─────────────────► bootstrap.bootstrap_problem()   agent authors it
        │
        ├─► webui.start_server() daemon thread, unless --no-web
        └─► Orchestrator(problem, cfg).run()
```

`prepare_problem` (`kernelthing/problem.py:97`) is load-bearing: the problem dir is **copied** into
`~/.cache/kernelthing/<name>/`, `git init`'d, and committed. Every worktree branches from that copy,
so the source repo is never mutated and the search always forks from committed state.

---

## 3. Run lifecycle

```
Orchestrator.run()
  │
  └─ _run()
       │
       ├─ setup()           run dir, run.json, live.lock, Journal, LoopControl
       │                    emits: run_start
       │
       ├─ emit phase=evolve
       │
       ├─ _evolve_seed()    score HEAD as member 0 (one remote submission)
       │                    emits: member_result, new_best
       │
       ├─ emit search_start
       │
       ├─ ╔══════════════════════════════════════════════╗
       │  ║  THE PUMP  (section 5)                       ║
       │  ║  dispatch ──► agents ──► score ──► collect   ║
       │  ╚══════════════════════════════════════════════╝
       │
       ├─ promote:  git reset --hard <best.commit>
       │            _copy_best_kernel() → ~/.cache/kernelthing/<name>-best/
       │            emits: promoted
       │
       ├─ _evolve_cleanup() delete refs/kernelthing/<ts>/*, worktree prune, rmtree
       │
       └─ _finish()         optional methodology retrospective; emits run_end
  │
  └─ finally: journal.close(), live_lock.release()
```

---

## 4. Seeding: the starting kernel becomes member 0

Before any agent runs, the starting `HEAD` is scored as member 0 (`_evolve_seed`). It is one remote
submission, same as any candidate:

```
  kernelthing score <seed worktree>
                │
                └─► becomes member 0's own correct/metric — the incumbent to beat
```

The metric is an absolute time (µs), so there is no baseline denominator to pin — the number the
service returns *is* the score. Consequence:

- If the seed fails to score, the population starts **empty**. That is handled, not fatal: with no
  elites, `choose_operator` forces `explore` forever and every task forks the base commit until one
  works.

---

## 5. The pump: steady-state async dispatch

This is the core loop (`orchestrator.py:976`). There are no rounds and no barriers — nothing waits
for a generation to finish.

```
   ┌─ prime ──────────────────────────────────────────────────────────┐
   │  while want_more() and len(futures) < target():                  │
   │        dispatch()                                                │
   └──────────────────────────────────────────────────────────────────┘
                              │
   ┌─ drain ──────────────────▼───────────────────────────────────────┐
   │  while futures:                                                  │
   │      done = wait(futures, timeout=15s, FIRST_COMPLETED)  ◄────┐   │
   │      for fut in done:  collect(fut)                          │   │
   │      while want_more() and len(futures) < target():          │   │
   │            dispatch()                ────────────────────────┘   │
   └──────────────────────────────────────────────────────────────────┘

   want_more()  =  not stop_requested
                   and (max_candidates == 0 or dispatched < max_candidates)
                   and (wall_clock == 0 or elapsed < wall_clock)
   target()     =  clamp(control.parallelism, 1, MAX_PARALLELISM=64)
```

Three deliberate details:

- **The thread pool is sized to `MAX_PARALLELISM` (64), not `-j`.** Threads spawn lazily, so `-j` is
  the only thing deciding how many agents run — which is what lets you *raise* it mid-run from the
  web UI, not just lower it.
- **The 15-second bounded wait** exists so a `-j` bump refills immediately instead of waiting for
  some agent to finish. Every `want_more()`/`target()` call re-reads `control.json` (mtime+inode
  guarded) before answering.
- **Budgets are checked at dispatch boundaries only.** Hitting the wall clock stops *new* dispatches;
  in-flight agents run to completion, then the drain loop exits. A run does not end at exactly `-w`.

Occupancy over time looks like this — a finished slot refills without waiting for a cohort:

```
  agent A  ├──── edit ────┤├score┤
  agent B     ├───── edit ─────┤├score┤
  agent C  ├── edit ──┤├score┤
  agent D                 ├───── edit ──────┤├score┤
           └──────────────────────────────────────────────► time
                     ▲ D dispatched the instant C's result landed
```
(`score` is a remote popcorn submission; several can be in flight at once, bounded by `-j` and the
service's own rate limits.)

---

## 6. Choosing what to try next

`_ev_dispatch` (`orchestrator.py:800`) makes two decisions per task: **which operator**, then **which
parent**.

### 6a. Operator

```
  explore_frac ──┬─ manual (UI slider off "auto") ─► explore_bias / 100
                 │
                 └─ auto ─► _auto_explore_frac():  0.8 − 0.6 · min(1, progress)
                              progress = max of whichever budgets are set —
                                 dispatched / live max_candidates
                                 elapsed    / wall_clock
                              neither set ──────────────────────► 0.5 (flat)
                                    │
                                    ▼
                       evolve.choose_operator()                     evolve.py:232
                                    │
                    ┌───────────────┴───────────────┐
                    │ no viable elites yet?         │──yes──► EXPLORE  (forced)
                    └───────────────┬───────────────┘
                                    │ no
                    ┌───────────────┴───────────────┐
                    │ len(niches) < min_niches (4)? │──yes──► weight[EXPLORE] × 2
                    └───────────────┬───────────────┘
                                    ▼
                        weighted random pick
```

The default schedule anneals from 80 % explore at the start to 20 % at budget exhaustion: broad early,
greedy late. Progress tracks whichever budget is nearest its limit, mirroring `_ev_want_more`'s
whichever-comes-first stop, so the anneal always completes exactly as the run ends — including for
wall-clock-only runs (`-m 0 -w 8h`). Budgets are read through `_ev_max_candidates()` / `_ev_wall_limit()`,
which return the **live** control values, so raising `-m` mid-run re-scales the schedule instead of
leaving it stuck at its tail value.

### 6b. Parent (bandit with virtual loss)

```
  EXPLOIT ──► pool = elites()          top-K viable by metric        evolve.py:185
  EXPLORE ──► pool = all viable        sorted best-first             evolve.py:213

              pool empty? ──► parent = None ──► operator downgraded to EXPLORE,
                                                task forks the base commit

  weight(m) for EXPLOIT:   ( normalized_metric(m) + c·√(ln(Σchildren+1)/(children(m)+1)) )
                           ────────────────────────────────────────────────────────────────
                                                1 + in_flight(m)                    c = 0.7

  weight(m) for EXPLORE:              √(ln(Σchildren+1)/(children(m)+1))
                                      ──────────────────────────────────
                                            1 + in_flight(m)
```

`children` counts tasks ever dispatched from a member; `in_flight` counts tasks dispatched from it
that have **not yet returned**. Dividing by `1 + in_flight` is the *virtual loss*: without it, four
concurrent dispatches would all pile onto whichever member currently looks best, because none of
their results have landed to update the statistics. Explore drops the exploitation term entirely, so
it fans out across neglected branches of the whole frontier rather than always restarting from the
seed.

### 6c. The prompt

```
  EXPLORE prompt          EXPLOIT prompt
  ─ parent metric         ─ parent metric
  ─ list of niche keys    ─ parent commit message
    already in the                 │
    archive, with        ─ "push further along the SAME approach"
    "pick a different            │
     angle"                      │
        └──────────┬─────────────┘
                   ▼
        + EVOLVE_DESCRIPTOR_FOOTER   (self-test command, "commit as soon as you have
        |                             ANY correct improvement", write candidate-summary.md)
        + _kernel_tools_block        (scoring + profiling surfaces, vendored skill
        |                             pointers, veloq verbs — cached per run)
```

These four templates are **inline constants in `orchestrator.py`**, not files under `prompts/`.

---

## 7. The life of one candidate

`_evolve_task` (`orchestrator.py:607`) runs on a worker thread. It never raises — every failure mode
becomes a dead `Member` with an `error` string.

```
  ┌────────────────────────────────────────────────────────────────────────────┐
  │ [git lock]  git worktree add --detach --force  wt/m<N>  <parent_commit>    │
  ├────────────────────────────────────────────────────────────────────────────┤
  │             write members/<N>/prompt.md                                    │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  ── AGENT TURN ── (minutes; API-bound)                                     │
  │     bwrap ─► opencode run --format json --auto                             │
  │       • writable: worktree, its own XDG state, main .git                   │
  │       • guard: PreToolUse plugin, fake loopDir inside the worktree         │
  │       • the agent edits, then tests/benchmarks/profiles via `popcorn`      │
  │     stdout ─► members/<N>/opencode.ndjson   (tail -f -able, live UI reads) │
  ├────────────────────────────────────────────────────────────────────────────┤
  │             git rev-parse HEAD                                             │
  │               ├─ unchanged ─► error "no commit" / "agent turn timed out"   │
  │               │               (exit 124) ─────────────────────► DEAD, return│
  │               └─ changed ──► record commit, subject, full commit list      │
  ├────────────────────────────────────────────────────────────────────────────┤
  │             git diff parent..HEAD -- <edit_files>  ─► members/<N>/diff.patch│
  │             git diff --name-only                   ─► changed_files         │
  │ [git lock]  git checkout -- <edit_files>    discard UNCOMMITTED edits      │
  ├────────────────────────────────────────────────────────────────────────────┤
  │  ── SCORE ──  _guarded_score()  (section 8)                                │
  ├────────────────────────────────────────────────────────────────────────────┤
  │ [git lock]  git worktree remove --force        (finally: always)           │
  └────────────────────────────────────────────────────────────────────────────┘
```

Two subtleties:

- **Only the last commit is scored.** The `git checkout -- <edit_files>` before scoring throws away
  uncommitted working-tree changes, so what gets measured is exactly what the agent committed. This
  is why the prompt insists on committing as soon as anything correct exists — a timeout then costs
  the improvement-in-progress, not the whole task.
- **`diff.patch` is the edit-files diff only.** Agents are told to `git add -A`, which drags in build
  output and profiler dumps; the full inventory survives as `changed_files` in `result.json`.

Then `_ev_collect` (`orchestrator.py:869`) folds the result back in:

```
  future done ──► release parent's in_flight
              ──► pop.insert(member)          reclassifies elite/live/dead
              ──► viable? git update-ref refs/kernelthing/<ts>/mem-<id>   (keeps the commit alive)
              ──► write members/<id>/{summary.md,result.json}; emit member_result
              ──► best changed? emit new_best
```

---

## 8. Scoring: the remote submission

```
  _guarded_score()
     │
     ├─ 1. kernelguard static scan  ── cheat found? ──► DEAD, nothing is submitted
     │      (timer monkeypatching, result/CUDA-graph replay, shape hardcoding, …)
     │
     └─ 2. _cli_score():  subprocess ──►  python -m kernelthing score <dir>
                │
                └─ popcorn.score()   (kernelthing/popcorn.py)
                          │
                          ├─ submit  --mode test       correctness first (cheap)
                          │            └─ fail? ──────► {"correct": false, …}, no benchmark
                          ├─ submit  --mode benchmark   timing, only if test passed
                          ├─ GET /user/submissions/<id>  full-precision ns from the API
                          │    (text --output is the fallback when the API is unreachable)
                          └─ stdout: {"correct":…, "metric":…, "bench":{per-shape times, source}}
```

Why the subprocess is not optional: `popcorn.score` mutates the process cwd and shares an on-disk
submission cache; running each score in its own process keeps concurrent scorings from stepping on
each other. It is also the *same* entry point agents call (`kernelthing score`). **The process
boundary is the isolation.**

Why two submissions: a broken kernel never pays for a benchmark — `--mode test` gates `--mode
benchmark`, and `correct` is the conjunction (the benchmark re-checks every shape, so a kernel that
only breaks at scale still fails). See `popcorn.py`'s module docstring for the API-vs-text data path,
the `benchmark_spec` index-drift pin, and the submission cache keyed on the file's sha256.

Concurrency: there is no local device to exclude on. Up to `-j` scores run concurrently, each a
separate `popcorn` submission; the only ceiling is the hosted service's own rate limits.

---

## 9. The archive

```
  Population (evolve.py:160)
     members[]  ─ every attempt ever, including dead ones

     viable  ≡  correct AND has a commit AND has a metric

     ┌─ elites() ──────────────┐   top-K viable by metric        → the EXPLOIT pool
     │   K = control.elite_k    │   (live-tunable; re-read before every selection)
     └──────────────────────────┘
     ┌─ niches() ──────────────┐   best viable member per niche key
     │   key = commit subject,  │   → MAP-Elites diversity grid
     │   lowercased, ≤60 chars  │   → also the "strategies already tried" list in
     └──────────────────────────┘     the EXPLORE prompt

     status:  ELITE ⊂ LIVE (viable)   |   DEAD (not viable)
```

The niche key being the **commit subject** is the whole diversity mechanism: the agent names its own
strategy when it commits, distinct names create distinct niches, and `min_niches` biases toward
explore while the grid is thin. It is cheap and it works, but it is only as good as the agent's
commit messages — two genuinely different strategies described identically collapse into one niche.

---

## 10. Feedback loops

```
     loop process                     files                      web UI
   ─────────────────────────────────────────────────────────────────────────
     Journal.emit()  ──────────►  events.ndjson  ──────────►  GET /api/events
                                  (append-only,               ?offset=  (incremental
                                   seq + t)                    fold, no polling cost)

     opencode stdout ──────────►  members/<id>/               GET /api/agents
                                  opencode.ndjson              (live tool call + cost)

     LoopControl.parallelism() ◄─  control.json   ◄──────────  POST /api/control
     .elite_k() .wall_clock()      (atomic replace,            (409 unless live.lock
     .max_candidates() .stop()      clamped keys)               is held)

     LiveLock (flock) ──────────►  live.lock      ──────────►  is_live()
```

Every observed control change is itself journaled as a `control_changed` event, so a run's history
records mid-run tuning. Event types emitted: `run_start`, `phase`, `search_start`, `dispatch`,
`member_result`, `new_best`, `control_changed`, `promoted`, `methodology_turn`, `run_end`.

`dispatch` events deliberately carry the selection context (`explore_frac`, `niches`, `elites`,
`parent_metric`) — with an RNG and a live population, *why* the search made a choice is otherwise
unreconstructable after the fact.

---

## 11. Termination

```
   pump exits (budget spent, or stop requested)
        │
        ├─ best = pop.best()
        │     ├─ found ──► git reset --hard <commit>  in the managed repo
        │     │            copy edit_files → ~/.cache/kernelthing/<name>-best/
        │     │            emit promoted
        │     └─ none ───► HEAD unchanged
        │
        ├─ _evolve_cleanup: delete refs/kernelthing/<ts>/*, worktree prune, rmtree wt/
        │
        └─ _finish(reason):
              stop_requested ──► "stopped"
              best is None ─────► "stalled_out"
              otherwise ────────► "maxiter"
                 │
                 └─ if --methodology: retrospective agent turn (up to 3 attempts)
                 └─ emit run_end
```

`Ctrl-C` is handled separately in `cli.run_loop`: it calls `persist_current_head()` so an interrupted
run still saves whatever kernel was last promoted, then exits 130.

---

## 12. Sharp edges

- **`EXIT_COMPLETE`, `EXIT_STOP`, and `EXIT_ERROR` are dead constants** — `_run()` only ever returns
  `"stopped"`, `"stalled_out"`, or `"maxiter"`.
- **A UI stop silently skips the methodology retrospective.** `_finish` gates it on `EXIT_STOP`
  (`"stop"`), but the reachable value is `EXIT_STOPPED` (`"stopped"`). `orchestrator.py:1062` vs
  `orchestrator.py:1005`.
- **`pool_cap` in `_run` is display-only** — it appears in the log banner, but the executor is sized
  to `MAX_PARALLELISM`.
- **`stalled_out` does not mean "stalled".** Despite the constant's comment about consecutive
  no-progress rounds, it is returned when the search finished with *nothing viable at all*. There is
  no stagnation detector; the run stops on budget alone.
- **Wall-clock and candidate budgets bound dispatch, not completion.** Expect the run to overshoot
  `-w` by roughly one agent turn.
- **Every gate fails open.** kernelguard unimportable → no violations. Guard config missing → no
  enforcement. popcorn API unreachable → the score falls back to scraping the CLI's text output. This
  is intentional: a gate that raises can wedge a whole run. Preserve it when adding gates.
- **`git checkout -- <edit_files>` before scoring is load-bearing.** Removing it would let
  uncommitted edits be measured, so the promoted commit would not be the thing that scored.

---

## 13. The whole loop in one picture

Everything above, with the function names taken out. Sections 5–8 are all inside the one box in
the middle.

```
  kernelthing <problem>
          │
          ▼
  ┌─────────────────────────────────────┐
  │ SETUP                               │
  │   problem dir ─copy─► managed repo  │   source repo is never mutated
  │   run dir + web server              │
  └─────────────────┬───────────────────┘
                    │
                    ▼
  ┌─────────────────────────────────────┐
  │ SEED                                │
  │   score the starting kernel         │   becomes member 0,
  │   (one remote submission)           │   the incumbent to beat
  └─────────────────┬───────────────────┘
                    │
                    ▼
╔═══════════════════════════════════════════════════════════════════════╗
║                          THE SEARCH LOOP                              ║
║                                                                       ║
║                 ┌─────────────────────────────┐                       ║
║        ┌───────►│         POPULATION          │────────┐              ║
║        │        │   all attempts · elites ·   │        │              ║
║        │        │   niche grid (diversity)    │        │              ║
║        │        └─────────────────────────────┘        │              ║
║        │                                               │              ║
║  measured verdict                          explore or exploit?        ║
║  correct? faster?                          which parent?              ║
║  ── the only signal ──                                 │              ║
║        │                                               ▼              ║
║  ┌─────┴─────────────┐                   ┌─────────────────────────┐  ║
║  │   REMOTE SCORE    │◄───── commit ─────│       AGENT TURN        │  ║
║  │  cheat scan first │                   │  own git worktree       │  ║
║  │  own subprocess   │                   │  sandboxed agent + LLM  │  ║
║  │  popcorn submit   │                   │  edits · submits        │  ║
║  └───────────────────┘                   └─────────────────────────┘  ║
║                                                                       ║
║   N in flight at once · no generations · no barriers ·                ║
║   a finished slot is refilled the instant its result lands            ║
╚═══════════════════════════════╤═══════════════════════════════════════╝
                                │  budget spent, or stop requested
                                ▼
                ┌───────────────────────────────────┐
                │ FINISH                            │
                │   promote best commit + copy out  │
                │   clean up refs and worktrees     │
                │   exit reason ─► process status   │
                └───────────────────────────────────┘
```

The run directory is the only output surface, and the only way back in:

```
   every stage appends ──► .humanize/rlcr/<ts>/
                              │
                              ├─ events.ndjson ──────► web UI folds it into live state
                              ├─ members/<id>/ ──────► prompt, transcript, diff, result
                              ├─ live.lock ──────────► "is this run still alive?"
                              └─ control.json ◄────── web UI writes knobs back in
                                                       (parallelism, budgets, stop)
```

Four invariants that explain the rest of this document:

1. **The measurement is the only feedback.** Nothing an agent claims about its own work affects
   selection — only the remote score verdict enters the population.
2. **Nothing runs a kernel locally.** There is no GPU, no benchmark engine, and no lock; every score
   is a remote popcorn submission graded on a real B200.
3. **The loop and the UI share no memory.** They communicate through files, which is why a finished
   run replays exactly like a live one.
4. **The search policy is pure.** Operator choice, parent selection, and the archive are
   side-effect-free and unit-testable; every worktree, subprocess, and lock lives in the controller.
