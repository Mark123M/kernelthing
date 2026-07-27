# kernelthing

A kernel optimization autoresearch loop built on top of [`opencode`](https://github.com/sst/opencode) and DeepSeek V4.

Iteratively improves a GPU kernel, scoring every attempt against a known-correct
baseline and keeping only genuine improvements. Originally a DSV4/Opencode
reimplementation of [Humanize](https://github.com/PolyArch/humanize), but now
is rather different and has many more features.

<br>

![kernelthing in action](images/kernel_autoresearch_run.png)

## Setup

```bash
python -m pip install -e .
git submodule update --init vendor/ncu-report-skill # fetch the vendored ncu skill

# external prerequisites
opencode --version     # CLI agent on PATH + DEEPSEEK_API_KEY
bwrap --version        # bubblewrap sandbox
nvcc --version         # CUDA compiler (for problems/gemm)
# As well as whatever else your kernels might need.
```

## Quick start

```bash
kernelthing                                                  # Jump into an opencode session to define a new problem interactively
kernelthing -j 8                                             # Launch with 8 concurrent agents working on improving kernels
kernelthing problems/gemm -j 8                               # opens web UI at localhost:8765
kernelthing problems/gemm --max-candidates 50                # headless, set budgets
```

Running without `--no-web` will open a WebUI at [http://localhost:8765](http://localhost:8765) from which to view progress and tweak parameters.


## Targeting your own problem

A **problem** is a directory inside a git repo with a `problem.json` manifest:

```json
{
  "name": "my-kernel",
  "plan": "plan.md",
  "edit_files": ["kernels/mykernel.cu"],
  "score_command": "bash score.sh",
  "metric_name": "pct_baseline", "unit": "%baseline",
  "direction": "maximize", "bench_runs": 3
}
```

You can write it manually, or have the agent write it on launch. Having the agent write it is recommended.

### Score scripts must benchmark honestly

A single timed run is not a measurement. The bundled `problems/gemm` harness shows
the bar: **multi-seed correctness**, **steady-state warmup**, an **L2-cache flush
between timed iterations**, and **median of many iters with min/max spread**.
kernelthing additionally re-runs your score command `bench_runs` times and keeps only
the best of all-correct runs. Hold your own problems to the same standard.

## Web UI

`http://127.0.0.1:<port>` (stdlib-only, zero deps). The UI is fully decoupled
from the run: every run journals everything it does to its run dir
(`.humanize/rlcr/<ts>/` — `run.json`, an append-only `events.ndjson`, and
`members/<id>/` with each candidate's exact prompt, agent transcript, diff,
summary, and result/cost record), and the server is a pure reader of those
dirs. The run picker lists every run — live or finished — so old runs replay
with the full chart/lineage/leaderboard, and `events.ndjson` can be debugged
with `tail`/`grep`/`jq`.

- **Best vs. submitted** — best-so-far staircase chart with each attempt a dot
  colored by operator (explore/exploit), failures marked.
- **Agents (live)** — cards showing operator, parent, tool-call count, cost, and
  the latest tool call + reasoning line from the streaming opencode log. Click for
  full transcript.
- **Lineage** (parent→child mutation tree), **Leaderboard** (with per-candidate
  cost), and a **member detail pane** (transcript / prompt / diff / summary /
  result).
- Live-tunable controls (parallelism, budgets, explore bias, stop) flow through
  the run dir's `control.json`, re-read by the loop at dispatch boundaries —
  only shown while the run holds its `live.lock`.

Disable with `--no-web`, or serve all runs standalone (no loop needed) with
`kernelthing web [--root ~/.cache/kernelthing]`.

## Search strategy: 

The search strategy is the biggest difference between kernelthing and humanize. Humanize takes an RLCR approach.
We take an async evolutionary population approach. This allows us to separate parallel tasks like code editing from
tasks like benchmarking, and allows us to maintain idea diversity.

- a **controller** owns the population/archive and a task queue;
- a pool of **mutation workers** (many, concurrent) each take a parent + operator,
  edit in an isolated worktree, and submit the result;
- **scoring is a remote submission** to the popcorn service — correctness then
  timing — so many candidates can be scored at once, bounded only by the service.

Results flow back continuously, so workers are always busy and slots refill the instant a result lands.

**Two operators**, with compute budget split across them:

| Operator | Purpose | How |
|---|---|---|
| **Explore** (breadth) | new lineages | start from a seed, take a strategy *not yet in the archive* |
| **Exploit** (depth) | deepen winners | refine a top scoring commit |

**Diversity via MAP-Elites** niches keyed on agent-reported strategy descriptors
(tiling, vectorization, tensor-core use, …): the best kernel *per niche* is kept,
and exploration targets empty niches so the search can't collapse onto one lineage.

**The measured benchmark is the only thing that decides what is elite or gets promoted.** The run stops on a global budget (wall-clock / candidate count), not a
round count. `-j` sets max concurrent agents.

**Remote scoring.** kernelthing does not run kernels locally: every attempt is submitted
to the hosted gpu-mode [popcorn](https://gpu-mode.com) service and graded on real
competition hardware (a B200), which is the only place the target number is
authoritative. A score is a `--mode test` submission (correctness) followed by a
`--mode benchmark` submission (timing) only if the test passed; the returned numbers
feed the search unchanged. There is no local GPU, no CUDA/torch dependency, and no
device to contend on — `-j` candidates edit and score concurrently, bounded only by the
service's own rate limits. The guard blocks a *ranked* leaderboard submission so agents
can measure freely but never publish; that stays a human decision.

## Sandboxing

Every edit-capable agent runs under **bubblewrap**: filesystem read-only except the
candidate's worktree, opencode's own state, and `/tmp`. No GPU device nodes are bound —
kernels are compiled and benchmarked remotely on the popcorn service, never in the
sandbox. Network stays up (the model API and the popcorn service need it); the
filesystem is the confinement boundary. opencode's `--dangerously-skip-permissions`
is only safe because of this.

## Kernel tooling (profiling references)

Vendored skills under `vendor/`, surfaced as paths in the agent's prompt:

- **ncu-report-skill** (`vendor/ncu-report-skill`) — B200/sm_100 analysis
  dimensions, a signal→cause→fix playbook, and sm_100 metric names. Prose only:
  its collection workflow and `helpers/` assume a local GPU.
- **ncu-profile-analysis / nsys-profile-analysis** (`vendor/veloq-ncu-skill`,
  `vendor/veloq-nsys-skill`) — how to interrogate the two reports with `veloq`.
- **ptx-skill** (`vendor/ptx-skill`) — PTX/CUDA ISA reference.

Disable with `--no-ncu` / `--no-veloq` / `--no-ptx`.

A full remote score automatically overlaps timing with two profiles of the scored shape:
hosted Nsight Compute via popcorn, and a minimal Modal/B200 Nsight Systems timeline for
Cholesky problems. Artifacts land under `profile/latest/` and `profile/latest/nsys/`.
`kernelthing score --no-profile` skips both captures.

## GPU profiling permission (for ncu)

NVIDIA drivers restrict performance counters to admins by default. The agent runs
non-root under bubblewrap (which sets `no_new_privs`), so profiling fails with
`ERR_NVGPUCTRPERM` until you allow non-root access:

```bash
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
  | sudo tee /etc/modprobe.d/nvidia-profiling.conf
sudo reboot
```

Verify: `cat /proc/driver/nvidia/params | grep Profiling` → `RmProfilingAdminOnly: 0`.
Run with `--no-ncu` to skip profiling entirely.
