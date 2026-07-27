# Context examples

Every distinct thing that enters a candidate agent's context during an autoresearch run, with
**one example of each**. Examples are verbatim captures from run
`~/.local/share/kernelthing/runs/cholesky_b256n128/.humanize/rlcr/2026-07-24_09-19-41` unless
marked *(reconstructed)* or *(template)*. Long ones are cut with `[…]`.

Layers, in the order the model sees them:

1. [Process-level, before turn 1](#1-process-level-before-turn-1) — opencode's own system prompt layer
2. [The dispatch prompt](#2-the-dispatch-prompt) — the single user message kernelthing writes
3. [Files in the worktree](#3-files-in-the-worktree)
4. [Read-only knowledge bases](#4-read-only-knowledge-bases)
5. [Measurement feedback](#5-measurement-feedback) — the remote popcorn service
6. [Guard interceptions](#6-guard-interceptions) — between-turn rejections
7. [Tool-layer results and errors](#7-tool-layer-results-and-errors)
8. [Non-candidate turns](#8-non-candidate-turns) — methodology, bootstrap

Then [Not agent context](#not-agent-context) and an [appendix](#appendix-how-the-pieces-actually-reach-the-agent)
on delivery mechanisms, KernelWiki, and measured context cost.

---

## 1. Process-level, before turn 1

### 1.1 opencode base system prompt
Selected by model id inside the opencode binary; `deepseek/deepseek-v4-pro` falls through to the
default variant (no `claude`/`gpt`/`gemini`/`kimi`/`trinity` match).

```
You are opencode, an interactive CLI tool that helps users with software engineering tasks. Use the instructions below and the tools available to you to assist the user.
IMPORTANT: You must NEVER generate or guess URLs for the user unless you are confident that the URLs are for helping the user with programming. You may use URLs provided by the user in their messages or local files.
If the user asks for help or wants to give feedback inform them of the following:
- To give feedback, users should report the issue at https://github.com/anomalyco/opencode/issues
When the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') or asks in second person (eg 'are you able...', 'can you do...'), first use the WebFetch tool to gather information to answer the question from opencode docs at https://opencode.ai
You should be concise, direct, and to the point. When you run a non-trivial bash command, you should explain what the command does and why you are running it [...]
IMPORTANT: Keep your responses short, since they will be displayed on a command line interface. You MUST answer concisely with fewer than 4 lines (not including tool use or code generation), unless user asks for detail. [...]
```

### 1.2 Environment block *(reconstructed from the binary's template + this run's paths)*

```
You are powered by the model named deepseek-v4-pro. The exact model ID is deepseek/deepseek-v4-pro
Here is some useful information about the environment you are running in:
<env>
  Working directory: /home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1
  Workspace root folder: /home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1
  Is directory a git repo: yes
  Platform: linux
  Today's date: Fri Jul 24 2026
</env>
```

### 1.3 Tool definitions
One per available tool (`bash`, `read`, `write`, `edit`, `grep`, `glob`, `list`, `patch`,
`todowrite`, `todoread`, `webfetch`, …). The `bash` one:

```
Executes a given bash command in a persistent shell session with optional timeout, ensuring proper handling and security measures.
All commands run in the current working directory by default. Use the `workdir` parameter if you need to run a command in a different directory. AVOID using `cd <directory> && <command>` patterns - use `workdir` instead.
```

### 1.4 Instruction files (AGENTS.md / CLAUDE.md / CONTEXT.md)
opencode globs `AGENTS.md`, `CLAUDE.md`, `CONTEXT.md` up from the worktree, plus
`~/.config/opencode/AGENTS.md` and `~/.claude/CLAUDE.md`. **Empty in this setup** — none of those
files exist. If one existed in the problem dir it would arrive as an extra system block *(template)*:

```
<project_instructions>
  ...contents of AGENTS.md...
</project_instructions>
```

### 1.5 Skills block
`SystemPrompt.skills` lists loadable skills. **Empty here** — `sandbox.py:21` tmpfs-masks
`~/.claude/skills` and `~/.agents/skills` so no user skill reaches the prompt. Shape it would take
*(template)*:

```
Skills provide specialized instructions and workflows for specific tasks.
Use the skill tool to load a skill when a task matches its description.
```

### 1.6 MCP instructions
`<mcp_instructions>` block, one `<server>` per configured MCP server. **Empty** — `opencode.json`
declares none *(template)*:

```
<mcp_instructions>
  <server name="example">
    ...server-supplied instructions...
  </server>
</mcp_instructions>
```

### 1.7 Project references
`<available_references>`. **Empty** — no references configured *(template)*:

```
Project references provide additional directories that can be accessed when relevant.
<available_references>
  <reference>
    <name>docs</name>
    <path>/path/to/docs</path>
  </reference>
</available_references>
```

### 1.8 Inherited process environment
`opencode_client.build_opencode_env` passes the parent env through verbatim, then overrides these:

```
OPENCODE_CONFIG_CONTENT={"snapshot": false, "plugin": ["/home/mark123/projects/kernelthing/kernelthing/oc_guard/guard.js"]}
KERNELTHING_GUARD={"loopDir": "/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/.humanize/rlcr/candidate", "projectRoot": "/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1", "planFile": "plan.md", "currentRound": 1, "phase": "impl", "editFiles": ["submission.py"], "editDir": "/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1", "protectedFiles": ["task.py"], "blockDir": "/home/mark123/projects/kernelthing/prompts/block"}
XDG_DATA_HOME=/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/.humanize/oc-data/share
XDG_STATE_HOME=.../oc-data/state
XDG_CACHE_HOME=.../oc-data/cache
```

Inherited untouched and load-bearing: `POPCORN_API_URL` / `POPCORN_CLI_ID` /
`KERNELTHING_POPCORN_{BIN,CACHE}` (all optional overrides — the defaults come from `~/.popcorn.yaml`,
readable through the sandbox's ro-bind), the model API key, `PATH`, `HOME`.

### 1.9 The sandbox filesystem view
Latent context — anything the agent chooses to read. `bwrap --ro-bind / /` makes the *whole*
filesystem readable; only these are writable:

```
bwrap --die-with-parent --unshare-pid --ro-bind / / --proc /proc --dev /dev --tmpfs /tmp \
  --bind /home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1 <same> \
  --bind .../oc-data/share <same> --bind .../oc-data/state <same> --bind .../oc-data/cache <same> \
  --bind /home/mark123/.cache/kernelthing/cholesky_b256n128/.git <same> \
  --tmpfs /home/mark123/.claude/skills --tmpfs /home/mark123/.agents/skills \
  --chdir /home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1 \
  opencode run --format json -m deepseek/deepseek-v4-pro --auto --dir <worktree>
```

---

## 2. The dispatch prompt

One user message per candidate, assembled in `orchestrator.py` and written verbatim to
`members/<id>/prompt.md`.

### 2.1 EXPLORE body (`orchestrator.py:100`)
From `members/1/prompt.md`:

```markdown
Your work is not finished. Read and execute the below with ultrathink.

## Plan
@plan.md

You are an **EXPLORE** candidate in an evolutionary kernel search: open a NEW line
of attack. The working tree is at a kernel that reaches 75.5us
(commit: baseline). Make ONE focused, correct improvement by
editing ONLY submission.py, using an optimization strategy DISTINCT from those
already tried:
baseline
Pick a genuinely different angle -- do not just retune one of the above.
```

`{{KNOWN_STRATEGIES}}` is the live niche-key list (commit subjects, lowercased and cut to 60 chars).
By member 24 it had grown to:

```
__ldg() for a10 panel column loads: extend l1-cache-bypass t, __ldg() for a10 panel loads: bypass l1 cache on streaming pa, __ldg() for a11 trailing block loads: bypass l1 cache on str, baseline, eliminate redundant sync barrier for ksmemtrailing variants:, extend __ldg() to a00 factorization loads + __stcg() output , halve local[] register array for simt/balanced/tf32 update v, hoist column panel reads out of inner q-loop in simt trailin, hoist invariant row computations out of k-loop in simt_full_, hoist redundant rhs panel load in simt_full_tile_update: col, increase simt trailing update k-loop unroll from 8 to 16 for, precompute full row/col index offsets (int) in simt_full_til, precompute shared-memory base pointers in simt trailing upda, relax launch bounds from 2 to 1 to free register allocation , restructure simt_diagonal_tile_update with column-based pane, tune simt trailing update k-loop unroll from 16 to 12 for be, use __ldg() for factor64 global input loads to bypass l1 cac, use __ldg() for panel loads to bypass l1 cache (70.46us, -6.
```

### 2.2 EXPLOIT body (`orchestrator.py:114`)
From `members/18/prompt.md`:

```markdown
Your work is not finished. Read and execute the below with ultrathink.

## Plan
@plan.md

You are an **EXPLOIT** candidate: deepen a current best kernel. The working tree
is ALREADY at that kernel, which reaches 69.4us. The parent
commit message was:

  __ldg() for A11 trailing block loads: bypass L1 cache on streaming trailing data staged to shared memory. 74.121us -> 70.986us (-4.2%)

Push it further along the SAME approach by editing ONLY submission.py. Make ONE
focused, correct, measured improvement on top of it -- keep correctness.
```

### 2.3 Descriptor footer (`orchestrator.py:129`)
Appended to both operators:

```markdown
---

## How to finish (REQUIRED)
Self-test by running the scorer:

    /home/mark123/projects/kernelthing/.venv/bin/kernelthing score .

It must report `"correct": true`. **Commit as soon as
you have ANY correct improvement** (`git add -A && git commit -m "..."`) -- your
last committed correct version is what gets scored, so a timeout never wastes the
run. Do NOT edit anything outside submission.py.

Write @candidate-summary.md with a brief description of the changes made -- 2-3
sentences or short bullet points, with the final measured metric (us).
Keep it under 1000 characters.
```

### 2.4 KernelWiki tools block (`prompts/claude/kernel-tools-wiki.md`, rendered)

> **Removed 2026-07-27** — too ML-focused for a linalg board. The prompt file, `cfg.wiki` and
> `--no-wiki` are gone, so no candidate receives this block any more. Kept here because it *was*
> in the context of the run this document captures; §A3/§A4 below are void for the same reason.

````markdown
---

## Kernel optimization tools (available in your sandbox)

### KernelWiki — Blackwell/Hopper kernel-optimization knowledge base

A cross-referenced wiki of GPU kernel optimization (2179 merged PRs from CUTLASS,
vLLM, SGLang, FlashInfer, PyTorch, DeepGEMM + 48 synthesis pages). Consult it
*before* guessing at a technique — cite the page/PR you took an idea from in your
summary. Use it for tensor-core GEMM/attention/MoE patterns, tcgen05/TMEM/TMA,
warp specialization, FP8/FP4 block scaling, CuTe-DSL/PTX/Triton on Blackwell.

Query it (runs read-only, from any directory):

```bash
# natural-language search
/home/mark123/projects/kernelthing/.venv/bin/python /home/mark123/projects/kernelthing/vendor/KernelWiki/scripts/query.py "how to overlap TMA loads with tcgen05 mma" --limit 5
# filtered search
/home/mark123/projects/kernelthing/.venv/bin/python .../scripts/query.py --tag gemm --architecture sm100 --limit 10
# fetch a specific page (id or path), optionally with its sources
/home/mark123/projects/kernelthing/.venv/bin/python .../scripts/get_page.py kernel-flash-attention-4 --follow-sources
# regex search across wiki + PR bodies
/home/mark123/projects/kernelthing/.venv/bin/python .../scripts/grep_wiki.py "tcgen05\.fence"
```

Start broad with `references/primer.md` under /home/mark123/projects/kernelthing/vendor/KernelWiki if unsure what to ask.
````

### 2.5 popcorn + Nsight Compute tools block (`prompts/claude/kernel-tools-popcorn-ncu.md`, rendered)

````markdown
### There is no local GPU — everything runs on the competition hardware

This box has no GPU worth measuring on. Correctness and timings both come from the
hosted evaluation service, on the same hardware the leaderboard ranks. So: **never**
try to build, run, or time the kernel locally [...]

- Only `submission.py` is sent. It must be a single self-contained Python file [...]

### Checking your work

    /home/mark123/projects/kernelthing/.venv/bin/kernelthing score . --test-only
    /home/mark123/projects/kernelthing/.venv/bin/kernelthing score .

### Profiling with Nsight Compute (measure, don't guess)

Golden rule: **Profile → Diagnose → Plan, in that order.**

```bash
mkdir -p profile && cd profile
POPCORN_BREV_PROFILER_URL=https://http--brev-profiler-proxy--dxfjds728w5v.code.run /home/mark123/.local/bin/popcorn submit ../submission.py \
    --leaderboard cholesky --profile-brev --benchmark-index 2 --no-tui --output brev.json
```

- **Profiled durations are not benchmark times.** [...] 75.1 µs by the scorer reports ~168 µs under the profiler.
- **The profiler output contains the rejected token.** [...] the evaluator rejects the whole submission on that substring.
- `--benchmark-index 2` is the shape this problem is scored on. [...]

For deeper interpretation — the six analysis dimensions and a signal→cause→fix playbook — read `/home/mark123/projects/kernelthing/vendor/ncu-report-skill/SKILL.md`, then its `reference/` docs as needed.

The target is a **B200 (sm_100)**. Cite specific metric values in your summary [...]

### Rules

- Keep `profile/` and the `.zip`/`.ncu-rep` files out of your commit [...]
- Do not submit with `--mode leaderboard`. [...]
````

---

## 3. Files in the worktree

### 3.1 `plan.md`
Pointed at by `@plan.md`; read on turn 1 in every transcript.

```
<path>/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/plan.md</path>
<type>file</type>
<content>
1: # Batched Cholesky, batch=256 n=128, on B200
2:
3: ## Goal
4:
5: Minimise the wall-clock time of benchmark entry **index 2** of the gpu-mode `cholesky`
6: leaderboard: `batch: 256; n: 128; cond: 2; seed: 41128`, fp32, on a **B200 (sm_100)**.
[...]
</content>
```

### 3.2 The kernel under optimization (`submission.py`, the one editable file)

```
<path>/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/submission.py</path>
<type>file</type>
<content>
1: import hashlib
2: import os
3: from functools import lru_cache
4:
5: import torch
6: from task import input_t, output_t
7: from torch.utils.cpp_extension import load_inline
8:
9:
10: # The Popcorn tuner replaces this exact line in temporary, untracked copies.
11: _DEFAULT_VARIANT = 23  # POPCORN_VARIANT
12: _VARIANT_COUNT = 28
[...]
</content>
```

### 3.3 Protected problem assets (`task.py`, `problem.json`)
Readable, not writable — the guard rejects edits.

```python
"""Local stand-in for the ``task`` module the popcorn runner injects at evaluation time.

``submission.py`` does ``from task import input_t, output_t``. On the evaluation box that
module is supplied by the harness; here it exists only so the file can be imported for a
syntax/structure check without a network round-trip. It is deliberately inert [...]
"""
input_t = torch.Tensor
output_t = torch.Tensor
```

### 3.4 `.gitignore`

```
# Nsight Compute captures downloaded by `popcorn submit --profile-brev`. Candidates
# commit with `git add -A`, and these are large. popcorn extracts relative to the
# process cwd, not to --output, so the capture dir lands wherever the agent ran it and
# is named `profile.<index>-<spec-slug>/` -- a bare `profile/` misses ncu-details.csv
# and ncu-details.txt inside it.
profile/
profile.*/
*.ncu-rep
*.zip
runs/
```

### 3.5 Git history of the worktree
Reachable via bash; the parent chain is the record of what already worked.

```
$ git log --oneline -3 && echo "---" && git diff HEAD~1 --stat
6e751c7 use __ldg() for panel loads to bypass L1 cache (70.46us, -6.2% vs baseline 75.14us)
60c11e9 initial problem
---
 submission.py | 10 +++++-----
 1 file changed, 5 insertions(+), 5 deletions(-)
```

---

## 4. Read-only knowledge bases

### 4.1 KernelWiki search (`query.py`)

```
$ .../python .../KernelWiki/scripts/query.py "how to overlap TMA loads with tcgen05 mma" --limit 5
# 5 result(s)

## Migrating from wgmma to tcgen05
- **id**: `migration-wgmma-to-tcgen05`
- **type**: `wiki-migration`
- **path**: `wiki/migration/wgmma-to-tcgen05.md`
- **confidence**: source-reported
- **reproducibility**: pseudocode
- **tags**: ['tcgen05', 'wgmma', 'tmem']
- **sources**: ['doc-nvidia-tuning-guide', 'blog-tcgen05-tutorial', 'blog-colfax-cutlass']

## Tensor Memory Accelerator (TMA)
- **id**: `hw-tma`
- **architectures**: ['sm100', 'sm100a', 'sm90', 'sm90a']
[...]
```

### 4.2 KernelWiki page fetch (`get_page.py`)

````
$ .../python .../KernelWiki/scripts/get_page.py technique-register-budgeting --follow-sources
# wiki/techniques/register-budgeting.md

---
id: technique-register-budgeting
title: "Register Budgeting for Occupancy"
architectures: [sm100, sm90]
blackwell_relevance: "TMEM eliminates accumulator register pressure on Blackwell, freeing ~100 registers/thread for other uses; technique still critical for memory-bound kernels."
---

## Pattern

```cuda
// Aggressive: 32 registers/thread → ~4 blocks per SM at 256 threads/block
__launch_bounds__(256, 4)
__global__ void gemv_memory_bound(...) { ... }
```
[...]
````

### 4.3 KernelWiki primer (`references/primer.md`)

```
$ cat .../KernelWiki/references/primer.md
# Topic Map / Primer

A compact, authoritative map of the knowledge base. [...]

## Hardware Features (SM100)

| Feature | Page ID | Path | Notes |
|---|---|---|---|
| tcgen05 MMA instruction | `hw-tcgen05-mma` | `wiki/hardware/tcgen05-mma.md` | Blackwell tensor core instruction; replaces wgmma. |
| Tensor Memory (TMEM) | `hw-tmem` | `wiki/hardware/tmem.md` | 256 KB/SM dedicated accumulator storage [...] |
| Cluster Launch Control (CLC) | `hw-clc` | `wiki/hardware/clc.md` | Hardware work queue for persistent kernels [...] |
[...]
```

### 4.4 ncu-report-skill (`vendor/ncu-report-skill/SKILL.md` + `reference/`)

```markdown
---
name: ncu-report-skill
description: Profile CUDA kernels with Nsight Compute on B200 / sm_100. [...]
---

## Golden rule

**Profile → Diagnose → Plan, in that order. Never guess.**

## Quickstart (what to do when someone says "profile this kernel")

3. **Run two profiles**: `--set full` (with `PmSampling` sections) for the overview, and
   `--set source --section SourceCounters` for per-line stall attribution.
5. **Work through the six analysis dimensions.** See `reference/05-analysis-dimensions.md`.
6. **Match patterns to the diagnosis playbook.** See `reference/06-diagnosis-playbook.md`.
[...]
```

---

## 5. Measurement feedback

### 5.1 Correctness pre-check (`kernelthing score . --test-only`)

```json
{"unit": "us", "correct": true, "metric": null, "error": null, "bench": {"backend": "popcorn", "leaderboard": "cholesky", "gpu": "B200", "metric_mode": "shape", "sha256": "fe872d5041f9e96a", "test": {"passed": 17, "failed": 0, "failures": []}, "wall_s": {"test": 6.3}, "submission_ids": {"test": 902568}, "cached": {"test": false}, "source": {"test": "api"}}}
```

### 5.2 Full authoritative score (`kernelthing score .`)

```json
{"unit": "us", "correct": true, "metric": 81.49974216376582, "error": null, "bench": {"backend": "popcorn", "leaderboard": "cholesky", "gpu": "B200", "metric_mode": "shape", "sha256": "fe872d5041f9e96a", "test": {"passed": 17, "failed": 0, "failures": []}, "shapes": [{"index": 0, "spec": "n: 32; cond: 2; seed: 41032; batch: 4096", "status": "pass", "mean_us": 113.1613, "err_us": 0.0715, "best_us": 112.882, "worst_us": 113.856}, {"index": 1, "spec": "n: 64; cond: 2; seed: 41064; batch: 1024", "status": "pass", "mean_us": 110.0816, "err_us": 0.0704, "best_us": 109.83, "worst_us": 110.83}, {"index": 2, "spec": "n: 128; cond: 2; seed: 41128; batch: 256", "status": "pass", "mean_us": 81.4997, "err_us": 0.0809, "best_us": 80.962, "worst_us": 82.682}, {"index": 3, "spec": "n: 256; cond: 2; seed: 41256; batch: 64", "status": "pass", "mean_us": 276.4084, ...}]}}
```

### 5.3 Failing score (remote re-check verdict)

```
popcorn benchmark: 1 shape(s) failed re-check (n: 128; cond: 2; seed: 41128; batch: 256): L @ L.T does not reconstruct A: relative_residual=0.21
{"unit": "us", "correct": false, "metric": null, "error": "popcorn benchmark: 1 shape(s) failed re-check (n: 128; cond: 2; seed: 41128; batch: 256): L @ L.T does not reconstruct A: relative_residual=0.21", "bench": {..., "shapes": [{"index": 2, "spec": "n: 128; cond: 2; seed: 41128; batch: 256", "status": "fail", "error": "L @ L.T does not reconstruct A: relative_residual=0.21"}, ...]}}
```

### 5.4 Local pre-submission rejection (`popcorn.py:768`) *(template)*
Saves a remote round-trip; two forms:

```
submission does not parse: invalid syntax (line 412)
submission contains 'stream', which the evaluation server rejects; remove it (including inside comments and longer words)
```

### 5.5 Remote profiler submission (`popcorn submit --profile-brev`)

```
$ mkdir -p profile && cd profile && POPCORN_BREV_PROFILER_URL=... popcorn submit ../submission.py \
    --leaderboard cholesky --profile-brev --benchmark-index 2 --no-tui --output brev.json
Submitting to leaderboard: cholesky
GPU: B200_Brev
Mode: profile
File: ../submission.py

Waiting for results...
Profile job c331523d39c14cb59690c730d2a0dc44 accepted. Waiting for results...
Profile job c331523d39c14cb59690c730d2a0dc44 status: running (0s)
Profile job c331523d39c14cb59690c730d2a0dc44 status: running (5s)
[...]
```

### 5.6 Nsight Compute report (`profile/profile.<idx>-<spec>/ncu-details.txt`)

```
<path>.../m10/profile/profile.2-batch-256-n-128-cond-2-seed-41128/ncu-details.txt</path>
<type>file</type>
<content>
1: ==WARNING== Could not deploy stock section files to "/home/ubuntu/Documents/NVIDIA Nsight Compute/2026.2.0/Sections".
5: [73] python3.12@127.0.0.1
6:   void <unnamed>::blocked_128_kernel<1, 2, 0, 4, 1, 0, 1>(const float *, float *) (256, 1, 1)x(128, 1, 1), Context 1, Stream 7, Device 0, CC 10.0
11:     Section: GPU Speed Of Light Throughput
12:     ----------------------- ----------- ------------
13:     Metric Name             Metric Unit Metric Value
15:     DRAM Frequency                  Ghz         4.00
16:     SM Frequency                    Ghz         1.14
17:     Elapsed Cycles                cycle       197710
18:     Memory Throughput                 %        26.47
19:     DRAM Throughput                   %         1.13
20:     Duration                         us       171.52
21:     L1/TEX Cache Throughput           %        26.88
22:     L2 Cache Throughput               %        13.93
23:     SM Active Cycles              cycle    188948.61
24:     Compute (SM) Throughput           %        16.47
27:     OPT   This kernel grid is too small to fill the available resources on this device, resulting
[...]
</content>
```

---

## 6. Guard interceptions

The `tool.execute.before` hook throws; opencode feeds the message back as a tool error.

### 6.1 Rendered block template (`prompts/block/git-add-humanize.md`)

```markdown
# Git Add Blocked: .humanize Protection

The `.humanize/` directory contains local loop state that should NOT be committed.
This directory is already listed in `.gitignore`.

Your command was blocked because it would add .humanize files to version control.

## Allowed Commands

Use specific file paths instead of broad patterns:

    git add <specific-file>
    git add src/
    git add -p  # patch mode

## Blocked Commands

These commands are blocked when .humanize exists:

    git add .humanize      # direct reference
    git add -A             # adds all including .humanize
    git add --all          # adds all including .humanize
    git add .              # may include .humanize if not gitignored
    git add -f .           # force bypasses gitignore

## Adding .humanize to .gitignore

If you need to add `.humanize*` to `.gitignore`, follow these steps:

1. Edit `.gitignore` to append `.humanize*`
2. Run: `git add .gitignore`
3. Run: `git commit -m "Add humanize local folder into gitignore"`

IMPORTANT: The commit message must NOT contain the literal string ".humanize" to avoid triggering this protection.
```

### 6.2 Inline guard fallback (no template file; `guard_core.js:147`)

```markdown
# Summary too long

candidate-summary.md must be under 1000 characters (yours is 1194). Shorten it to 2-3 sentences or a few bullet points.
```

Other reachable blocks in this configuration, same delivery path: `edit-file-protected`,
`plan-file-modified`, `state-file-modification`, `git-push`, `popcorn-leaderboard`, `gpu-tamper`.

---

## 7. Tool-layer results and errors

### 7.1 `read`

```
<path>/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/plan.md</path>
<type>file</type>
<content>
1: # Batched Cholesky, batch=256 n=128, on B200
[...]
</content>
```

### 7.2 `grep`

```
Found 2 matches
/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m1/submission.py:
  Line 1313:       TORCH_CHECK(false, "native variant must be in [0, 27]");

  Line 1410:               "native variant must be in [0, 28]");
```

### 7.3 `edit` (success)

```
Edit applied successfully.
```

### 7.4 `edit` (failure)

```
Found multiple matches for oldString. Provide more surrounding context to make the match unique.
```

### 7.5 `write`

```
Wrote file successfully.
```

### 7.6 `todowrite`
Echoes the list back — the agent's own plan re-entering context.

```json
[
  { "content": "Add variant 28: AsyncLoad + SimtBalancedUpdate + OverlapOutput", "status": "in_progress", "priority": "high" },
  { "content": "Test with --test-only for correctness", "status": "pending", "priority": "high" },
  { "content": "Test with full scorer for metric", "status": "pending", "priority": "high" },
  { "content": "Commit if correct improvement", "status": "pending", "priority": "high" },
  { "content": "Write candidate-summary.md", "status": "pending", "priority": "medium" }
]
```

### 7.7 `bash` failure / missing-file error

```
NotFound: FileSystem.access (/home/mark123/.cache/kernelthing/wt/2026-07-24_09-19-41/evolve/m5/profile)
```

```
Traceback (most recent call last):
  File "<string>", line 2, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
```

### 7.8 `webfetch`
Allowed (`opencode.json` permits it; the sandbox leaves the network up), unused in this run.
Result shape *(template)*:

```
<url>https://docs.nvidia.com/...</url>
<content>
...markdown-converted page body...
</content>
```

### 7.9 Own prior turns
Within one candidate's session, the agent's own reasoning, text, and every tool call/result above
accumulate. Reasoning is carried as provider metadata:

```json
{"type": "reasoning.text", "text": "Let me start by understanding the current state of the project. I need to:\n\n1. Read the plan file\n2. Read the current submission.py\n3. Read the KernelWiki primer for background\n4. Understand what optimizations have already been tried (baseline)\n5. Find a genuinely different optimization angle\n6. Make ONE focused improvement\n7. Test and commit", "format": "unknown", "index": 0}
```

---

## 8. Non-candidate turns

### 8.1 Methodology retrospective (`orchestrator.py:65`, `--methodology`)
One extra turn after the loop exits. Phase is `methodology`, so the guard restricts reads to the
run dir and writes to two files.

```markdown
# Methodology Analysis (loop exit)

The optimization loop has exited.
- Exit reason: maxiter -- evolutionary search budget spent: 24 candidates, best 69.4us [exploit]
- Candidates dispatched: 24 (search budget: 24 candidates)
- Best result reached: 69.4us

Perform a retrospective on the *methodology* of this run (HOW the loop worked),
not the project itself. Read the development records in @runs/2026-07-24_09-19-41:
- the candidate summaries and results (`members/*/summary.md`, `members/*/result.json`)
- the structured event journal (`events.ndjson`)
- the full narrative (`loop.log`)

Analyze from a pure methodology perspective. Focus areas:
- Iteration efficiency: were rounds productive, or repetitive?
- Best-of-N effectiveness: did parallel candidates explore genuinely diverse
  strategies, and did winners beat the incumbent by real margins vs. noise?
- Stagnation / plateau: where did progress slow, and why?
- Feedback quality: did reviewer feedback lead to real improvements?
- Benchmark trust: any signs of noisy or misleading measurements?
- Plan-to-execution alignment and round-count vs. progress.

Write a structured retrospective of general, transferable improvements to the
optimization methodology [...] to:
  `runs/2026-07-24_09-19-41/methodology-analysis-report.md`
[...]
Do NOT edit any source files and do NOT commit -- only write those two files.
```

### 8.2 Methodology retry follow-up (`orchestrator.py:1114`)
Second user message in the same session when the artifacts are missing:

```
The methodology analysis is incomplete. Write the retrospective to runs/2026-07-24_09-19-41/methodology-analysis-report.md and a one-line completion note to runs/2026-07-24_09-19-41/methodology-analysis-done.md.
```

### 8.3 Bootstrap turn (`prompts/claude/bootstrap-problem.md`)
Only when the target has no `problem.json` — a different agent turn that authors the problem.

````markdown
Read and execute the below.

# Bootstrap phase: author a new kernel problem

There is no problem directory yet. We are creating one from scratch so that
kernelthing can optimize a GPU kernel against a well-defined, cheat-resistant
scoring objective. This is a one-time setup turn, you are NOT optimizing yet.

## The objective
{{OBJECTIVE}}

{{MODE_DIRECTIVE}}

## Where to write
Author the problem in this directory (it already exists, empty):

    {{TARGET_DIR}}

## What a problem is
[...]
1. **`problem.json`** — the manifest. Follow this schema (a real example):
   ```jsonc
   {
     "name": "<short-slug>",
     "edit_files": ["kernel.py"],                  // the kernel source — NEVER submission.py
     "bench": { "submission_qualname": "submission.kernel" },
     "metric": { "kind": "pct_baseline", "baseline_qualname": "baseline.kernel" },
     "direction": "maximize"
   }
   ```
2. **`plan.md`** — [...] It **must** include a `## Build & test` section [...]
````

### 8.4 Bootstrap mode directive (`prompts/claude/bootstrap-mode-auto.md`)
Substituted into `{{MODE_DIRECTIVE}}` above; the interactive variant is
`bootstrap-mode-interactive.md`.

```markdown
## Operating mode: AUTONOMOUS (`--auto-setup`)

There is **no operator** in this session. You cannot ask questions — nothing you
say will be answered — so drive the problem to a finish in this single turn:

- The objective above is guaranteed present. Resolve every ambiguity yourself by
  choosing the most defensible interpretation, and record each such assumption in
  your closing summary. **Never** end your turn waiting for input or asking a
  question.
- Before you emit `COMPLETE` you MUST have run `kernelthing score .` and seen
  `"correct": true`. [...]
- If you genuinely cannot establish a reference you are confident in, emit
  `SETUP_BLOCKED`. Blocking beats shipping a silently-wrong green.
```

---

## Not agent context

Written by the loop, never read back by an agent: `events.ndjson`, `control.json`, `run.json`,
`live.lock`, `members/<id>/{result.json,summary.md,diff.patch}`, `loop.log`, the web UI, and the
kernelguard scan verdict. Selection pressure reaches the agent only indirectly — as the parent
commit it forks from, its metric, and the niche-key list in the EXPLORE prompt.

---

# Appendix: how the pieces actually reach the agent

## A1. Two delivery mechanisms, and kernelthing uses only one

opencode has four native context channels — skills, project references, MCP instructions, and
instruction files (AGENTS.md / CLAUDE.md / CONTEXT.md). **All four are empty in this setup**
(§1.4–1.7). `sandbox.py:18` states the reason: user-level skills are "pure noise," so they are
tmpfs-masked and the loop supplies its own tooling.

Everything domain-specific therefore arrives as **prompt text** injected by
`Orchestrator._kernel_tools_block` (`orchestrator.py:264`), bypassing opencode entirely. Within
that, two sub-mechanisms:

**Embedded** — in-context, paid for on every turn:

- opencode base system prompt
- `<env>` block
- Tool definitions
- EXPLORE / EXPLOIT operator body
- Descriptor footer
- KernelWiki tools block
- popcorn + Nsight Compute tools block
- `KERNELTHING_GUARD` / `OPENCODE_CONFIG_CONTENT` (into the process, not the model)
- Guard block messages (only on rejection)
- Methodology prompt / retry (exit turn)
- Bootstrap prompt + mode directive (setup turn)

**Referenced by path** — costs a tool call, and only if the agent bites:

- `plan.md`
- `submission.py`
- `task.py`, `problem.json`, `.gitignore`
- ncu-report-skill `SKILL.md` + `reference/`
- KernelWiki `references/primer.md`
- KernelWiki pages via `query.py` / `get_page.py` / `grep_wiki.py`
- `profile/*/ncu-details.txt`, `ncu-details.csv`, `brev.json`
- Git history
- `~/.popcorn.yaml`
- The entire read-only filesystem (bwrap `--ro-bind / /`)

## A2. ncu-report-skill is a path, not a skill

§1.5 (opencode's skill system) being empty and §4.4 (ncu-report-skill) both being true is not a
contradiction — they are different mechanisms. The skill reaches the agent as one sentence
appended by `orchestrator.py:301`:

```
For deeper interpretation — the six analysis dimensions and a signal→cause→fix playbook —
read `/home/mark123/projects/kernelthing/vendor/ncu-report-skill/SKILL.md`, then its
`reference/` docs as needed.
```

~35 words. `SKILL.md` and the `reference/*.md` behind it are never embedded. Gated on `_vendored()`
(`orchestrator.py:149`) — an uninitialised submodule drops the pointer rather than pointing at a
missing file.

**Measured uptake: 3 of 25 candidates followed it.** A bare path is a weak delivery mechanism next
to embedded text.

KernelWiki (§2.4) is the same story in reverse: the *pitch and query commands* are embedded
(~230 words), but no wiki content is — only paths.

## A3. KernelWiki structure, and adding linalg / MAGMA

*(Void — the block was removed 2026-07-27; see §2.4. Retained as a record of what was measured.)*

Three layers (`vendor/KernelWiki/CLAUDE.md`):

| Layer | Path | Contents |
|---|---|---|
| 1. Sources | `sources/` | `prs/{repo}/PR-{N}.md`, `docs/`, `blogs/`, `contests/` |
| 2. Wiki | `wiki/` | `hardware/` (8), `techniques/` (15), `kernels/` (12), `patterns/` (7), `languages/` (4), `migration/` (2) |
| 3. Queries | `queries/` | auto-generated indices — **do not hand-edit** |

Everything is YAML-frontmatter-driven. `data/schemas.yaml` fixes the required fields per page type;
`data/tags.yaml` is a controlled vocabulary that `scripts/validate.py` rejects unknowns against.

To add linear algebra and MAGMA:

1. `data/tags.yaml` — add `kernel_types: cholesky, lu, qr, trsm, syrk`; add `magma` as a repo.
2. `sources/docs/magma-*.md` or `sources/prs/magma/PR-N.md` — id prefixes `doc-` / `pr-`.
3. `wiki/kernels/batched-cholesky.md` — id `kernel-batched-cholesky`, `type: kernel`. The schema
   requires `performance_claims`, `reproducibility >= snippet`, and `sources:` pointing at step 2.
4. `data/aliases.yaml` — add `potrf → cholesky`, `magma → linalg`. **This is what makes queries
   hit** (see A4); without it a search for "potrf" scores zero against a page titled "Cholesky".
5. `python3 scripts/generate-indices.py && python3 scripts/validate.py`

KernelWiki is a git submodule — those commits land in its own repo, not kernelthing's.

## A4. KernelWiki's "semantic search" is lexical

`scripts/query.py:114 score_keyword_match`:

```
title hit  +10
tag hit    +5     (tags, techniques, hardware_features, kernel_types, languages, aliases, symptoms)
body hits  +min(count, 3)
```

Summed per keyword, best alias variant wins. No embeddings, no vector index — it walks the tree and
parses frontmatter on every invocation.

The only semantic layer is `data/aliases.yaml` expansion via `expand_keyword`: `UMMA → tcgen05`,
`B200 → sm100`. Alias coverage *is* the retrieval quality.

## A5. .gitignore, and where context actually goes

Respected asymmetrically:

- **`grep` / `glob` respect it.** opencode embeds the Rust `ignore` crate (ripgrep); `ignore::gitignore`
  and `info/exclude` both appear in the binary. Sweeps will not surface `profile/`, `*.ncu-rep`, `*.zip`.
- **`read` and `bash` do not.** Nothing stops `cat profile/…`. Ignore rules are a *search* filter,
  not a read barrier.

Measured across all 25 members of the reference run:

| | |
|---|---|
| Largest single tool output | **59 KB — `submission.py` itself** (~15k tokens) |
| Profiler-artifact-touching calls | 18, largest 35.5 KB (`ncu-details.txt`) |
| `.ncu-rep` / `.zip` bytes reaching context | **zero** |
| Peak total tool output, one candidate | 171 KB (~43k tokens) |

No artifact pollution occurred. The dominant context cost is the kernel under optimization —
59 KB re-read on nearly every turn, ~4× the largest profiler read. One agent capped its
`ncu-details.txt` read with `limit: 300`; the rest read it whole.

## A6. The `git add -A` contradiction

`EVOLVE_DESCRIPTOR_FOOTER` (`orchestrator.py:138`) instructs:

```
git add -A && git commit -m "..."
```

`gitAddsHumanize` (`guard_core.js:118`) blocks `-a|--all` unconditionally — not only when
`.humanize` is present — so that instruction is always rejected on first use:

```
# Git Add Blocked: .humanize Protection
[...]
    git add -A             # adds all including .humanize
```

**Fired in 15 of 25 members**, once each. Every agent recovered by naming files explicitly (which
incidentally is what keeps profiler artifacts out of commits), so nothing was lost — but the prompt
and the guard disagree, and each candidate spends a turn discovering it.
