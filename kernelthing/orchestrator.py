"""The orchestration layer for the asynchronous evolutionary kernel search.

The search logic (population, operators, selection) is pure in ``evolve.py``;
this module is the side-effecting controller: it owns git worktrees, the agent
turns, the remote scoring submissions, and the loop budget. Problem-agnostic --
the objective fitness comes from the problem's ``score`` command (JSON {correct,
metric}); see problem.py and ``run()``. Grading is remote (the hosted popcorn
service), so there is no local GPU pool or benchmark to serialize.

Everything that happens is journaled to the run dir (see journal.py/state.py):
events to ``events.ndjson``, per-candidate artifacts to ``members/<id>/``. The
web UI is a pure reader of those files; live tuning (``-j``, budgets, stop)
arrives through ``control.json``, re-read at dispatch boundaries.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import yaml

from . import archive, evolve, gates, opencode_client, prompts
from .config import Config, format_duration
from .journal import MAX_PARALLELISM, Journal, LiveLock, LoopControl
from .problem import Problem
from .state import LoopDirs, State, new_timestamp, save_run

EXIT_COMPLETE = "complete"
EXIT_MAXITER = "maxiter"
EXIT_STOP = "stop"
EXIT_ERROR = "error"
EXIT_STALL = "stalled_out"  # several consecutive no-progress rounds; HEAD kept
EXIT_STOPPED = "stopped"  # user requested stop via UI


@dataclass
class RunContext:
    """Mutable state owned by a single evolutionary-search invocation.

    Separated from Orchestrator so the five closures that were tangled inside
    ``run()`` are regular methods taking this as a parameter.
    """

    dispatched: int = 0
    in_flight: dict[int, int] = field(default_factory=dict)
    futures: dict[Any, Any] = field(default_factory=dict)
    search_start: float | None = None


# --- prompts (rendered with problem fields) ---

# Methodology Analysis phase (ported from Humanize; adapted for headless opencode
# -- the Opus-subagent / AskUserQuestion / gh-issue flow is replaced by the agent
# writing the retrospective itself, since there is no interactive user here).
METHODOLOGY_PROMPT = """# Methodology Analysis (loop exit)

The optimization loop has exited.
- Exit reason: {{EXIT_REASON}} -- {{EXIT_REASON_DESCRIPTION}}
- Candidates dispatched: {{DISPATCHED}} (search budget: {{BUDGET}})
- Best result reached: {{BEST}}{{UNIT}}

Perform a retrospective on the *methodology* of this run (HOW the loop worked),
not the project itself. Read the development records in @{{LOOP_DIR}}:
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
optimization methodology (describe patterns and process changes; avoid dumping
project-specific code) to:
  `{{LOOP_DIR}}/methodology-analysis-report.md`
If the methodology worked well, say so briefly. Then write a one-line completion
note to:
  `{{LOOP_DIR}}/methodology-analysis-done.md`

Do NOT edit any source files and do NOT commit -- only write those two files.
"""

# --- evolutionary-search operator prompts (Orchestrator.run) ---

EVOLVE_EXPLORE_PROMPT = """Your work is not finished. Read and execute the below with ultrathink.

## Plan
@{{PLAN}}

You are an **EXPLORE** candidate in an evolutionary kernel search: open a NEW line
of attack. The working tree is at a kernel that reaches {{PARENT_METRIC}}{{UNIT}}
(commit: {{PARENT_COMMIT_MESSAGE}}). Make ONE focused, correct improvement by
editing ONLY {{EDIT_FILES}}, using an optimization strategy DISTINCT from those
already tried:
{{KNOWN_STRATEGIES}}
Pick a genuinely different angle -- do not just retune one of the above.
"""

EVOLVE_EXPLOIT_PROMPT = """Your work is not finished. Read and execute the below with ultrathink.

## Plan
@{{PLAN}}

You are an **EXPLOIT** candidate: deepen a current best kernel. The working tree
is ALREADY at that kernel, which reaches {{PARENT_METRIC}}{{UNIT}}. The parent
commit message was:

  {{PARENT_COMMIT_MESSAGE}}

Push it further along the SAME approach by editing ONLY {{EDIT_FILES}}. Make ONE
focused, correct, measured improvement on top of it -- keep correctness.
"""

EVOLVE_DESCRIPTOR_FOOTER = """

---

## How to finish (REQUIRED)
Self-test by running the scorer:

    {{SCORE_CMD}}

It must report `"correct": true`. **Commit as soon as
you have ANY correct improvement** (`git add -A && git commit -m "..."`) -- your
last committed correct version is what gets scored, so a timeout never wastes the
run. Do NOT edit anything outside {{EDIT_FILES}}.

Write @candidate-summary.md with a brief description of the changes made -- 2-3
sentences or short bullet points, with the final measured metric ({{UNIT}}).
Keep it under 1000 characters.
"""


def _vendored(path: Path) -> bool:
    """Is a vendored skill dir actually populated?

    ``vendor/`` holds git submodules that are frequently left uninitialised (and
    ``git submodule update --init`` fails repo-wide here — see CLAUDE.md), so an existing
    but empty directory is the normal broken state, not a missing one.
    """
    return path.is_dir() and any(path.iterdir())


def _skill_description(skill_dir: Path) -> str:
    """A vendored skill's own ``description:``, read verbatim from SKILL.md frontmatter.

    Read from the file rather than transcribed into a constant. A hand-written summary
    of someone else's document is worse than theirs by construction, and it goes stale
    silently the moment the vendored copy is refreshed -- the copies here are byte
    identical to their sources, so the text is already on disk. Returns '' on any IO or
    parse failure, like every other gate in this module.
    """
    try:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    try:
        meta: Any = yaml.safe_load(text.split("---", 2)[1])
    except (yaml.YAMLError, IndexError):
        return ""
    if not isinstance(meta, dict):
        return ""
    return str(meta.get("description", "")).strip()


def _skill_section(skill_dir: Path, title: str, level: int = 2) -> str:
    """Verbatim body of one ``<level>-deep <title>`` section of a vendored SKILL.md, or ''.

    The heading is dropped and the body returned exactly as written -- links, relative
    paths and all -- so the prompt carries the skill's own reference index instead of a
    paraphrase. Every vendored skill maintains one; the paraphrase they replaced got the
    ordering wrong (veloq's SKILL.md leads with the routing table, not the workflow) and
    quietly dropped entries.

    ``level`` exists for ncu-report-skill, whose ``## File index`` holds a ``### Reference
    docs`` table and a ``### Helpers`` table. Pasting the whole section would advertise
    helpers we spend a line telling the agent to skip -- finding them on its own is what
    costs it a turn -- so that note asks for the subsection instead. A section that is
    renamed upstream returns '' and drops the index, which is the fail-open behaviour
    every gate in this module has.
    """
    mark = "#" * level + " "
    try:
        lines = (skill_dir / "SKILL.md").read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    out: list[str] = []
    inside = False
    for line in lines:
        # Any heading at or above this level ends the section; deeper ones are its own.
        if line.startswith("#") and len(line) - len(line.lstrip("#")) <= level:
            if inside:
                break
            inside = line.startswith(mark) and line[len(mark) :].strip() == title
            continue
        if inside:
            out.append(line)
    return "\n".join(out).strip()


# Nsight Compute ships ``ncu_report`` as a plain module under its install tree, not on
# PyPI -- ``pip install ncu_report`` does not exist. This used to feed the prompt a
# ``PYTHONPATH=`` for the vendored ncu-report-skill's helpers/; that pointer is gone
# (``veloq ncu`` supersedes them, and they read only the first launch), so nothing in a
# run depends on the local Nsight any more -- veloq brings its own pinned reader. Kept
# as the diagnostic for the version trap documented in CLAUDE.md: it answers "which
# ncu_report would a local script actually import?", which ``ncu --version`` does not.
_NSIGHT_PYTHON_GLOBS = (
    "/opt/nvidia/nsight-compute/*/extras/python",
    "/usr/local/cuda*/nsight-compute*/extras/python",
    "/usr/local/NVIDIA-Nsight-Compute*/extras/python",
)


def _ncu_report_pythonpath() -> str:
    """Newest Nsight Compute ``extras/python`` dir holding ncu_report, or ''.

    Returns '' when Nsight Compute is not installed -- like every other gate here
    that degrades rather than raises, a missing profiler just drops the note.
    """
    found: list[Path] = []
    for pattern in _NSIGHT_PYTHON_GLOBS:
        try:
            found += [
                p for p in Path("/").glob(pattern.lstrip("/")) if (p / "ncu_report.py").is_file()
            ]
        except OSError:
            continue
    if not found:
        return ""
    # Version dirs sort lexically well enough (2025.4.0 < 2025.4.1); take the newest.
    return str(sorted(found)[-1])


class Orchestrator:
    def __init__(self, problem: Problem, cfg: Config):
        self.problem = problem
        self.wd = Path(problem.repo_root).resolve()
        self.cfg = cfg
        # Run-dir plumbing, created in setup(): the journal is the event record,
        # control the live-knob channel, live_lock the liveness beacon.
        self.journal: Journal | None = None
        self.control: LoopControl | None = None
        self._live_lock: LiveLock | None = None
        self.impl_session: str | None = None
        self._best: float | None = None
        self._dispatched = 0
        self._logfile: Path | None = None
        self._dirs: LoopDirs | None = None  # set in setup(); read by _archive_run()
        self._git_lock = threading.Lock()

    # --- helpers ---
    def _log(self, msg: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        print(f"[kernelthing] {msg}", file=sys.stderr, flush=True)
        if self._logfile is not None:
            try:
                with open(self._logfile, "a", encoding="utf-8") as f:
                    f.write(f"{stamp}  {msg}\n")
            except OSError:
                pass

    def _emit(self, type: str, **fields: Any) -> None:
        if self.journal is not None:
            self.journal.emit(type, **fields)

    def _rel(self, path: Path) -> str:
        return os.path.relpath(Path(path).resolve(), self.wd)

    def _git(self, args: list[str]) -> str:
        return gates.git(args, self.wd).stdout.strip()

    def _parallelism(self) -> int:
        return self.control.parallelism() if self.control else self.cfg.parallelism

    def _budget_desc(self) -> str:
        """Human description of the real search budget -- the candidate/wall-clock
        bound ``want_more()`` enforces -- for agent prompts and logs."""
        parts = []
        if self.cfg.max_candidates:
            parts.append(f"{self.cfg.max_candidates} candidates")
        if self.cfg.wall_clock_s:
            parts.append(f"{format_duration(self.cfg.wall_clock_s)} wall-clock")
        if not parts:
            return "unbounded (runs until a manual stop)"
        return " or ".join(parts) + (" (whichever comes first)" if len(parts) > 1 else "")

    def _guard(
        self,
        dirs: LoopDirs,
        rnd: int,
        phase: str,
        project_root: Path | None = None,
        loop_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Context handed to the opencode PreToolUse guard (oc_guard/guard.js).

        Lets the guard reject writes/edits/reads/bash that would corrupt loop
        infrastructure (state file, plan, plan backup, prompt/summary/contract
        files, git push). ``phase`` is one of impl/review/finalize/methodology.
        """
        from .bootstrap import protected_files

        root = Path(project_root or self.wd).resolve()
        return {
            "loopDir": str((loop_dir or dirs.base).resolve()),
            "projectRoot": str(root),
            "planFile": self.problem.plan,
            "currentRound": rnd,
            "phase": phase,
            "editFiles": [str(Path(f)) for f in self.problem.edit_files],
            "editDir": str(root / self.problem.rel_dir)
            if self.problem.rel_dir not in ("", ".")
            else str(root),
            "protectedFiles": sorted(protected_files(self.problem)),
        }

    def _score_cmd_str(self) -> str:
        """Absolute-path scoring command the agent can self-test with.

        Constructs one from the venv so the agent never has to discover kernelthing
        on PATH. The problem's own ``score_command`` (if set) takes priority.

        ``--brief`` is here rather than in the prompt template because the template
        appends flags to this string (``{{SCORE_CMD}} --test-only``), so one place
        covers both invocations. It drops the ``bench`` record, which is 98% of the
        verdict line and which only the archive reads -- ``_cli_score`` runs its own
        invocation *without* it, so nothing is lost from result.json or the journal.
        A problem that brings its own ``score_command`` is returned untouched: we do
        not know that an arbitrary command understands the flag."""
        venv_bin = Path(sys.executable).parent
        kt = str(venv_bin / "kernelthing")
        if self.problem.score_command:
            return self.problem.score_command
        return f"{kt} score . --brief"

    def _preflight(self) -> None:
        """Fail loudly on a scoring command the agents cannot actually run.

        Every agent gets ``sys.executable``'s sibling ``kernelthing`` baked into its
        prompt as the scoring command. Moving or renaming the checkout while its venv
        is active breaks that script without breaking *us*: the parent process already
        resolved its interpreter at import, so the loop starts happily and hands all N
        agents a command that dies. The failure is silent and total -- every candidate
        burns a turn discovering it cannot score, and the run produces nothing.

        Existence is not enough to check. A console script left behind by a renamed
        checkout still exists; it is its ``#!`` line that dangles, and the kernel
        reports that as ENOENT *on the script*. So actually execute it. One ~0.3s
        subprocess at setup buys certainty about the string every agent will run.
        """
        if self.problem.score_command:
            return  # the problem owns its own command; not ours to validate
        kt = Path(sys.executable).parent / "kernelthing"
        hint = (
            f"\n  sys.executable = {sys.executable}"
            "\nThis usually means the checkout was renamed or moved while a venv from the "
            "old path was still active, leaving the console script's #! line dangling. "
            "Re-activate and reinstall:"
            "\n  deactivate; source .venv/bin/activate; pip install -e '.[dev]'"
        )
        try:
            proc = subprocess.run(
                [str(kt), "--help"],
                capture_output=True,
                text=True,
                timeout=60,
            )
        except OSError as e:
            # ENOENT here means either the script or its interpreter is missing.
            raise RuntimeError(
                f"scoring command is not runnable -- every agent would be handed a dead "
                f"path.\n  {kt}: {e}{hint}"
            ) from e
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"scoring command hung on --help: {kt}{hint}") from e
        if proc.returncode != 0:
            raise RuntimeError(
                f"scoring command exits {proc.returncode} on --help -- every agent would "
                f"be handed a broken command.\n  {kt}\n"
                f"{(proc.stderr or proc.stdout).strip()[:500]}{hint}"
            )

    def _prompt_common(self, state: State) -> dict[str, Any]:
        """Template fields shared by every operator prompt + the descriptor footer."""

        return {
            "PLAN": state.plan_file,
            "PLAN_FILE": state.plan_file,
            "EDIT_FILES": ", ".join(self.problem.edit_files),
            "SCORE_CMD": self._score_cmd_str(),
            "UNIT": self.problem.unit or self.problem.metric_name,
        }

    @cached_property
    def _kernel_tools_block(self) -> str:
        """Assemble the optional kernel-domain tool guidance (KDA skills) appended
        to implementer prompts. Each part is included only when its flag is on;
        with both off this returns '' (no section). Absolute paths into the
        kernelthing install's ``vendor/`` are used -- the whole filesystem is
        read-only-bound in the sandbox, so they resolve from any worktree.

        Cached: every input (cfg flags, ``sys.executable``, ``REPO_ROOT``) is
        fixed for the run, so the block is built once instead of per candidate.

        Scoring is remote-only: ``cli.score_command`` hard-errors on a problem
        without the popcorn backend, so a problem that has no popcorn config
        cannot be scored and there is no other tool story to tell it. The local
        story this used to carry (a shared-GPU arbitration notice, a local ``ncu``
        recipe) described a benchmark stack that no longer exists.
        """
        from . import popcorn
        from .config import REPO_ROOT, veloq_python

        pop = popcorn.config_or_none(self.problem)
        if pop is None:
            return ""

        parts: list[str] = []
        pyexe = sys.executable or "python3"
        wiki_dir = REPO_ROOT / "vendor" / "KernelWiki"
        ncu_dir = REPO_ROOT / "vendor" / "ncu-report-skill"
        # Ordering is three bands: what the agent *does* every turn, then the reference
        # material it consults when stuck, then the command surfaces it types. Commands
        # last is deliberate -- the verb lists are the most actionable thing here, and
        # this block sits immediately above the task, so last is the recency slot. It is
        # also why the skills that interpret a report are allowed to precede the verbs
        # that produce it: each section is self-contained (`_veloq_ref_note` opens with
        # its own vendored path), so nothing reads as a forward reference. The two
        # backward ones do still hold -- kernel-tools-veloq.md cites "the profiling
        # section above" for the rejected token, and kernel-tools-nsys.md defers to the
        # ncu section on treating durations as ratios; both stay above their citers.
        #
        # Band 1: the turn loop. Unconditional, unlike everything after it -- this block
        # carries the scoring command and the submission rules as well as the profile,
        # and a popcorn problem cannot be worked on without them. Gating it on cfg.ncu
        # (as the profiling how-to it replaced was) would leave `--no-ncu` agents unable
        # to score at all. cfg.ncu now narrows to what it names: the ncu-report-skill
        # section below.
        parts.append(
            prompts.load_and_render_safe(
                "claude/kernel-tools-profile.md",
                "",
                SUBMISSION_FILE=pop.submission_file,
                SCORE_CMD=self._score_cmd_str(),
            )
        )
        # Band 2: reference documents, one `###` section each. The veloq-backed ones are
        # gated on the binary *and* its bundled reader: the Nsight installed here is too
        # old to open a capture from the hosted profiler (see config.veloq_python), so
        # without the venv they would only mislead.
        veloq_bin = shutil.which("veloq")
        veloq_ok = bool(self.cfg.veloq and veloq_bin and veloq_python())
        if veloq_ok:
            parts.append(
                self._skill_part(
                    "Diagnosing an ncu report — `ncu-profile-analysis`",
                    self._veloq_ref_note(REPO_ROOT / "vendor" / "veloq-ncu-skill"),
                )
            )
        # B200-specific prose, not tooling -- and gated separately from the veloq blocks
        # because it reads a report the same way whether or not veloq resolved. `--no-ncu`
        # drops exactly this.
        if self.cfg.ncu:
            parts.append(
                self._skill_part(
                    "B200 profiling reference — `ncu-report-skill`",
                    self._ncu_skill_note(ncu_dir),
                )
            )
        if veloq_ok and pop.nsys:
            parts.append(
                self._skill_part(
                    "Reading an nsys timeline — `nsys-profile-analysis`",
                    self._veloq_ref_note(REPO_ROOT / "vendor" / "veloq-nsys-skill"),
                )
            )
        # The CUDA-docs MCP is the one tool here we cannot verify from this side: it is
        # declared in opencode's config, but whether any tool actually materialises
        # depends on an OAuth flow that happens outside kernelthing. So the block says
        # what to do when the tool is absent rather than asserting that it is there.
        if self.cfg.mcp_cuda_docs:
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-cuda-docs.md",
                    "",
                    MCP_SERVER=opencode_client.CUDA_DOCS_MCP_SERVER,
                    MCP_TOOL=opencode_client.CUDA_DOCS_MCP_TOOL,
                    MCP_DESCRIPTION=opencode_client.CUDA_DOCS_MCP_INSTRUCTIONS,
                )
            )
        parts.append(
            self._skill_part(
                "PTX / CUDA ISA reference — `ptx-skill`",
                self._ptx_note(REPO_ROOT / "vendor" / "ptx-skill"),
            )
        )
        # Band 3: command surfaces. One section per report, never a combined one --
        # the nsys section is dropped whole for a problem with `bench.popcorn.nsys` off,
        # and a merged section would either leak a timeline that never lands or carry a
        # conditional the .md cannot express.
        #
        # A report section is commands and constraints only. It deliberately does *not*
        # enumerate what the capture holds: the verb list already says what each verb
        # answers, and a prose list of "per-launch metrics, rule findings, warp-stall
        # histograms, SASS/PTX" is the digest problem in miniature -- it pre-picks the
        # dimensions and invites the agent to look no further than the ones named.
        if veloq_ok:
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-veloq.md",
                    "",
                    VELOQ_BIN=veloq_bin,
                    REPORT=f"{popcorn.PROFILE_DIR}/{popcorn.PROFILE_SUBDIR}/profile.ncu-rep",
                    # The verb lists live in popcorn.py because that is where they are
                    # checked against a real capture; rendering them here is what keeps
                    # the prompt from carrying a second copy that drifts.
                    NCU_VERBS=popcorn.veloq_verb_block("ncu"),
                    SUBMISSION_FILE=pop.submission_file,
                )
            )
        if veloq_ok and pop.nsys:
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-nsys.md",
                    "",
                    VELOQ_BIN=veloq_bin,
                    NSYS_REPORT=(
                        f"{popcorn.PROFILE_DIR}/{popcorn.PROFILE_SUBDIR}/"
                        f"{popcorn.NSYS_SUBDIR}/profile.nsys-rep"
                    ),
                    NSYS_VERBS=popcorn.veloq_verb_block("nsys"),
                )
            )
        if self.cfg.wiki and _vendored(wiki_dir):
            parts.append(
                prompts.load_and_render_safe(
                    "claude/kernel-tools-wiki.md",
                    "",
                    PYTHON=pyexe,
                    WIKI_DIR=str(wiki_dir),
                )
            )
        return self._tools_section(parts)

    @staticmethod
    def _skill_part(title: str, note: str) -> str:
        """One vendored skill as its own headed section, or '' when its note is empty.

        Every skill renders through the same template so the four sections cannot drift
        apart in shape. The empty-note check is what makes the heading safe to put in the
        .md: these trees are routinely absent (submodules, or a `cp -r` not yet done) and
        each note gates itself on that, so without this a missing tree would render a
        section header with nothing under it.
        """
        if not note.strip():
            return ""
        return prompts.load_and_render_safe(
            "claude/kernel-tools-skill.md", "", SKILL_TITLE=title, SKILL_NOTE=note
        )

    def _ptx_note(self, ptx_dir: Path) -> str:
        """Pointer to the vendored PTX/CUDA reference tree, or '' when absent.

        Every word of substance is lifted from the skill's own SKILL.md -- its
        ``description:`` and its two index sections -- rather than summarised here. Two
        mechanical excisions from the description, both visible in the code below rather
        than done by retyping it: the harness name it was written for (a different agent
        runs here, and ``tests/test_prompts.py`` bans that role word from the prompt tree
        for the same reason), and the trailing "Triggers on ..." list, which is
        skill-router dispatch metadata and means nothing to an agent already holding the
        path.

        The one addition is a single sentence about the local-collection material, which
        is where this tree genuinely contradicts the setup: it opens with
        compute-sanitizer, cuda-gdb and ``nvcc -g -G``, and those binaries *are*
        installed on this box and will happily run against whatever consumer GPU it has,
        at the wrong architecture, returning numbers that look real.
        """
        if not (self.cfg.ptx and _vendored(ptx_dir)):
            return ""
        desc = _skill_description(ptx_dir).split("Triggers on")[0].strip()
        desc = desc.replace(" for Claude Code", "")
        body = "\n\n".join(
            s
            for s in (
                _skill_section(ptx_dir, "Local API Documentation"),
                _skill_section(ptx_dir, "Additional References"),
            )
            if s
        )
        return (
            f"\nFor decoding an unfamiliar instruction in `veloq ncu disasm` output. "
            f"Vendored at `{ptx_dir}/`; every path below is relative to it.\n"
            f"The full skill is `{ptx_dir}/SKILL.md`; below are excerpts.\n"
            f"{('> ' + desc) if desc else ''}\n\n"
            "Its local-collection material does not apply here: the ncu capture is made "
            "for you by the remote profiler"
            f"\n{body}\n"
        )

    @staticmethod
    def _veloq_ref_note(skill_dir: Path) -> str:
        """Pointer to one vendored veloq analysis skill, or '' when absent.

        Description and reference index are read straight out of the skill's SKILL.md;
        nothing here is a summary of them. Neither veloq skill contradicts this setup --
        both are pure readers of a report the loop already downloaded -- so unlike
        ``_ptx_note`` there is no caveat to add, and every entry of each index carries
        through, in its order. ``ncu-profile-analysis`` and ``nsys-profile-analysis`` are
        the same document shape, which is why one function serves both with **no**
        per-skill parameter: differing text would be a paraphrase creeping back in, and
        what distinguishes the two sections is their heading, not their prose.
        """
        if not _vendored(skill_dir):
            return ""
        desc = _skill_description(skill_dir)
        refs = _skill_section(skill_dir, "References")
        return (
            f"\nVendored at `{skill_dir}/`; every path below is relative to it.\n"
            f"The full skill is `{skill_dir}/SKILL.md` (verb matrix, JSON envelope "
            "contract, workflow, when to stop trusting it); below are excerpts.\n"
            f"{('> ' + desc) if desc else ''}\n"
            f"\n{refs}\n"
        )

    @staticmethod
    def _ncu_skill_note(skill_dir: Path) -> str:
        """Pointer to the vendored ncu-report-skill, or '' when absent.

        A second ncu skill next to ``ncu-profile-analysis``, kept for a different reason:
        it is B200/sm_100-specific prose (the six analysis dimensions, a
        signal->cause->fix playbook, and sm_100 metric names that differ from every older
        GPU's) rather than tooling. Everything *executable* in it is a trap here, and the
        two caveats below are why this cannot be a bare "go read the skill":

        - Its SKILL.md quickstart and ``reference/03-collection.md`` build a harness and
          run ``ncu`` locally. Those binaries are installed on this box and will run, on a
          consumer GPU at the wrong architecture, returning numbers that look real.
        - Its ``helpers/*.py`` wrap ``ncu_report`` for a local capture: they read
          ``range_by_idx(0).action_by_idx(0)`` and stop, carry no SASS/PTX or per-line
          attribution, and ``rule_speedups()`` reads keys Nsight 2026.2 no longer emits,
          so it ranks every finding at 0.0/'?' rather than erroring.

        Hence the index comes from the ``### Reference docs`` subsection, not the whole
        ``## File index`` -- pasting that table would list all seven helpers by name and
        purpose one line above the sentence telling the agent to skip them.

        The description's trailing router metadata (a list of Chinese trigger phrases) is
        excised the same way ``_ptx_note`` drops its "Triggers on ..." tail: it is
        skill-router dispatch data and means nothing to an agent already holding the path.
        """
        if not _vendored(skill_dir):
            return ""
        desc = _skill_description(skill_dir).split("— including variants in Chinese")[0]
        desc = desc.strip().rstrip(",").strip()
        index = _skill_section(skill_dir, "Reference docs (read these when you need details)", 3)
        return (
            f"\nVendored at `{skill_dir}/`; every path below is relative to it.\n"
            f"{('> ' + desc) if desc else ''}\n\n"
            "Read it for its prose only. Ignore its collection workflow and its "
            f"`helpers/*.py`: `{skill_dir}/SKILL.md` assumes you profile locally, and the "
            "helpers wrap a local capture, read only the first launch, and add nothing "
            "over the report tooling you already have.\n"
            f"\n{index}\n"
        )

    @staticmethod
    def _tools_section(parts: list[str]) -> str:
        """Join the rendered tool blocks under one heading; '' when none are enabled."""
        parts = [p for p in parts if p.strip()]
        if not parts:
            return ""
        return "\n\n---\n\n## Kernel optimization tools (available in your sandbox)\n" + "\n".join(
            parts
        )

    # --- setup ---
    def setup(self) -> tuple[State, LoopDirs]:
        self._preflight()
        if not (self.wd / self.problem.plan).is_file():
            raise FileNotFoundError(f"plan not found: {self.wd / self.problem.plan}")
        if gates.git(["rev-parse", "--git-dir"], self.wd).returncode != 0:
            raise RuntimeError(f"{self.wd} is not a git repository")

        start_branch = self._git(["rev-parse", "--abbrev-ref", "HEAD"]) or "HEAD"
        base_commit = self._git(["rev-parse", "HEAD"])
        ts = new_timestamp()
        dirs = LoopDirs(self.wd, ts).ensure()
        self._logfile = dirs.logfile
        self._dirs = dirs
        state = State(
            timestamp=ts,
            plan_file=self.problem.plan,
            model=self.cfg.model,
            start_branch=start_branch,
            base_branch=start_branch,
            base_commit=base_commit,
            methodology=self.cfg.methodology,
        )
        shutil.copyfile(self.wd / self.problem.plan, dirs.plan_backup)
        problem_meta = {
            "name": self.problem.name,
            "unit": self.problem.unit,
            "direction": self.problem.direction,
            "metric_name": self.problem.metric_name,
        }
        config_meta = {
            "parallelism": self.cfg.parallelism,
            "max_candidates": self.cfg.max_candidates,
            "wall_clock_s": self.cfg.wall_clock_s,
            "elite_k": self.cfg.elite_k,
            "min_niches": self.cfg.min_niches,
            "sandbox": self.cfg.sandbox,
            "kernelguard": self.cfg.kernelguard,
        }
        save_run(dirs, state, problem=problem_meta, config=config_meta)
        self._live_lock = LiveLock(dirs.live_lock)
        self._live_lock.acquire()
        self.journal = Journal(dirs.events_file)
        self.control = LoopControl(
            dirs.control_file,
            self.journal,
            parallelism=self.cfg.parallelism,
            elite_k=self.cfg.elite_k,
            wall_clock_s=self.cfg.wall_clock_s,
            max_candidates=self.cfg.max_candidates,
        )
        self._emit(
            "run_start",
            problem=problem_meta,
            config=config_meta,
            model=self.cfg.model,
            base_commit=base_commit,
            base_branch=start_branch,
        )
        self._log("=" * 72)
        self._log(f"SETUP  problem '{self.problem.name}'")
        self._log(f"  repo        {self.wd}")
        self._log(
            f"  metric      {self.problem.metric_name or self.problem.unit} "
            f"({self.problem.unit}, {self.problem.direction})"
        )
        self._log(f"  start       branch {start_branch} @ {base_commit[:8]}")
        self._log(
            f"  config      budget {self.cfg.max_candidates or '∞'} candidates"
            + (f"/{format_duration(self.cfg.wall_clock_s)}" if self.cfg.wall_clock_s else "")
            + f" | parallelism {self._parallelism()} | elite_k {self.cfg.elite_k} | "
            f"model {self.cfg.model}"
        )
        self._log(f"  artifacts   {dirs.base}  (full log: loop.log)")
        self._log("=" * 72)
        return state, dirs

    def _run_implementer(
        self,
        prompt: str,
        log_path: Path,
        *,
        guard: dict[str, Any] | None = None,
    ) -> opencode_client.OpencodeResult:
        res = opencode_client.run(
            prompt,
            working_dir=self.wd,
            model=self.cfg.model,
            session=self.impl_session,
            timeout=self.cfg.opencode_timeout,
            writable=True,
            sandboxed=self.cfg.sandbox,
            log_path=log_path,
            err_path=Path(str(log_path) + ".stderr"),
            guard=guard,
            mcp_cuda_docs=self.cfg.mcp_cuda_docs,
        )
        if res.session_id:
            self.impl_session = res.session_id
        return res

    def _cli_score(self, wt: Path) -> dict[str, Any]:
        """Score a worktree by shelling out to ``kernelthing score`` and parsing its
        JSON. Runs the *same* code path agents use, and -- crucially -- in its own
        process, so concurrent scorings never race on the shared in-process import
        state that ``popcorn.score`` touches (submission cache, cwd).

        Returns the full verdict dict: ``{correct, metric, error, unit, bench}``
        plus ``stderr_tail`` when the scorer wrote to stderr. ``bench`` is the raw
        measurement record from the popcorn service (per-shape timings). Grading is
        remote, so no local GPU is involved.
        """
        prob_dir = wt / self.problem.rel_dir
        cmd = [sys.executable, "-m", "kernelthing", "score", str(prob_dir)]
        try:
            r = subprocess.run(
                cmd, cwd=str(prob_dir), capture_output=True, text=True, timeout=1800
            )
        except subprocess.TimeoutExpired:
            return {"correct": False, "metric": None, "error": "score timeout"}
        stderr_tail = (r.stderr or "").strip()[-2000:]
        line = next(
            (ln.strip() for ln in reversed(r.stdout.splitlines()) if ln.strip().startswith("{")),
            "",
        )
        if not line:
            err = stderr_tail[-500:] or "score emitted no JSON"
            return {"correct": False, "metric": None, "error": err, "stderr_tail": stderr_tail}
        try:
            d: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError:
            return {
                "correct": False,
                "metric": None,
                "error": "score emitted no parseable JSON",
                "stderr_tail": stderr_tail,
            }
        if stderr_tail:
            d["stderr_tail"] = stderr_tail
        return d

    @staticmethod
    def _score_tuple(d: dict[str, Any]) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Split a ``_cli_score`` verdict into (correct, metric, err, detail); the
        detail keeps everything else (bench timings, stderr tail) for result.json."""
        detail = {k: v for k, v in d.items() if k not in ("correct", "metric", "error", "unit")}
        return bool(d.get("correct")), d.get("metric"), d.get("error"), detail

    def _score_worktree(
        self, wt: Path
    ) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Score the worktree, returning (correct, metric, err, detail).

        Shells out to ``kernelthing score`` (one process per score -> no shared-state
        race), which submits to the hosted popcorn service.
        """
        return self._score_tuple(self._cli_score(wt))

    def _guarded_score(
        self, wt: Path
    ) -> tuple[bool, float | None, str | None, dict[str, Any]]:
        """Run the cheap static cheat gate (kernelguard) BEFORE the expensive remote
        score: a detected cheat is disqualified outright and never submitted. Scoring
        shells out to ``kernelthing score`` (see ``_cli_score``)."""
        if self.cfg.kernelguard:
            cheats = gates.kernelguard_violations(
                self.problem.edit_files,
                wt,
                profile=self.cfg.kernelguard_profile,
                metadata={"problem_name": self.problem.name},
            )
            if cheats:
                err = "kernelguard: " + ", ".join(x["file"] for x in cheats)
                return False, None, err, {"kernelguard": cheats}
        return self._score_worktree(wt)

    # --- async evolutionary search ---
    @staticmethod
    def _fmt(x: float | None) -> str:
        return f"{x:.1f}" if isinstance(x, (int, float)) else "?"

    def _evolve_ref(self, ts: str, member_id: int) -> str:
        return f"refs/kernelthing/{ts}/mem-{member_id}"

    def _evolve_prompt(
        self, operator: str, parent: evolve.Member | None, state: State, known_strategies: str
    ) -> str:
        common = self._prompt_common(state)
        if operator == evolve.OP_EXPLORE:
            body = prompts.render(
                EVOLVE_EXPLORE_PROMPT,
                PARENT_METRIC=self._fmt(parent.metric) if parent else "?",
                PARENT_COMMIT_MESSAGE=(parent.commit_message if parent else "baseline"),
                KNOWN_STRATEGIES=known_strategies or "(none yet)",
                **common,
            )
        else:  # exploit
            assert parent is not None
            body = prompts.render(
                EVOLVE_EXPLOIT_PROMPT,
                PARENT_METRIC=self._fmt(parent.metric),
                PARENT_COMMIT_MESSAGE=parent.commit_message or "(no commit message)",
                **common,
            )
        return (
            body + prompts.render(EVOLVE_DESCRIPTOR_FOOTER, **common) + self._kernel_tools_block
        )

    def _evolve_task(
        self, task: evolve.Task, base: str, wt_root: Path, dirs: LoopDirs
    ) -> evolve.Member:
        """Worker: fork a worktree, run one agent turn, then score it remotely.

        Pure git plumbing is taken under ``self._git_lock``; the agent turn and the
        remote score run outside it. Returns the scored Member; never raises.

        Everything about the attempt lands in ``members/<id>/`` -- the exact
        prompt, the live agent transcript, the summary, and (when it committed)
        the diff against its parent, which survives the run's ref cleanup.
        """
        m = evolve.Member(
            id=task.member_id, operator=task.operator, parent_id=task.parent_id
        )
        parent_commit = task.parent_commit or base
        wt = wt_root / f"m{task.member_id}"
        mdir = dirs.ensure_member(task.member_id)
        (mdir / "prompt.md").write_text(task.prompt, encoding="utf-8")
        try:
            with self._git_lock:
                rc = gates.git(
                    ["worktree", "add", "--detach", "--force", str(wt), parent_commit], self.wd
                ).returncode
            if rc != 0:
                m.error = "worktree add failed"
                return m
            guard = self._guard(
                dirs,
                task.member_id,
                "impl",
                project_root=wt,
                loop_dir=wt / ".humanize" / "rlcr" / "candidate",
            )
            t0 = time.time()
            res = opencode_client.run(
                task.prompt,
                working_dir=wt,
                model=self.cfg.model,
                session=None,
                timeout=self.cfg.opencode_timeout,
                writable=True,
                sandboxed=self.cfg.sandbox,
                log_path=dirs.member_log(task.member_id),
                err_path=dirs.member_stderr(task.member_id),
                data_dir=wt / ".humanize" / "oc-data",
                extra_writable=[self.wd / ".git"],
                guard=guard,
                mcp_cuda_docs=self.cfg.mcp_cuda_docs,
            )
            m.agent_s = round(time.time() - t0, 1)
            m.cost = res.cost
            m.tokens = res.tokens
            m.tool_calls = res.tool_calls
            m.agent_exit = res.exit_code
            head = gates.git(["rev-parse", "HEAD"], wt).stdout.strip()
            m.commit = head if head and head != parent_commit else None
            if m.commit:
                m.commit_message = gates.git(
                    ["log", "-1", "--format=%s", "HEAD"], wt
                ).stdout.strip()
                # Every commit of the turn, oldest first -- intermediate attempts
                # are part of the record even though only HEAD gets scored.
                m.commits = list(
                    reversed(
                        gates.git(
                            ["log", "--format=%h %s", f"{parent_commit}..HEAD"], wt
                        ).stdout.strip().splitlines()
                    )
                )
            sm = wt / "candidate-summary.md"
            m.summary_text = sm.read_text(encoding="utf-8") if sm.exists() else ""
            if m.commit is None:
                if res.exit_code == 124:
                    default_error = "agent turn timed out"
                elif res.error:
                    default_error = f"opencode: {res.error}"
                else:
                    default_error = "no commit"
                m.error = m.error or default_error
                return m
            # diff.patch is the *solution* diff only: agents are told to commit
            # with `git add -A`, so the raw parent..HEAD diff drags in profiling
            # dumps and other worktree junk. The full inventory of what else was
            # committed is kept as a name list (changed_files) in result.json.
            diff = gates.git(
                ["diff", f"{parent_commit}..HEAD", "--", *self.problem.edit_files], wt
            ).stdout
            if diff:
                (mdir / "diff.patch").write_text(diff, encoding="utf-8")
            m.changed_files = gates.git(
                ["diff", "--name-only", f"{parent_commit}..HEAD"], wt
            ).stdout.split()
            with self._git_lock:
                gates.git(["checkout", "--", *self.problem.edit_files], wt)
            t0 = time.time()
            m.correct, m.metric, m.error, m.score_detail = self._guarded_score(wt)
            m.score_s = round(time.time() - t0, 1)
        except Exception as e:
            m.error = repr(e)
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        return m

    def _evolve_seed(self, base: str, wt_root: Path, pop: evolve.Population) -> evolve.Member:
        """Score the starting HEAD as member 0 so explore/exploit have an incumbent.

        If it scores, it is the first elite; if not, the population starts empty and
        every operator falls back to explore (forking the base) until one works.
        Scoring shells out to ``kernelthing score`` (see ``_cli_score``), which
        submits the seed to the hosted popcorn service.
        """
        m = evolve.Member(id=pop.next_id(), operator="seed", commit=base, commit_message="baseline")
        wt = wt_root / "seed"
        with self._git_lock:
            gates.git(["worktree", "add", "--detach", "--force", str(wt), base], self.wd)
        try:
            if self.cfg.kernelguard:
                cheats = gates.kernelguard_violations(
                    self.problem.edit_files,
                    wt,
                    profile=self.cfg.kernelguard_profile,
                    metadata={"problem_name": self.problem.name},
                )
                if cheats:
                    m.error = "kernelguard: " + ", ".join(x["file"] for x in cheats)
                    m.score_detail = {"kernelguard": cheats}
                    return m

            m.correct, m.metric, m.error, m.score_detail = self._score_tuple(self._cli_score(wt))
        finally:
            with self._git_lock:
                gates.git(["worktree", "remove", "--force", str(wt)], self.wd)
        pop.insert(m)
        return m

    def _record_member(self, dirs: LoopDirs, m: evolve.Member) -> None:
        """Persist a settled member: summary + result.json + the journal event."""
        dirs.ensure_member(m.id)
        if m.summary_text:
            dirs.member_summary(m.id).write_text(m.summary_text, encoding="utf-8")
        rec = m.record()
        dirs.member_result(m.id).write_text(json.dumps(rec, indent=2), encoding="utf-8")
        self._emit("member_result", **rec)

    # --- evolutionary-search control (extracted from run() closures) ---

    def _ev_wall_limit(self) -> int:
        """Live wall-clock budget in seconds (0 = off)."""
        return self.control.wall_clock() if self.control else self.cfg.wall_clock_s

    def _ev_target(self) -> int:
        """Live -j: how many agents to keep in flight right now. The worker pool
        is sized to MAX_PARALLELISM (threads spawn lazily), so this can be
        raised as well as lowered mid-run."""
        return max(1, min(self._parallelism(), MAX_PARALLELISM))

    def _ev_max_candidates(self) -> int:
        """Live candidate budget (0 = unbounded).

        Every consumer must read the budget through here. The stop condition and
        the explore/exploit schedule previously disagreed -- one read the live
        value, the other ``cfg.max_candidates`` -- so raising -m in the web UI
        extended the run but left the schedule pinned at its tail value.
        """
        return self.control.max_candidates() if self.control else self.cfg.max_candidates

    def _ev_want_more(self, rc: RunContext) -> bool:
        """True while neither the candidate budget, wall-clock, nor stop flag is hit."""
        if self.control and self.control.stop_requested():
            return False
        maxc = self._ev_max_candidates()
        if maxc and rc.dispatched >= maxc:
            return False
        limit = self._ev_wall_limit()
        assert rc.search_start is not None  # set before dispatch loop begins
        return not (limit and time.time() - rc.search_start >= limit)

    def _auto_explore_frac(self, rc: RunContext) -> float:
        """Annealed explore fraction: broad early (0.8), greedy late (0.2).

        Progress is measured against whichever budget is actually in force, using
        the same live values ``_ev_want_more`` stops on. With both a candidate cap
        and a wall clock, the one nearer exhaustion drives the schedule -- matching
        the whichever-comes-first stop, so the anneal always finishes just as the
        run does. A wall-clock-only run (``-m 0 -w 8h``) therefore anneals on time
        rather than sitting at a flat 0.5 for its whole life. With neither budget
        set there is nothing to anneal against, so hold the midpoint.
        """
        maxc = self._ev_max_candidates()
        limit = self._ev_wall_limit()
        progress = 0.0
        bounded = False
        if maxc:
            progress = max(progress, rc.dispatched / maxc)
            bounded = True
        if limit and rc.search_start is not None:
            progress = max(progress, (time.time() - rc.search_start) / limit)
            bounded = True
        if not bounded:
            return 0.5
        return 0.8 - 0.6 * min(1.0, progress)

    def _ev_dispatch(
        self,
        rc: RunContext,
        pop: evolve.Population,
        state: State,
        base: str,
        wt_root: Path,
        dirs: LoopDirs,
        rng: random.Random,
        ex: ThreadPoolExecutor,
    ) -> None:
        """Pick an operator + parent, fork a worker, and register the future."""
        cfg = self.cfg
        if self.control:
            # Live -k: resize the exploit frontier before selection sees it.
            pop.elite_k = self.control.elite_k()
        have_elites = bool(pop.elites())
        if self.control and not self.control.explore_auto():
            explore_frac = self.control.explore_bias() / 100.0
        else:
            explore_frac = self._auto_explore_frac(rc)
        op = evolve.choose_operator(
            rng,
            {evolve.OP_EXPLORE: explore_frac, evolve.OP_EXPLOIT: 1.0 - explore_frac},
            have_elites=have_elites,
            n_niches=len(pop.niches()),
            min_niches=cfg.min_niches,
        )
        parent = pop.select_parent(op, rng, rc.in_flight)
        if parent is None:
            op = evolve.OP_EXPLORE
        mid = pop.next_id()
        prompt = self._evolve_prompt(op, parent, state, ", ".join(sorted(pop.niches().keys())))
        task = evolve.Task(
            member_id=mid,
            operator=op,
            parent_id=(parent.id if parent else None),
            parent_commit=(parent.commit if parent else None),
            prompt=prompt,
        )
        fut = ex.submit(self._evolve_task, task, base, wt_root, dirs)
        rc.futures[fut] = task
        if parent is not None:
            rc.in_flight[parent.id] = rc.in_flight.get(parent.id, 0) + 1
            parent.children += 1
        rc.dispatched += 1
        self._emit(
            "dispatch",
            member=mid,
            op=op,
            parent=(parent.id if parent else None),
            parent_metric=(parent.metric if parent else None),
            in_flight=len(rc.futures),
            dispatched=rc.dispatched,
            # Selection context at dispatch time -- why the search made this
            # choice is otherwise unreconstructable (RNG + live population).
            explore_frac=round(explore_frac, 3),
            niches=len(pop.niches()),
            elites=len(pop.elites()),
        )
        ptxt = f" <- mem {parent.id}" if parent else ""
        self._log(f"dispatch mem {mid}: {op}{ptxt}  (in-flight {len(rc.futures)})")

    def _ev_collect(
        self,
        rc: RunContext,
        pop: evolve.Population,
        state: State,
        dirs: LoopDirs,
        unit: str,
        fut: Any,
    ) -> None:
        """Absorb one completed future into the population (the run loop refills)."""
        task = rc.futures.pop(fut)
        if task.parent_id is not None:
            rc.in_flight[task.parent_id] = max(0, rc.in_flight.get(task.parent_id, 1) - 1)
        try:
            m = fut.result()
        except Exception as e:
            m = evolve.Member(
                id=task.member_id,
                operator=task.operator,
                parent_id=task.parent_id,
                error=repr(e),
            )
        pop.insert(m)
        if m.viable:
            assert m.commit is not None
            with self._git_lock:
                gates.git(
                    ["update-ref", self._evolve_ref(state.timestamp, m.id), m.commit],
                    self.wd,
                )
        self._record_member(dirs, m)
        best = pop.best()
        prev_best = self._best
        self._best = best.metric if best else self._best
        if best is not None and self._best != prev_best:
            self._emit("new_best", member=best.id, metric=best.metric)
        ptxt = f"<-{task.parent_id} " if task.parent_id is not None else ""
        res = (
            f"{m.metric:.1f}{unit} ✓ [{m.commit_message[:50]}]"
            if m.viable
            else f"✗ {m.error or 'no result'}"
        )
        self._log(
            f"result mem {m.id} ({m.operator} {ptxt}): {res}  · best {self._fmt(self._best)}{unit}"
        )

    # --- run: the evolutionary search loop ---

    def run(self) -> str:
        """Steady-state asynchronous evolutionary search (see kernelthing/evolve.py).

        Keeps up to ``-j``/parallelism agents editing at once (live-tunable both
        ways via the web UI, along with -k/-m/-w); scoring is a remote submission
        per candidate. Dispatches explore/exploit tasks against a durable
        population until the budget is spent or a stop is requested, then
        promotes the best kernel to HEAD.
        """
        try:
            return self._run()
        finally:
            if self.journal is not None:
                self.journal.close()
            if self._live_lock is not None:
                self._live_lock.release()
            # After the journal closes, so the archived copy has the last events;
            # in the finally, so a stop, a stall, an exception and a Ctrl-C all
            # archive too -- the runs worth keeping are rarely the tidy ones.
            self._archive_run()

    def _archive_run(self) -> None:
        """Copy this run's artifacts somewhere the next run cannot reach.

        Best-effort by construction: ``archive.export_run`` swallows its own
        failures, and the run's exit status must not depend on whether a copy
        succeeded. Only a hard power-loss escapes this path -- and that case is
        covered from the other side, by ``prepare_problem`` preserving the run
        record in the managed root instead of deleting it."""
        if self.cfg.archive_root is None or self._dirs is None:
            return
        archive.export_run(
            self._dirs.base,
            self.cfg.archive_root,
            problem_name=self.problem.name,
            repo=self.wd,
            edit_files=self.problem.edit_files,
            log=self._log,
        )

    def _run(self) -> str:
        state, dirs = self.setup()
        cfg = self.cfg
        unit = self.problem.unit
        rng = random.Random(cfg.evolve_seed)
        pop = evolve.Population(direction=self.problem.direction, elite_k=cfg.elite_k)
        base = self._git(["rev-parse", "HEAD"])
        self._base_commit = base
        # Namespaced by problem: the managed root is shared by every problem, and
        # the timestamp is only second-resolution, so two loops on *different*
        # problems launched in the same second would otherwise share this dir --
        # and the rmtree below (plus _evolve_cleanup's) would delete the other
        # run's live worktrees out from under it.
        wt_root = cfg.problem_root / "wt" / self.problem.name / state.timestamp / "evolve"
        shutil.rmtree(wt_root, ignore_errors=True)
        wt_root.mkdir(parents=True, exist_ok=True)

        pool_cap = max(1, cfg.parallelism)
        self._log("")
        self._log(
            f"──── EVOLVE  parallelism {pool_cap} · budget "
            f"{cfg.max_candidates or '∞'} candidates"
            + (f" / {format_duration(cfg.wall_clock_s)}" if cfg.wall_clock_s else "")
            + " ────"
        )

        rc = RunContext()
        self._emit("phase", phase="evolve")

        seed = self._evolve_seed(base, wt_root, pop)
        self._best = seed.metric if seed.viable else None
        self._record_member(dirs, seed)
        if seed.viable:
            self._emit("new_best", member=seed.id, metric=seed.metric)
        self._log(
            f"seed (HEAD {base[:8]}): "
            + (f"{seed.metric:.1f}{unit} ✓" if seed.viable else f"✗ {seed.error or 'no score'}")
        )

        rc.search_start = time.time()
        self._emit("search_start")

        # The pool is sized to the hard cap, not -j: threads spawn lazily, so the
        # live parallelism target (_ev_target) alone decides how many agents run.
        # That is what lets -j be raised mid-run, not just lowered.
        with ThreadPoolExecutor(max_workers=MAX_PARALLELISM) as ex:
            while self._ev_want_more(rc) and len(rc.futures) < self._ev_target():
                self._ev_dispatch(rc, pop, state, base, wt_root, dirs, rng, ex)
            while rc.futures:
                # Bounded wait: wake periodically to re-read control.json so a
                # raised -j refills immediately instead of at the next result.
                done, _ = wait(list(rc.futures), timeout=15, return_when=FIRST_COMPLETED)
                for fut in done:
                    self._ev_collect(rc, pop, state, dirs, unit, fut)
                while self._ev_want_more(rc) and len(rc.futures) < self._ev_target():
                    self._ev_dispatch(rc, pop, state, base, wt_root, dirs, rng, ex)

        self._dispatched = rc.dispatched
        best = pop.best()
        if best and best.commit:
            with self._git_lock:
                gates.git(["reset", "--hard", best.commit], self.wd)
            self._emit("promoted", member=best.id, commit=best.commit, metric=best.metric)
            self._log(
                f"evolve: promoted mem {best.id} @ {best.metric:.1f}{unit}"
                + (f" [{best.commit_message[:50]}]" if best.commit_message else "")
                + " to HEAD"
            )
            self._copy_best_kernel()
        else:
            self._log("evolve: no viable kernel found; HEAD unchanged")
        self._evolve_cleanup(state, pop, wt_root)

        if self.control and self.control.stop_requested():
            return self._finish(state, dirs, EXIT_STOPPED, "stopped by user via the web UI")
        if best is None:
            return self._finish(
                state,
                dirs,
                EXIT_STALL,
                f"evolutionary search ({rc.dispatched} candidates) found nothing viable",
            )
        return self._finish(
            state,
            dirs,
            EXIT_MAXITER,
            f"evolutionary search budget spent: {rc.dispatched} candidates, "
            f"best {best.metric:.1f}{unit} [{best.operator}]",
        )

    def _copy_best_kernel(self) -> Path:
        """Copy the current HEAD kernel files to the stable best-kernel path."""
        out = self.cfg.problem_root / f"{self.problem.name}-best"
        out.mkdir(parents=True, exist_ok=True)
        for f in self.problem.edit_files:
            src = self.wd / self.problem.rel_dir / f
            if src.is_file():
                shutil.copy(src, out / Path(f).name)
        self._log(f"persisted best kernel to {out}")
        return out

    def persist_current_head(self) -> None:
        """Copy whatever kernel is at HEAD to the stable best-kernel path.

        Called on KeyboardInterrupt so a killed run still saves the last promoted
        kernel, not just a clean exit. Best-effort; swallows all errors."""
        try:
            head = self._git(["rev-parse", "HEAD"])
        except Exception:
            return
        base = getattr(self, "_base_commit", "")
        if base and head == base:
            self._log("interrupted: HEAD unchanged from baseline, nothing to persist")
            return
        if not base:
            self._log("interrupted before seed scoring, nothing to persist")
            return
        self._copy_best_kernel()

    def _evolve_cleanup(self, state: State, pop: evolve.Population, wt_root: Path) -> None:
        with self._git_lock:
            for m in pop.members:
                if m.viable:
                    gates.git(
                        ["update-ref", "-d", self._evolve_ref(state.timestamp, m.id)], self.wd
                    )
            gates.git(["worktree", "prune"], self.wd)
        shutil.rmtree(wt_root, ignore_errors=True)

    # --- terminal exit + optional methodology analysis (ported from Humanize) ---
    def _finish(self, state: State, dirs: LoopDirs, reason: str, desc: str) -> str:
        if self.cfg.methodology and reason in (EXIT_COMPLETE, EXIT_MAXITER, EXIT_STALL, EXIT_STOP):
            try:
                self._methodology_phase(state, dirs, reason, desc)
            except Exception as e:  # methodology must never break the exit
                self._log(f"methodology phase error (ignored): {e!r}")
        self._log(f"loop exit: {reason} ({desc})")
        self._emit(
            "run_end", reason=reason, desc=desc, dispatched=self._dispatched, best=self._best
        )
        return reason

    def _methodology_phase(self, state: State, dirs: LoopDirs, exit_reason: str, desc: str) -> None:
        """Final retrospective on the run's methodology, written to the loop dir.

        Faithful to Humanize: analyze the round summaries/reviews from a pure
        methodology perspective and write a report + completion marker; gate on
        both existing with content (retry a few times), then exit.
        """
        done = dirs.base / "methodology-analysis-done.md"
        report = dirs.base / "methodology-analysis-report.md"
        if done.is_file() and done.read_text(encoding="utf-8").strip():
            return  # already done
        self._log("")
        self._log(f"──── METHODOLOGY ANALYSIS (exit: {exit_reason}) ────")
        self._emit("phase", phase="methodology")
        prompt = prompts.render(
            METHODOLOGY_PROMPT,
            LOOP_DIR=self._rel(dirs.base),
            EXIT_REASON=exit_reason,
            EXIT_REASON_DESCRIPTION=desc,
            DISPATCHED=self._dispatched,
            BUDGET=self._budget_desc(),
            BEST=(self._best if self._best is not None else "n/a"),
            UNIT=self.problem.unit,
        )
        log_path = dirs.base / "methodology-opencode.log"
        guard = self._guard(dirs, state.current_round, "methodology")
        for _ in range(3):
            res = self._run_implementer(prompt, log_path, guard=guard)
            self._emit(
                "methodology_turn",
                cost=res.cost,
                tokens=res.tokens,
                tool_calls=res.tool_calls,
                exit=res.exit_code,
            )
            self._log(
                f"methodology: analysis turn done (tools={res.tool_calls}, cost=${res.cost:.4f})"
            )
            if (
                report.is_file()
                and report.read_text(encoding="utf-8").strip()
                and done.is_file()
                and done.read_text(encoding="utf-8").strip()
            ):
                self._log(f"methodology: retrospective written -> {self._rel(report)}")
                return
            prompt = (
                f"The methodology analysis is incomplete. Write the retrospective to "
                f"{self._rel(report)} and a one-line completion note to {self._rel(done)}."
            )
        self._log("methodology: still incomplete after retries; continuing exit")
