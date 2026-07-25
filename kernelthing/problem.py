"""Problem manifest: how kernelthing targets an arbitrary kernel problem.

A *problem* is a directory inside a git repo containing a ``problem.json``
manifest, a plan, the editable kernel file(s), and a self-contained ``score``
command that builds + checks correctness against the problem's own baseline
(cuBLAS, torch, a CPU reference, ...) and prints a JSON line:

    {"correct": true, "metric": 88.0, "unit": "%cuBLAS", ...}

kernelthing stays language/baseline-agnostic: it only runs the score command
and reads that JSON. All paths the orchestrator uses are resolved relative to
the enclosing git repo root (so worktrees and @file mentions work).
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Problem:
    name: str
    repo_root: Path  # git toplevel == orchestrator working dir
    rel_dir: str  # problem dir, relative to repo_root
    plan: str  # repo-relative path to the plan
    edit_files: list[str]  # repo-relative paths the agent may edit
    # Plain score command (cwd = <worktree>/<rel_dir>), used as the agent's
    # self-test command; the authoritative scoring always goes through pygpubench.
    # Problems may set it to a custom check or leave it empty (the loop fills in
    # ``kernelthing score .`` for the agent).
    score_command: str = ""
    metric_name: str = "metric"
    unit: str = ""
    direction: str = "maximize"  # or "minimize"
    # Expected GPU model name (matched against nvidia-smi --query-gpu=name).
    # Empty means "no restriction" — pre-existing or hand-authored problems
    # are allowed to run on any GPU. The bootstrap process fills this in
    # automatically from the GPU it runs on, and kernelthing rejects --gpu
    # indices whose model name doesn't match.
    gpu: str = ""
    # pygpubench config: submission_qualname, task_module, generator,
    # test_args, repeats, seed, timeout, landlock/mseal/allow_root. See bench.py.
    bench: dict[str, Any] = field(default_factory=dict)
    # metric derivation for pygpubench: kind + (flops|baseline_qualname).
    metric: dict[str, Any] = field(default_factory=dict)

    @property
    def dir(self) -> Path:
        return self.repo_root / self.rel_dir


def git_toplevel(path: Path) -> Path:
    r = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"{path} is not inside a git repository")
    return Path(r.stdout.strip())


def load_problem(path: str | Path) -> Problem:
    """Load a problem from a directory or a problem.json path."""
    p = Path(path).resolve()
    manifest = p / "problem.json" if p.is_dir() else p
    if not manifest.is_file():
        raise FileNotFoundError(f"no problem.json at {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    prob_dir = manifest.parent
    repo_root = git_toplevel(prob_dir)
    rel_dir = str(prob_dir.relative_to(repo_root))

    def repo_rel(rel_to_problem: str) -> str:
        # manifest paths are relative to the problem dir; make them repo-relative
        return str(Path(rel_dir) / rel_to_problem) if rel_dir != "." else rel_to_problem

    return Problem(
        name=data["name"],
        repo_root=repo_root,
        rel_dir=rel_dir,
        plan=repo_rel(data["plan"]),
        edit_files=[repo_rel(f) for f in data["edit_files"]],
        score_command=data.get("score_command", ""),
        metric_name=data.get("metric_name", "metric"),
        unit=data.get("unit", ""),
        direction=data.get("direction", "maximize"),
        gpu=data.get("gpu", ""),
        bench=dict(data.get("bench", {})),
        metric=dict(data.get("metric", {})),
    )


# Never copied from the source problem dir into the managed repo: build noise,
# and the artifact trees (a run's own record, archives of past runs, and rendered
# transcripts of past agents). Copying those in would commit them to the initial
# commit, so every worktree would then materialise every past run -- and the
# agent would be editing its kernel next to a verbatim log of what 25 previous
# agents tried, which is context nobody chose to give it. This is a name filter,
# not a gitignore consult: the copy happens before the managed repo exists, so a
# .gitignore in the source dir does not gate it.
NO_COPY = frozenset({"__pycache__", ".humanize", "runs", "transcripts", ".git"})

# Kept across the managed repo's rebuild -- this is the run record (journal,
# members, results). See kernelthing/archive.py for why losing it was expensive.
PRESERVE = frozenset({".humanize"})


def prepare_problem(problem: Problem, managed_root: Path) -> Problem:
    """Copy the problem dir into a standalone git repo at ``managed_root/<name>/``
    and return a new Problem rooted there. All worktrees branch from this repo,
    so the source repo (kernelthing itself) is never touched.

    The managed repo is rebuilt from scratch on every run -- fresh ``git init``,
    one initial commit -- but rebuilding is *not* the same as erasing. Past runs'
    artifacts under ``.humanize/`` are preserved: they are the only record that a
    run ever happened, they are what the web UI replays, and a run that died to a
    crashed machine has nothing else left. Deleting them here used to be how you
    lost a finished experiment by starting the next one."""
    dest = managed_root / problem.name
    _clear_except(dest, PRESERVE)
    dest.mkdir(parents=True, exist_ok=True)
    for item in problem.dir.iterdir():
        if item.name in NO_COPY:
            continue
        if item.is_dir():
            shutil.copytree(item, dest / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest / item.name)

    rewrite_plan_for_worktree(dest, problem)

    subprocess.run(["git", "init", "-b", "main"], cwd=dest, check=True, capture_output=True)
    # Written before the first `git add -A`: the preserved artifact tree is
    # sitting in the working dir now, and must never enter the index (it would
    # put every past run into the initial commit, and into every worktree).
    _write_git_exclude(dest, PRESERVE)
    subprocess.run(["git", "add", "-A"], cwd=dest, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "initial problem"],
        cwd=dest,
        check=True,
        capture_output=True,
    )

    return load_problem(dest / "problem.json")


def _clear_except(dest: Path, keep: frozenset[str]) -> None:
    """Empty ``dest`` of everything but ``keep``.

    Deleting entry-by-entry rather than ``rmtree(dest)`` + recreate is what makes
    the preservation safe: the kept subtree is never moved, so there is no window
    where a crash strands it somewhere the next run will not look."""
    if not dest.is_dir():
        return
    for item in dest.iterdir():
        if item.name in keep:
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item, ignore_errors=True)
        else:
            with contextlib.suppress(OSError):
                item.unlink()


def _write_git_exclude(dest: Path, names: frozenset[str]) -> None:
    """Mark the preserved artifact dirs untracked-forever for this repo.

    ``.git/info/exclude`` rather than ``.gitignore``: the exclude file is not a
    tracked file, so it never shows up in a candidate's diff, and it is shared by
    every worktree -- which is also where a candidate's own ``.humanize/`` scratch
    dir would otherwise land in its ``git add -A``."""
    info = dest / ".git" / "info"
    with contextlib.suppress(OSError):
        info.mkdir(parents=True, exist_ok=True)
        (info / "exclude").write_text(
            "# written by kernelthing: run artifacts are never tracked\n"
            + "".join(f"/{n}/\n" for n in sorted(names)),
            encoding="utf-8",
        )


def rewrite_plan_for_worktree(dest: Path, problem: Problem) -> None:
    """Update the copied plan.md for standalone worktree context.

    The source plan references the kernelthing repo layout (e.g. ``kernelthing
    score problems/<name>``).  In the managed worktree the problem files live at
    the repo root, not under ``problems/<name>/``, so these references are wrong.
    Replace them with the worktree-appropriate equivalent and prepend a context
    notice so agents know where they are.
    """
    import sys

    plan_path = dest / problem.plan
    if not plan_path.is_file():
        return
    text = plan_path.read_text(encoding="utf-8")
    original = text

    source_dir = f"problems/{problem.name}"
    if source_dir in text:
        text = text.replace(source_dir, ".")
        text = text.replace("./plan.md", "plan.md")

    venv_bin = Path(sys.executable).parent
    kt = str(venv_bin / "kernelthing")
    text = text.replace("kernelthing score", f"{kt} score")

    if text != original:
        plan_path.write_text(text, encoding="utf-8")
