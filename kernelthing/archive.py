"""Durable export of a run's artifacts out of the disposable managed root.

Everything a run produces lands under ``<problem_root>/<name>/`` -- which is
``~/.cache/kernelthing`` by default. Two things follow from that, and both cost
real work: ``~/.cache`` is XDG-disposable by convention, and
``problem.prepare_problem`` rebuilds the managed repo from the source problem
dir on every run. The rebuild used to take ``.humanize/`` with it, so starting a
second run silently destroyed the first one's entire record; it now preserves
that subtree, but "the cache still has it" is not a promise anyone should plan
around. So a run is copied here when it ends, outside the cache, where nothing
kernelthing does will touch it again.

The layout deliberately mirrors the managed root -- ``<archive>/<problem>/
.humanize/rlcr/<ts>/`` -- which makes an archive root a drop-in for
``kernelthing web --root``: ``journal.discover_runs`` globs
``<root>/*/.humanize/rlcr/*``, so archived runs replay exactly like live ones
with no special-casing anywhere in the UI. Alongside the run dir go the two
things the run dir does *not* contain: a git bundle of every member commit (the
managed repo is rebuilt on the next run, taking those commits with it) and the
winning kernel files.

Export is best-effort and never raises. A failed copy must not change a run's
exit status, and the cache copy is still on disk to retry from -- see
``kernelthing archive`` for exporting a run whose process died before it could
export itself.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from . import journal

# Subdirs of <archive>/<problem>/ that sit beside the mirrored .humanize tree.
BUNDLES = "bundles"
BEST = "best"
ARCHIVE_JSON = "archive.json"


def default_archive_root() -> Path:
    """``~/.local/share/kernelthing/runs`` -- XDG *data*, not cache: the whole
    point is to land somewhere nothing sweeps."""
    return Path.home() / ".local" / "share" / "kernelthing" / "runs"


def _bundle_repo(repo: Path, out: Path) -> bool:
    """Bundle every ref in ``repo`` (member commits live under
    ``refs/kernelthing/<ts>/mem-N``, so ``--all`` catches them)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            ["git", "bundle", "create", str(out), "--all"],
            cwd=repo,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return r.returncode == 0 and out.is_file()


def export_run(
    run_dir: Path,
    archive_root: Path,
    *,
    problem_name: str,
    repo: Path | None = None,
    edit_files: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> Path | None:
    """Copy one run's artifacts to ``<archive_root>/<problem_name>/``.

    Returns the archived run dir, or None if nothing was exported. Never raises:
    every failure mode here (disk full, permissions, a repo that has gone away)
    is strictly less important than the caller's own exit path.
    """

    def say(msg: str) -> None:
        if log is not None:
            log(msg)

    try:
        run_dir = Path(run_dir).resolve()
        if not (run_dir / journal.RUN_JSON).is_file():
            say(f"archive skipped: {run_dir} is not a run dir")
            return None

        ts = run_dir.name
        base = Path(archive_root).expanduser() / problem_name
        dest = base / ".humanize" / "rlcr" / ts
        dest.parent.mkdir(parents=True, exist_ok=True)

        # dirs_exist_ok so re-exporting a run refreshes it in place rather than
        # erroring -- a crashed run gets exported again by `kernelthing archive`.
        shutil.copytree(run_dir, dest, dirs_exist_ok=True, symlinks=True)

        bundled = False
        if repo is not None and (Path(repo) / ".git").exists():
            bundled = _bundle_repo(Path(repo), base / BUNDLES / f"{ts}.bundle")

        copied: list[str] = []
        best: dict[str, object] | None = None
        if repo is not None and edit_files:
            best_dir = base / BEST / ts
            best_dir.mkdir(parents=True, exist_ok=True)
            # Prefer the journal's own verdict over the repo's HEAD. They agree
            # for a run that exited cleanly (the loop promotes the winner to HEAD
            # on the way out) -- but a run that was killed never got to promote
            # anything, so HEAD is still the seed. The journal knows regardless.
            commit, best = _best_scored_member(run_dir)
            if commit:
                copied = _extract_at(Path(repo), commit, edit_files, best_dir)
            if not copied:
                for f in edit_files:
                    src = Path(repo) / f
                    if src.is_file():
                        shutil.copy2(src, best_dir / Path(f).name)
                        copied.append(Path(f).name)
                best = None if commit else best

        (base / ARCHIVE_JSON).write_text(
            json.dumps(
                {
                    "problem": problem_name,
                    "archived_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source_run_dir": str(run_dir),
                    "source_repo": str(repo) if repo else None,
                    "best": best,
                    "note": "serve with: kernelthing web --root "
                    + str(Path(archive_root).expanduser()),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        extra = []
        if bundled:
            extra.append("git bundle")
        if copied:
            extra.append(f"best kernel ({', '.join(copied)})")
        say(f"archived run to {dest}" + (f" + {' + '.join(extra)}" if extra else ""))
        return dest
    except Exception as e:  # never let archiving change a run's outcome
        say(f"archive failed (ignored): {e!r}")
        return None


def _best_scored_member(run_dir: Path) -> tuple[str | None, dict[str, object] | None]:
    """The best member the journal actually recorded, as ``(commit, info)``.

    A pure fold over ``events.ndjson`` -- the same source the web UI folds -- so
    it is right for a crashed run, a stopped run, and a clean one alike. Returns
    ``(None, None)`` if nothing scored.
    """
    try:
        meta = json.loads((run_dir / journal.RUN_JSON).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None
    minimize = str((meta.get("problem") or {}).get("direction", "maximize")) == "minimize"

    best: dict[str, object] | None = None
    best_metric: float | None = None
    try:
        with open(run_dir / journal.EVENTS_NDJSON, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a torn last line is normal after a hard kill
                if e.get("type") != "member_result" or not e.get("correct"):
                    continue
                metric, commit = e.get("metric"), e.get("commit")
                if metric is None or not commit:
                    continue
                metric = float(metric)
                if best_metric is None or (
                    metric < best_metric if minimize else metric > best_metric
                ):
                    best_metric = metric
                    best = {
                        "member": e.get("member"),
                        "metric": metric,
                        "commit": commit,
                        "message": e.get("message", ""),
                    }
    except OSError:
        return None, None
    return (str(best["commit"]), best) if best else (None, None)


def _extract_at(repo: Path, commit: str, files: list[str], out: Path) -> list[str]:
    """Write ``files`` as they were at ``commit`` into ``out``."""
    written: list[str] = []
    for f in files:
        try:
            r = subprocess.run(
                ["git", "show", f"{commit}:{f}"],
                cwd=repo,
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if r.returncode == 0:
            dest = out / Path(f).name
            dest.write_bytes(r.stdout)
            written.append(dest.name)
    return written


def export_all(
    source_root: Path,
    archive_root: Path,
    *,
    run_ids: list[str] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    """Export every run under ``source_root`` (or just ``run_ids``).

    The problem name and repo are recovered from the run dir's own position:
    a run dir is ``<repo>/.humanize/rlcr/<ts>``, so the repo is three parents up
    and its basename is the problem name.
    """
    source_root = Path(source_root).expanduser().resolve()
    wanted = set(run_ids or [])
    out: list[Path] = []
    for entry in journal.discover_runs(source_root):
        run_id = str(entry["id"])
        if wanted and run_id not in wanted:
            continue
        run_dir = source_root / run_id
        repo = run_dir.parent.parent.parent
        meta = entry.get("run") or {}
        problem = str((meta.get("problem") or {}).get("name") or repo.name)
        edit_files = _edit_files_of(repo)
        dest = export_run(
            run_dir,
            archive_root,
            problem_name=problem,
            repo=repo if repo.is_dir() else None,
            edit_files=edit_files,
            log=log,
        )
        if dest is not None:
            out.append(dest)
    return out


def _edit_files_of(repo: Path) -> list[str]:
    """Best-effort ``edit_files`` from the repo's problem.json (absent for a
    managed repo that has already been rebuilt -- then we just skip the kernel
    copy and still take the run dir)."""
    try:
        manifest = json.loads((repo / "problem.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    files = manifest.get("edit_files") or []
    return [str(f) for f in files]
