"""Artifact durability: the managed root is disposable, so run records must
survive both the next run (prepare_problem) and leaving the cache (archive)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kernelthing import archive, journal
from kernelthing.problem import load_problem, prepare_problem

MANIFEST = {
    "name": "toy",
    "plan": "plan.md",
    "edit_files": ["submission.py"],
    "metric_name": "t",
    "unit": "us",
    "direction": "minimize",
    "bench": {"backend": "popcorn"},
}


def make_source_problem(tmp_path: Path) -> Path:
    """A problem dir inside a git repo, the shape prepare_problem consumes."""
    repo = tmp_path / "src"
    (repo / "problems" / "toy").mkdir(parents=True)
    d = repo / "problems" / "toy"
    (d / "problem.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (d / "plan.md").write_text("# plan\n", encoding="utf-8")
    (d / "submission.py").write_text("kernel = 'v1'\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    return d / "problem.json"


def make_run_dir(base: Path, ts: str, *, members: int = 2) -> Path:
    """A minimally valid run dir: run.json + journal + members."""
    d = base / ".humanize" / "rlcr" / ts
    (d / "members").mkdir(parents=True)
    (d / "run.json").write_text(
        json.dumps({"timestamp": ts, "problem": {"name": "toy"}}), encoding="utf-8"
    )
    (d / "events.ndjson").write_text('{"seq":1,"type":"run_start"}\n', encoding="utf-8")
    for i in range(members):
        m = d / "members" / str(i)
        m.mkdir()
        (m / "result.json").write_text(json.dumps({"id": i}), encoding="utf-8")
    return d


# --- prepare_problem must not destroy prior runs -----------------------------


def test_prepare_problem_preserves_prior_run_artifacts(tmp_path: Path) -> None:
    """The regression that cost a 4.5h run: rebuilding the managed repo used to
    rmtree the whole dir, taking every past run's journal with it."""
    manifest = make_source_problem(tmp_path)
    managed = tmp_path / "managed"
    prepared = prepare_problem(load_problem(manifest), managed)

    run = make_run_dir(prepared.repo_root, "2026-01-01_00-00-00")
    assert run.is_dir()

    # Second run of the same problem rebuilds the managed repo.
    prepare_problem(load_problem(manifest), managed)

    assert (run / "run.json").is_file(), "run.json destroyed by the rebuild"
    assert (run / "events.ndjson").read_text(encoding="utf-8").startswith('{"seq":1')
    assert (run / "members" / "1" / "result.json").is_file()


def test_prepare_problem_still_rebuilds_the_repo(tmp_path: Path) -> None:
    """Preserving artifacts must not turn into preserving stale kernel state:
    everything that is not an artifact is still rebuilt from source."""
    manifest = make_source_problem(tmp_path)
    managed = tmp_path / "managed"
    prepared = prepare_problem(load_problem(manifest), managed)

    stale = prepared.repo_root / "leftover.txt"
    stale.write_text("junk", encoding="utf-8")
    (prepared.repo_root / "submission.py").write_text("kernel = 'mutated'\n", encoding="utf-8")

    src = Path(manifest).parent
    (src / "submission.py").write_text("kernel = 'v2'\n", encoding="utf-8")
    again = prepare_problem(load_problem(manifest), managed)

    assert not stale.exists(), "stale file survived the rebuild"
    assert (again.repo_root / "submission.py").read_text(encoding="utf-8") == "kernel = 'v2'\n"
    log = subprocess.run(
        ["git", "log", "--oneline"], cwd=again.repo_root, capture_output=True, text=True
    )
    assert log.stdout.count("\n") == 1, "expected exactly one initial commit"


def test_preserved_artifacts_never_enter_the_index(tmp_path: Path) -> None:
    """.humanize/ in the working dir at `git add -A` time would put every past
    run into the initial commit -- and thus into every candidate worktree."""
    manifest = make_source_problem(tmp_path)
    managed = tmp_path / "managed"
    prepared = prepare_problem(load_problem(manifest), managed)
    make_run_dir(prepared.repo_root, "2026-01-01_00-00-00")
    again = prepare_problem(load_problem(manifest), managed)

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=again.repo_root, capture_output=True, text=True
    ).stdout
    assert ".humanize" not in tracked
    # ...and it does not show up as untracked noise in a candidate's status either
    status = subprocess.run(
        ["git", "status", "--short"], cwd=again.repo_root, capture_output=True, text=True
    ).stdout
    assert ".humanize" not in status


def test_source_side_archives_are_not_copied_into_the_managed_repo(tmp_path: Path) -> None:
    """A `runs/` archive kept next to the problem must not be dragged into the
    managed repo (and committed, and materialised in every worktree)."""
    manifest = make_source_problem(tmp_path)
    src = Path(manifest).parent
    (src / "runs" / "old").mkdir(parents=True)
    (src / "runs" / "old" / "big.ndjson").write_text("x" * 1000, encoding="utf-8")

    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    assert not (prepared.repo_root / "runs").exists()


# --- export ------------------------------------------------------------------


def test_export_run_copies_artifacts_and_bundle(tmp_path: Path) -> None:
    manifest = make_source_problem(tmp_path)
    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    run = make_run_dir(prepared.repo_root, "2026-02-02_11-11-11", members=3)

    dest = archive.export_run(
        run,
        tmp_path / "arch",
        problem_name="toy",
        repo=prepared.repo_root,
        edit_files=prepared.edit_files,
    )

    assert dest is not None
    assert (dest / "run.json").is_file()
    assert (dest / "events.ndjson").is_file()
    assert (dest / "members" / "2" / "result.json").is_file()
    base = tmp_path / "arch" / "toy"
    assert (base / "bundles" / "2026-02-02_11-11-11.bundle").is_file()
    assert (base / "best" / "2026-02-02_11-11-11" / "submission.py").is_file()
    assert json.loads((base / archive.ARCHIVE_JSON).read_text(encoding="utf-8"))["problem"] == "toy"


def test_archived_run_is_discoverable_by_the_web_ui(tmp_path: Path) -> None:
    """The archive layout mirrors the managed root so an archive root is a
    drop-in for `kernelthing web --root` -- no UI special-casing."""
    manifest = make_source_problem(tmp_path)
    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    make_run_dir(prepared.repo_root, "2026-03-03_09-09-09")
    archive.export_run(
        prepared.repo_root / ".humanize" / "rlcr" / "2026-03-03_09-09-09",
        tmp_path / "arch",
        problem_name="toy",
        repo=prepared.repo_root,
        edit_files=prepared.edit_files,
    )

    found = journal.discover_runs(tmp_path / "arch")
    assert [r["id"] for r in found] == ["toy/.humanize/rlcr/2026-03-03_09-09-09"]
    assert found[0]["live"] is False


def test_best_kernel_comes_from_the_journal_not_head(tmp_path: Path) -> None:
    """A killed run never promotes its winner to HEAD, so HEAD is still the seed.
    The journal recorded the winner anyway -- archive that, not HEAD."""
    manifest = make_source_problem(tmp_path)
    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    repo = prepared.repo_root

    # Two scored candidates committed, neither merged to HEAD (the loop died).
    commits = {}
    for tag, body in (("slow", "kernel = 'slow'\n"), ("fast", "kernel = 'fast'\n")):
        (repo / "submission.py").write_text(body, encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", tag], cwd=repo, check=True, capture_output=True)
        commits[tag] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True
        ).stdout.strip()
    subprocess.run(["git", "reset", "--hard", "HEAD~2"], cwd=repo, check=True, capture_output=True)
    assert (repo / "submission.py").read_text(encoding="utf-8") == "kernel = 'v1'\n"

    ts = "2026-07-07_07-07-07"
    run = make_run_dir(repo, ts, members=0)
    run.joinpath("run.json").write_text(
        json.dumps({"timestamp": ts, "problem": {"name": "toy", "direction": "minimize"}}),
        encoding="utf-8",
    )
    run.joinpath("events.ndjson").write_text(
        json.dumps(
            {"type": "member_result", "member": 1, "correct": True, "metric": 90.0,
             "commit": commits["slow"], "message": "slow"}
        )
        + "\n"
        + json.dumps(
            {"type": "member_result", "member": 2, "correct": True, "metric": 70.0,
             "commit": commits["fast"], "message": "fast"}
        )
        + "\n"
        + json.dumps({"type": "member_result", "member": 3, "correct": False, "metric": None})
        + "\n"
        + '{"type":"member_result","member":4,"corr',  # torn line, as after a kill
        encoding="utf-8",
    )

    dest = archive.export_run(
        run, tmp_path / "arch", problem_name="toy", repo=repo, edit_files=["submission.py"]
    )
    assert dest is not None

    best = tmp_path / "arch" / "toy" / "best" / ts / "submission.py"
    assert best.read_text(encoding="utf-8") == "kernel = 'fast'\n", "archived the wrong candidate"
    info = json.loads((tmp_path / "arch" / "toy" / archive.ARCHIVE_JSON).read_text("utf-8"))
    assert info["best"]["member"] == 2
    assert info["best"]["metric"] == 70.0


def test_best_kernel_maximize_direction(tmp_path: Path) -> None:
    """direction is read per-problem; a maximize problem must not pick the min."""
    manifest = make_source_problem(tmp_path)
    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    ts = "2026-08-08_08-08-08"
    run = make_run_dir(prepared.repo_root, ts, members=0)
    run.joinpath("run.json").write_text(
        json.dumps({"timestamp": ts, "problem": {"name": "toy", "direction": "maximize"}}),
        encoding="utf-8",
    )
    run.joinpath("events.ndjson").write_text(
        json.dumps({"type": "member_result", "member": 1, "correct": True, "metric": 90.0,
                    "commit": "deadbeef", "message": "hi"})
        + "\n"
        + json.dumps({"type": "member_result", "member": 2, "correct": True, "metric": 70.0,
                      "commit": "cafe", "message": "lo"})
        + "\n",
        encoding="utf-8",
    )
    commit, best = archive._best_scored_member(run)
    assert best is not None and best["member"] == 1 and commit == "deadbeef"


def test_export_run_is_idempotent(tmp_path: Path) -> None:
    """Re-exporting refreshes in place: a crashed run gets archived by hand, then
    again if it is ever resumed and finished."""
    manifest = make_source_problem(tmp_path)
    prepared = prepare_problem(load_problem(manifest), tmp_path / "managed")
    run = make_run_dir(prepared.repo_root, "2026-04-04_08-08-08")

    first = archive.export_run(run, tmp_path / "arch", problem_name="toy")
    (run / "events.ndjson").write_text('{"seq":2,"type":"run_end"}\n', encoding="utf-8")
    second = archive.export_run(run, tmp_path / "arch", problem_name="toy")

    assert first == second
    assert second is not None
    assert '"run_end"' in (second / "events.ndjson").read_text(encoding="utf-8")


def test_export_run_never_raises(tmp_path: Path) -> None:
    """Archiving runs in the orchestrator's finally block; if it could raise it
    would mask the run's real outcome."""
    assert archive.export_run(tmp_path / "nope", tmp_path / "arch", problem_name="toy") is None
    plain = tmp_path / "plain"
    plain.mkdir()
    assert archive.export_run(plain, tmp_path / "arch", problem_name="toy") is None


def test_export_all_recovers_every_run_under_a_root(tmp_path: Path) -> None:
    manifest = make_source_problem(tmp_path)
    managed = tmp_path / "managed"
    prepared = prepare_problem(load_problem(manifest), managed)
    make_run_dir(prepared.repo_root, "2026-05-05_01-01-01")
    make_run_dir(prepared.repo_root, "2026-05-05_02-02-02")

    done = archive.export_all(managed, tmp_path / "arch")

    assert len(done) == 2
    ids = {r["id"] for r in journal.discover_runs(tmp_path / "arch")}
    assert ids == {
        "toy/.humanize/rlcr/2026-05-05_01-01-01",
        "toy/.humanize/rlcr/2026-05-05_02-02-02",
    }


def test_export_all_can_select_one_run(tmp_path: Path) -> None:
    manifest = make_source_problem(tmp_path)
    managed = tmp_path / "managed"
    prepared = prepare_problem(load_problem(manifest), managed)
    make_run_dir(prepared.repo_root, "2026-06-06_01-01-01")
    make_run_dir(prepared.repo_root, "2026-06-06_02-02-02")

    done = archive.export_all(
        managed, tmp_path / "arch", run_ids=["toy/.humanize/rlcr/2026-06-06_02-02-02"]
    )
    assert len(done) == 1
    assert done[0].name == "2026-06-06_02-02-02"
