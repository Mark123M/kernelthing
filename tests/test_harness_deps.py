"""Harness prerequisites that fail silently at runtime rather than loudly at startup.

Each case here is a real failure recovered from a run's artifacts:

- a renamed checkout left every agent with a ``kernelthing`` path that no longer
  existed, so all 25 candidates burned a turn discovering they could not score;
- ``popcorn`` extracts profile captures relative to the process cwd, producing
  ``profile.<index>-<slug>/``, which a bare ``profile/`` ignore rule misses;
- ``ncu_report`` ships inside Nsight Compute rather than on PyPI, so the vendored
  skill's ``helpers/`` are dead unless the prompt names the path.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from kernelthing.config import Config, veloq_python
from kernelthing.opencode_client import build_opencode_env
from kernelthing.orchestrator import Orchestrator, _ncu_report_pythonpath
from kernelthing.problem import Problem

REPO = Path(__file__).resolve().parent.parent


def _problem(tmp_path: Path, *, score_command: str | None = None) -> Problem:
    return Problem(
        name="p",
        repo_root=tmp_path,
        rel_dir=".",
        plan="plan.md",
        edit_files=["submission.py"],
        score_command=score_command,
    )


# --- preflight: a dead interpreter path must not reach the agents ---


def _fake_venv(tmp_path: Path, script: str | None) -> Path:
    """A bin/ dir with a python and, optionally, a kernelthing console script."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "python").write_text("")
    if script is not None:
        kt = bin_dir / "kernelthing"
        kt.write_text(script)
        kt.chmod(0o755)
    return bin_dir


def test_preflight_rejects_missing_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "executable", str(_fake_venv(tmp_path, None) / "python"))
    with pytest.raises(RuntimeError, match="dead path"):
        Orchestrator(_problem(tmp_path), Config())._preflight()


def test_preflight_rejects_dangling_shebang(tmp_path, monkeypatch):
    # The real failure: a renamed checkout leaves the console script in place with a
    # #! pointing at an interpreter that is gone. The file EXISTS -- only executing it
    # reveals the break, and the kernel reports it as ENOENT on the script itself.
    bin_dir = _fake_venv(tmp_path, f"#!{tmp_path}/gone/bin/python\nprint('hi')\n")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    with pytest.raises(RuntimeError, match="dead path"):
        Orchestrator(_problem(tmp_path), Config())._preflight()


def test_preflight_rejects_nonzero_help(tmp_path, monkeypatch):
    bin_dir = _fake_venv(tmp_path, "#!/bin/sh\necho boom >&2\nexit 3\n")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    with pytest.raises(RuntimeError, match="exits 3"):
        Orchestrator(_problem(tmp_path), Config())._preflight()


def test_preflight_passes_on_a_runnable_entrypoint(tmp_path, monkeypatch):
    bin_dir = _fake_venv(tmp_path, "#!/bin/sh\nexit 0\n")
    monkeypatch.setattr(sys, "executable", str(bin_dir / "python"))
    Orchestrator(_problem(tmp_path), Config())._preflight()  # must not raise


def test_preflight_accepts_the_installed_entrypoint():
    # Guards the developer's own environment: if this fails, `kernelthing <problem>`
    # would start a run whose every candidate cannot score.
    prob = Problem(
        name="p", repo_root=REPO, rel_dir=".", plan="plan.md", edit_files=["submission.py"]
    )
    Orchestrator(prob, Config())._preflight()


def test_preflight_defers_to_an_explicit_score_command(tmp_path, monkeypatch):
    # The problem owns its command; we have no business validating its paths.
    monkeypatch.setattr(sys, "executable", str(tmp_path / "gone" / "bin" / "python"))
    orch = Orchestrator(_problem(tmp_path, score_command="echo"), Config())
    orch._preflight()  # must not raise


# --- ncu_report discovery ---


def test_ncu_report_pythonpath_is_either_valid_or_empty():
    # Fails open like every other gate: no Nsight Compute installed -> '' -> the
    # prompt drops the helper note instead of pointing at a path that is not there.
    found = _ncu_report_pythonpath()
    if found:
        assert (Path(found) / "ncu_report.py").is_file()


def test_veloq_python_is_either_valid_or_empty():
    # Same fail-open contract, but this one is load-bearing rather than a nicety:
    # the hosted profiler emits Nsight Compute 2026.2 captures and the Nsight
    # installed here (2025.4.1) cannot open one -- it dies building the sidecar on
    # `'IAction' object has no attribute 'timed_warp_samples'`, which fails every
    # veloq verb. So either veloq's own bundled venv resolves or the prompt block is
    # dropped; there is no third outcome and no usable fallback.
    found = veloq_python()
    if found:
        py = Path(found)
        assert py.is_file(), found
        assert list(py.parent.parent.glob("lib/python*/site-packages/ncu_report*")), found


def test_veloq_env_pin_survives_the_per_candidate_xdg_repoint(monkeypatch):
    # build_opencode_env repoints XDG_DATA_HOME at an isolated per-candidate dir,
    # which is exactly where veloq would otherwise look for its bundled reader. The
    # pin has to be an absolute path resolved from the *parent* env, or every agent
    # gets a veloq that cannot open a report.
    #
    # Clear any inherited value first: the pin is a setdefault, so a developer with
    # VELOQ_PYTHON exported would otherwise test their own shell, not this resolver.
    monkeypatch.delenv("VELOQ_PYTHON", raising=False)
    resolved = veloq_python()
    env, _ = build_opencode_env(data_dir=Path("/tmp/kt-xdg-probe"))
    if not resolved:
        assert "VELOQ_PYTHON" not in env  # nothing to pin -> pin nothing
        return
    assert env["VELOQ_PYTHON"] == resolved
    pinned = Path(env["VELOQ_PYTHON"])
    assert pinned.is_absolute() and pinned.is_file()
    assert not str(pinned).startswith(env["XDG_DATA_HOME"])


def test_veloq_env_pin_defers_to_an_operator_override(monkeypatch):
    # An explicitly exported VELOQ_PYTHON wins: the agent inherits the parent env
    # verbatim by design, and the pin only fills the gap the XDG repoint creates.
    monkeypatch.setenv("VELOQ_PYTHON", "/custom/python3")
    env, _ = build_opencode_env(data_dir=Path("/tmp/kt-xdg-probe"))
    assert env["VELOQ_PYTHON"] == "/custom/python3"


# --- the ignore rules must match the layout popcorn actually produces ---

# popcorn names the extracted capture after the shape, at the cwd it ran in.
# The `.veloq/` sidecar is written next to the report the first time veloq reads it.
_ARTIFACTS = (
    "profile.2-batch-256-n-128-cond-2-seed-41128/ncu-details.csv",
    "profile.2-batch-256-n-128-cond-2-seed-41128/ncu-details.txt",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.ncu-rep",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.ncu-rep.veloq/ncu-native.json.gz",
    "profile.2-batch-256-n-128-cond-2-seed-41128.zip",
    "profile/brev.json",
)


def _gitignores() -> list[Path]:
    return sorted((REPO / "problems").glob("*/.gitignore"))


def test_every_problem_ignores_profile_captures():
    ignores = _gitignores()
    assert ignores, "no problem .gitignore found"
    for gi in ignores:
        proc = subprocess.run(
            ["git", "check-ignore", *_ARTIFACTS],
            cwd=str(gi.parent),
            capture_output=True,
            text=True,
        )
        matched = set(proc.stdout.split())
        missing = [a for a in _ARTIFACTS if a not in matched]
        assert not missing, f"{gi} does not ignore {missing}"


def test_problem_sources_are_not_ignored():
    # The widened rule must not start swallowing the files the agent actually edits.
    for gi in _gitignores():
        proc = subprocess.run(
            ["git", "check-ignore", "submission.py", "task.py", "problem.json"],
            cwd=str(gi.parent),
            capture_output=True,
            text=True,
        )
        assert not proc.stdout.strip(), f"{gi} ignores problem sources: {proc.stdout}"
