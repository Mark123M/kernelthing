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

from kernelthing import opencode_client
from kernelthing.config import Config, veloq_python
from kernelthing.opencode_client import _seed_auth_for_isolated_data, build_opencode_env
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


def test_isolated_data_dir_gets_every_opencode_auth_file(tmp_path):
    # opencode splits credentials across two files under $XDG_DATA_HOME/opencode:
    # auth.json (provider keys) and mcp-auth.json (per-server OAuth tokens, written by
    # `opencode mcp auth`). Seeding only the first leaves every candidate pointed at an
    # unauthenticated MCP server, which fails *silently* -- the server simply
    # contributes no tools, so the run looks normal and the docs are just never there.
    src = tmp_path / "src"
    (src / "opencode").mkdir(parents=True)
    (src / "opencode" / "auth.json").write_text('{"openrouter": {"type": "api"}}')
    (src / "opencode" / "mcp-auth.json").write_text('{"nvidia-cuda-docs": {"tokens": {}}}')
    dst = tmp_path / "dst"
    _seed_auth_for_isolated_data(src, dst)
    for name in ("auth.json", "mcp-auth.json"):
        copied = dst / "opencode" / name
        assert copied.is_file(), f"{name} was not seeded into the isolated data dir"
        assert copied.read_text() == (src / "opencode" / name).read_text()


def test_auth_seeding_survives_a_partial_source(tmp_path):
    # Only one of the two files existing is the normal state before anyone has run
    # `opencode mcp auth`; it must copy what is there rather than skip both.
    src = tmp_path / "src"
    (src / "opencode").mkdir(parents=True)
    (src / "opencode" / "auth.json").write_text("{}")
    dst = tmp_path / "dst"
    _seed_auth_for_isolated_data(src, dst)
    assert (dst / "opencode" / "auth.json").is_file()
    assert not (dst / "opencode" / "mcp-auth.json").exists()


def test_bash_timeout_default_clears_a_hosted_profile(monkeypatch):
    # opencode's bash tool caps a command at 120s by default. A hosted --profile-brev
    # run is 245-270s alone and longer behind a queue, so on the default every profile
    # dies mid-poll -- and silently: the job finishes server-side, but the CLI downloads
    # the artifacts only after it sees `succeeded`, so the loop pays and gets nothing.
    monkeypatch.delenv(opencode_client.BASH_TIMEOUT_ENV, raising=False)
    env, _ = build_opencode_env(data_dir=Path("/tmp/kt-xdg-probe"))
    assert int(env[opencode_client.BASH_TIMEOUT_ENV]) >= 600_000


def test_bash_timeout_defers_to_an_operator_override(monkeypatch):
    monkeypatch.setenv(opencode_client.BASH_TIMEOUT_ENV, "42")
    env, _ = build_opencode_env(data_dir=Path("/tmp/kt-xdg-probe"))
    assert env[opencode_client.BASH_TIMEOUT_ENV] == "42"


def test_veloq_env_pin_defers_to_an_operator_override(monkeypatch):
    # An explicitly exported VELOQ_PYTHON wins: the agent inherits the parent env
    # verbatim by design, and the pin only fills the gap the XDG repoint creates.
    monkeypatch.setenv("VELOQ_PYTHON", "/custom/python3")
    env, _ = build_opencode_env(data_dir=Path("/tmp/kt-xdg-probe"))
    assert env["VELOQ_PYTHON"] == "/custom/python3"


# --- the ignore rules must match the layout popcorn actually produces ---

# What `kernelthing score` now writes for every full score (popcorn.PROFILE_DIR /
# PROFILE_SUBDIR). These are 80MB of .ncu-rep per candidate; a problem whose .gitignore
# missed them would commit one per member.
_SCORER_ARTIFACTS = (
    "profile/latest/ncu-details.txt",
    "profile/latest/digest.txt",
    "profile/latest/profile.ncu-rep",
    "profile/latest/profile.ncu-rep.veloq/ncu-native.json.gz",
    "profile/latest/nsys/profile.nsys-rep",
    "profile/latest/nsys/profile.sqlite",
    "profile/latest/nsys/profile.sqlite-journal",
    "profile/latest/nsys/stats.txt",
)

# popcorn names the extracted capture after the shape, at the cwd it ran in. The scorer
# no longer lets that reach the worktree (it extracts in a temp dir), but a capture taken
# by hand still lands this way, so the rule has to keep covering it.
# The `.veloq/` sidecar is written next to the report the first time veloq reads it.
_ARTIFACTS = (
    *_SCORER_ARTIFACTS,
    "profile.2-batch-256-n-128-cond-2-seed-41128/ncu-details.csv",
    "profile.2-batch-256-n-128-cond-2-seed-41128/ncu-details.txt",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.ncu-rep",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.ncu-rep.veloq/ncu-native.json.gz",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.nsys-rep",
    "profile.2-batch-256-n-128-cond-2-seed-41128/profile.sqlite",
    "profile.2-batch-256-n-128-cond-2-seed-41128.zip",
    "profile/brev.json",
    "manual.nsys-rep",
    "manual.sqlite",
    "manual.sqlite-journal",
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
