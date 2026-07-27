"""The static cheat gate (kernelguard) must run BEFORE the expensive benchmark in
the evolutionary search, and a detected cheat must skip scoring entirely
(_guarded_score, used by every evolve worker)."""

from kernelthing import gates
from kernelthing.config import Config
from kernelthing.orchestrator import Orchestrator
from kernelthing.problem import Problem


def _orch(tmp_path):
    prob = Problem(
        name="p",
        repo_root=tmp_path,
        rel_dir=".",
        plan="plan.md",
        edit_files=["k.cu"],
        score_command="echo",
    )
    return Orchestrator(prob, Config(kernelguard=True))


def test_cheat_disqualified_without_scoring(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    monkeypatch.setattr(gates, "kernelguard_violations", lambda *a, **k: [{"file": "k.cu"}])
    scored = []
    monkeypatch.setattr(
        orch, "_score_worktree", lambda wt, **kw: scored.append(wt) or (True, 99.0, None, {})
    )
    correct, metric, err, detail = orch._guarded_score(tmp_path)
    assert correct is False
    assert metric is None
    assert err.startswith("kernelguard")
    assert detail["kernelguard"] == [{"file": "k.cu"}]  # full violation record kept
    assert scored == []


def test_clean_candidate_is_scored(tmp_path, monkeypatch):
    orch = _orch(tmp_path)
    monkeypatch.setattr(gates, "kernelguard_violations", lambda *a, **k: [])
    monkeypatch.setattr(orch, "_score_worktree", lambda wt, **kw: (True, 99.0, None, {}))
    correct, metric, err, _detail = orch._guarded_score(tmp_path)
    assert correct is True and metric == 99.0 and err is None


def test_the_seed_score_does_not_profile_but_a_candidate_score_does(tmp_path, monkeypatch):
    """The seed's worktree is removed seconds after it scores, so a capture taken
    there is deleted before anything reads it -- and because profiles are cached on
    the submission sha256, its only lasting effect was to turn the first candidate's
    capture (scored on the *unmodified* seed file) into a cache hit. A hit carries no
    report, only flat text, so the agent's veloq verbs and both analysis skills go
    dark on its first look at the problem. Skipping it makes that first score a
    deliberate miss."""
    orch = _orch(tmp_path)
    seen: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = '{"correct": true, "metric": 1.0, "error": null}'
        stderr = ""

    monkeypatch.setattr(
        "kernelthing.orchestrator.subprocess.run",
        lambda cmd, **kw: seen.append(list(cmd)) or _Proc(),
    )

    orch._cli_score(tmp_path, no_profile=True)
    orch._cli_score(tmp_path)
    seed_cmd, candidate_cmd = seen
    assert "--no-profile" in seed_cmd, "the seed must not populate the profile cache"
    assert "--no-profile" not in candidate_cmd, "every other score still captures"
    # test/benchmark are deliberately still cached -- the incumbent metric comes from
    # them and the hit is pure win, so --test-only must not have crept in here.
    assert "--test-only" not in seed_cmd


def test_no_profile_is_a_flag_the_score_cli_actually_accepts():
    """_cli_score builds this string by hand; a flag that stopped parsing would make
    every seed score fail at argparse, before any submission."""
    import pytest

    from kernelthing import cli

    with pytest.raises(SystemExit) as e:
        cli.score_command(["--no-profile", "--help"])
    assert e.value.code == 0
