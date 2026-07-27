"""``kernelthing score --dry-run`` -- the replay of what an agent reads after a score.

The command exists because that text is the loop's tightest feedback channel and is
otherwise invisible until a real submission has been paid for. These tests hold the two
properties that make the replay worth trusting: it goes through the production
formatters, and it neither submits nor writes anything.
"""

import json
import shlex

import pytest

from kernelthing import cli, dryrun, popcorn
from kernelthing.config import REPO_ROOT
from kernelthing.problem import load_problem

PROBLEM = REPO_ROOT / "problems" / "cholesky_b256n128"

# The service's figure for benchmark index 2, in microseconds -- the same number
# tests/fixtures/popcorn/api-benchmark.json carries in nanoseconds.
TARGET_US = 75.14144521620538


def _run(capsys, *argv):
    """Drive the real CLI and split stdout into (prose blocks, parsed verdict)."""
    rc = cli.score_command([str(PROBLEM), *argv])
    lines = capsys.readouterr().out.splitlines()
    return rc, "\n".join(lines[:-1]), json.loads(lines[-1])


def _scored(scenario="ok", **kw):
    """``popcorn.score`` under the replay -> ``(correct, metric, err, detail)``.

    How a test reaches the ``bench`` record now that no CLI invocation prints it. This
    is the better route regardless: it checks production's own return value instead of
    a flag that would exist only to let tests look at it.
    """
    problem = load_problem(PROBLEM)
    with dryrun._replayed(scenario):
        return popcorn.score(problem, problem.repo_root, **kw)


def test_a_full_dry_run_reads_like_a_full_score(capsys):
    rc, blocks, verdict = _run(capsys, "--dry-run")
    assert rc == 0
    assert verdict["correct"] is True
    assert verdict["metric"] == pytest.approx(TARGET_US)
    assert "--- Nsight Compute capture of the scored shape ---" in blocks
    assert "--- Nsight Systems timeline of the scored shape ---" in blocks
    # the directive leads, then the captures it refers to
    assert blocks.startswith("--- Task ---")
    assert blocks.index("--- Task ---") < blocks.index("--- Nsight Compute capture")


def test_stdout_is_the_production_fold_and_carries_no_dry_run_marker(capsys):
    """The whole point: stdout is byte-for-byte what a real score prints.

    Re-rendering the blocks from ``score``'s own detail dict is the check -- if the
    replay ever grew a marker of its own, or hand-rolled a banner instead of calling
    ``format_*``, the two would stop matching. The "this is a replay" notice lives on
    stderr for exactly this reason.
    """
    _, blocks, _ = _run(capsys, "--dry-run")
    _, _, _, detail = _scored()
    expected = "\n".join(
        b
        for b in (
            popcorn.format_analysis_directive(detail),
            popcorn.format_profile_block(detail),
            popcorn.format_nsys_block(detail),
        )
        if b
    )
    assert blocks == expected
    assert "dry" not in blocks.lower()


def test_a_dry_run_submits_nothing_and_writes_nothing(capsys, monkeypatch):
    """No network, no submission cache, no ``profile/latest/`` -- and the patches are
    handed back, since a leaked one would score every later test against a fixture."""

    def explode(*a, **k):
        raise AssertionError("the network boundary was reached during a dry run")

    real = (popcorn.submit, popcorn.profile_submission, popcorn.profile_nsys_submission)
    monkeypatch.setattr(popcorn, "submit", explode)
    monkeypatch.setattr(popcorn, "profile_submission", explode)
    monkeypatch.setattr(popcorn, "profile_nsys_submission", explode)
    monkeypatch.setattr(popcorn, "_cache_store", explode)
    monkeypatch.setattr(popcorn, "_cache_load", explode)

    before = set(PROBLEM.rglob("*"))
    rc, _, _ = _run(capsys, "--dry-run")
    assert rc == 0
    assert set(PROBLEM.rglob("*")) == before
    assert not (PROBLEM / popcorn.PROFILE_DIR).exists()
    # restored to whatever they were on entry, not to the module originals
    assert (popcorn.submit, popcorn.profile_submission, popcorn.profile_nsys_submission) == (
        explode,
        explode,
        explode,
    )
    assert real  # the module originals are what monkeypatch puts back after this test


def test_a_cache_hit_offers_the_text_view_instead_of_a_report(capsys):
    """The 80MB capture is deliberately not re-downloaded, so there is no path to query
    -- the block has to say so rather than print one that is not there."""
    _, blocks, _ = _run(capsys, "--dry-run", "cached")
    assert "(cached: identical submission)" in blocks
    assert "REP=" not in blocks
    assert "ncu-details.txt" in blocks


def test_a_failed_capture_gives_a_reason_and_drops_out_of_the_directive(capsys):
    rc, blocks, verdict = _run(capsys, "--dry-run", "nsys-fail")
    assert rc == 0, "a capture must never change whether a kernel scored"
    assert "--- Nsight Systems profile unavailable: Modal CLI not found" in blocks
    assert verdict["correct"] is True


def test_a_broken_kernel_exits_nonzero_and_never_reaches_a_capture(capsys):
    """Profilers fire at test-pass, so a failing kernel never spends a profiler slot.

    ``blocks == ""`` only proves nothing printed; that no profiler *ran* is a claim
    about the detail dict, so this one reads it straight off ``score``.
    """
    rc, blocks, verdict = _run(capsys, "--dry-run", "test-fail")
    assert rc == 1
    assert verdict["correct"] is False and verdict["metric"] is None
    assert blocks == ""
    _, _, _, detail = _scored("test-fail")
    assert "profile" not in detail and "nsys" not in detail


def test_the_real_flags_apply_to_the_replay(capsys):
    """--test-only and --no-profile are not re-implemented here; they reach ``score``
    unchanged, which is what makes the replay a demonstration rather than a mock-up."""
    _, blocks, verdict = _run(capsys, "--dry-run", "--test-only")
    assert blocks == "" and verdict["metric"] is None and verdict["correct"] is True
    _, blocks, verdict = _run(capsys, "--dry-run", "--no-profile")
    assert blocks == "" and verdict["metric"] == pytest.approx(TARGET_US)


def test_brief_drops_the_bench_record_and_nothing_else(capsys):
    """`bench` is the forensic archive -- carried opaquely into result.json and the
    journal, with no consumer reading a field of it -- and 98% of the line. An agent
    reads the verdict; it should not have to read the archive.

    Tested on the real score path, since that is where the flag does its work -- the
    replay is unconditionally brief and could not tell the two apart.
    """

    class Args:
        test_only = False
        profile = None
        brief = False

    def emit(brief):
        Args.brief = brief
        with dryrun._replayed("ok"):
            popcorn.score_command(load_problem(PROBLEM), Args())
        lines = capsys.readouterr().out.splitlines()
        return "\n".join(lines[:-1]), json.loads(lines[-1])

    blocks, full = emit(False)
    brief_blocks, brief = emit(True)
    assert set(brief) == {"unit", "correct", "metric", "error"}
    assert all(brief[k] == full[k] for k in brief)
    assert "bench" not in brief and "bench" in full
    # the banners are the signal, so --brief must not touch them
    assert brief_blocks == blocks
    assert len(json.dumps(brief)) < len(json.dumps(full)) / 20


def test_the_replay_is_exactly_what_an_agent_runs(capsys):
    """A replay that faithfully rendered output no candidate ever sees would be worse
    than no replay at all -- so the bare command matches `_score_cmd_str`'s flags, and
    there is deliberately no flag to opt out of that."""
    from kernelthing.config import Config
    from kernelthing.orchestrator import Orchestrator

    agent_flags = shlex.split(Orchestrator(load_problem(PROBLEM), Config())._score_cmd_str())[3:]
    bare = _run(capsys, "--dry-run")
    assert bare == _run(capsys, *agent_flags, "--dry-run")
    assert "bench" not in bare[2]


def test_a_failure_still_says_why_under_brief(capsys):
    """The reason a kernel failed lives in `error`, never in `bench` -- which is what
    makes the record safe to drop for the caller that has to read the output."""
    rc, _, brief = _run(capsys, "--dry-run", "test-fail")
    assert rc == 1
    assert brief["correct"] is False
    assert "17/17 shapes failed" in brief["error"]


def test_the_scoring_command_handed_to_agents_is_one_the_cli_accepts(capsys):
    """`_score_cmd_str` is baked into every candidate's prompt, and the prompt appends
    to it (`{{SCORE_CMD}} --test-only`). A flag that stopped parsing would fail the
    same way a dangling path does -- silently, on every candidate, for a whole run --
    which is why `_preflight` executes the command at setup rather than stat-ing it."""
    from kernelthing.config import Config
    from kernelthing.orchestrator import Orchestrator

    cmd = shlex.split(Orchestrator(load_problem(PROBLEM), Config())._score_cmd_str())
    assert cmd[1] == "score" and cmd[2] == "."
    flags = cmd[3:]
    assert "--brief" in flags
    # every flag in that string parses, both bare and with what the prompt appends
    assert _run(capsys, *flags, "--dry-run")[0] == 0
    assert _run(capsys, *flags, "--test-only", "--dry-run")[0] == 0


def test_an_unknown_scenario_is_rejected_by_the_parser(capsys):
    with pytest.raises(SystemExit) as e:
        cli.score_command([str(PROBLEM), "--dry-run", "no-such-scenario"])
    assert e.value.code == 2
    assert set(dryrun.SCENARIOS) == {"ok", "cached", "nsys-fail", "test-fail"}
