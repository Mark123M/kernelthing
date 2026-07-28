"""``kernelthing score --dry-run``: print a score's output without submitting anything.

What a candidate actually reads after a full score -- the two capture banners, the
analysis directive, the verdict JSON -- is the loop's tightest feedback channel and the
one thing that is invisible until a real submission has already been paid for. This
replays it from the captures in ``tests/fixtures/popcorn/``, so the shape an agent sees
can be inspected, diffed and reviewed for free.

Three properties make it worth having rather than a scratch script:

- **It drives the real ``popcorn.score_command``.** Only the four network boundaries are
  replaced (``submit``, the exact-shape check, and the two profilers); block ordering, wording, the
  ``format_*`` fold and the verdict dict all come from production code, so a change to
  any of them shows up here without this file being touched. A hand-rolled sample would
  drift the day someone edits a banner.
- **stdout is byte-exact; the "this is a replay" banner goes to stderr.** That is the
  whole point -- a marker on stdout would defeat the comparison the command exists for,
  and stderr is this CLI's logger anyway. Nothing downstream ever passes ``--dry-run``,
  so the fake verdict has no path into a journal.
- **It writes nothing and reads no network.** ``submit`` is replaced above the
  submission cache, so a dry run neither loads nor stores one, and the profilers are
  replaced above the code that creates ``profile/latest/``.

What is still real: ``resolve_config`` and ``_precheck`` run against the problem's own
``submission.py``, so a dry run does catch a config error or a submission carrying the
substring the evaluator rejects -- before it costs a round trip.

The wall-clock figures are nominal (the medians recorded in CLAUDE.md), not measurements
of this invocation; a replay takes milliseconds.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from . import popcorn
from .config import REPO_ROOT
from .problem import Problem

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "popcorn"

# Every scenario is reachable from the captures already in the fixture dir. They exist
# because the degraded blocks are the ones nobody sees until they happen in a run: a
# cache hit does not re-download the 80MB report, and a capture that failed must leave
# the agent a reason rather than a path to a file that is not there.
SCENARIOS = ("ok", "cached", "nsys-fail", "test-fail")

# The four functions in popcorn.py that reach the network, in the order _replayed
# builds their stand-ins. Everything else on the score path is the real thing.
_BOUNDARIES = (
    "submit",
    "deadlock_check_submission",
    "profile_submission",
    "profile_nsys_submission",
)

# Nominal, from the figures in CLAUDE.md: test ~11s, benchmark ~275s over 15 shapes,
# hosted ncu 245-270s. They are printed nowhere on stdout -- only the verdict's
# ``bench.wall_s`` -- but a plausible number there keeps the sample readable.
_WALL = {
    "test": 11.0,
    "benchmark": 275.0,
    "leaderboard": 275.0,
    "deadlock": 18.0,
    "ncu": 262.0,
    "nsys": 98.0,
}


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def _api_result(name: str, mode: str) -> dict[str, Any] | None:
    """The ``runs[].result`` dict of a verbatim ``GET /user/submissions/<id>`` capture."""
    run = popcorn.select_run(json.loads(_fixture(f"{name}.json")), mode)
    return run["result"] if run else None


def _replay_submit(scenario: str) -> Callable[..., popcorn.Submission]:
    """Stand in for ``popcorn.submit``: one canned submission per mode.

    Test and benchmark replay the API channel (``raw`` set), which is the one a run
    normally takes. A ranked run has no API capture on file, so it replays the text
    scraper instead -- which is also the only way to see that fallback's output.
    """
    failing = scenario == "test-fail"

    def fake(cfg: Any, sub_path: Path, mode: str, digest: str) -> popcorn.Submission:
        if mode == popcorn.MODE_TEST:
            name = "test-fail" if failing else "test-pass"
            raw = _api_result(f"api-{name}", "test")
            text = _fixture(f"{name}.txt")
        elif mode == popcorn.MODE_BENCHMARK:
            raw = _api_result("api-benchmark", "benchmark")
            text = _fixture("benchmark.txt")
        elif mode == popcorn.MODE_LEADERBOARD:
            raw = None
            text = _fixture("leaderboard.txt")
        else:
            raise popcorn.PopcornError(f"no fixture for submission mode {mode}")
        return popcorn.Submission(
            mode=mode, text=text, returncode=0, wall_s=_WALL.get(mode, 0.0), raw=raw
        )

    return fake


def _replay_ncu(scenario: str) -> Callable[..., popcorn.Profile]:
    def fake(cfg: Any, sub_path: Path, dest: Path, digest: str) -> popcorn.Profile:
        if scenario == "cached":
            # A cache hit restores the text views and nothing else: the report itself is
            # 80MB and deliberately not re-downloaded, so there is no path to query.
            return popcorn.Profile(
                ok=True,
                details_path=str(dest / "ncu-details.txt"),
                digest_path=str(dest / "digest.txt"),
                cached=True,
                wall_s=0.1,
            )
        return popcorn.Profile(
            ok=True,
            details_path=str(dest / "ncu-details.txt"),
            report_path=str(dest / "profile.ncu-rep"),
            digest_path=str(dest / "digest.txt"),
            wall_s=_WALL["ncu"],
        )

    return fake


def _replay_deadlock(scenario: str) -> Callable[..., popcorn.DeadlockCheck]:
    def fake(cfg: Any, sub_path: Path) -> popcorn.DeadlockCheck:
        batch, n, seed = popcorn._cholesky_modal_args(cfg)
        return popcorn.DeadlockCheck(
            ok=True,
            status="passed",
            batch=batch,
            n=n,
            seed=seed,
            wall_s=_WALL["deadlock"],
        )

    return fake


def _replay_nsys(scenario: str) -> Callable[..., popcorn.NsysProfile]:
    def fake(cfg: Any, sub_path: Path, dest: Path, digest: str) -> popcorn.NsysProfile:
        if scenario == "nsys-fail":
            # Verbatim from profile_nsys_submission's own early return -- an invented
            # message would show a banner no run can actually produce.
            return popcorn.NsysProfile(
                error="Modal CLI not found on PATH (set KERNELTHING_MODAL_BIN)"
            )
        return popcorn.NsysProfile(
            ok=True,
            report_path=str(dest / "profile.nsys-rep"),
            sqlite_path=str(dest / "profile.sqlite"),
            stats_path=str(dest / "stats.txt"),
            wall_s=_WALL["nsys"],
        )

    return fake


@contextlib.contextmanager
def _replayed(scenario: str) -> Iterator[None]:
    """Swap the three network boundaries for the duration of one score, then restore.

    Restoring matters more than it looks: the tests exercise this in-process, and a
    leaked patch would make every later popcorn test score against a fixture.
    """
    real = {name: getattr(popcorn, name) for name in _BOUNDARIES}
    fakes = (
        _replay_submit(scenario),
        _replay_deadlock(scenario),
        _replay_ncu(scenario),
        _replay_nsys(scenario),
    )
    for name, fake in zip(_BOUNDARIES, fakes, strict=True):
        setattr(popcorn, name, fake)
    try:
        yield
    finally:
        for name, fn in real.items():
            setattr(popcorn, name, fn)


def score_command(problem: Problem, args: Any) -> int:
    """``kernelthing score --dry-run [SCENARIO]``: the replay, with the real formatters.

    Returns what the real command would: 0 when the replayed verdict is correct. The
    honoured flags are the real ones -- ``--test-only`` and ``--no-profile`` change the
    output here exactly as they would against the service.

    **A replay is always brief**, with no flag to opt out. The command exists to show
    what an agent reads, and agents run ``--brief``
    (``Orchestrator._score_cmd_str``); rendering the full verdict here would faithfully
    reproduce output no candidate ever sees, which is the one thing this must not do.
    So ``--dry-run`` has exactly one meaning. A real score is unaffected -- its default
    stays full, so ``_cli_score`` still archives everything.

    Nothing needs an escape hatch for the ``bench`` record: a test that wants it takes
    ``popcorn.score`` under ``_replayed`` directly, which is a better check anyway, and
    every archived run already has 20-odd ``result.json`` files showing its shape.
    """
    scenario = str(getattr(args, "dry_run", None) or "ok")
    if scenario not in SCENARIOS:
        print(f"error: unknown dry-run scenario '{scenario}'", file=sys.stderr)
        return 2
    if not FIXTURES.is_dir():
        print(
            f"error: --dry-run needs the popcorn captures at {FIXTURES}, which ship with "
            "the source tree (an editable install has them; a wheel does not)",
            file=sys.stderr,
        )
        return 2
    args.brief = True
    print(
        f"[dry-run:{scenario}] replaying {FIXTURES}; nothing is submitted, downloaded or "
        "written. The paths below are where a real score would land its captures. "
        "stdout is exactly what an agent reads.",
        file=sys.stderr,
    )
    with _replayed(scenario):
        return popcorn.score_command(problem, args)
