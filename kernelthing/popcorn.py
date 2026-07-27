"""Popcorn (gpu-mode) remote scorer -- the only benchmark backend.

Kernels cannot be scored on the box the loop runs on. The gpu-mode competitions grade on
hardware (B200) reachable only through the hosted popcorn service, so no local measurement
is authoritative -- a laptop RTX card tells you nothing about the ranking. A problem sets
``bench.backend = "popcorn"`` in ``problem.json`` and ``kernelthing score`` routes here;
there is no local benchmark engine.

The backend is deliberately confined to this module. ``cli.score_command`` routes here
right after ``load_problem``, and the verdict printed is the
``{correct, metric, error, unit, bench}`` JSON line ``Orchestrator._cli_score`` parses --
so the search, the journal, the run dir and the web UI never learn how a number was
produced. Nothing local ever opens libcuda on this path.

Scoring is two remote submissions, in order:

  1. ``popcorn submit --mode test``      -- correctness over the public test shapes.
  2. ``popcorn submit --mode benchmark`` -- timings; only reached if (1) passed.

...plus two profilers fired *concurrently with* (2) the moment (1) passes: hosted
Nsight Compute via ``--profile-brev`` and a minimal Modal/B200 Nsight Systems
capture. See ``profile_submission`` and ``profile_nsys_submission``. Overlapping is
the whole trick -- the profile jobs are the long pole and the benchmark is ~172s,
so running them together costs the difference rather than the sum.

Splitting them is what makes a broken kernel cheap: ``test`` runs small shapes and
returns fast, so most failures never pay for a benchmark. ``benchmark`` re-checks every
shape anyway (``run_benchmarking`` passes ``recheck=True``), so a kernel that only breaks
at benchmark scale is still caught -- ``correct`` is the conjunction of both.

**Numbers come from the API, not from the CLI's printed text.** The CLI formats results for
humans -- ``⏱ 75.1 ± 0.07 µs`` -- keeping three significant figures, and that formatted row
is the only thing ``--output`` writes. But the service keeps the real measurement and the
CLI already fetches it: ``GET /user/submissions/<id>`` (header ``X-Popcorn-Cli-Id``) returns
``runs[].result``, a flat dict of ``benchmark.N.{spec,mean,err,best,worst,std,runs}`` in
nanoseconds, plus ``benchmark-count`` and an authoritative ``check``. ``get_user_submission``
in the CLI parses that dict and then drops it on the floor while formatting. So: submit with
the CLI (it owns auth, upload and polling), then make one authenticated GET for the id it
reports.

Concretely that is ``75141.44521620538`` ns rather than ``75.1 µs``, and ``benchmark-count``
makes a shape-index mix-up impossible instead of merely detectable.

``parse_benchmark_output`` / ``parse_test_output`` scrape the formatted text and survive as a
**fallback** for when the API is unreachable (they are validated against 111 real captures).
Falling back keeps a run going at ~0.5% worst-case quantisation instead of failing outright;
``bench.source`` in the verdict records which channel produced the number, so a silent
downgrade is visible afterwards.

Repeat submissions are cached on the submission file's sha256 (see ``_cache_path``). This
is not a micro-optimisation: an agent self-tests with ``kernelthing score .`` and the
orchestrator then scores the commit it produced, which is usually the identical file, so
the cache roughly halves the remote traffic of a run. Cached scores keep a benchmark's
noise frozen rather than re-rolling it -- ``bench.cached`` records when that happened.
"""

from __future__ import annotations

import ast
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .problem import Problem

# Wall-clock caps, split because the two runs are wildly different sizes. Measured on the
# live cholesky board: `test` took 11s, `benchmark` 275s across 15 shapes. The server caps
# a benchmark shape at 180s, so a pathologically slow kernel can push the timing run
# towards 2700s -- hence the generous default, which mostly exists so such a kernel fails
# as "too slow" rather than hanging.
#
# Both must fit inside Orchestrator._cli_score's 1800s kill of the whole `kernelthing
# score` process: 300 + 1200 = 1500 leaves margin. Raise DEFAULT_TIMEOUT_S past 1500 and
# the orchestrator starts killing scores mid-poll, losing the submission's result. The
# popcorn CLI's own poll timeout is an hour, so ours is always the binding one.
TEST_TIMEOUT_S = 300
DEFAULT_TIMEOUT_S = 1200

# Hosted Nsight Compute endpoint the CLI's --profile-brev talks to (popcorn-cli
# docs/profiling.md). Only used to fill in the agent's prompt; the CLI reads it from the
# environment, and an operator export of either name wins over this default.
DEFAULT_BREV_PROFILER_URL = "https://http--brev-profiler-proxy--dxfjds728w5v.code.run"

# Service API. The CLI hardcodes this same default in main.rs when POPCORN_API_URL is
# unset, and authenticates every request with the cli_id from ~/.popcorn.yaml.
DEFAULT_API_URL = "https://site--bot--dxfjds728w5v.code.run"
API_TIMEOUT_S = 30
CLI_ID_HEADER = "X-Popcorn-Cli-Id"

# Where the numbers came from, recorded in the verdict so a silent downgrade is visible.
SOURCE_API = "api"
SOURCE_TEXT = "cli-text"

# Submission modes we drive.
MODE_TEST = "test"
MODE_BENCHMARK = "benchmark"
MODE_LEADERBOARD = "leaderboard"
# Not a submission mode: --profile-brev is its own service (see profile_submission).
MODE_PROFILE = "profile"
MODE_NSYS = "nsys"

# The profile runs concurrently with the timing submission, so this cap does not add to
# the budget -- the pair costs max(timeout_s, PROFILE_TIMEOUT_S), and holding them equal
# keeps the total at the same 300 + 1200 that already fits _cli_score's 1800s kill.
#
# Sized off the 2026-07-24 run: a job is 245-270s with an empty queue and grows with
# depth (515s observed at depth 2). The service itself never killed a job in that run;
# all four losses were the *client* giving up early, and it gives up after paying the
# full wait, because the CLI downloads artifacts only once the job reports `succeeded`.
PROFILE_TIMEOUT_S = 1200

# Where the capture lands inside the worktree. Every problem's .gitignore already covers
# `profile/` (test_harness_deps asserts it), so the artifacts cannot reach a commit. It
# cannot live in the submission cache instead: the sandbox ro-binds everything except the
# worktree, so an agent's own `kernelthing score` could not write there.
PROFILE_DIR = "profile"
PROFILE_SUBDIR = "latest"
NSYS_SUBDIR = "nsys"

# Which verb answers which question -- never the capture text itself. This list used to be
# a ~12KB digest inlined in every score, which put a fixed handful of pre-chosen numbers in
# front of the agent and invited it to stop there; one report holds ~22.5k metrics and ~108
# rule findings that no digest can reach. Hints leave the choice of question to the agent.
# `<N>` placeholders are deliberate: the first verb in each list hands out the row ids for
# the rest.
#
# Rendered into the *candidate prompt* (`kernel-tools-veloq.md`, once per member), not into
# each score. It is static -- same 30 lines regardless of what was measured -- and the
# 2026-07-24 run took a median of 5 full scores per member (12 at the tail), so printing it
# per score meant ~22KB of byte-identical text resident in a member's context, up to ~53KB,
# versus 4.4KB once in a cached prompt prefix. What a score prints instead is the *state*
# no constant can carry: whether each capture landed, and where. Each capture block names
# its own `veloq <group>`, which is what sends the agent back here -- see `_capture_block`.
VELOQ_NCU_HINTS = (
    ("launches $REP --limit 20", "every captured launch + its row id -- START HERE"),
    ("summary $REP", "totals: launch / metric / rule / disasm counts, NCU version"),
    ("inspect $REP --row-id launch:<N>", "one launch: all metrics + rule findings"),
    ("metrics $REP --counter 'sm__throughput*,dram__throughput*'", "one counter family, all launches"),
    ("warp-stalls $REP --row-id launch:<N> --by reason", "why warps stalled (--by line|sass)"),
    ("source-metrics $REP --row-id launch:<N> --counter '*bank_conflict*'", "per-source-line attribution"),
    ("sources $REP", "which cubin each launch ran out of; has_disasm flag"),
    ("disasm $REP --row-id launch:<N>", "SASS (+ PTX when the cubin embeds it)"),
    ("ranges $REP", "range workloads, if captured under --replay-mode range"),
    ("graphs $REP", "CUDA-graph workloads, if captured under --graph-profiling graph"),
    ("schema <verb>", "JSON schema of any verb's response; reads no report"),
)
VELOQ_NSYS_HINTS = (
    ("stats $NSYS --type kernel --limit 20", "hottest kernels by total GPU time -- START HERE"),
    ("summary $NSYS", "what the trace contains: tables, span, capability flags"),
    ("stats $NSYS --type runtime --collapse-versioned --limit 20", "CPU-side CUDA API cost"),
    ("stats $NSYS --type memcpy --sort gbps:desc", "transfer bandwidth, H2D vs D2H"),
    ("search $NSYS --type kernel --limit 20", "filter events -> row ids for inspect/correlate"),
    ("inspect $NSYS kernel:<N>", "full detail for one event (row id is positional here)"),
    ("correlate $NSYS kernel:<N>", "the CPU launch call behind a GPU kernel, and back"),
    ("gaps $NSYS --scope device --limit 20", "idle bubbles (--scope stream|trace)"),
    ("concurrency $NSYS", "GPU overlap: union vs sum busy time, compute/copy split"),
    ("timeline $NSYS --interval 1ms", "bucketed GPU activity over time"),
    ("slices $NSYS", "per-NVTX-range CPU bounds + attributed GPU work"),
    ("graph-replays $NSYS", "CUDA graph replays and the nodes dominating them"),
    ("metrics $NSYS --type gpu", "PM-sampling series (--type nic|cpu-sampling|cpu-sched)"),
    ("hardware $NSYS", "profiled CPU / GPU / NIC inventory"),
    ("ncu-command $NSYS kernel:<N>", "the Nsight Compute rerun command for one kernel"),
    ("correlation-stats $NSYS", "per-kind row stats of the CPU<->GPU correlation index"),
    ("prep $NSYS", "warm the parquet/sidecar caches up front (--status to check)"),
    ("viz timeline $NSYS", "export a bounded timeline window as an SVG"),
    ("schema <verb>", "JSON schema of any verb's response; reads no trace"),
)

# ncu-details.txt sections worth quoting back in full. The rule findings (OPT/INF/WRN)
# are always kept and are what actually names a bottleneck; these three carry the numbers
# an agent needs to check one. The rest of a --set full dump (roofline, PM sampling,
# instruction mix, memory tables) is ~14 sections of context for a question nobody asked.
PROFILE_KEEP_SECTIONS = (
    "GPU Speed Of Light Throughput",
    "Launch Statistics",
    "Occupancy",
    "Warp State Statistics",
)
# `[73] python3.12@127.0.0.1` opens a launch; the kernel signature is the line after it.
PROFILE_LAUNCH_RE = re.compile(r"^\[\d+\]\s")
PROFILE_SECTION_RE = re.compile(r"^\s{4}Section:\s*(?P<name>.+?)\s*$")
PROFILE_RULE_RE = re.compile(r"^\s{4}(?:OPT|INF|WRN)\s")

# How the parsed per-shape timings become a single number.
#   shape       -- one benchmark index's mean (the default; a per-shape specialisation
#                  only moves its own entry, and a whole-board geomean would bury it)
#   geomean     -- geometric mean over every shape, mirroring the official ranking
#   leaderboard -- ask the server for the ranked geomean instead of timing locally
METRIC_MODES = ("shape", "geomean", "leaderboard")

# Value+unit as rendered by the CLI's format_time (ns / us / ms, auto-scaled by
# magnitude). Both the micro sign and a Greek mu are accepted -- they render identically
# and which one appears is not worth depending on.
_NUM = r"[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?"
_UNIT = r"(?:ns|µs|μs|us|ms)"
TIME_RE = re.compile(rf"({_NUM})\s*({_UNIT})")
# `<mean> ± <err> <unit>` or `<mean> <unit>`: format_time prints ONE unit, after the
# error term, so the mean's scale has to be read off the end of the line.
MEAN_ERR_RE = re.compile(rf"({_NUM})(?:\s*±\s*({_NUM}))?\s*({_UNIT})")
_TO_US = {"ns": 1e-3, "µs": 1.0, "μs": 1.0, "us": 1.0, "ms": 1e3}

# A benchmark/test spec line, matching the grammar the eval driver parses test cases
# with: `key: value` parts joined by `;`. Anchoring on this is what lets the parsers tell
# a spec apart from a line of a failure's free-form error text.
_PART = r"[a-zA-Z_][a-zA-Z0-9_]*:\s*(?:[a-zA-Z_][a-zA-Z0-9_]*|[+-]?[0-9]+)"
SPEC_RE = re.compile(rf"\s*{_PART}\s*(?:;\s*{_PART}\s*)*")
# Two-field form. A single `Something: value` also matches SPEC_RE, and Python
# exception lines ("RuntimeError: foo") do too -- so inside a failed shape's error body,
# where such lines are expected, only the unambiguous multi-field form opens a new shape.
SPEC_MULTI_RE = re.compile(rf"\s*{_PART}\s*(?:;\s*{_PART}\s*)+")

# `❌ <spec> failed testing:` opens a failed benchmark row; the error body follows.
BENCH_FAIL_RE = re.compile(r"^❌\s*(?P<spec>.*?)\s+failed testing:\s*$")
# Test rows: `✅ <spec>` / `❌ <spec>` / `? <spec> (<status>)`.
TEST_ROW_RE = re.compile(r"^(?P<mark>✅|❌|\?)\s+(?P<spec>.*?)\s*$")
# `Geomean score (public) on B200: 0.0023 s` -- full precision, unlike the ⏱ rows.
GEOMEAN_RE = re.compile(
    r"Geomean score\s*\((?P<scope>public|secret)\)\s*on\s*(?P<runner>[^:]+):\s*"
    rf"(?P<value>{_NUM})\s*s"
)

_STOPWATCH = "⏱"  # mean ± err
_FAST = "⚡"  # best
_SLOW = "\U0001f40c"  # worst


class PopcornError(RuntimeError):
    """A popcorn invocation could not be turned into a verdict."""


@dataclass
class PopcornConfig:
    """Resolved ``problem.bench.popcorn`` block."""

    leaderboard: str
    gpu: str = "B200"
    submission_file: str = "submission.py"
    benchmark_index: int = 0
    # Expected spec of benchmark_index, e.g. "n: 128; cond: 2; seed: 41128; batch: 256".
    # Optional but strongly recommended: it is the only thing standing between a parser
    # that drops a row and a run that silently optimises a different shape. Compared
    # field-wise, so upstream reordering is not a false alarm.
    benchmark_spec: str = ""
    metric_mode: str = "shape"
    timeout_s: int = DEFAULT_TIMEOUT_S
    reject_substrings: list[str] = field(default_factory=list)
    cache: bool = True
    bin: str = ""
    # Capture an Nsight Compute profile alongside every full score. On by default: the
    # point is that an agent never has to decide to profile, so it never reasons about a
    # bottleneck it did not measure.
    profile: bool = True
    # Capture an Nsight Systems timeline on Modal/B200 alongside the same full score.
    # Separate from ``profile`` so operators can keep the existing ncu path without
    # Modal, or vice versa. ``kernelthing score --no-profile`` still disables both.
    nsys: bool = True

    @property
    def timing_mode(self) -> str:
        """The submission mode that produces the metric."""
        return MODE_LEADERBOARD if self.metric_mode == "leaderboard" else MODE_BENCHMARK

    def timeout_for(self, mode: str) -> int:
        """Wall-clock cap for one submission. The test run is ~25x cheaper than timing."""
        return min(self.timeout_s, TEST_TIMEOUT_S) if mode == MODE_TEST else self.timeout_s


@dataclass
class Shape:
    """One benchmark entry as the CLI rendered it."""

    index: int
    spec: str
    status: str = "pass"
    mean_us: float | None = None
    err_us: float | None = None
    best_us: float | None = None
    worst_us: float | None = None
    error: str = ""

    def record(self) -> dict[str, Any]:
        d: dict[str, Any] = {"index": self.index, "spec": self.spec, "status": self.status}
        for key in ("mean_us", "err_us", "best_us", "worst_us"):
            value = getattr(self, key)
            if value is not None:
                d[key] = round(value, 4)
        if self.error:
            d["error"] = self.error[:1000]
        return d


@dataclass
class TestReport:
    """The ✅/❌ rows of a ``--mode test`` run."""

    passed: int = 0
    failed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {"passed": self.passed, "failed": self.failed, "failures": self.failures}


@dataclass
class Submission:
    """One completed ``popcorn submit`` invocation.

    ``raw`` is the service's own result dict for the non-secret run of this mode, fetched
    from the API. When it is present the numbers come from there; ``text`` is only parsed
    when it is not.
    """

    mode: str
    text: str
    returncode: int
    wall_s: float
    cached: bool = False
    raw: dict[str, Any] | None = None
    score: float | None = None  # the run's own score field (ranked geomean, seconds)

    @property
    def submission_id(self) -> int | None:
        summary = parse_summary_json(self.text)
        sid = summary.get("submission_id")
        return int(sid) if isinstance(sid, (int, float)) else None

    @property
    def source(self) -> str:
        return SOURCE_API if self.raw else SOURCE_TEXT


# --- configuration ----------------------------------------------------------


def resolve_config(problem: Problem) -> PopcornConfig:
    """Extract and validate the popcorn block from a problem manifest."""
    raw = dict((problem.bench or {}).get("popcorn") or {})
    leaderboard = str(raw.get("leaderboard", "")).strip()
    if not leaderboard:
        raise PopcornError("problem.bench.popcorn.leaderboard is required")
    metric_mode = str(raw.get("metric_mode", "shape"))
    if metric_mode not in METRIC_MODES:
        raise PopcornError(
            f"unknown metric_mode '{metric_mode}'; expected one of {', '.join(METRIC_MODES)}"
        )
    return PopcornConfig(
        leaderboard=leaderboard,
        gpu=str(raw.get("gpu", "B200")),
        submission_file=str(raw.get("submission_file", "submission.py")),
        benchmark_index=int(raw.get("benchmark_index", 0)),
        benchmark_spec=str(raw.get("benchmark_spec", "")),
        metric_mode=metric_mode,
        timeout_s=int(raw.get("timeout_s", DEFAULT_TIMEOUT_S)),
        reject_substrings=[str(s) for s in raw.get("reject_substrings", [])],
        cache=bool(raw.get("cache", True)),
        bin=str(raw.get("bin", "")),
        profile=bool(raw.get("profile", True)),
        nsys=bool(raw.get("nsys", True)),
    )


def config_or_none(problem: Problem) -> PopcornConfig | None:
    """``resolve_config`` for callers that only want to know 'is this a popcorn problem'.

    Returns ``None`` for a non-popcorn problem *and* for a malformed popcorn block --
    prompt assembly must not raise, and a bad manifest surfaces properly at score time.
    """
    if (problem.bench or {}).get("backend") != "popcorn":
        return None
    try:
        return resolve_config(problem)
    except (PopcornError, TypeError, ValueError):
        return None


def popcorn_bin(cfg: PopcornConfig | None = None) -> str:
    """Path to the popcorn CLI, or '' when it cannot be found.

    ``bench.popcorn.bin`` wins (the problem declared it), then
    ``KERNELTHING_POPCORN_BIN``, then PATH. The env override exists because the CLI
    usually lives in ``~/.local/bin``, which a sandboxed agent's PATH may not carry.
    """
    if cfg and cfg.bin:
        return cfg.bin
    override = os.environ.get("KERNELTHING_POPCORN_BIN", "").strip()
    if override:
        return override
    return shutil.which("popcorn") or shutil.which("popcorn-cli") or ""


def modal_bin() -> str:
    """Path to the Modal CLI, or '' when nsys profiling cannot be started."""
    override = os.environ.get("KERNELTHING_MODAL_BIN", "").strip()
    if override:
        return override
    return shutil.which("modal") or ""


def api_base() -> str:
    """Service base URL. Mirrors the CLI: env wins, else its own hardcoded default."""
    return (os.environ.get("POPCORN_API_URL", "").strip() or DEFAULT_API_URL).rstrip("/")


def cli_id() -> str:
    """The client id the CLI authenticates with, or '' if not registered.

    ``~/.popcorn.yaml`` holds a single ``cli_id: <uuid>`` key. Parsed by hand rather than
    with PyYAML so fetching a result never depends on an optional import being present --
    a missing id degrades to text scraping, it must not raise.
    """
    override = os.environ.get("POPCORN_CLI_ID", "").strip()
    if override:
        return override
    try:
        text = (Path.home() / ".popcorn.yaml").read_text(encoding="utf-8")
    except OSError:
        return ""
    m = re.search(r"^\s*cli_id\s*:\s*[\"']?([^\"'\s]+)", text, re.MULTILINE)
    return m.group(1) if m else ""


def fetch_submission(submission_id: int, *, timeout: int = API_TIMEOUT_S) -> dict[str, Any]:
    """``GET /user/submissions/<id>`` -- the full record, including ``runs[].result``.

    Raises ``PopcornError`` on any failure; callers treat that as "fall back to text".
    """
    ident = cli_id()
    if not ident:
        raise PopcornError("no popcorn cli_id (run `popcorn register`)")
    req = urllib.request.Request(
        f"{api_base()}/user/submissions/{submission_id}",
        headers={CLI_ID_HEADER: ident, "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
        raise PopcornError(f"popcorn API fetch failed: {e}") from e
    if not isinstance(payload, dict):
        raise PopcornError("popcorn API returned a non-object")
    return payload


def select_run(payload: dict[str, Any], mode: str) -> dict[str, Any] | None:
    """The public run for *mode*.

    A ranked submission carries six runs -- test/benchmark/leaderboard x public/secret --
    so both the mode and the secret flag have to be matched. The secret run is measured on
    a different seed and is never the number to report.
    """
    for run in payload.get("runs") or []:
        if not isinstance(run, dict):
            continue
        if str(run.get("mode", "")).lower() == mode.lower() and not run.get("secret"):
            return run
    return None


def _num(value: Any) -> float | None:
    """Result-dict values arrive as strings; anything unparseable is simply absent."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def shapes_from_result(result: dict[str, Any]) -> list[Shape]:
    """Per-shape timings from the service's own result dict, in nanoseconds.

    ``benchmark-count`` is authoritative, so a shape that failed (and therefore has no
    timings) still occupies its index -- which is what makes positional selection safe
    here in a way it can never be when scraping rows out of formatted text.
    """
    count = _num(result.get("benchmark-count"))
    if count is None:
        return []
    shapes: list[Shape] = []
    for i in range(int(count)):
        base = f"benchmark.{i}"
        status = str(result.get(f"{base}.status", "") or "")
        mean_ns = _num(result.get(f"{base}.mean"))
        shapes.append(
            Shape(
                index=i,
                spec=str(result.get(f"{base}.spec", "") or ""),
                status="fail" if status == "fail" or mean_ns is None else "pass",
                mean_us=None if mean_ns is None else mean_ns / 1000.0,
                err_us=(lambda v: None if v is None else v / 1000.0)(_num(result.get(f"{base}.err"))),
                best_us=(lambda v: None if v is None else v / 1000.0)(_num(result.get(f"{base}.best"))),
                worst_us=(lambda v: None if v is None else v / 1000.0)(
                    _num(result.get(f"{base}.worst"))
                ),
                error=str(result.get(f"{base}.error", "") or ""),
            )
        )
    return shapes


def test_report_from_result(result: dict[str, Any]) -> TestReport:
    """Test outcomes from the service's own result dict."""
    count = _num(result.get("test-count"))
    if count is None:
        return TestReport()
    report = TestReport()
    for i in range(int(count)):
        base = f"test.{i}"
        if str(result.get(f"{base}.status", "") or "") == "fail":
            report.failed += 1
            report.failures.append(
                {
                    "spec": str(result.get(f"{base}.spec", "") or ""),
                    "error": str(result.get(f"{base}.error", "") or ""),
                }
            )
        else:
            report.passed += 1
    return report


def brev_profiler_url() -> str:
    """Hosted profiler endpoint: an operator's export wins, else the documented default."""
    for name in ("POPCORN_BREV_PROFILER_URL", "BREV_PROFILER_URL"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    return DEFAULT_BREV_PROFILER_URL


# --- parsing ----------------------------------------------------------------


def parse_time_us(text: str) -> float | None:
    """Read the first ``<value> <unit>`` in *text* and return it in microseconds."""
    m = TIME_RE.search(text)
    if not m:
        return None
    try:
        value = float(m.group(1))
    except ValueError:
        return None
    return value * _TO_US[m.group(2)]


def _is_spec(line: str, *, strict: bool = False) -> bool:
    """Does *line* look like a benchmark/test spec? ``strict`` demands >1 field."""
    text = line.strip()
    pattern = SPEC_MULTI_RE if strict else SPEC_RE
    return bool(text) and pattern.fullmatch(text) is not None


def parse_mean_err_us(line: str) -> tuple[float | None, float | None]:
    """Read a ``⏱ <mean> ± <err> <unit>`` line, returning both in microseconds."""
    m = MEAN_ERR_RE.search(line)
    if not m:
        return None, None
    try:
        scale = _TO_US[m.group(3)]
        mean = float(m.group(1)) * scale
        err = float(m.group(2)) * scale if m.group(2) else None
    except (ValueError, KeyError):
        return None, None
    return mean, err


def split_summary(text: str) -> tuple[str, str]:
    """Split a popcorn result into ``(rows_text, summary_json_text)``.

    The CLI appends a pretty-printed JSON object after the formatted rows, so the
    summary always starts at a line that is exactly ``{``. Later candidates are tried
    first: a failure's error text may itself contain JSON.
    """
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].rstrip() != "{":
            continue
        candidate = "\n".join(lines[i:])
        try:
            json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return "\n".join(lines[:i]), candidate
    return text, ""


def parse_summary_json(text: str) -> dict[str, Any]:
    """The trailing summary object (submission_id, job status, runs), or ``{}``."""
    _, summary = split_summary(text)
    if not summary:
        return {}
    try:
        parsed = json.loads(summary)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_test_output(text: str) -> TestReport:
    """Parse the ✅/❌ rows of a ``--mode test`` result.

    A row's detail is introduced by ``> `` and runs until the next row, so failures keep
    their whole (multi-line) message -- that text is the most useful thing the next agent
    turn can be handed.
    """
    rows, _ = split_summary(text)
    report = TestReport()
    current: dict[str, str] | None = None
    detail: list[str] = []

    def flush() -> None:
        nonlocal current, detail
        if current is not None:
            current["error"] = "\n".join(detail).strip()
            report.failures.append(current)
        current, detail = None, []

    for line in rows.splitlines():
        m = TEST_ROW_RE.match(line)
        if m and _is_spec(m.group("spec")):
            flush()
            if m.group("mark") == "✅":
                report.passed += 1
            else:
                report.failed += 1
                current = {"spec": m.group("spec")}
            continue
        if current is None:
            continue
        stripped = line.lstrip()
        detail.append(stripped[2:] if stripped.startswith("> ") else line)
    flush()
    return report


def parse_benchmark_output(text: str) -> list[Shape]:
    """Parse the per-shape rows of a ``--mode benchmark`` result, in index order.

    Walks lines rather than splitting on blank lines: a failed shape's error body is
    free-form and may contain blank lines of its own, which would fragment a chunk-based
    split and silently drop later shapes.
    """
    rows, _ = split_summary(text)
    shapes: list[Shape] = []
    error_lines: list[str] = []

    def collecting() -> bool:
        return bool(shapes) and shapes[-1].status == "fail"

    def flush_error() -> None:
        nonlocal error_lines
        if shapes and error_lines:
            shapes[-1].error = "\n".join(error_lines).strip()
        error_lines = []

    for line in rows.splitlines():
        fail = BENCH_FAIL_RE.match(line.strip())
        if fail:
            flush_error()
            shapes.append(Shape(index=len(shapes), spec=fail.group("spec"), status="fail"))
            continue
        # While inside a failure's error body, demand the unambiguous multi-field spec
        # form: an exception line would otherwise open a phantom shape and shift every
        # later index -- which silently repoints the metric at the wrong benchmark.
        if _is_spec(line, strict=collecting()):
            flush_error()
            shapes.append(Shape(index=len(shapes), spec=line.strip()))
            continue
        if not shapes:
            continue
        if _STOPWATCH in line:
            shapes[-1].mean_us, shapes[-1].err_us = parse_mean_err_us(line)
            continue
        if _FAST in line or _SLOW in line:
            head, _, tail = line.partition(_SLOW)
            shapes[-1].best_us = parse_time_us(head)
            shapes[-1].worst_us = parse_time_us(tail) if tail else None
            continue
        if collecting():
            error_lines.append(line)
    flush_error()
    return shapes


def parse_leaderboard_score_s(text: str) -> float | None:
    """The public ranked geomean in seconds, or ``None``.

    Match on *scope*, never on position. The CLI prints one line per scored run in
    ``details.runs`` order, which the server does not stabilise: across 54 real captures
    26 printed the secret line first. A parser that took the first ``Geomean score`` line
    would therefore report the secret score -- measured on a different seed, and not what
    the public board ranks on -- roughly half the time, with nothing to flag it.
    """
    for m in GEOMEAN_RE.finditer(text):
        if m.group("scope") == "public":
            try:
                return float(m.group("value"))
            except ValueError:
                return None
    return None


def spec_fields(spec: str) -> dict[str, str]:
    """Split a ``key: value; key: value`` spec into a dict.

    Comparison is field-wise rather than string-wise because the service does not emit a
    consistent field order -- the benchmark rows print ``n; cond; seed; batch`` while the
    profiler's artifact slug uses ``batch-n-cond-seed``.
    """
    fields: dict[str, str] = {}
    for part in spec.split(";"):
        key, sep, value = part.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def geomean_us(shapes: list[Shape]) -> float | None:
    """Geometric mean of every timed shape, in microseconds."""
    values = [s.mean_us for s in shapes if s.mean_us and s.mean_us > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


# --- submission cache -------------------------------------------------------


def _cache_root() -> Path:
    override = os.environ.get("KERNELTHING_POPCORN_CACHE", "").strip()
    if override:
        return Path(override)
    return Path.home() / ".cache" / "kernelthing" / "popcorn-cache"


def _cache_path(cfg: PopcornConfig, digest: str, mode: str) -> Path:
    return _cache_root() / cfg.leaderboard / cfg.gpu / mode / f"{digest}.json"


def _cache_load(cfg: PopcornConfig, digest: str, mode: str) -> Submission | None:
    if not cfg.cache:
        return None
    path = _cache_path(cfg, digest, mode)
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    raw = d.get("raw")
    return Submission(
        mode=mode,
        text=str(d.get("text", "")),
        returncode=int(d.get("returncode", 0)),
        wall_s=float(d.get("wall_s", 0.0)),
        cached=True,
        raw=raw if isinstance(raw, dict) else None,
        score=_num(d.get("score")),
    )


def _cache_store(cfg: PopcornConfig, digest: str, sub: Submission) -> None:
    if not cfg.cache:
        return
    path = _cache_path(cfg, digest, sub.mode)
    payload = {
        "text": sub.text,
        "returncode": sub.returncode,
        "wall_s": sub.wall_s,
        "raw": sub.raw,
        "score": sub.score,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass  # a cache that cannot be written must never fail a score


# --- submitting -------------------------------------------------------------


def submit(cfg: PopcornConfig, sub_path: Path, mode: str, digest: str) -> Submission:
    """Run one ``popcorn submit`` and return its combined output.

    Leaderboard/GPU/mode are passed as explicit flags so they override any ``#!POPCORN``
    directives inside the submission file -- the agent edits that file, and the target it
    is graded against is not the agent's to choose.

    The CLI writes artifact files into its working directory, so it runs in a throwaway
    temp dir; ``--output`` captures the result text, and stdout is used as the fallback
    when the CLI exits before writing the file.
    """
    cached = _cache_load(cfg, digest, mode)
    if cached is not None:
        return cached
    exe = popcorn_bin(cfg)
    if not exe:
        raise PopcornError("popcorn CLI not found on PATH (set bench.popcorn.bin)")

    with tempfile.TemporaryDirectory(prefix="kt-popcorn-") as tmp:
        out_file = Path(tmp) / "result.txt"
        cmd = [
            exe, "submit", str(sub_path.resolve()),
            "--leaderboard", cfg.leaderboard,
            "--gpu", cfg.gpu,
            "--mode", mode,
            "--no-tui",
            "--output", str(out_file),
        ]
        budget = cfg.timeout_for(mode)
        t0 = time.time()
        try:
            r = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=budget)
        except subprocess.TimeoutExpired as e:
            raise PopcornError(f"popcorn --mode {mode} timed out after {budget}s") from e
        wall = round(time.time() - t0, 1)
        text = ""
        with contextlib.suppress(OSError):
            text = out_file.read_text(encoding="utf-8")
        if not text.strip():
            text = r.stdout or ""
        if not text.strip():
            raise PopcornError(
                f"popcorn --mode {mode} produced no result "
                f"(exit {r.returncode}): {(r.stderr or '').strip()[-500:]}"
            )
        sub = Submission(mode=mode, text=text, returncode=r.returncode, wall_s=wall)

    _attach_raw(sub)
    # Only memoise a submission the CLI completed. A nonzero exit means the job failed,
    # timed out, or the network broke -- caching that would pin a transient error to this
    # file's hash forever, so an unlucky blip would permanently mark a good kernel broken.
    # A *correctness* failure still exits 0 and is cached on purpose: it is deterministic.
    if r.returncode == 0:
        _cache_store(cfg, digest, sub)
    return sub


def _attach_raw(sub: Submission) -> None:
    """Best-effort upgrade from formatted text to the service's own numbers.

    Never raises: an unreachable API degrades the run to text scraping rather than
    failing it. ``Submission.source`` records which channel won.
    """
    sid = sub.submission_id
    if sid is None:
        return
    try:
        run = select_run(fetch_submission(sid), sub.mode)
    except PopcornError:
        return
    if not run:
        return
    result = run.get("result")
    if isinstance(result, dict):
        sub.raw = result
    sub.score = _num(run.get("score"))


# --- scoring ----------------------------------------------------------------


# --- profiling --------------------------------------------------------------


@dataclass
class Profile:
    """One hosted Nsight Compute capture, as far as the scorer is concerned."""

    ok: bool = False
    details_path: str = ""  # ncu-details.txt, the flat --set full dump
    report_path: str = ""  # profile.ncu-rep, for veloq; absent on a cache hit
    digest_path: str = ""  # digest.txt, what score_command echoes to the agent
    digest: str = ""  # the same text, in memory
    wall_s: float = 0.0
    cached: bool = False
    error: str = ""

    def record(self) -> dict[str, Any]:
        """The verdict's ``bench.profile`` entry: status and *paths*, never capture text.

        The JSON line is the seam between this module and everything downstream, so the
        capture must not cross it -- the orchestrator, the journal and the web UI would
        each grow a copy of a dump they have no use for, on every score. The agent gets
        the digest on stdout instead (see ``score_command``), and anything that wants the
        rest opens ``details_path``.
        """
        d: dict[str, Any] = {"ok": self.ok, "wall_s": round(self.wall_s, 1)}
        if self.cached:
            d["cached"] = True
        for key in ("details_path", "report_path", "digest_path"):
            if getattr(self, key):
                d[key] = getattr(self, key)
        if self.error:
            d["error"] = self.error[:500]
        return d


@dataclass
class NsysProfile:
    """One Modal/B200 Nsight Systems capture, as far as the scorer is concerned."""

    ok: bool = False
    report_path: str = ""  # profile.nsys-rep, for the GUI
    sqlite_path: str = ""  # profile.sqlite, for nsys stats / sqlite inspection
    stats_path: str = ""  # stats.txt, what score_command echoes in compact form
    wall_s: float = 0.0
    cached: bool = False
    error: str = ""

    def record(self) -> dict[str, Any]:
        """The verdict's ``bench.nsys`` entry: status and paths, never stats text."""
        d: dict[str, Any] = {"ok": self.ok, "wall_s": round(self.wall_s, 1)}
        if self.cached:
            d["cached"] = True
        for key in ("report_path", "sqlite_path", "stats_path"):
            if getattr(self, key):
                d[key] = getattr(self, key)
        if self.error:
            d["error"] = self.error[:500]
        return d


def brev_defuse(text: str, cfg: PopcornConfig) -> str:
    """Break any evaluator-rejected substring in profiler output before an agent sees it.

    Nsight prints a kernel header as ``..., Context 1, S-t-r-e-a-m 7, Device 0, ...``
    (hyphenated here so this source file does not carry the token either). The evaluator
    rejects a submission containing that substring *anywhere* -- inside a longer word,
    inside a comment -- so quoting a capture verbatim into an agent's context hands it a
    string that silently poisons any file it lands in. Hyphenating costs nothing to read
    and cannot be pasted into a rejection.
    """
    for bad in cfg.reject_substrings:
        if len(bad) < 2:
            continue
        text = re.sub(re.escape(bad), "-".join(bad), text, flags=re.IGNORECASE)
    return text


def profile_digest(details: str, cfg: PopcornConfig) -> str:
    """Reduce a ``--set full`` dump to the part that names a bottleneck.

    A real capture is ~21KB over 14 sections per launch. Almost all of the signal is in
    the profiler's own OPT/INF/WRN rule findings (16 of them in that capture) plus the
    handful of tables you would check them against; the rest is a wall of numbers that
    would be re-quoted on every score. Keeps launch headers, ``PROFILE_KEEP_SECTIONS``,
    and every rule finding with its continuation lines.

    Unrecognised input degrades to '' rather than raising -- a profiler that changes its
    output format must cost the run a digest, not a score.
    """
    keep: list[str] = []
    section = ""
    in_rule = False
    want_signature = False
    for raw_line in details.splitlines():
        line = raw_line.rstrip()
        if PROFILE_LAUNCH_RE.match(line):
            section, in_rule, want_signature = "", False, True
            keep += ["", line]
            continue
        if want_signature:
            # The mangled kernel name plus `(grid)x(block), Context, Stream, ..., CC`.
            # It is the only place the launch's shape appears, and the only place the
            # evaluator's rejected token appears -- brev_defuse handles that below.
            if line.strip():
                keep.append(line)
                want_signature = False
            continue
        m = PROFILE_SECTION_RE.match(line)
        if m:
            section, in_rule = m.group("name"), False
            if section in PROFILE_KEEP_SECTIONS:
                keep += ["", line]
            continue
        if PROFILE_RULE_RE.match(line):
            in_rule = True
            keep.append(line)
            continue
        # A rule's wrapped continuation is indented past its 4-space marker; a blank line
        # ends it. Section tables sit at exactly 4, so they cannot be mistaken for one.
        if in_rule:
            if line.strip() and line.startswith("     "):
                keep.append(line)
                continue
            in_rule = False
        if section in PROFILE_KEEP_SECTIONS and line.strip():
            keep.append(line)
    text = "\n".join(keep).strip()
    return brev_defuse(text, cfg) if text else ""


def _profile_cache_path(cfg: PopcornConfig, digest: str) -> Path:
    return _cache_root() / cfg.leaderboard / cfg.gpu / MODE_PROFILE / f"{digest}.txt"


def _profile_cache_load(cfg: PopcornConfig, digest: str) -> str:
    """The cached ``ncu-details.txt`` for this exact submission, or ''.

    Only the flat text is memoised, not the ``.ncu-rep`` -- the report is tens of MB and
    a cache of them would outgrow its own directory within a run. So a cache hit gives
    back the digest and the details file but no report for veloq to open; ``Profile.cached``
    records that, and re-profiling is one `--no-cache` away.
    """
    if not cfg.cache:
        return ""
    try:
        return _profile_cache_path(cfg, digest).read_text(encoding="utf-8")
    except OSError:
        return ""


def _profile_cache_store(cfg: PopcornConfig, digest: str, details: str) -> None:
    if not cfg.cache or not details.strip():
        return
    path = _profile_cache_path(cfg, digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(details, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        # Expected inside the sandbox: bwrap ro-binds everything but the worktree, so an
        # agent's own score cannot populate this. Reads still work; a miss just profiles.
        pass


def _nsys_cache_path(cfg: PopcornConfig, digest: str) -> Path:
    return _cache_root() / cfg.leaderboard / cfg.gpu / MODE_NSYS / f"{digest}.txt"


def _nsys_cache_load(cfg: PopcornConfig, digest: str) -> str:
    """The cached ``nsys stats`` text for this exact submission, or ''."""
    if not cfg.cache:
        return ""
    try:
        return _nsys_cache_path(cfg, digest).read_text(encoding="utf-8")
    except OSError:
        return ""


def _nsys_cache_store(cfg: PopcornConfig, digest: str, stats: str) -> None:
    if not cfg.cache or not stats.strip():
        return
    path = _nsys_cache_path(cfg, digest)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".txt.tmp")
        tmp.write_text(stats, encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass


def _find_capture(root: Path) -> Path | None:
    """The ``profile.<index>-<spec-slug>/`` directory the CLI extracted, if any."""
    for child in sorted(root.glob("profile.*")):
        if child.is_dir() and (child / "ncu-details.txt").is_file():
            return child
    return None


def profile_submission(cfg: PopcornConfig, sub_path: Path, dest: Path, digest: str) -> Profile:
    """Capture one hosted Nsight Compute profile of ``benchmark_index``.

    Best-effort by construction: every failure path returns a ``Profile`` with ``ok``
    false and an ``error``, and none of them raise. A profile is evidence, not a verdict
    -- losing one must never change whether a kernel scored.

    ``--profile-brev`` is a *different service* from the one ``submit`` talks to: it POSTs
    to ``$POPCORN_BREV_PROFILER_URL/profile``, gets a job id, polls it, and downloads a
    zip. There is no submission id and nothing in the popcorn API to correlate it with,
    which is why this cannot be another ``mode`` of ``submit``.

    The CLI extracts **relative to its own cwd**, ignoring ``--output`` -- so it runs in a
    temp dir and the capture is moved to ``dest`` afterwards. Doing that here is what
    removes the whole class of bug the agents used to hit by hand: no stray ``profile.*/``
    at the worktree root, no ``workdir`` that must exist before the command that creates it.
    """
    prof = Profile()
    cached = _profile_cache_load(cfg, digest)
    if cached:
        prof.cached = True
        _land(prof, dest, cached, cfg)
        if not prof.ok:
            prof.error = f"could not write cached capture under {dest}"
        return prof

    exe = popcorn_bin(cfg)
    if not exe:
        prof.error = "popcorn CLI not found on PATH (set bench.popcorn.bin)"
        return prof

    # Printed only past the cache check, and only once the run is really going to cost
    # minutes: the CLI's own poll chatter is captured below, so without this the caller
    # stares at a silent five minutes and cannot tell a queued profile from a hung score.
    print(
        f"profiling benchmark index {cfg.benchmark_index} on the hosted Nsight service "
        "(minutes; runs alongside the timing submission)...",
        file=sys.stderr,
    )
    env = dict(os.environ)
    env.setdefault("POPCORN_BREV_PROFILER_URL", brev_profiler_url())
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="kt-profile-") as tmp:
        cmd = [
            exe, "submit", str(sub_path.resolve()),
            "--leaderboard", cfg.leaderboard,
            "--profile-brev",
            "--benchmark-index", str(cfg.benchmark_index),
            "--no-tui",
            "--output", "brev.json",
        ]
        try:
            r = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True,
                timeout=PROFILE_TIMEOUT_S, env=env,
            )
        except subprocess.TimeoutExpired:
            prof.wall_s = time.time() - t0
            prof.error = f"profile timed out after {PROFILE_TIMEOUT_S}s"
            return prof
        except OSError as e:
            prof.wall_s = time.time() - t0
            prof.error = f"profile could not start: {e}"
            return prof
        prof.wall_s = time.time() - t0
        capture = _find_capture(Path(tmp))
        if capture is None:
            tail = ((r.stderr or "") + (r.stdout or "")).strip()[-500:]
            prof.error = f"profile produced no capture (exit {r.returncode}): {tail}"
            return prof
        details = ""
        with contextlib.suppress(OSError):
            details = (capture / "ncu-details.txt").read_text(encoding="utf-8", errors="replace")
        _land(prof, dest, details, cfg)
        report = capture / "profile.ncu-rep"
        if prof.ok and report.is_file():
            with contextlib.suppress(OSError):
                shutil.copy2(report, dest / report.name)
                prof.report_path = str(dest / report.name)

    if not prof.ok:
        prof.error = prof.error or f"could not write capture under {dest}"
    else:
        _profile_cache_store(cfg, digest, details)
    return prof


def _cholesky_nsys_args(cfg: PopcornConfig) -> tuple[int, int, int]:
    """Return ``(batch, n, seed)`` for the Modal nsys harness, or raise."""
    if cfg.leaderboard != "cholesky":
        raise PopcornError(
            f"Nsight Systems Modal profiling supports only cholesky in v1, not {cfg.leaderboard}"
        )
    fields = spec_fields(cfg.benchmark_spec)
    missing = [name for name in ("batch", "n", "seed") if not fields.get(name)]
    if missing:
        raise PopcornError(
            "Nsight Systems Modal profiling requires benchmark_spec fields: "
            + ", ".join(missing)
        )
    try:
        batch = int(fields["batch"])
        n = int(fields["n"])
        seed = int(fields["seed"])
    except ValueError as e:
        raise PopcornError(
            "Nsight Systems Modal profiling requires integer batch, n, and seed "
            f"in benchmark_spec: {cfg.benchmark_spec}"
        ) from e
    if batch <= 0 or n <= 0:
        raise PopcornError(
            "Nsight Systems Modal profiling requires positive batch and n "
            f"in benchmark_spec: {cfg.benchmark_spec}"
        )
    return batch, n, seed


def profile_nsys_submission(
    cfg: PopcornConfig,
    sub_path: Path,
    dest: Path,
    digest: str,
) -> NsysProfile:
    """Capture one Modal/B200 Nsight Systems timeline of the scored Cholesky shape."""
    prof = NsysProfile()
    cached = _nsys_cache_load(cfg, digest)
    if cached:
        prof.cached = True
        _land_nsys(prof, dest, cached, cfg)
        if not prof.ok:
            prof.error = f"could not write cached Nsight Systems stats under {dest}"
        return prof

    try:
        batch, n, seed = _cholesky_nsys_args(cfg)
    except PopcornError as e:
        prof.error = str(e)
        return prof

    exe = modal_bin()
    if not exe:
        prof.error = "Modal CLI not found on PATH (set KERNELTHING_MODAL_BIN)"
        return prof
    script = Path(__file__).resolve().with_name("nsys_modal.py")
    if not script.is_file():
        prof.error = f"Nsight Systems Modal runner missing: {script}"
        return prof

    print(
        f"profiling benchmark index {cfg.benchmark_index} with Nsight Systems on Modal/B200 "
        "(minutes; runs alongside the timing submission)...",
        file=sys.stderr,
    )
    cmd = [
        exe,
        "run",
        str(script),
        "--submission",
        str(sub_path.resolve()),
        "--output",
        str(dest),
        "--batch",
        str(batch),
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--digest",
        digest,
    ]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROFILE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        prof.wall_s = time.time() - t0
        prof.error = f"Nsight Systems profile timed out after {PROFILE_TIMEOUT_S}s"
        return prof
    except OSError as e:
        prof.wall_s = time.time() - t0
        prof.error = f"Nsight Systems profile could not start: {e}"
        return prof
    prof.wall_s = time.time() - t0
    if r.returncode != 0:
        tail = ((r.stderr or "") + (r.stdout or "")).strip()[-500:]
        prof.error = f"Nsight Systems profile failed (exit {r.returncode}): {tail}"
        return prof

    report_path = dest / "profile.nsys-rep"
    sqlite_path = dest / "profile.sqlite"
    stats_path = dest / "stats.txt"
    missing = [
        str(path)
        for path in (report_path, sqlite_path, stats_path)
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        prof.error = f"Nsight Systems profile produced missing/empty artifacts: {missing}"
        return prof

    try:
        stats = stats_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        prof.error = f"Nsight Systems stats not readable: {e}"
        return prof
    _land_nsys(prof, dest, stats, cfg)
    if prof.ok:
        prof.report_path = str(report_path)
        prof.sqlite_path = str(sqlite_path)
        with contextlib.suppress(OSError):
            _nsys_cache_store(
                cfg,
                digest,
                stats_path.read_text(encoding="utf-8", errors="replace"),
            )
    else:
        prof.error = prof.error or f"could not write Nsight Systems stats under {dest}"
    return prof


def _land(prof: Profile, dest: Path, details: str, cfg: PopcornConfig) -> None:
    """Write the capture and its digest into ``dest`` and fill in ``prof``.

    ``digest.txt`` is written beside the full dump rather than returned in memory so the
    text has exactly one home: ``score_command`` echoes it to the agent, the orchestrator
    can read it off a finished member, and the verdict JSON carries only its path.
    """
    if not details.strip():
        return
    prof.digest = profile_digest(details, cfg)
    try:
        dest.mkdir(parents=True, exist_ok=True)
        details_path = dest / "ncu-details.txt"
        details_path.write_text(details, encoding="utf-8")
        prof.details_path = str(details_path)
        prof.ok = True
    except OSError:
        return
    if prof.digest:
        with contextlib.suppress(OSError):
            digest_path = dest / "digest.txt"
            digest_path.write_text(prof.digest + "\n", encoding="utf-8")
            prof.digest_path = str(digest_path)


def _land_nsys(prof: NsysProfile, dest: Path, stats: str, cfg: PopcornConfig) -> None:
    """Write nsys stats into ``dest`` and fill in ``prof``."""
    stats = brev_defuse(stats, cfg).strip()
    if not stats:
        return
    try:
        dest.mkdir(parents=True, exist_ok=True)
        stats_path = dest / "stats.txt"
        stats_path.write_text(stats + "\n", encoding="utf-8")
        prof.stats_path = str(stats_path)
        prof.ok = True
    except OSError:
        return


def _precheck(cfg: PopcornConfig, source: str) -> str | None:
    """Local rejections that would otherwise cost a remote round-trip."""
    try:
        ast.parse(source)
    except SyntaxError as e:
        return f"submission does not parse: {e.msg} (line {e.lineno})"
    for needle in cfg.reject_substrings:
        if needle and needle in source:
            return (
                f"submission contains '{needle}', which the evaluation server rejects; "
                "remove it (including inside comments and longer words)"
            )
    return None


def _test_error(report: TestReport, total: int) -> str:
    """A compact, actionable failure string built from the failing test rows."""
    parts = [f"popcorn test: {report.failed}/{total} shapes failed"]
    for f in report.failures[:3]:
        detail = " ".join(f.get("error", "").split())[:400]
        parts.append(f"  {f['spec']}: {detail}" if detail else f"  {f['spec']}")
    if len(report.failures) > 3:
        parts.append(f"  ... and {len(report.failures) - 3} more")
    return "\n".join(parts)


def _metric_from(
    cfg: PopcornConfig, shapes: list[Shape], timing: Submission
) -> tuple[float | None, str | None, dict[str, Any]]:
    """Turn a parsed timing submission into ``(metric_us, err, extra_detail)``."""
    if cfg.metric_mode == "leaderboard":
        # The run's own score field is the ranked geomean at full precision; the printed
        # line is the same number formatted, so it only matters when the API is down.
        seconds = timing.score if timing.score is not None else parse_leaderboard_score_s(
            timing.text
        )
        if seconds is None:
            return None, "popcorn leaderboard run reported no public geomean score", {}
        return seconds * 1e6, None, {"geomean_s": seconds}

    extra: dict[str, Any] = {}
    gm = geomean_us(shapes)
    if gm is not None:
        extra["geomean_us"] = round(gm, 4)
    if cfg.metric_mode == "geomean":
        if gm is None:
            return None, "popcorn benchmark produced no usable timings", extra
        return gm, None, extra

    idx = cfg.benchmark_index
    if idx >= len(shapes):
        return None, (
            f"benchmark_index {idx} is out of range: the run reported "
            f"{len(shapes)} benchmark shapes"
        ), extra
    target = shapes[idx]
    # Positional selection is only safe while the row list is complete. If a row is ever
    # dropped -- an unparsed failure row is the realistic way -- every later index shifts
    # and the metric silently comes from a different shape, which the search would then
    # happily optimise. Pinning the spec turns that into a loud failure.
    if cfg.benchmark_spec:
        want = spec_fields(cfg.benchmark_spec)
        if want and spec_fields(target.spec) != want:
            found = next(
                (s.index for s in shapes if spec_fields(s.spec) == want), None
            )
            where = (
                f"it is at index {found} instead"
                if found is not None
                else f"it is absent from the {len(shapes)} shapes reported"
            )
            return None, (
                f"benchmark_index {idx} is '{target.spec}', not the pinned "
                f"'{cfg.benchmark_spec}' -- {where}. Refusing to score a shape this "
                f"problem does not target; fix benchmark_index/benchmark_spec, or "
                f"check whether a benchmark row failed to parse."
            ), extra
    extra.update(
        {
            "target_index": idx,
            "target_spec": target.spec,
            "mean_us": target.mean_us,
            "err_us": target.err_us,
            "best_us": target.best_us,
            "worst_us": target.worst_us,
        }
    )
    if target.mean_us is None:
        return None, (
            f"benchmark shape {idx} ({target.spec}) reported no time"
            + (f": {target.error[:300]}" if target.error else "")
        ), extra
    return target.mean_us, None, extra


def score(
    problem: Problem,
    worktree: Path,
    *,
    test_only: bool = False,
    profile: bool | None = None,
) -> tuple[bool, float | None, str | None, dict[str, Any]]:
    """Score a worktree through the hosted popcorn service.

    Returns ``(correct, metric, err, detail)`` -- the same tuple ``bench.score`` returns,
    so ``cli.score_command`` can emit an identical verdict either way. ``metric`` is in
    microseconds and lower is better, so a popcorn problem sets ``direction: minimize``.

    ``profile`` overrides the automatic profilers (``None`` defers to the manifest, and
    is what the CLI passes unless ``--no-profile`` was given). When on, a hosted Nsight
    Compute capture and, when configured, a Modal/B200 Nsight Systems capture are started
    as soon as the test submission passes and joined after the timing submission returns,
    so all three overlap; captures land in ``<worktree>/profile/latest/`` and their
    paths land in ``detail["profile"]`` / ``detail["nsys"]``.

    Starting it at test-pass rather than at entry is the one place this is *not* maximally
    concurrent, and deliberately: overlapping the test as well would buy ~10s, while a
    kernel that fails correctness would already have taken a ~300s slot in a queue that
    runs one job at a time.

    A profile is never allowed to affect the verdict. Profilers fire only after
    correctness is established, their failures are recorded rather than raised, and they
    are joined in a ``finally`` so an exception on the timing path cannot leak a thread.
    """
    try:
        cfg = resolve_config(problem)
    except PopcornError as e:
        return False, None, str(e), {}

    detail: dict[str, Any] = {
        "backend": "popcorn",
        "leaderboard": cfg.leaderboard,
        "gpu": cfg.gpu,
        "metric_mode": cfg.metric_mode,
    }
    sub_path = Path(worktree) / problem.rel_dir / cfg.submission_file
    try:
        source = sub_path.read_text(encoding="utf-8")
    except OSError as e:
        return False, None, f"submission not readable at {sub_path}: {e}", detail

    err = _precheck(cfg, source)
    if err:
        return False, None, err, detail
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    detail["sha256"] = digest[:16]

    wall: dict[str, float] = {}
    ids: dict[str, int] = {}
    cached: dict[str, bool] = {}
    sources: dict[str, str] = {}

    def run(mode: str) -> Submission:
        sub = submit(cfg, sub_path, mode, digest)
        wall[mode] = sub.wall_s
        cached[mode] = sub.cached
        sources[mode] = sub.source
        sid = sub.submission_id
        if sid is not None:
            ids[mode] = sid
        return sub

    # Default: capture on a full score, never on --test-only. There is nothing to overlap
    # a profile with on the test path -- it is a ~10s call, so a capture would make it
    # ~25x slower -- and nothing to reason about either, since --test-only produces no
    # timing. That is the call an agent iterating on correctness makes most often, and
    # keeping it cheap is what stops it from being avoided.
    want_profile = (cfg.profile and not test_only) if profile is None else bool(profile)
    want_nsys = (cfg.nsys and not test_only) if profile is None else (bool(profile) and cfg.nsys)
    pool: ThreadPoolExecutor | None = None
    pending_profile: Any = None  # Future[Profile], typed loosely to keep imports small
    pending_nsys: Any = None  # Future[NsysProfile]

    def start_profiles() -> None:
        """Kick off captures in the background. Called once correctness is known."""
        nonlocal pool, pending_profile, pending_nsys
        if pending_profile is not None or pending_nsys is not None:
            return
        workers = int(bool(want_profile)) + int(bool(want_nsys))
        if workers <= 0:
            return
        dest = Path(worktree) / problem.rel_dir / PROFILE_DIR / PROFILE_SUBDIR
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kt-profile")
        if want_profile:
            pending_profile = pool.submit(profile_submission, cfg, sub_path, dest, digest)
        if want_nsys:
            pending_nsys = pool.submit(
                profile_nsys_submission,
                cfg,
                sub_path,
                dest / NSYS_SUBDIR,
                digest,
            )

    try:
        test_sub = run(MODE_TEST)
        report = (
            test_report_from_result(test_sub.raw)
            if test_sub.raw
            else parse_test_output(test_sub.text)
        )
        detail["test"] = report.record()
        total = report.passed + report.failed
        if total == 0:
            job = parse_summary_json(test_sub.text).get("job") or {}
            reason = str(job.get("error") or "").strip()
            raise PopcornError(
                "popcorn test reported no shapes" + (f": {reason[:500]}" if reason else "")
            )
        if report.failed:
            detail.update(
                {"wall_s": wall, "submission_ids": ids, "cached": cached, "source": sources}
            )
            return False, None, _test_error(report, total), detail

        # Correctness is established, so the kernel is worth measuring. Start the
        # profilers *here*, before the timing submission, so all three run together. A
        # broken kernel never reaches this line and so never spends a profiler slot.
        start_profiles()
        if test_only:
            detail.update(
                {"wall_s": wall, "submission_ids": ids, "cached": cached, "source": sources}
            )
            return True, None, None, detail

        timing = run(cfg.timing_mode)
        shapes = (
            shapes_from_result(timing.raw) if timing.raw else parse_benchmark_output(timing.text)
        )
        if shapes:
            detail["shapes"] = [s.record() for s in shapes]
        failed = [s for s in shapes if s.status == "fail"]
        metric, metric_err, extra = _metric_from(cfg, shapes, timing)
        detail.update(extra)
        detail.update(
            {"wall_s": wall, "submission_ids": ids, "cached": cached, "source": sources}
        )
        if failed:
            specs = ", ".join(s.spec for s in failed[:3])
            body = " ".join(failed[0].error.split())[:400]
            return False, metric, (
                f"popcorn {cfg.timing_mode}: {len(failed)} shape(s) failed re-check "
                f"({specs})" + (f": {body}" if body else "")
            ), detail
        if metric_err:
            return True, None, metric_err, detail
        return True, metric, None, detail
    except PopcornError as e:
        detail.update(
            {"wall_s": wall, "submission_ids": ids, "cached": cached, "source": sources}
        )
        return False, None, str(e), detail
    finally:
        # Joined here rather than at any single return so that every exit -- pass, shape
        # re-check failure, PopcornError -- reclaims the thread and reports what the
        # capture did. ``detail`` is returned by reference, so mutating it after the
        # return expression has been evaluated still reaches the caller.
        if pending_profile is not None:
            try:
                prof: Profile = pending_profile.result()
            except Exception as e:  # a capture must never fail a score
                # profile_submission handles its own expected failures; this catches the
                # unexpected ones. Raising here would happen *inside* finally, replacing
                # an already-computed verdict with a traceback -- the one way a profile
                # could still change whether a kernel scored.
                prof = Profile(error=f"{type(e).__name__}: {e}"[:500])
            detail["profile"] = prof.record()
        if pending_nsys is not None:
            try:
                nsys_prof: NsysProfile = pending_nsys.result()
            except Exception as e:
                nsys_prof = NsysProfile(error=f"{type(e).__name__}: {e}"[:500])
            detail["nsys"] = nsys_prof.record()
        if pool is not None:
            pool.shutdown(wait=True)


def _veloq_bin() -> str:
    """``"veloq"`` when it is on PATH, else ``""``.

    Returning the bare name rather than the resolved path is the point: ``which``
    succeeding *proves* the bare name works from here, and the hint blocks are read
    inside the sandbox where an absolute path from some other PATH entry would only be
    noise. Empty means the hints are dropped -- never printed as a command that is not
    there.
    """
    return "veloq" if shutil.which("veloq") else ""


def _one_line(text: Any) -> str:
    """Collapse a message to a single line for a ``--- ... ---`` banner.

    ``_cli_score`` finds the verdict by scanning stdout in reverse for the last line
    starting with ``{``, so an error string carrying a newline followed by a brace could
    otherwise shadow it. Profiler errors are the one thing here we do not author.
    """
    return " ".join(str(text or "unknown").split())


def _hint_lines(
    hints: tuple[tuple[str, str], ...], veloq: str, group: str, indent: str = ""
) -> list[str]:
    """Render ``VELOQ_*_HINTS`` as an aligned ``veloq <group> <verb>  # what for`` list.

    The comment column is capped rather than set by the longest command: two of the ncu
    verbs carry a ``--counter`` glob and would otherwise push every comment 40 columns
    right, which is what makes a list like this read as a wall.
    """
    cmds = [f"{indent}{veloq} {group} {verb}" for verb, _ in hints]
    width = min(max(len(c) for c in cmds), 62)
    return [f"{c.ljust(width)}  # {why}" for c, (_, why) in zip(cmds, hints, strict=True)]


def veloq_verb_block(group: str) -> str:
    """The ``veloq <group>`` verb list for ``kernel-tools-veloq.md``, or ``''``.

    The prompt renders this rather than carrying its own copy: keeping the verbs in two
    places is how the prompt and reality drift apart, and this list is checked against a
    real capture. Commands are written against ``$V``/``$REP``/``$NSYS``, which the prompt
    assigns just above -- the same variable names a score's block prints, so a path copied
    out of a score drops straight into a command copied out of the prompt.
    """
    hints = {"ncu": VELOQ_NCU_HINTS, "nsys": VELOQ_NSYS_HINTS}.get(group)
    return "\n".join(_hint_lines(hints, "$V", group)) if hints else ""


def _capture_block(prof: Any, *, kind: str, noun: str, var: str, group: str, text_key: str) -> str:
    """One capture's status in a score: did it land, and where.

    Deliberately just the state. The verbs that read it are static, so they live in the
    candidate prompt (see ``VELOQ_NCU_HINTS``) and this points back at them; what cannot
    live there is whether *this* score produced a queryable report. Two ways it did not:
    no ``veloq`` on PATH, and a cached score (the 80MB capture is not re-downloaded).
    Both leave the agent the flat text view's path rather than a dead command.
    """
    if not isinstance(prof, dict):
        return ""
    if not prof.get("ok"):
        return f"--- {kind} profile unavailable: {_one_line(prof.get('error'))} ---"
    cached = " (cached: identical submission)" if prof.get("cached") else ""
    head = f"--- {kind} {noun} of the scored shape{cached} ---"
    report = str(prof.get("report_path") or "")
    if report and _veloq_bin():
        return f"{head}\n  {var}={report}\n  Query it with the `veloq {group}` verbs in your tools section."
    text = str(prof.get(text_key) or "")
    if not text:
        return f"{head}\n  No readable artifact this score."
    return f"{head}\n  No queryable report this score. Flat text view: {text}"


def format_profile_block(detail: dict[str, Any]) -> str:
    """The Nsight Compute status ``score_command`` prints above the verdict, or ''."""
    return _capture_block(
        detail.get("profile"),
        kind="Nsight Compute",
        noun="capture",
        var="REP",
        group="ncu",
        text_key="details_path",
    )


def format_nsys_block(detail: dict[str, Any]) -> str:
    """The Nsight Systems status ``score_command`` prints above the verdict, or ''."""
    return _capture_block(
        detail.get("nsys"),
        kind="Nsight Systems",
        noun="timeline",
        var="NSYS",
        group="nsys",
        text_key="stats_path",
    )


# Printed once, above the capture blocks: the marching order first, then what landed and
# where. It is deliberately singular on both counts -- the 2026-07-24 run's members
# routinely named three bottlenecks and changed four things at once, which makes a
# regression un-attributable and a win unrepeatable.
#
# Leading rather than closing puts the instruction before the data it applies to, and the
# whole score is ~800 chars now, so nothing here is far enough from anything else for
# recency to decide it. What the position *does* cost is the anchor role: the capture
# blocks below each name their own `veloq <group>` verbs, so the pointer back to the
# prompt's verb list survives this being first.
ANALYSIS_DIRECTIVE = (
    "Analyze the ncu and nsys reports using ncu-profile-analysis, ncu-report-skill, and nsys-profile-analysis skills. "
    "Design and implement exactly one optimization strategy, targetting exactly one high impact performance bottleneck. "
    "Use the nvidia-cuda-docs MCP and the vendored ptx-skill to answer API and hardware questions."
)


def format_analysis_directive(detail: dict[str, Any]) -> str:
    """The marching order above the capture blocks, or '' when nothing was captured.

    The empty case is the load-bearing one: a score that captured nothing must not open
    with an instruction to go read reports that are not there.
    """
    landed = [
        kind
        for kind in ("profile", "nsys")
        if isinstance(detail.get(kind), dict) and detail[kind].get("ok")
    ]
    if not landed:
        return ""
    text = ANALYSIS_DIRECTIVE
    if landed == ["profile"]:
        text = text.replace(" and Nsight Systems", "")
    elif landed == ["nsys"]:
        text = text.replace("Nsight Compute and ", "")
    # Not wrapped. The reader is a model, not a terminal -- the capture blocks below run
    # to 111 chars unwrapped -- and every width breaks one of the hyphenated skill names
    # the directive exists to hand over verbatim (70 splits `ncu-report-skill`, 84
    # `nvidia-cuda-docs`, 100 `nsys-profile-analysis`).
    # A noun phrase for what follows, like the capture banners below it -- "Next" named a
    # position this block no longer holds, now that it opens the score instead of closing
    # it. Purely a visual section marker: nothing parses it, and the only structure in
    # this output that is load-bearing is the verdict line _cli_score scans back for.
    return "--- Task ---\n" + text


def score_command(problem: Problem, args: Any) -> int:
    """``kernelthing score`` for a popcorn-backed problem: print the verdict JSON line.

    Mirrors the pygpubench branch's contract exactly -- one JSON object on stdout, exit 0
    only when the submission is correct. ``--emit-baseline`` / ``--baseline-median`` are
    accepted and ignored: the metric is an absolute time, so there is no denominator to
    pin, and the orchestrator's seed path already tolerates a missing baseline.

    The profile block is printed **before** the verdict on purpose: ``_cli_score`` finds
    the JSON by scanning stdout in reverse for the last line starting with ``{``, so
    anything emitted after it would have to be guaranteed brace-free forever. Printing
    first makes that impossible to get wrong.

    ``--brief`` drops ``bench`` and nothing else. That record is the run's forensic
    archive -- ``_cli_score`` carries it opaquely into ``members/<id>/result.json`` and
    the journal, and no consumer anywhere reads a field of it -- but it is 98% of the
    line, and 63% is ``shapes``: fourteen per-shape rows an agent cannot act on, since
    ``metric_mode = "shape"`` means one index scores. ``profile``/``nsys`` are another
    20%, repeating paths the banners above already printed. So the caller that archives
    keeps the default and the caller that *reads* asks for brief -- see
    ``Orchestrator._score_cmd_str``. Everything an agent acts on survives: a failure's
    reason is in ``error`` (and on stderr), not in ``bench``.
    """
    correct, metric, err, detail = score(
        problem,
        problem.repo_root,
        test_only=bool(getattr(args, "test_only", False)),
        profile=getattr(args, "profile", None),
    )
    # Directive first, then the captures it refers to: instruction before data.
    for block in (
        format_analysis_directive(detail),
        format_profile_block(detail),
        format_nsys_block(detail),
    ):
        if block:
            print(block)
    result: dict[str, Any] = {
        "unit": problem.unit,
        "correct": correct,
        "metric": metric,
        "error": err,
    }
    if not getattr(args, "brief", False):
        result["bench"] = detail
    if getattr(args, "emit_baseline", False):
        result["baseline_median"] = None
    print(json.dumps(result))
    if err and not correct:
        print(err, file=sys.stderr)
    return 0 if correct else 1
