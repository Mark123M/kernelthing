"""Parsing and scoring for the hosted-popcorn benchmark backend.

The popcorn CLI only ever hands us *formatted* text -- the raw per-shape nanoseconds never
leave the service -- so every number the search ranks on comes out of a regex over that
text. That makes an upstream formatting change a silent-wrong-metric bug, and these tests
the thing that catches it.

``tests/fixtures/popcorn/*.txt`` are **verbatim captures** of real ``--output`` files from
the live cholesky leaderboard on B200 -- submissions 900985 / 900988 / 900990 (2026-07-23)
and 889961 (a linalg tuning sweep). Do not hand-edit them to make a test pass -- re-capture
instead:

    popcorn submit <file> --leaderboard cholesky --gpu B200 --mode <mode> \\
        --no-tui --output <fixture>.txt

One fixture is still synthetic because no live run produced it: a benchmark shape that
fails its re-check. It is spliced into the real benchmark capture and marked below.

No network: ``popcorn.submit`` is faked wherever a score is exercised.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest

from kernelthing import popcorn
from kernelthing.problem import Problem

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "popcorn"


def _fixture(name: str) -> str:
    return (FIXTURES / f"{name}.txt").read_text(encoding="utf-8")


# Real captures. Note the field order the service actually emits -- `n` first, `batch`
# last -- and that *passing* test rows carry a `> ` detail line too, not just failures.
BENCHMARK_OK = _fixture("benchmark")
TEST_OK = _fixture("test-pass")
TEST_FAIL = _fixture("test-fail")

# SYNTHETIC (no live capture): a benchmark shape failing its re-check. Built by splicing a
# failure row into the real capture, so everything around it is still ground truth. The
# error body deliberately contains a blank line and a `RuntimeError:` line -- both are
# things that must not be mistaken for the next shape's spec.
BENCHMARK_FAIL = BENCHMARK_OK.replace(
    """n: 64; cond: 2; seed: 41064; batch: 1024
 ⏱ 110 ± 0.0 µs
 ⚡ 110 µs 🐌 110 µs""",
    """❌ n: 64; cond: 2; seed: 41064; batch: 1024 failed testing:
output is not lower triangular enough: relative_residual=0.807

RuntimeError: CUDA error: an illegal memory access was encountered""",
)

# A real ranked run (submission 889961, from a linalg tuning sweep). Deliberately one of
# the ~half of captures where the SECRET line is printed first -- see
# test_leaderboard_score_ignores_line_order.
LEADERBOARD = _fixture("leaderboard")
LEADERBOARD_PUBLIC_S = 0.0019501559313196228
LEADERBOARD_SECRET_S = 0.0019876696598447596

# Index 2 is the shape problems/cholesky_b256n128 targets, confirmed against the live
# benchmark grid; the seed kernel measured 75.1 us there.
TARGET_INDEX = 2
TARGET_SPEC = "n: 128; cond: 2; seed: 41128; batch: 256"
TARGET_US = 75.1


def _problem(tmp_path, **popcorn_cfg) -> Problem:
    cfg = {
        "leaderboard": "cholesky",
        "gpu": "B200",
        "submission_file": "submission.py",
        "benchmark_index": TARGET_INDEX,
        "benchmark_spec": TARGET_SPEC,
        # Off by default here, on by default in production. `score` shells out to the
        # real popcorn CLI to profile, and the CLI is installed on a dev box -- leaving
        # this on would have every scoring test queue a live job on gpu-mode's hosted
        # profiler. The tests that do exercise profiling stub `profile_submission`.
        "profile": False,
    }
    cfg.update(popcorn_cfg)
    (tmp_path / "submission.py").write_text("def custom_kernel(data):\n    return data\n")
    return Problem(
        name="cholesky_b256n128",
        repo_root=tmp_path,
        rel_dir=".",
        plan="plan.md",
        edit_files=["submission.py"],
        unit="us",
        direction="minimize",
        bench={"backend": "popcorn", "popcorn": cfg},
    )


def _fake_submit(monkeypatch, texts: dict[str, str], seen: list[str] | None = None, raw=None):
    """Replace the network call with canned per-mode output.

    ``raw=None`` simulates an unreachable API, i.e. the text-scraping fallback. Pass a
    ``{mode: result_dict}`` to exercise the primary (API) path.
    """

    def fake(cfg, sub_path, mode, digest):
        if seen is not None:
            seen.append(mode)
        if mode not in texts:
            raise popcorn.PopcornError(f"unexpected submission mode {mode}")
        return popcorn.Submission(
            mode=mode,
            text=texts[mode],
            returncode=0,
            wall_s=1.0,
            raw=(raw or {}).get(mode),
        )

    monkeypatch.setattr(popcorn, "submit", fake)


def _api(name: str) -> dict:
    """A verbatim `GET /user/submissions/<id>` response (code stripped)."""
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _api_result(name: str, mode: str) -> dict:
    run = popcorn.select_run(_api(name), mode)
    assert run is not None, f"{name} has no public {mode} run"
    return run["result"]


API_BENCHMARK = _api_result("api-benchmark", "benchmark")
API_TEST_OK = _api_result("api-test-pass", "test")
API_TEST_FAIL = _api_result("api-test-fail", "test")

# The exact figure the service measured for the target shape, in nanoseconds. The CLI
# would have printed this as "75.1 µs".
TARGET_NS = 75141.44521620538


# --- the API path (primary) -------------------------------------------------


def test_api_gives_full_precision_where_the_text_gives_three_digits():
    """The whole reason the API path exists."""
    shapes = popcorn.shapes_from_result(API_BENCHMARK)
    api_us = shapes[TARGET_INDEX].mean_us
    text_us = popcorn.parse_benchmark_output(BENCHMARK_OK)[TARGET_INDEX].mean_us
    assert api_us == pytest.approx(TARGET_NS / 1000.0)
    assert text_us == pytest.approx(75.1)  # what the formatted row rounds to
    assert api_us != text_us
    assert abs(api_us - text_us) / api_us < 0.001  # same measurement, coarser rendering


def test_api_shapes_match_the_text_shapes():
    """Both channels must describe the same run -- the fallback cannot mean something else."""
    api = popcorn.shapes_from_result(API_BENCHMARK)
    text = popcorn.parse_benchmark_output(BENCHMARK_OK)
    assert len(api) == len(text) == 15
    for a, t in zip(api, text, strict=True):
        assert a.index == t.index
        assert popcorn.spec_fields(a.spec) == popcorn.spec_fields(t.spec)
        # the text is the same number at three significant figures
        assert a.mean_us == pytest.approx(t.mean_us, rel=0.005)


def test_api_shape_count_is_authoritative():
    """`benchmark-count` fixes the index space, so a shape cannot shift position."""
    assert API_BENCHMARK["benchmark-count"] == "15"
    shapes = popcorn.shapes_from_result(API_BENCHMARK)
    assert [s.index for s in shapes] == list(range(15))
    assert shapes[TARGET_INDEX].spec == TARGET_SPEC


def test_api_test_reports():
    ok = popcorn.test_report_from_result(API_TEST_OK)
    assert (ok.passed, ok.failed) == (17, 0)
    bad = popcorn.test_report_from_result(API_TEST_FAIL)
    assert (bad.passed, bad.failed) == (0, 17)
    assert bad.failures[0]["error"] == (
        "output is not lower triangular enough: relative_residual=0.807"
    )
    # and it agrees with what the text parser makes of the same submission
    assert bad.failures[0] == popcorn.parse_test_output(TEST_FAIL).failures[0]


def test_select_run_skips_secret_runs():
    """A ranked submission has six runs; the secret ones use a different seed."""
    payload = {
        "runs": [
            {"mode": "leaderboard", "secret": True, "score": 0.9, "result": {"a": 1}},
            {"mode": "leaderboard", "secret": False, "score": 0.1, "result": {"b": 2}},
        ]
    }
    run = popcorn.select_run(payload, "leaderboard")
    assert run is not None and run["score"] == 0.1
    assert popcorn.select_run(payload, "benchmark") is None
    assert popcorn.select_run({}, "test") is None


def test_score_prefers_the_api_and_records_the_source(tmp_path, monkeypatch):
    _fake_submit(
        monkeypatch,
        {"test": TEST_OK, "benchmark": BENCHMARK_OK},
        raw={"test": API_TEST_OK, "benchmark": API_BENCHMARK},
    )
    correct, metric, err, detail = popcorn.score(_problem(tmp_path), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(TARGET_NS / 1000.0)  # full precision, not 75.1
    assert detail["source"] == {"test": "api", "benchmark": "api"}


def test_score_falls_back_to_text_when_the_api_is_unreachable(tmp_path, monkeypatch):
    """A downgrade must keep the run going, and must be visible afterwards."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})  # raw=None
    correct, metric, err, detail = popcorn.score(_problem(tmp_path), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(75.1)  # the coarse number, still usable
    assert detail["source"] == {"test": "cli-text", "benchmark": "cli-text"}


def test_attach_raw_never_raises_when_the_api_is_down(monkeypatch):
    def boom(sid, **kw):
        raise popcorn.PopcornError("network is down")

    monkeypatch.setattr(popcorn, "fetch_submission", boom)
    sub = popcorn.Submission(mode="benchmark", text=BENCHMARK_OK, returncode=0, wall_s=1.0)
    popcorn._attach_raw(sub)  # must not propagate
    assert sub.raw is None and sub.source == "cli-text"


def test_cli_id_is_read_from_the_popcorn_config(tmp_path, monkeypatch):
    monkeypatch.delenv("POPCORN_CLI_ID", raising=False)
    monkeypatch.setattr(popcorn.Path, "home", staticmethod(lambda: tmp_path))
    (tmp_path / ".popcorn.yaml").write_text("cli_id: 4cdcbba4-7514-4b71-bbb3-766a21aec68e\n")
    assert popcorn.cli_id() == "4cdcbba4-7514-4b71-bbb3-766a21aec68e"
    (tmp_path / ".popcorn.yaml").unlink()
    assert popcorn.cli_id() == ""  # missing config degrades, never raises
    monkeypatch.setenv("POPCORN_CLI_ID", "from-env")
    assert popcorn.cli_id() == "from-env"


# --- time parsing -----------------------------------------------------------


@pytest.mark.parametrize(
    "line,mean,err",
    [
        # every form the real captures contain
        (" ⏱ 75.1 ± 0.07 µs", 75.1, 0.07),
        (" ⏱ 113 ± 0.0 µs", 113.0, 0.0),
        (" ⏱ 3.78 ± 0.001 ms", 3780.0, 1.0),
        (" ⏱ 221 ± 0.0 ms", 221000.0, 0.0),
        (" ⏱ 11.9 ± 0.00 ms", 11900.0, 0.0),
        # forms format_time can emit that the captures happened not to hit
        (" ⏱ 950 ± 3.1 ns", 0.95, 0.0031),
        (" ⏱ 98.4 µs", 98.4, None),
        (" ⏱ 0.9876543 ± 0.0012 µs", 0.9876543, 0.0012),
    ],
)
def test_parse_mean_err_us(line, mean, err):
    got_mean, got_err = popcorn.parse_mean_err_us(line)
    assert got_mean == pytest.approx(mean)
    if err is None:
        assert got_err is None
    else:
        assert got_err == pytest.approx(err)


def test_parse_mean_err_us_no_match():
    assert popcorn.parse_mean_err_us("no timing here") == (None, None)


def test_parse_time_us_scales():
    assert popcorn.parse_time_us("113 µs") == pytest.approx(113.0)
    assert popcorn.parse_time_us("3.78 ms") == pytest.approx(3780.0)
    assert popcorn.parse_time_us("512 ns") == pytest.approx(0.512)
    assert popcorn.parse_time_us("97.2 us") == pytest.approx(97.2)


# --- benchmark rows ---------------------------------------------------------


def test_parse_benchmark_shapes():
    shapes = popcorn.parse_benchmark_output(BENCHMARK_OK)
    assert len(shapes) == 15  # the live cholesky benchmark grid
    assert [s.index for s in shapes] == list(range(15))
    assert all(s.status == "pass" for s in shapes)
    assert all(s.mean_us is not None for s in shapes)


def test_parse_benchmark_target_shape():
    """The one row the search actually ranks on."""
    target = popcorn.parse_benchmark_output(BENCHMARK_OK)[TARGET_INDEX]
    assert target.spec == TARGET_SPEC
    assert target.mean_us == pytest.approx(TARGET_US)
    assert target.err_us == pytest.approx(0.07)
    assert target.best_us == pytest.approx(74.6)
    assert target.worst_us == pytest.approx(76.2)


def test_millisecond_rows_are_normalised_to_microseconds():
    """Shapes span 75 us to 221 ms; format_time rescales, the metric must not."""
    shapes = popcorn.parse_benchmark_output(BENCHMARK_OK)
    assert shapes[5].mean_us == pytest.approx(3780.0)  # " ⏱ 3.78 ± 0.001 ms"
    assert shapes[14].mean_us == pytest.approx(221000.0)  # " ⏱ 221 ± 0.0 ms"
    assert shapes[14].best_us == pytest.approx(221000.0)


def test_benchmark_failure_does_not_shift_indices():
    """A phantom shape would silently repoint the metric at the wrong benchmark."""
    shapes = popcorn.parse_benchmark_output(BENCHMARK_FAIL)
    assert len(shapes) == 15
    assert shapes[1].status == "fail"
    assert "not lower triangular" in shapes[1].error
    assert "illegal memory access" in shapes[1].error  # exception line stays error text
    # ...and the target shape is still at index 2, with its timings intact.
    assert shapes[TARGET_INDEX].spec == TARGET_SPEC
    assert shapes[TARGET_INDEX].mean_us == pytest.approx(TARGET_US)


def test_dropped_row_is_caught_instead_of_scoring_the_wrong_shape(tmp_path, monkeypatch):
    """Positional selection is only safe while the row list is complete.

    An unparsed failure row would shorten the list and shift every later index -- index 2
    would silently become the 276 us shape instead of the 75.1 us one, and the search
    would optimise the wrong kernel with nothing to flag it. The pinned spec must turn
    that into a refusal.
    """
    dropped = BENCHMARK_OK.replace(
        "n: 64; cond: 2; seed: 41064; batch: 1024\n ⏱ 110 ± 0.0 µs\n ⚡ 110 µs 🐌 110 µs\n\n",
        "",
    )
    assert len(popcorn.parse_benchmark_output(dropped)) == 14  # a row really is gone

    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": dropped})
    correct, metric, err, _ = popcorn.score(_problem(tmp_path), tmp_path)
    assert metric is None, "must not report a metric from the wrong shape"
    assert "not the pinned" in err and "index 1 instead" in err
    assert correct is True  # the kernel is fine; our view of the row list is not


def test_spec_pin_is_order_insensitive(tmp_path, monkeypatch):
    """The service does not emit a stable field order, so pinning compares fields."""
    reordered = "batch: 256; n: 128; seed: 41128; cond: 2"
    assert popcorn.spec_fields(reordered) == popcorn.spec_fields(TARGET_SPEC)
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _, metric, err, _ = popcorn.score(_problem(tmp_path, benchmark_spec=reordered), tmp_path)
    assert err is None and metric == pytest.approx(TARGET_US)


def test_unpinned_problem_still_scores(tmp_path, monkeypatch):
    """benchmark_spec is optional; without it we fall back to bare positional selection."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _, metric, err, _ = popcorn.score(_problem(tmp_path, benchmark_spec=""), tmp_path)
    assert err is None and metric == pytest.approx(TARGET_US)


def test_geomean_us():
    shapes = popcorn.parse_benchmark_output(BENCHMARK_OK)
    assert popcorn.geomean_us(shapes) == pytest.approx(2007.56, rel=1e-4)
    assert popcorn.geomean_us([]) is None


# --- test rows --------------------------------------------------------------


def test_parse_test_output_all_pass():
    """Passing rows carry a `> ` detail line of their own; it must not become a failure."""
    report = popcorn.parse_test_output(TEST_OK)
    assert (report.passed, report.failed) == (17, 0)
    assert report.failures == []


def test_parse_test_output_failures_keep_detail():
    report = popcorn.parse_test_output(TEST_FAIL)
    assert (report.passed, report.failed) == (0, 17)
    assert report.failures[0]["spec"] == "n: 32; cond: 2; seed: 53124; batch: 16"
    assert report.failures[0]["error"] == (
        "output is not lower triangular enough: relative_residual=0.807"
    )
    # specs carrying a `case:` field parse like any other
    assert report.failures[-1]["spec"] == "n: 1024; case: lowrank; cond: 4; seed: 4330; batch: 2"


# --- summary / leaderboard --------------------------------------------------


def test_split_summary_and_submission_id():
    rows, summary = popcorn.split_summary(BENCHMARK_OK)
    assert "submission_id" not in rows
    assert json.loads(summary)["submission_id"] == 900990
    assert json.loads(summary)["job"]["status"] == "succeeded"
    sub = popcorn.Submission(mode="benchmark", text=BENCHMARK_OK, returncode=0, wall_s=1.0)
    assert sub.submission_id == 900990


def test_split_summary_without_json():
    rows, summary = popcorn.split_summary("just rows\nno json\n")
    assert summary == ""
    assert rows.startswith("just rows")


def test_stdout_fallback_matches_the_output_file():
    """submit() falls back to stdout when --output is unreadable. The CLI prints
    `println!("\\n{}", content)`, so stdout is the output file plus a newline either
    side -- verified against all three live captures -- and must parse identically."""
    as_stdout = "\n" + BENCHMARK_OK + "\n"
    assert popcorn.parse_benchmark_output(as_stdout) == popcorn.parse_benchmark_output(
        BENCHMARK_OK
    )
    assert popcorn.parse_summary_json(as_stdout)["submission_id"] == 900990


def test_parse_leaderboard_score_prefers_public():
    got = popcorn.parse_leaderboard_score_s(LEADERBOARD)
    assert got == LEADERBOARD_PUBLIC_S  # exact: the line carries a full-precision f64
    assert got != LEADERBOARD_SECRET_S
    assert popcorn.parse_leaderboard_score_s(BENCHMARK_OK) is None


def test_leaderboard_score_ignores_line_order():
    """The server emits the public/secret lines in `runs` order, which is not stable.

    Across 54 real captures from the linalg sweeps, 26 printed SECRET first -- so a
    parser that took the first `Geomean score` line would report the wrong number 48% of
    the time, and the secret score is measured on a different seed. Scope must be matched,
    never position. This fixture is one of the secret-first files; the assertion below
    pins the other order too.
    """
    scopes = [m.group("scope") for m in popcorn.GEOMEAN_RE.finditer(LEADERBOARD)]
    assert scopes == ["secret", "public"], "fixture must keep the harder ordering"
    assert popcorn.parse_leaderboard_score_s(LEADERBOARD) == LEADERBOARD_PUBLIC_S

    swapped = "\n".join(reversed(LEADERBOARD.splitlines()[:2]))
    assert popcorn.parse_leaderboard_score_s(swapped) == LEADERBOARD_PUBLIC_S


def test_leaderboard_result_has_no_benchmark_rows():
    """Leaderboard mode emits no per-shape rows (format_submission_rows returns None for
    it), so the metric has to come from the geomean line -- and the summary's Geomean
    lines must not be mistaken for shape specs."""
    assert popcorn.parse_benchmark_output(LEADERBOARD) == []
    assert popcorn.parse_test_output(LEADERBOARD).passed == 0
    assert popcorn.parse_summary_json(LEADERBOARD)["submission_id"] == 889961


# --- config -----------------------------------------------------------------


def test_resolve_config_defaults(tmp_path):
    cfg = popcorn.resolve_config(_problem(tmp_path))
    assert (cfg.leaderboard, cfg.gpu, cfg.benchmark_index) == ("cholesky", "B200", TARGET_INDEX)
    assert cfg.metric_mode == "shape"
    assert cfg.timing_mode == popcorn.MODE_BENCHMARK


def test_timeout_budgets_fit_inside_the_orchestrator_kill(tmp_path):
    """_cli_score kills `kernelthing score` at 1800s; both submissions must fit."""
    cfg = popcorn.resolve_config(_problem(tmp_path))
    assert cfg.timeout_for(popcorn.MODE_TEST) == popcorn.TEST_TIMEOUT_S
    assert cfg.timeout_for(popcorn.MODE_BENCHMARK) == cfg.timeout_s
    assert cfg.timeout_for(popcorn.MODE_TEST) + cfg.timeout_s < 1800
    # a problem asking for less than the test default gets the smaller number
    tight = popcorn.resolve_config(_problem(tmp_path, timeout_s=60))
    assert tight.timeout_for(popcorn.MODE_TEST) == 60


def test_resolve_config_leaderboard_metric_mode_changes_timing_mode(tmp_path):
    cfg = popcorn.resolve_config(_problem(tmp_path, metric_mode="leaderboard"))
    assert cfg.timing_mode == popcorn.MODE_LEADERBOARD


def test_resolve_config_rejects_bad_input(tmp_path):
    with pytest.raises(popcorn.PopcornError):
        popcorn.resolve_config(_problem(tmp_path, leaderboard=""))
    with pytest.raises(popcorn.PopcornError):
        popcorn.resolve_config(_problem(tmp_path, metric_mode="nonsense"))


def test_config_or_none_never_raises(tmp_path):
    assert popcorn.config_or_none(_problem(tmp_path, metric_mode="nonsense")) is None
    prob = _problem(tmp_path)
    prob.bench = {}
    assert popcorn.config_or_none(prob) is None


# --- scoring ----------------------------------------------------------------


def test_score_shape_metric(tmp_path, monkeypatch):
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK}, seen)
    correct, metric, err, detail = popcorn.score(_problem(tmp_path), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(TARGET_US)  # the target shape, not the geomean
    assert seen == ["test", "benchmark"]
    assert detail["target_spec"] == TARGET_SPEC
    assert detail["submission_ids"] == {"test": 900985, "benchmark": 900990}
    assert len(detail["shapes"]) == 15  # every shape kept, so the index can change later
    assert detail["test"]["passed"] == 17


def test_score_geomean_metric(tmp_path, monkeypatch):
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _, metric, err, detail = popcorn.score(_problem(tmp_path, metric_mode="geomean"), tmp_path)
    assert err is None
    assert metric == pytest.approx(detail["geomean_us"])
    assert metric != pytest.approx(TARGET_US)


def test_score_leaderboard_metric_uses_full_precision(tmp_path, monkeypatch):
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_OK, "leaderboard": LEADERBOARD}, seen)
    correct, metric, err, detail = popcorn.score(
        _problem(tmp_path, metric_mode="leaderboard"), tmp_path
    )
    assert (correct, err) == (True, None)
    assert seen == ["test", "leaderboard"]  # benchmark is skipped entirely
    assert metric == pytest.approx(LEADERBOARD_PUBLIC_S * 1e6)  # seconds -> microseconds
    assert detail["geomean_s"] == LEADERBOARD_PUBLIC_S


def test_failing_test_short_circuits_before_the_timing_run(tmp_path, monkeypatch):
    """A broken kernel must not cost a benchmark submission (11s vs 275s, measured)."""
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_FAIL}, seen)
    correct, metric, err, detail = popcorn.score(_problem(tmp_path), tmp_path)
    assert (correct, metric) == (False, None)
    assert seen == ["test"]
    assert "17/17 shapes failed" in err
    assert "not lower triangular" in err  # the agent gets the real failure text back
    assert detail["test"]["failed"] == 17


def test_failed_benchmark_recheck_is_not_correct(tmp_path, monkeypatch):
    """test can pass while a benchmark-scale re-check fails; correct is the conjunction."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_FAIL})
    correct, _metric, err, detail = popcorn.score(_problem(tmp_path), tmp_path)
    assert correct is False
    assert "failed re-check" in err
    assert detail["shapes"][1]["status"] == "fail"


def test_test_only_skips_the_timing_run(tmp_path, monkeypatch):
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_OK}, seen)
    correct, metric, err, _ = popcorn.score(_problem(tmp_path), tmp_path, test_only=True)
    assert (correct, metric, err) == (True, None, None)
    assert seen == ["test"]


def test_benchmark_index_out_of_range_is_an_error_not_a_crash(tmp_path, monkeypatch):
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    correct, metric, err, _ = popcorn.score(_problem(tmp_path, benchmark_index=99), tmp_path)
    assert correct is True and metric is None
    assert "out of range" in err


def test_precheck_rejects_locally_without_submitting(tmp_path, monkeypatch):
    seen: list[str] = []
    _fake_submit(monkeypatch, {}, seen)
    prob = _problem(tmp_path, reject_substrings=["stream"])
    (tmp_path / "submission.py").write_text("# uses a cuda stream\ndef custom_kernel(d): return d\n")
    correct, _metric, err, _ = popcorn.score(prob, tmp_path)
    assert correct is False and "stream" in err
    assert seen == []


def test_precheck_rejects_syntax_error(tmp_path, monkeypatch):
    seen: list[str] = []
    _fake_submit(monkeypatch, {}, seen)
    prob = _problem(tmp_path)
    (tmp_path / "submission.py").write_text("def custom_kernel(:\n")
    correct, _metric, err, _ = popcorn.score(prob, tmp_path)
    assert correct is False and "does not parse" in err
    assert seen == []


def test_missing_submission_file_is_reported(tmp_path, monkeypatch):
    _fake_submit(monkeypatch, {})
    prob = _problem(tmp_path, submission_file="nope.py")
    correct, _metric, err, _ = popcorn.score(prob, tmp_path)
    assert correct is False and "not readable" in err


# --- cache ------------------------------------------------------------------


def test_cache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELTHING_POPCORN_CACHE", str(tmp_path / "cache"))
    cfg = popcorn.resolve_config(_problem(tmp_path))
    sub = popcorn.Submission(mode="benchmark", text=BENCHMARK_OK, returncode=0, wall_s=275.0)
    popcorn._cache_store(cfg, "deadbeef", sub)
    got = popcorn._cache_load(cfg, "deadbeef", "benchmark")
    assert got is not None and got.cached is True
    assert got.text == BENCHMARK_OK and got.wall_s == 275.0
    assert popcorn._cache_load(cfg, "deadbeef", "test") is None  # keyed per mode


def test_cache_disabled_is_never_read(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELTHING_POPCORN_CACHE", str(tmp_path / "cache"))
    cfg = popcorn.resolve_config(_problem(tmp_path))
    popcorn._cache_store(
        cfg, "deadbeef", popcorn.Submission("benchmark", BENCHMARK_OK, 0, 1.0)
    )
    cfg.cache = False
    assert popcorn._cache_load(cfg, "deadbeef", "benchmark") is None


def test_transient_failure_is_not_cached(tmp_path, monkeypatch):
    """A nonzero exit means the job failed / timed out / the network broke.

    Memoising that against the file's hash would pin a transient error to a kernel
    forever -- every later score of the identical file would replay the failure, so one
    unlucky blip would permanently mark a good kernel broken.
    """
    monkeypatch.setenv("KERNELTHING_POPCORN_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(popcorn, "_attach_raw", lambda sub: None)
    cfg = popcorn.resolve_config(_problem(tmp_path))

    class R:
        stdout, stderr, returncode = "server exploded", "", 1

    monkeypatch.setattr(popcorn.subprocess, "run", lambda *a, **k: R())
    popcorn.submit(cfg, tmp_path / "submission.py", "benchmark", "beef")
    assert popcorn._cache_load(cfg, "beef", "benchmark") is None, "failure was cached"

    # a *correctness* failure exits 0 and is deterministic, so it IS cached
    class Ok:
        stdout, stderr, returncode = TEST_FAIL, "", 0

    monkeypatch.setattr(popcorn.subprocess, "run", lambda *a, **k: Ok())
    popcorn.submit(cfg, tmp_path / "submission.py", "test", "beef")
    assert popcorn._cache_load(cfg, "beef", "test") is not None


def test_cache_roundtrips_the_api_result(tmp_path, monkeypatch):
    """The cached entry must carry the raw dict, or a cache hit silently downgrades."""
    monkeypatch.setenv("KERNELTHING_POPCORN_CACHE", str(tmp_path / "cache"))
    cfg = popcorn.resolve_config(_problem(tmp_path))
    popcorn._cache_store(
        cfg,
        "abc",
        popcorn.Submission("benchmark", BENCHMARK_OK, 0, 1.0, raw=API_BENCHMARK, score=0.5),
    )
    got = popcorn._cache_load(cfg, "abc", "benchmark")
    assert got is not None and got.source == "api"
    assert got.score == 0.5
    assert popcorn.shapes_from_result(got.raw)[TARGET_INDEX].mean_us == pytest.approx(
        TARGET_NS / 1000.0
    )


def test_submit_reuses_the_cache_instead_of_shelling_out(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELTHING_POPCORN_CACHE", str(tmp_path / "cache"))
    cfg = popcorn.resolve_config(_problem(tmp_path))
    popcorn._cache_store(
        cfg, "cafe", popcorn.Submission("benchmark", BENCHMARK_OK, 0, 3.0)
    )

    def boom(*a, **k):  # pragma: no cover - fails the test if reached
        raise AssertionError("submit shelled out despite a warm cache")

    monkeypatch.setattr(popcorn.subprocess, "run", boom)
    sub = popcorn.submit(cfg, tmp_path / "submission.py", "benchmark", "cafe")
    assert sub.cached is True and sub.text == BENCHMARK_OK


# --- CLI verdict ------------------------------------------------------------


def test_score_command_emits_one_parseable_verdict(tmp_path, monkeypatch, capsys):
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})

    class Args:
        test_only = False
        emit_baseline = True

    rc = popcorn.score_command(_problem(tmp_path), Args())
    out = capsys.readouterr().out.strip().splitlines()
    verdict = json.loads(out[-1])
    assert rc == 0
    assert verdict["correct"] is True
    assert verdict["metric"] == pytest.approx(TARGET_US)
    assert verdict["unit"] == "us"
    # --emit-baseline is accepted so _evolve_seed works unchanged; there is no
    # denominator to pin for an absolute-time metric, so it reports none.
    assert verdict["baseline_median"] is None
    assert verdict["bench"]["backend"] == "popcorn"


def test_score_command_exit_code_is_nonzero_when_incorrect(tmp_path, monkeypatch, capsys):
    _fake_submit(monkeypatch, {"test": TEST_FAIL})

    class Args:
        test_only = False
        emit_baseline = False

    rc = popcorn.score_command(_problem(tmp_path), Args())
    verdict = json.loads(capsys.readouterr().out.strip().splitlines()[0])
    assert rc == 1
    assert verdict["correct"] is False and verdict["metric"] is None


# --- automatic profiling ----------------------------------------------------
#
# `ncu-details.txt` is a verbatim capture too: benchmark index 2 of the cholesky board,
# read back out of member 10's `opencode.ndjson` in the 2026-07-24 run. Re-capture rather
# than hand-edit, same as the `--output` fixtures above.

NCU_DETAILS = _fixture("ncu-details")


def _fake_profile(monkeypatch, prof: popcorn.Profile, seen: list | None = None, gate=None):
    """Replace the hosted profiler with a canned result.

    ``gate`` is set the moment the profile starts, so a test can prove the capture is
    already running while the timing submission is still in flight.
    """

    def fake(cfg, sub_path, dest, digest):
        if seen is not None:
            seen.append(digest)
        if gate is not None:
            gate.set()
        return prof

    monkeypatch.setattr(popcorn, "profile_submission", fake)


def _ok_profile(digest: str = "OPT   grid too small") -> popcorn.Profile:
    return popcorn.Profile(
        ok=True,
        details_path="/wt/profile/latest/ncu-details.txt",
        digest_path="/wt/profile/latest/digest.txt",
        digest=digest,
        wall_s=251.0,
    )


def test_digest_keeps_every_rule_finding_and_drops_the_bulk():
    """The findings name the bottleneck; the other ten sections are context nobody asked
    for, re-quoted on every score."""
    cfg = popcorn.PopcornConfig(leaderboard="cholesky")
    digest = popcorn.profile_digest(NCU_DETAILS, cfg)
    # every OPT/INF/WRN survives
    rules = len([ln for ln in NCU_DETAILS.splitlines() if popcorn.PROFILE_RULE_RE.match(ln)])
    assert rules == 16
    assert len([ln for ln in digest.splitlines() if popcorn.PROFILE_RULE_RE.match(ln)]) == rules
    # the launch header and its kernel signature (the only place the shape appears)
    assert "blocked_128_kernel" in digest and "(256, 1, 1)x(128, 1, 1)" in digest
    # kept sections, with their numbers
    assert "Compute (SM) Throughput" in digest and "16.47" in digest
    assert "Achieved Occupancy" in digest and "Registers Per Thread" in digest
    # dropped sections
    for gone in ("Section: PM Sampling", "Section: Instruction Statistics",
                 "Section: Memory Workload Analysis Tables"):
        assert gone not in digest
    assert len(digest) < len(NCU_DETAILS)


def test_digest_defuses_the_token_the_evaluator_rejects():
    """Nsight prints it in every kernel header. Quoting a capture into an agent's context
    verbatim hands it a string that silently poisons any file it is pasted into."""
    cfg = popcorn.PopcornConfig(leaderboard="cholesky", reject_substrings=["stream"])
    assert len(re.findall("stream", NCU_DETAILS, re.I)) == 1
    digest = popcorn.profile_digest(NCU_DETAILS, cfg)
    assert not re.search("stream", digest, re.I)
    assert "s-t-r-e-a-m" in digest


def test_digest_survives_output_it_cannot_parse():
    """A profiler that changes its format must cost a digest, not a score."""
    cfg = popcorn.PopcornConfig(leaderboard="cholesky")
    assert popcorn.profile_digest("", cfg) == ""
    assert popcorn.profile_digest("total gibberish\nno sections here\n", cfg) == ""


def test_profile_runs_concurrently_with_the_timing_submission(tmp_path, monkeypatch):
    """The whole point of firing at test-pass rather than after the score: the profile is
    the long pole (245s+) and the benchmark is ~172s, so they have to overlap."""
    started = threading.Event()

    def fake_submit(cfg, sub_path, mode, digest):
        if mode == "benchmark":
            # Serial ordering would deadlock here: the profile is only started after the
            # test submission returns, so if it waited on the benchmark this never fires.
            assert started.wait(timeout=10), "benchmark ran without the profile in flight"
        return popcorn.Submission(
            mode=mode, text={"test": TEST_OK, "benchmark": BENCHMARK_OK}[mode],
            returncode=0, wall_s=1.0,
        )

    monkeypatch.setattr(popcorn, "submit", fake_submit)
    _fake_profile(monkeypatch, _ok_profile(), gate=started)
    correct, metric, err, detail = popcorn.score(_problem(tmp_path, profile=True), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(75.1)
    assert detail["profile"]["ok"] is True


def test_a_broken_kernel_never_reaches_the_profiler_queue(tmp_path, monkeypatch):
    """A failed test returns before the capture starts -- the profiler is a single-worker
    queue, and a kernel that does not run has nothing worth measuring anyway."""
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_FAIL})
    _fake_profile(monkeypatch, _ok_profile(), seen=seen)
    correct, _metric, err, detail = popcorn.score(_problem(tmp_path, profile=True), tmp_path)
    assert correct is False and err
    assert seen == [] and "profile" not in detail


def test_test_only_stays_cheap_by_default(tmp_path, monkeypatch):
    """--test-only is the ~10s correctness call an agent makes most; a capture would make
    it ~25x slower and produces no timing to reason about. Explicit --profile overrides."""
    seen: list[str] = []
    _fake_submit(monkeypatch, {"test": TEST_OK})
    _fake_profile(monkeypatch, _ok_profile(), seen=seen)
    prob = _problem(tmp_path, profile=True)

    correct, _m, err, detail = popcorn.score(prob, tmp_path, test_only=True)
    assert (correct, err) == (True, None)
    assert seen == [] and "profile" not in detail

    correct, _m, err, detail = popcorn.score(prob, tmp_path, test_only=True, profile=True)
    assert (correct, err) == (True, None)
    assert len(seen) == 1 and detail["profile"]["ok"] is True


def test_a_failed_profile_never_changes_the_verdict(tmp_path, monkeypatch):
    """Evidence, not a verdict. Every failure path records itself and returns."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _fake_profile(monkeypatch, popcorn.Profile(ok=False, error="profile timed out after 1200s"))
    correct, metric, err, detail = popcorn.score(_problem(tmp_path, profile=True), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(75.1)
    assert detail["profile"] == {"ok": False, "wall_s": 0.0, "error": "profile timed out after 1200s"}


def test_the_verdict_carries_paths_not_capture_text(tmp_path, monkeypatch):
    """The JSON line is the seam. A 12KB digest crossing it would land in every
    result.json and every journal event, on every score, for no reader."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _fake_profile(monkeypatch, _ok_profile(digest="OPT   this must not appear in the JSON"))
    _c, _m, _e, detail = popcorn.score(_problem(tmp_path, profile=True), tmp_path)
    assert "this must not appear" not in json.dumps(detail)
    assert detail["profile"]["digest_path"].endswith("digest.txt")


def test_score_command_prints_the_profile_above_the_verdict(tmp_path, monkeypatch, capsys):
    """_cli_score finds the verdict by scanning stdout in reverse for the last line
    starting with '{'. Printing the capture first makes that impossible to break."""
    digest_file = tmp_path / "digest.txt"
    digest_file.write_text("OPT   {this line starts with a brace}\n", encoding="utf-8")
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    prof = _ok_profile()
    prof.digest_path = str(digest_file)
    _fake_profile(monkeypatch, prof)

    class Args:
        test_only = False
        emit_baseline = False
        profile = True

    rc = popcorn.score_command(_problem(tmp_path), Args())
    out = capsys.readouterr().out
    assert rc == 0
    verdict = json.loads(out.strip().splitlines()[-1])
    assert verdict["correct"] is True and verdict["metric"] == pytest.approx(TARGET_US)
    assert "Nsight Compute profile" in out
    assert "{this line starts with a brace}" in out
    assert out.index("Nsight Compute profile") < out.index('"correct"')


def test_score_command_says_so_when_the_capture_failed(tmp_path, monkeypatch, capsys):
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})
    _fake_profile(monkeypatch, popcorn.Profile(ok=False, error="popcorn CLI not found"))

    class Args:
        test_only = False
        emit_baseline = False
        profile = True

    assert popcorn.score_command(_problem(tmp_path), Args()) == 0
    out = capsys.readouterr().out
    assert "profile unavailable: popcorn CLI not found" in out
    assert json.loads(out.strip().splitlines()[-1])["correct"] is True


def test_an_exploding_profile_thread_never_fails_the_score(tmp_path, monkeypatch):
    """The join happens inside `finally`, so an escaped exception there would replace an
    already-computed verdict with a traceback."""
    _fake_submit(monkeypatch, {"test": TEST_OK, "benchmark": BENCHMARK_OK})

    def boom(cfg, sub_path, dest, digest):
        raise ZeroDivisionError("unexpected")

    monkeypatch.setattr(popcorn, "profile_submission", boom)
    correct, metric, err, detail = popcorn.score(_problem(tmp_path, profile=True), tmp_path)
    assert (correct, err) == (True, None)
    assert metric == pytest.approx(75.1)
    assert detail["profile"]["ok"] is False
    assert "ZeroDivisionError" in detail["profile"]["error"]
