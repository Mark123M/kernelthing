"""Exercise the opencode guard plugin's decide() against synthetic tool calls.

guard.js is JS, so these tests shell out to a JS runtime via tests/guard_driver.mjs.
They are skipped if none is available. The decision logic is the port of
Humanize's PreToolUse validators; see kernelthing/oc_guard/guard.js.

At runtime the guard needs no system runtime at all -- opencode loads the plugin with
its own embedded one -- so a box with no node still runs the guard correctly and only
loses this coverage. That asymmetry hides a broken guard, so the skip names the fix.
Set ``KERNELTHING_NODE`` to point at a runtime pytest cannot find on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "tests" / "guard_driver.mjs"
BLOCK_DIR = REPO / "prompts" / "block"


def _node_bin() -> str | None:
    """First usable JS runtime: an explicit override, then the usual names."""
    override = os.environ.get("KERNELTHING_NODE")
    if override:
        return override if Path(override).exists() or shutil.which(override) else None
    for name in ("node", "nodejs", "bun", "deno"):
        found = shutil.which(name)
        if found:
            return found
    return None


NODE = _node_bin()

pytestmark = pytest.mark.skipif(
    NODE is None,
    reason="no JS runtime (node/nodejs/bun/deno) -- guard decisions are UNTESTED. "
    "Install node (apt install nodejs) or set KERNELTHING_NODE=/path/to/node.",
)


def _run_cases(cases: list[dict]) -> list[dict]:
    assert NODE is not None
    proc = subprocess.run(
        [NODE, str(DRIVER)],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def _cfg(tmp_path: Path, *, phase: str = "impl", rnd: int = 2) -> dict:
    loop = tmp_path / ".humanize" / "rlcr" / "20240101-000000"
    loop.mkdir(parents=True, exist_ok=True)
    return {
        "loopDir": str(loop),
        "projectRoot": str(tmp_path),
        "planFile": "problems/x/plan.md",
        "currentRound": rnd,
        "phase": phase,
        "blockDir": str(BLOCK_DIR),
    }


def _decide(cfg: dict, tool: str, args: dict) -> dict:
    return _run_cases([{"cfg": cfg, "tool": tool, "args": args}])[0]


def _loop_path(cfg: dict, *parts: str) -> str:
    return str(Path(cfg["loopDir"], *parts))


def test_write_state_file_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "state.json"), "content": "{}"})
    assert r["blocked"] and "State File" in r["message"]


def test_write_run_files_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    for name in ("run.json", "events.ndjson", "control.json", "live.lock"):
        r = _decide(cfg, "write", {"filePath": _loop_path(cfg, name), "content": "{}"})
        assert r["blocked"], name


def test_write_members_store_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg, "write", {"filePath": _loop_path(cfg, "members", "3", "result.json"), "content": "{}"}
    )
    assert r["blocked"]


def test_bash_redirect_to_events_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "echo x >> .humanize/rlcr/x/events.ndjson"})
    assert r["blocked"]


def test_methodology_allows_member_records_read(tmp_path):
    cfg = _cfg(tmp_path, phase="methodology")
    for rel in (("members", "3", "summary.md"), ("events.ndjson",), ("loop.log",)):
        r = _decide(cfg, "read", {"filePath": _loop_path(cfg, *rel)})
        assert not r["blocked"], rel


def test_write_finalize_state_blocked(tmp_path):
    cfg = _cfg(tmp_path, phase="finalize")
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "finalize-state.json"), "content": "{}"})
    assert r["blocked"] and "Finalize" in r["message"] and "State File" in r["message"]


def test_write_todos_file_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "round-2-todos.md"), "content": "x"})
    assert r["blocked"] and "Todos File Access" in r["message"]


def test_bash_create_todos_file_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "echo hi > round-2-todos.md"})
    assert r["blocked"] and "Todos File Access" in r["message"]


def test_write_current_round_summary_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "round-2-summary.md"), "content": "x"})
    assert not r["blocked"]


def test_write_wrong_round_summary_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "round-1-summary.md"), "content": "x"})
    assert r["blocked"] and "Wrong Round Number" in r["message"]


def test_write_summary_outside_loop_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": str(tmp_path / "round-2-summary.md"), "content": "x"})
    assert r["blocked"] and "Wrong Summary Location" in r["message"]


def test_write_prompt_file_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "round-3-prompt.md"), "content": "x"})
    assert r["blocked"] and "Prompt File Write Blocked" in r["message"]


def test_write_plan_file_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": "problems/x/plan.md", "content": "x"})
    assert r["blocked"] and "Plan File" in r["message"]


def test_write_plan_backup_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "plan.md"), "content": "x"})
    assert r["blocked"] and "Plan Backup" in r["message"]


def test_write_normal_source_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "write", {"filePath": "src/kernel.cu", "content": "x"})
    assert not r["blocked"]


def test_bash_git_push_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "git push origin main"})
    assert r["blocked"] and "Git Push" in r["message"]


def test_bash_git_add_all_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "git add -A"})
    assert r["blocked"] and "Git Add" in r["message"]


def test_bash_redirect_to_summary_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "echo hi > round-2-summary.md"})
    assert r["blocked"]


def test_bash_sed_state_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "sed -i s/a/b/ .humanize/rlcr/x/state.json"})
    assert r["blocked"]


def test_bash_normal_command_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "echo hi > kernel.c && make"})
    assert not r["blocked"]


def test_read_wrong_round_in_loop_blocked_during_impl(tmp_path):
    cfg = _cfg(tmp_path)  # impl phase, current round 2
    r = _decide(cfg, "read", {"filePath": _loop_path(cfg, "round-1-summary.md")})
    assert r["blocked"] and "Wrong Round File" in r["message"]


def test_read_current_round_in_loop_allowed_during_impl(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "read", {"filePath": _loop_path(cfg, "round-2-summary.md")})
    assert not r["blocked"]


def test_read_prior_round_allowed_during_review(tmp_path):
    # reviewer must keep history access (@-referenced prior summaries)
    cfg = _cfg(tmp_path, phase="review")
    r = _decide(cfg, "read", {"filePath": _loop_path(cfg, "round-1-summary.md")})
    assert not r["blocked"]


def test_finalize_blocks_contract_write(tmp_path):
    cfg = _cfg(tmp_path, phase="finalize")
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "round-2-contract.md"), "content": "x"})
    assert r["blocked"] and "Finalize Contract Access" in r["message"]


def test_finalize_blocks_contract_read(tmp_path):
    cfg = _cfg(tmp_path, phase="finalize")
    r = _decide(cfg, "read", {"filePath": _loop_path(cfg, "round-2-contract.md")})
    assert r["blocked"] and "Finalize Contract Access" in r["message"]


def test_finalize_allows_summary_write(tmp_path):
    cfg = _cfg(tmp_path, phase="finalize")
    r = _decide(cfg, "write", {"filePath": _loop_path(cfg, "finalize-summary.md"), "content": "x"})
    assert not r["blocked"]


def test_read_round_file_outside_loop_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "read", {"filePath": str(tmp_path / "round-1-summary.md")})
    assert r["blocked"] and "Wrong File Location" in r["message"]


def test_methodology_allows_report(tmp_path):
    cfg = _cfg(tmp_path, phase="methodology")
    r = _decide(
        cfg,
        "write",
        {"filePath": _loop_path(cfg, "methodology-analysis-report.md"), "content": "x"},
    )
    assert not r["blocked"]


def test_methodology_blocks_source_write(tmp_path):
    cfg = _cfg(tmp_path, phase="methodology")
    r = _decide(cfg, "write", {"filePath": "src/kernel.cu", "content": "x"})
    assert r["blocked"]


def test_methodology_blocks_project_read(tmp_path):
    cfg = _cfg(tmp_path, phase="methodology")
    r = _decide(cfg, "read", {"filePath": str(tmp_path / "src" / "kernel.cu")})
    assert r["blocked"]


def test_no_config_allows_everything(tmp_path):
    r = _decide(None, "write", {"filePath": "anything", "content": "x"})
    assert not r["blocked"]


# --- env-tampering hardening (bash) -----------------------------------------
# Scoring is remote (no local GPU-lock shim), but the CUDA/loader environment and the
# guard's own internals stay off-limits as defensive hygiene. Running GPU tools directly
# is fine; what is blocked is any attempt to set/unset or inspect that environment.


def test_bash_ncu_runs_unwrapped(tmp_path):
    # Profiling tools run without a wrapper; only env tampering is blocked.
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "/usr/local/cuda/bin/ncu --set full ./bench"})
    assert not r["blocked"]


def test_bash_nsys_runs_unwrapped(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "nsys profile ./bench"})
    assert not r["blocked"]


def test_bash_bench_runs_unwrapped(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "python3 -c 'import torch; torch.randn(8).cuda()'"})
    assert not r["blocked"]


def test_bash_set_cuda_visible_devices_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "CUDA_VISIBLE_DEVICES=0 python3 train.py"})
    assert r["blocked"] and "gpu" in r["message"].lower()


def test_bash_unset_cuda_visible_devices_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "unset CUDA_VISIBLE_DEVICES && ./bench"})
    assert r["blocked"]


def test_bash_ld_preload_override_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "LD_PRELOAD= python3 bench.py"})
    assert r["blocked"]


def test_bash_read_shim_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "cat /x/libktgpu.so | strings"})
    assert r["blocked"]


def test_bash_read_guard_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "cat kernelthing/oc_guard/guard_core.js"})
    assert r["blocked"]


def test_bash_bare_printenv_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "printenv"})
    assert r["blocked"]


def test_bash_env_prefix_allowed(tmp_path):
    # `env FOO=bar cmd` is a normal invocation and must not be mistaken for a dump.
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "env OMP_NUM_THREADS=8 ./bench"})
    assert not r["blocked"]


def test_bash_ncu_report_parser_not_blocked(tmp_path):
    # The CPU-only report parser lives under ncu-report-skill/ -- pure CPU work.
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg,
        "bash",
        {"command": "python vendor/ncu-report-skill/helpers/analyze_reports.py --run-dir profile"},
    )
    assert not r["blocked"]


# --- remote (popcorn) submissions -------------------------------------------
# Remotely-scored problems submit to a public competition service. Correctness,
# timing and profiling runs are the candidate's to make; a *ranked* submission
# publishes to the board and draws a separate rate limit, so it stays a human call.


def test_bash_popcorn_leaderboard_submission_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg,
        "bash",
        {"command": "popcorn submit sub.py --leaderboard cholesky --gpu B200 --mode leaderboard"},
    )
    assert r["blocked"] and "leaderboard" in r["message"].lower()


def test_bash_popcorn_leaderboard_equals_form_blocked(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "popcorn-cli submit s.py --mode=leaderboard --no-tui"})
    assert r["blocked"]


def test_bash_popcorn_test_and_benchmark_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    for mode in ("test", "benchmark"):
        r = _decide(
            cfg,
            "bash",
            {"command": f"popcorn submit sub.py --leaderboard cholesky --mode {mode} --no-tui"},
        )
        assert not r["blocked"], mode


def test_bash_popcorn_profile_brev_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg,
        "bash",
        {"command": "popcorn submit ../submission.py --leaderboard cholesky --profile-brev "
                    "--benchmark-index 2 --no-tui --output brev.json"},
    )
    assert not r["blocked"]


def test_bash_popcorn_leaderboard_flag_alone_allowed(tmp_path):
    # `--leaderboard <name>` merely names the competition; only `--mode leaderboard`
    # is a ranked submission.
    cfg = _cfg(tmp_path)
    r = _decide(cfg, "bash", {"command": "popcorn submissions list --leaderboard cholesky"})
    assert not r["blocked"]


# --- veloq: reading a downloaded .ncu-rep is not a GPU operation ---------------
# The guard is a denylist, so `veloq` is allowed by construction. These pin that:
# a new binary in the profiling path must not trip gpu-tamper or the env-dump rule.


def test_bash_veloq_ncu_read_allowed(tmp_path):
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg,
        "bash",
        {"command": "veloq ncu inspect profile.0-batch-4096/profile.ncu-rep --row-id launch:2"},
    )
    assert r is None or not r.get("blocked")


def test_bash_veloq_python_pin_allowed(tmp_path):
    # The loop pins VELOQ_PYTHON itself; an agent echoing it back in a command is
    # not env tampering. Only CUDA_VISIBLE_DEVICES / LD_PRELOAD / KERNELTHING_* are.
    cfg = _cfg(tmp_path)
    r = _decide(
        cfg,
        "bash",
        {"command": "VELOQ_PYTHON=/home/u/.local/share/veloq/ncu-report-2026.2/bin/python3 "
                    "veloq ncu warp-stalls r.ncu-rep --row-id launch:0 --by reason"},
    )
    assert r is None or not r.get("blocked")
