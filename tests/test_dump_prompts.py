from __future__ import annotations

import json
import subprocess
from pathlib import Path

from kernelthing import cli, dump_prompts, evolve, opencode_client
from kernelthing.config import Config
from kernelthing.problem import Problem, load_problem

MANIFEST = {
    "name": "toy",
    "plan": "plan.md",
    "edit_files": ["submission.py"],
    "metric_name": "latency",
    "unit": "us",
    "direction": "minimize",
    "bench": {
        "backend": "popcorn",
        "popcorn": {
            "leaderboard": "toy",
            "submission_file": "submission.py",
            "benchmark_index": 0,
        },
    },
}


def make_problem(tmp_path: Path) -> Problem:
    repo = tmp_path / "repo"
    prob = repo / "problems" / "toy"
    prob.mkdir(parents=True)
    (prob / "problem.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    (prob / "plan.md").write_text("# Plan\n\nUse {{PLAN_DETAIL}} when it appears.\n")
    (prob / "submission.py").write_text("def kernel():\n    return None\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    return load_problem(prob / "problem.json")


def test_render_candidate_prompt_preserves_placeholders() -> None:
    assert Config().model == dump_prompts.DEFAULT_MODEL

    prompt = dump_prompts.render_candidate_prompt(Config(), evolve.OP_EXPLORE)
    for placeholder in (
        "{{PARENT_METRIC}}",
        "{{PARENT_COMMIT_MESSAGE}}",
        "{{KNOWN_STRATEGIES}}",
        "{{EDIT_FILES}}",
        "{{SCORE_CMD}}",
        "{{UNIT}}",
        "{{SUBMISSION_FILE}}",
    ):
        assert placeholder in prompt


def test_build_opencode_dump_env_appends_dump_plugin_after_guard(tmp_path: Path) -> None:
    dump_plugin = tmp_path / "dump.mjs"
    capture_file = tmp_path / "capture.json"
    env, _state_dirs, oc_config = dump_prompts.build_opencode_dump_env(
        data_dir=tmp_path / "data",
        guard={
            "loopDir": str(tmp_path / "loop"),
            "projectRoot": str(tmp_path),
            "planFile": "plan.md",
            "currentRound": 1,
            "phase": "impl",
            "editFiles": ["submission.py"],
            "editDir": str(tmp_path),
            "protectedFiles": [],
        },
        dump_plugin=dump_plugin,
        capture_file=capture_file,
        mcp_cuda_docs=True,
    )

    assert oc_config["plugin"] == [str(opencode_client.GUARD_PLUGIN), str(dump_plugin)]
    assert oc_config["mcp"]["nvidia-cuda-docs"]["enabled"] is True
    assert env["KERNELTHING_PROMPT_DUMP_PATH"] == str(capture_file)
    assert env["KERNELTHING_PROMPT_DUMP_SENTINEL"] == dump_prompts.SENTINEL
    assert json.loads(env["OPENCODE_CONFIG_CONTENT"]) == oc_config


def test_normalize_json_replaces_paths_and_today() -> None:
    replacements = {
        "/tmp/kernelthing-dump-prompts-123/worktree": "{{WORKTREE}}",
        "/tmp/kernelthing-dump-prompts-123/worktree/.humanize/rlcr/prompt-dump": "{{LOOP_DIR}}",
    }
    normalized = dump_prompts.normalize_json(
        {
            "text": (
                "Working directory: /tmp/kernelthing-dump-prompts-123/worktree\n"
                "Loop: /tmp/kernelthing-dump-prompts-123/worktree/.humanize/rlcr/prompt-dump\n"
                "Today's date: Sunday, July 26, 2026\n"
                "Keep {{PLAN}} untouched."
            )
        },
        replacements,
    )

    assert normalized["text"] == (
        "Working directory: {{WORKTREE}}\n"
        "Loop: {{LOOP_DIR}}\n"
        "Today's date: {{TODAY}}\n"
        "Keep {{PLAN}} untouched."
    )


def test_dump_prompts_cli_writes_expected_files(tmp_path: Path, monkeypatch) -> None:
    problem = make_problem(tmp_path)
    out = tmp_path / "dump"

    def fake_capture(
        prompt_text: str,
        worktree: Path,
        loop_dir: Path,
        data_dir: Path,
        cfg: Config,
        operator: str,
    ) -> dict:
        assert operator == evolve.OP_EXPLORE
        assert cfg.model == "test/model"
        assert data_dir.name == "dump-explore"
        assert "{{PARENT_METRIC}}" in prompt_text
        assert "{{SCORE_CMD}}" in prompt_text
        return {
            "completed": True,
            "system_transforms": [
                {
                    "output": {
                        "system": [
                            f"OpenCode system from {worktree}\n"
                            "Today's date: Sunday, July 26, 2026"
                        ]
                    }
                }
            ],
            "messages_transforms": [
                {
                    "output": {
                        "messages": [
                            {
                                "role": "user",
                                "text": f"{prompt_text}\nLoop lives at {loop_dir}",
                            }
                        ]
                    }
                }
            ],
            "tool_definitions": [
                {
                    "input": {"toolID": "bash"},
                    "output": {
                        "description": f"Run shell in {worktree}",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "chat_params": {"input": {"agent": "build"}, "output": {"temperature": 0}},
        }

    monkeypatch.setattr(dump_prompts, "capture_opencode_stack", fake_capture)
    monkeypatch.setattr(
        dump_prompts,
        "opencode_info",
        lambda: {"path": "/tmp/opencode", "version": "1.test"},
    )

    rc = cli.main(
        [
            "dump-prompts",
            str(problem.dir),
            "-o",
            str(out),
            "--operator",
            "explore",
            "--model",
            "test/model",
        ]
    )

    assert rc == 0
    assert (out / "candidate-explore.md").is_file()
    assert not (out / "candidate-exploit.md").exists()
    assert (out / "opencode-explore.json").is_file()
    assert (out / "opencode-explore.md").is_file()
    assert (out / "guard-blocks.md").is_file()
    assert (out / "manifest.json").is_file()

    candidate = (out / "candidate-explore.md").read_text(encoding="utf-8")
    assert "{{PARENT_METRIC}}" in candidate
    assert "{{KNOWN_STRATEGIES}}" in candidate
    assert "{{SCORE_CMD}}" in candidate

    readable = (out / "opencode-explore.md").read_text(encoding="utf-8")
    assert "{{WORKTREE}}" in readable
    assert "{{LOOP_DIR}}" in readable
    assert "{{TODAY}}" in readable

    guard_blocks = (out / "guard-blocks.md").read_text(encoding="utf-8")
    assert "# Edit Blocked" in guard_blocks
    assert "{{EDIT_FILES}}" in guard_blocks

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "test/model"
    assert manifest["operators"] == ["explore"]
    assert manifest["opencode"]["version"] == "1.test"
    assert manifest["flags"]["mcp_cuda_docs"] is True
