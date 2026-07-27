"""Dump the prompt stack a candidate opencode turn sees.

This is an inspection command only. It renders kernelthing's candidate prompts
with volatile fields left as ``{{PLACEHOLDER}}`` values, then asks opencode to
compose its own system/messages/tools stack under a temporary plugin that aborts
before the provider call.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import evolve, opencode_client, popcorn, prompts
from .bootstrap import protected_files
from .config import REPO_ROOT, Config
from .orchestrator import (
    EVOLVE_DESCRIPTOR_FOOTER,
    EVOLVE_EXPLOIT_PROMPT,
    EVOLVE_EXPLORE_PROMPT,
    Orchestrator,
)
from .problem import NO_COPY, Problem, load_problem

SENTINEL = "KERNELTHING_PROMPT_DUMP_COMPLETE"
DEFAULT_OUT = Path("prompt-dump")
DEFAULT_MODEL = Config().model

JSONDict = dict[str, Any]
CaptureFunc = Callable[[str, Path, Path, Path, Config, str], JSONDict]


@dataclass(frozen=True)
class DumpScratch:
    """Temporary candidate-like problem checkout."""

    problem: Problem
    worktree: Path
    loop_dir: Path


def command(argv: list[str]) -> int:
    """``kernelthing dump-prompts`` CLI entry."""

    parser = argparse.ArgumentParser(
        prog="kernelthing dump-prompts",
        description="Dump candidate prompt context for explore/exploit turns without "
        "calling the model provider.",
    )
    parser.add_argument("problem", help="problem dir or problem.json to inspect")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output directory (default: ./{DEFAULT_OUT})",
    )
    parser.add_argument(
        "--operator",
        choices=("both", evolve.OP_EXPLORE, evolve.OP_EXPLOIT),
        default="both",
        help="candidate operator to dump (default: both)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="opencode model id to compose the prompt for (default: %(default)s)",
    )
    parser.add_argument("--no-ncu", action="store_true", help="omit the ncu analysis-skill note")
    parser.add_argument("--no-veloq", action="store_true", help="omit the veloq block")
    parser.add_argument("--no-ptx", action="store_true", help="omit the PTX reference block")
    parser.add_argument("--no-cuda-docs", action="store_true", help="omit CUDA-docs MCP guidance")
    args = parser.parse_args(argv)

    try:
        problem = load_problem(Path(args.problem))
        cfg = Config(
            model=args.model,
            ncu=not args.no_ncu,
            veloq=not args.no_veloq,
            ptx=not args.no_ptx,
            mcp_cuda_docs=not args.no_cuda_docs,
        )
        operators = operators_for(args.operator)
        dump_candidate_prompts(problem, cfg, args.out, operators=operators)
    except (FileNotFoundError, RuntimeError, OSError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"[kernelthing] prompt dump -> {args.out}", file=sys.stderr)
    return 0


def operators_for(choice: str) -> list[str]:
    if choice == "both":
        return [evolve.OP_EXPLORE, evolve.OP_EXPLOIT]
    return [choice]


def dump_candidate_prompts(
    problem: Problem,
    cfg: Config,
    out_dir: Path,
    *,
    operators: list[str],
    capture_func: CaptureFunc | None = None,
) -> list[Path]:
    """Write the prompt dump files and return the paths written."""

    capture = capture_func or capture_opencode_stack
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="kernelthing-dump-prompts-") as td:
        scratch = create_scratch_problem(problem, Path(td) / "worktree")
        normalizers = base_normalizers(scratch)

        guard_blocks = out_dir / "guard-blocks.md"
        guard_blocks.write_text(render_guard_blocks(), encoding="utf-8")
        written.append(guard_blocks)

        for operator in operators:
            prompt_text = render_candidate_prompt(cfg, operator)
            prompt_path = out_dir / f"candidate-{operator}.md"
            prompt_path.write_text(prompt_text, encoding="utf-8")
            written.append(prompt_path)

            data_dir = scratch.worktree / ".humanize" / "oc-data" / f"dump-{operator}"
            raw_capture = capture(
                prompt_text,
                scratch.worktree,
                scratch.loop_dir,
                data_dir,
                cfg,
                operator,
            )
            normalized = normalize_json(raw_capture, normalizers)
            json_path = out_dir / f"opencode-{operator}.json"
            json_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")
            written.append(json_path)

            md_path = out_dir / f"opencode-{operator}.md"
            md_path.write_text(render_capture_markdown(operator, normalized), encoding="utf-8")
            written.append(md_path)

    manifest_path = out_dir / "manifest.json"
    manifest = build_manifest(problem, cfg, operators)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    written.append(manifest_path)
    return written


def create_scratch_problem(problem: Problem, worktree: Path) -> DumpScratch:
    """Copy the problem into a temporary standalone git repo."""

    worktree.mkdir(parents=True, exist_ok=True)
    for item in problem.dir.iterdir():
        if item.name in NO_COPY:
            continue
        dest = worktree / item.name
        if item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dest)
    subprocess.run(["git", "init", "-b", "main"], cwd=worktree, check=True, capture_output=True)
    loop_dir = worktree / ".humanize" / "rlcr" / "prompt-dump"
    loop_dir.mkdir(parents=True, exist_ok=True)
    return DumpScratch(load_problem(worktree / "problem.json"), worktree, loop_dir)


def render_candidate_prompt(cfg: Config, operator: str) -> str:
    """Render one candidate prompt with volatile values left as placeholders."""

    common = {
        "PLAN": "{{PLAN}}",
        "PLAN_FILE": "{{PLAN_FILE}}",
        "EDIT_FILES": "{{EDIT_FILES}}",
        "SCORE_CMD": "{{SCORE_CMD}}",
        "UNIT": "{{UNIT}}",
    }
    if operator == evolve.OP_EXPLORE:
        body = prompts.render(
            EVOLVE_EXPLORE_PROMPT,
            PARENT_METRIC="{{PARENT_METRIC}}",
            PARENT_COMMIT_MESSAGE="{{PARENT_COMMIT_MESSAGE}}",
            KNOWN_STRATEGIES="{{KNOWN_STRATEGIES}}",
            **common,
        )
    elif operator == evolve.OP_EXPLOIT:
        body = prompts.render(
            EVOLVE_EXPLOIT_PROMPT,
            PARENT_METRIC="{{PARENT_METRIC}}",
            PARENT_COMMIT_MESSAGE="{{PARENT_COMMIT_MESSAGE}}",
            **common,
        )
    else:
        raise ValueError(f"unknown operator: {operator}")
    return body + prompts.render(EVOLVE_DESCRIPTOR_FOOTER, **common) + render_kernel_tools(cfg)


def render_kernel_tools(cfg: Config) -> str:
    """Render kernel tooling blocks for dump mode, assuming enabled tools work."""

    parts: list[str] = []

    def skill(title: str, note: str, tree: str, placeholder: str) -> str:
        # Same headed-section template the orchestrator uses, with the absolute vendor
        # path swapped back out for its placeholder -- a dump is of the configuration, so
        # nothing in it should be specific to this checkout's location.
        return Orchestrator._skill_part(
            title, note.replace(str(REPO_ROOT / "vendor" / tree), placeholder)
        )

    # Band order mirrors _kernel_tools_block exactly (turn loop -> reference -> commands);
    # a dump that reordered them would be a dump of a prompt no candidate receives.
    parts.append(
        prompts.load_and_render_safe(
            "claude/kernel-tools-profile.md",
            "",
            SUBMISSION_FILE="{{SUBMISSION_FILE}}",
            SCORE_CMD="{{SCORE_CMD}}",
        )
    )

    if cfg.veloq:
        parts.append(
            skill(
                "Diagnosing an ncu report — `ncu-profile-analysis`",
                Orchestrator._veloq_ref_note(REPO_ROOT / "vendor" / "veloq-ncu-skill"),
                "veloq-ncu-skill",
                "{{VELOQ_SKILL_DIR}}",
            )
        )

    if cfg.ncu:
        parts.append(
            skill(
                "B200 profiling reference — `ncu-report-skill`",
                Orchestrator._ncu_skill_note(REPO_ROOT / "vendor" / "ncu-report-skill"),
                "ncu-report-skill",
                "{{NCU_SKILL_DIR}}",
            )
        )

    # Dump mode assumes every enabled tool works, so unlike the orchestrator the nsys
    # sections are not also gated on the problem's bench.popcorn.nsys -- a dump is of the
    # configuration, not of one problem's capture plan.
    if cfg.veloq:
        parts.append(
            skill(
                "Reading an nsys timeline — `nsys-profile-analysis`",
                Orchestrator._veloq_ref_note(REPO_ROOT / "vendor" / "veloq-nsys-skill"),
                "veloq-nsys-skill",
                "{{NSYS_SKILL_DIR}}",
            )
        )

    if cfg.mcp_cuda_docs:
        parts.append(
            prompts.load_and_render_safe(
                "claude/kernel-tools-cuda-docs.md",
                "",
                MCP_SERVER=opencode_client.CUDA_DOCS_MCP_SERVER,
                MCP_TOOL=opencode_client.CUDA_DOCS_MCP_TOOL,
                MCP_DESCRIPTION=opencode_client.CUDA_DOCS_MCP_INSTRUCTIONS,
            )
        )

    parts.append(
        skill(
            "PTX / CUDA ISA reference — `ptx-skill`",
            Orchestrator._ptx_note(
                _PlaceholderOrchestrator(cfg), REPO_ROOT / "vendor" / "ptx-skill"
            ),
            "ptx-skill",
            "{{PTX_SKILL_DIR}}",
        )
    )

    if cfg.veloq:
        parts.append(
            prompts.load_and_render_safe(
                "claude/kernel-tools-veloq.md",
                "",
                VELOQ_BIN="{{VELOQ_BIN}}",
                REPORT="{{REPORT}}",
                NCU_VERBS=popcorn.veloq_verb_block("ncu"),
                SUBMISSION_FILE="{{SUBMISSION_FILE}}",
            )
        )
        parts.append(
            prompts.load_and_render_safe(
                "claude/kernel-tools-nsys.md",
                "",
                VELOQ_BIN="{{VELOQ_BIN}}",
                NSYS_REPORT="{{NSYS_REPORT}}",
                NSYS_VERBS=popcorn.veloq_verb_block("nsys"),
            )
        )
    return Orchestrator._tools_section(parts)


class _PlaceholderOrchestrator:
    """Small adapter for calling Orchestrator note helpers in dump mode."""

    def __init__(self, cfg: Config):
        self.cfg = cfg


def render_guard_blocks() -> str:
    blocks: list[str] = ["# Guard block-message prompts\n"]
    for path in sorted((REPO_ROOT / "prompts" / "block").glob("*.md")):
        rel = path.relative_to(REPO_ROOT)
        blocks.append(f"\n## {rel}\n")
        blocks.append(path.read_text(encoding="utf-8"))
        if not blocks[-1].endswith("\n"):
            blocks.append("\n")
    return "".join(blocks)


def build_opencode_dump_env(
    *,
    data_dir: Path,
    guard: JSONDict,
    dump_plugin: Path,
    capture_file: Path,
    mcp_cuda_docs: bool,
) -> tuple[dict[str, Any], list[Path], JSONDict]:
    """Build opencode env/config with the dump plugin appended last."""

    env, oc_state = opencode_client.build_opencode_env(
        data_dir=data_dir,
        guard=guard,
        mcp_cuda_docs=mcp_cuda_docs,
    )
    oc_config = json.loads(str(env["OPENCODE_CONFIG_CONTENT"]))
    plugins = list(oc_config.get("plugin") or [])
    plugins.append(str(dump_plugin))
    oc_config["plugin"] = plugins
    env["OPENCODE_CONFIG_CONTENT"] = json.dumps(oc_config)
    env["KERNELTHING_PROMPT_DUMP_PATH"] = str(capture_file)
    env["KERNELTHING_PROMPT_DUMP_SENTINEL"] = SENTINEL
    return env, oc_state, oc_config


def capture_opencode_stack(
    prompt_text: str,
    worktree: Path,
    loop_dir: Path,
    data_dir: Path,
    cfg: Config,
    operator: str,
) -> JSONDict:
    """Run opencode until the dump plugin aborts before the provider call."""

    opencode = shutil.which("opencode")
    if opencode is None:
        raise FileNotFoundError("opencode not found on PATH")

    data_dir.mkdir(parents=True, exist_ok=True)
    plugin = data_dir / "kernelthing_prompt_dump_plugin.mjs"
    capture_file = data_dir / f"{operator}.json"
    plugin.write_text(DUMP_PLUGIN_JS, encoding="utf-8")
    guard = guard_config(worktree, loop_dir, operator)
    env, _oc_state, _oc_config = build_opencode_dump_env(
        data_dir=data_dir,
        guard=guard,
        dump_plugin=plugin,
        capture_file=capture_file,
        mcp_cuda_docs=cfg.mcp_cuda_docs,
    )
    cmd = [
        opencode,
        "run",
        "--format",
        "json",
        "-m",
        cfg.model,
        "--auto",
        "--dir",
        str(worktree.resolve()),
    ]
    proc = subprocess.run(
        cmd,
        input=prompt_text,
        cwd=worktree,
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    if not capture_file.is_file():
        raise RuntimeError(
            "opencode prompt dump failed before capture "
            f"(exit {proc.returncode}): {(proc.stderr or proc.stdout).strip()[-2000:]}"
        )
    data = json.loads(capture_file.read_text(encoding="utf-8"))
    data["opencode_exit"] = proc.returncode
    data["opencode_stdout_tail"] = (proc.stdout or "").strip()[-2000:]
    data["opencode_stderr_tail"] = (proc.stderr or "").strip()[-2000:]
    if not data.get("completed"):
        tails = " ".join(
            part
            for part in (
                data["opencode_stderr_tail"],
                data["opencode_stdout_tail"],
            )
            if part
        )
        detail = f": {tails}" if tails else ""
        raise RuntimeError(
            "opencode prompt dump captured partial data but did not reach chat.params: "
            f"{data.get('last_hook', 'unknown')}{detail}"
        )
    return data


def guard_config(worktree: Path, loop_dir: Path, operator: str) -> JSONDict:
    problem = load_problem(worktree / "problem.json")
    return {
        "loopDir": str(loop_dir.resolve()),
        "projectRoot": str(worktree.resolve()),
        "planFile": problem.plan,
        "currentRound": 1 if operator == evolve.OP_EXPLORE else 2,
        "phase": "impl",
        "editFiles": [str(Path(f)) for f in problem.edit_files],
        "editDir": str(worktree.resolve()),
        "protectedFiles": sorted(protected_files(problem)),
    }


def base_normalizers(scratch: DumpScratch) -> dict[str, str]:
    data_root = scratch.worktree / ".humanize" / "oc-data"
    return {
        str(scratch.worktree.resolve()): "{{WORKTREE}}",
        str(scratch.loop_dir.resolve()): "{{LOOP_DIR}}",
        str(data_root.resolve()): "{{OPENCODE_DATA_DIR}}",
    }


def normalize_json(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return normalize_text(value, replacements)
    if isinstance(value, list):
        return [normalize_json(v, replacements) for v in value]
    if isinstance(value, dict):
        return {str(k): normalize_json(v, replacements) for k, v in value.items()}
    return value


def normalize_text(text: str, replacements: dict[str, str]) -> str:
    out = text
    for src, dst in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        out = out.replace(src, dst)
    out = re.sub(r"(Today's date:\s*)[^\n<]+", r"\1{{TODAY}}", out)
    return out


def render_capture_markdown(operator: str, capture: JSONDict) -> str:
    out = [f"# OpenCode prompt stack: {operator}\n"]
    out.append("\n## System Prompt\n")
    systems = capture.get("system_transforms") or []
    if systems:
        system = ((systems[-1].get("output") or {}).get("system")) or []
        for idx, block in enumerate(system, 1):
            out.append(f"\n### System block {idx}\n")
            out.append(_fence(str(block), "text"))
    else:
        out.append("\n(no system transform captured)\n")

    out.append("\n## Messages\n")
    messages = capture.get("messages_transforms") or []
    if messages:
        msg_payload = ((messages[-1].get("output") or {}).get("messages")) or []
        out.append(_fence(json.dumps(msg_payload, indent=2, sort_keys=True), "json"))
    else:
        out.append("\n(no messages transform captured)\n")

    out.append("\n## Tool Definitions\n")
    tools = capture.get("tool_definitions") or []
    if tools:
        for item in tools:
            tool_id = (item.get("input") or {}).get("toolID", "tool")
            output = item.get("output") or {}
            out.append(f"\n### {tool_id}\n")
            desc = output.get("description")
            if desc:
                out.append(str(desc).rstrip() + "\n")
            params = output.get("parameters")
            if params is not None:
                out.append(_fence(json.dumps(params, indent=2, sort_keys=True), "json"))
    else:
        out.append("\n(no tool definitions captured)\n")

    out.append("\n## Chat Params\n")
    out.append(_fence(json.dumps(capture.get("chat_params"), indent=2, sort_keys=True), "json"))
    return "".join(out)


def _fence(text: str, info: str) -> str:
    return f"\n```{info}\n{text.rstrip()}\n```\n"


def build_manifest(problem: Problem, cfg: Config, operators: list[str]) -> JSONDict:
    return {
        "problem": {
            "name": problem.name,
            "source_dir": str(problem.dir),
            "plan": problem.plan,
            "edit_files": problem.edit_files,
        },
        "model": cfg.model,
        "operators": operators,
        "flags": {
            "ncu": cfg.ncu,
            "veloq": cfg.veloq,
            "ptx": cfg.ptx,
            "mcp_cuda_docs": cfg.mcp_cuda_docs,
        },
        "opencode": opencode_info(),
        "sources": source_inventory(cfg),
    }


def opencode_info() -> JSONDict:
    opencode = shutil.which("opencode") or ""
    version = ""
    if opencode:
        try:
            proc = subprocess.run(
                [opencode, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            version = proc.stdout.strip() if proc.returncode == 0 else ""
        except (OSError, subprocess.TimeoutExpired):
            version = ""
    return {"path": opencode, "version": version}


def source_inventory(cfg: Config) -> JSONDict:
    prompt_files = [
        "kernelthing/orchestrator.py:EVOLVE_EXPLORE_PROMPT",
        "kernelthing/orchestrator.py:EVOLVE_EXPLOIT_PROMPT",
        "kernelthing/orchestrator.py:EVOLVE_DESCRIPTOR_FOOTER",
    ]
    prompt_files.append("prompts/claude/kernel-tools-profile.md")
    if cfg.veloq:
        prompt_files.append("prompts/claude/kernel-tools-veloq.md")
        prompt_files.append("prompts/claude/kernel-tools-nsys.md")
    # One template, rendered once per vendored skill (ncu-profile-analysis,
    # ncu-report-skill, nsys-profile-analysis, ptx-skill) so the sections cannot drift.
    if cfg.veloq or cfg.ncu or cfg.ptx:
        prompt_files.append("prompts/claude/kernel-tools-skill.md")
    if cfg.mcp_cuda_docs:
        prompt_files.append("prompts/claude/kernel-tools-cuda-docs.md")
    return {
        "candidate_prompt_sources": prompt_files,
        "opencode_runtime_sources": [
            "OpenCode runtime system prompt via experimental.chat.system.transform",
            "OpenCode message payloads via experimental.chat.messages.transform",
            "OpenCode tool schemas/descriptions via tool.definition",
            "OpenCode chat params via chat.params",
        ],
        "opencode_config_source": "kernelthing/opencode_client.py:build_opencode_env",
        "guard_plugin": str(opencode_client.GUARD_PLUGIN),
        "guard_block_dir": str(opencode_client.GUARD_BLOCK_DIR),
        "guard_block_templates": [
            str(path.relative_to(REPO_ROOT))
            for path in sorted((REPO_ROOT / "prompts" / "block").glob("*.md"))
        ],
        "dump_plugin": "kernelthing/dump_prompts.py:DUMP_PLUGIN_JS",
        "mcp": {
            "nvidia_cuda_docs": {
                "enabled": cfg.mcp_cuda_docs,
                "server": opencode_client.CUDA_DOCS_MCP_SERVER,
                "tool": opencode_client.CUDA_DOCS_MCP_TOOL,
                "url": opencode_client.CUDA_DOCS_MCP_URL,
                "instructions": opencode_client.CUDA_DOCS_MCP_INSTRUCTIONS,
            }
        },
    }


DUMP_PLUGIN_JS = r"""
import fs from "node:fs";
import path from "node:path";

const capturePath = process.env.KERNELTHING_PROMPT_DUMP_PATH;
const sentinel = process.env.KERNELTHING_PROMPT_DUMP_SENTINEL || "KERNELTHING_PROMPT_DUMP_COMPLETE";
const state = {
  completed: false,
  last_hook: "init",
  config: null,
  chat_messages: [],
  system_transforms: [],
  messages_transforms: [],
  tool_definitions: [],
  chat_params: null,
};

function clone(value) {
  const seen = new WeakSet();
  return JSON.parse(JSON.stringify(value, (_key, val) => {
    if (typeof val === "bigint") return val.toString();
    if (typeof val === "function") return `[Function ${val.name || "anonymous"}]`;
    if (val && typeof val === "object") {
      if (seen.has(val)) return "[Circular]";
      seen.add(val);
    }
    return val;
  }));
}

function writeCapture(hook) {
  state.last_hook = hook;
  if (!capturePath) return;
  fs.mkdirSync(path.dirname(capturePath), { recursive: true });
  fs.writeFileSync(capturePath, JSON.stringify(state, null, 2));
}

export const KernelthingPromptDump = async () => {
  return {
    config: async (cfg) => {
      state.config = clone(cfg);
      writeCapture("config");
    },
    "chat.message": async (input, output) => {
      state.chat_messages.push({ input: clone(input), output: clone(output) });
      writeCapture("chat.message");
    },
    "experimental.chat.system.transform": async (input, output) => {
      state.system_transforms.push({ input: clone(input), output: clone(output) });
      writeCapture("experimental.chat.system.transform");
    },
    "experimental.chat.messages.transform": async (input, output) => {
      state.messages_transforms.push({ input: clone(input), output: clone(output) });
      writeCapture("experimental.chat.messages.transform");
    },
    "tool.definition": async (input, output) => {
      state.tool_definitions.push({ input: clone(input), output: clone(output) });
      writeCapture("tool.definition");
    },
    "chat.params": async (input, output) => {
      state.chat_params = { input: clone(input), output: clone(output) };
      state.completed = true;
      writeCapture("chat.params");
      throw new Error(sentinel);
    },
  };
};
"""
