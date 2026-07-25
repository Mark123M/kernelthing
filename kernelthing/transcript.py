"""Flatten a run's agent NDJSON logs into readable transcripts.

``members/<id>/opencode.ndjson`` is opencode's raw ``--format json`` event stream,
written straight to the file as the turn happens. It is the *complete* record of
one agent -- assistant prose, reasoning, and every tool call with its full input
and output -- but it is one JSON object per part, so reading a 750KB turn by eye
means writing a parser first. Everyone who wants to know what an agent actually
did ends up writing that parser: the web UI has one (``webui.transcript_items``),
and it clips output and only serves the tail because it feeds a browser pane.

This module is the same fold with nothing dropped, aimed at a file instead of a
socket: one markdown file per member, prompt first, then the stream in order. It
is a pure reader of the run dir like everything else on that side of the fence
(see webui's docstring) -- it opens no repo, spawns no process, and works exactly
the same on a live run, a finished one in the managed root, and an archived copy.

The parsing helpers live here rather than in ``webui`` because they describe
opencode's wire format, not HTTP; webui imports them back.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Run-level files copied beside the rendered members so the output dir stands on
# its own -- the controller narrative and the journal explain *why* an agent was
# spawned with the prompt it got.
RUN_FILES = ("run.json", "loop.log", "events.ndjson", "methodology-opencode.log")


def is_tool(d: dict[str, Any], part: dict[str, Any]) -> bool:
    return d["type"] in ("tool", "tool_use") or (
        "type" in part and part["type"] in ("tool", "tool-invocation")
    )


def tool_line(part: dict[str, Any]) -> str:
    """One readable line for a tool event: name + its salient argument.

    opencode nests the call args under ``part.state.input`` (e.g. a read tool is
    ``{"tool":"read","state":{"input":{"filePath":...}}}``); only some shapes put
    them at ``part.input``. Check both, else the line is just the bare tool name."""
    name = part.get("tool") or part.get("name", "tool")
    inp = tool_input(part)
    arg = ""
    for key in (
        "command",
        "filePath",
        "file_path",
        "path",
        "pattern",
        "url",
        "query",
        "description",
        "prompt",
    ):
        if inp.get(key):
            arg = inp[key]
            break
    return " ".join((str(name) + " " + str(arg)).split())


def tool_input(part: dict[str, Any]) -> dict[str, Any]:
    """The call arguments, from either shape opencode emits."""
    state = part["state"] if isinstance(part.get("state"), dict) else {}
    if isinstance(state.get("input"), dict):
        return dict(state["input"])
    if isinstance(part.get("input"), dict):
        return dict(part["input"])
    return {}


def clip(text: str, head: int = 16000, tail: int = 8000) -> str:
    """Clip huge tool output, keeping head and tail (errors usually sit at the end)."""
    if len(text) <= head + tail + 64:
        return text
    omitted = len(text) - head - tail
    return text[:head] + f"\n… [{omitted} chars omitted] …\n" + text[-tail:]


def parts(log: Path) -> Iterator[dict[str, Any]]:
    """Yield one normalised item per NDJSON part, in stream order.

    Kinds: ``text`` (assistant prose), ``think`` (reasoning), ``tool`` (with the
    complete input and output), ``step`` (cost/token accounting), ``error``.
    Unparseable lines are skipped rather than raising -- a log tailed mid-write
    ends in a partial line, and a truncated transcript beats no transcript.
    """
    if not log.is_file():
        return
    for raw in log.read_text(encoding="utf-8", errors="replace").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(d, dict) or "type" not in d:
            continue
        part = d["part"] if isinstance(d.get("part"), dict) else {}
        ptype = part.get("type")
        ts = d.get("timestamp")
        if d["type"] in ("reasoning", "thinking") or ptype in ("reasoning", "thinking"):
            if part.get("text"):
                yield {"kind": "think", "ts": ts, "text": part["text"]}
        elif d["type"] == "text" and part.get("text"):
            yield {"kind": "text", "ts": ts, "text": part["text"]}
        elif is_tool(d, part):
            state = part["state"] if isinstance(part.get("state"), dict) else {}
            out = state.get("output") or state.get("error") or part.get("output") or ""
            yield {
                "kind": "tool",
                "ts": ts,
                "tool": str(part.get("tool") or part.get("name") or "tool"),
                "line": tool_line(part),
                "status": str(state.get("status") or ""),
                "input": tool_input(part),
                "output": str(out),
            }
        elif d["type"] == "step_finish":
            yield {"kind": "step", "ts": ts, "cost": part.get("cost"), "tokens": part.get("tokens")}
        elif d["type"] == "error":
            yield {"kind": "error", "ts": ts, "text": json.dumps(d.get("error"))}


def member_ids(run_dir: Path) -> list[int]:
    """Member ids present in a run dir, in numeric order."""
    d = Path(run_dir) / "members"
    if not d.is_dir():
        return []
    return sorted(int(p.name) for p in d.iterdir() if p.is_dir() and p.name.isdigit())


def _fence(body: str, lang: str = "") -> str:
    """Fence a block, lengthening the fence so embedded ``` can't break out."""
    ticks = "`" * max(3, max((len(m) for m in _tick_runs(body)), default=0) + 1)
    return f"{ticks}{lang}\n{body}\n{ticks}\n"


def _tick_runs(body: str) -> Iterator[str]:
    run = ""
    for ch in body:
        if ch == "`":
            run += ch
        elif run:
            yield run
            run = ""
    if run:
        yield run


def render_member(member_dir: Path, *, max_output: int = 0) -> str:
    """One member's whole turn as markdown: verdict, prompt, then the stream.

    ``max_output`` clips tool input/output to that many characters (0 = verbatim,
    the default -- the point of this over the web UI's pane is that nothing is
    dropped).
    """
    member_dir = Path(member_dir)
    out: list[str] = [f"# member {member_dir.name}\n"]

    verdict = _read_json(member_dir / "result.json")
    if verdict is not None:
        out.append("## verdict\n")
        out.append(_fence(json.dumps(verdict, indent=2), "json"))

    prompt = member_dir / "prompt.md"
    if prompt.is_file():
        out.append("## prompt\n")
        out.append(prompt.read_text(encoding="utf-8", errors="replace"))

    out.append("\n## transcript\n")
    cost, tools = 0.0, 0
    for item in parts(member_dir / "opencode.ndjson"):
        kind = item["kind"]
        if kind == "text":
            out.append(f"\n### assistant\n\n{item['text']}\n")
        elif kind == "think":
            out.append(f"\n### reasoning\n\n{item['text']}\n")
        elif kind == "tool":
            tools += 1
            arg = json.dumps(item["input"], indent=2) if item["input"] else "{}"
            out.append(f"\n### tool {tools}: {item['line']} [{item['status']}]\n")
            out.append("input:\n" + _fence(_clip_to(arg, max_output), "json"))
            out.append("output:\n" + _fence(_clip_to(item["output"], max_output)))
        elif kind == "error":
            out.append("\n### error\n\n" + _fence(item["text"]))
        elif kind == "step" and isinstance(item.get("cost"), (int, float)):
            cost += float(item["cost"])

    stderr = member_dir / "stderr.log"
    if stderr.is_file() and stderr.stat().st_size:
        body = stderr.read_text(encoding="utf-8", errors="replace")
        out.append("\n## stderr\n\n" + _fence(clip(body)))

    out.append(f"\n---\n\n{tools} tool call(s), ${cost:.4f}\n")
    return "\n".join(out)


def _clip_to(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}\n… [{len(text) - limit} chars omitted] …"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _index(run_dir: Path, ids: list[int]) -> str:
    """A table of the members, so the output dir is navigable without grepping."""
    meta = _read_json(run_dir / "run.json") or {}
    problem = (meta.get("problem") or {}).get("name", "?") if isinstance(meta, dict) else "?"
    rows = [
        f"# {problem} / {run_dir.name}\n",
        "| member | op | parent | correct | metric | tools | cost | summary |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i in ids:
        r = _read_json(run_dir / "members" / str(i) / "result.json") or {}
        r = r if isinstance(r, dict) else {}
        metric = r.get("metric")
        note = str(r.get("message") or r.get("error") or "").replace("|", "\\|")
        rows.append(
            f"| [{i}](member-{i}.md) | {r.get('op', '')} | {r.get('parent', '')} "
            f"| {r.get('correct', '')} | {metric if metric is not None else ''} "
            f"| {r.get('tool_calls', '')} | {r.get('cost', '')} | {note[:80]} |"
        )
    return "\n".join(rows) + "\n"


def export_transcripts(
    run_dir: Path,
    out_dir: Path,
    *,
    max_output: int = 0,
    jsonl: bool = False,
) -> list[int]:
    """Render every member of ``run_dir`` into ``out_dir``; return the ids written.

    Writes ``member-<id>.md`` (plus ``member-<id>.jsonl`` of the normalised items
    when ``jsonl``), an ``index.md``, and copies of the run-level logs.
    """
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = member_ids(run_dir)
    for i in ids:
        mdir = run_dir / "members" / str(i)
        (out_dir / f"member-{i}.md").write_text(
            render_member(mdir, max_output=max_output), encoding="utf-8"
        )
        if jsonl:
            with open(out_dir / f"member-{i}.jsonl", "w", encoding="utf-8") as fh:
                for item in parts(mdir / "opencode.ndjson"):
                    fh.write(json.dumps(item) + "\n")
    (out_dir / "index.md").write_text(_index(run_dir, ids), encoding="utf-8")
    for name in RUN_FILES:
        src = run_dir / name
        if src.is_file():
            (out_dir / name).write_bytes(src.read_bytes())
    return ids
