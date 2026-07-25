"""Flattening the agents' NDJSON logs: what a member actually did must survive
into a file nobody needs a parser (or a running web server) to read."""

from __future__ import annotations

import json
from pathlib import Path

from kernelthing import archive, transcript

BIG_OUTPUT = "x" * 40000


def ev(etype: str, part: dict[str, object]) -> str:
    return json.dumps({"type": etype, "timestamp": 1, "sessionID": "s1", "part": part})


def tool_part(tool: str, inp: dict[str, object], out: str, status: str = "completed") -> dict:
    return {"type": "tool", "tool": tool, "state": {"status": status, "input": inp, "output": out}}


def make_member(run_dir: Path, mid: int, *, lines: list[str], result: dict | None = None) -> Path:
    m = run_dir / "members" / str(mid)
    m.mkdir(parents=True)
    (m / "prompt.md").write_text(f"# prompt for {mid}\nmake it faster\n", encoding="utf-8")
    (m / "opencode.ndjson").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (m / "result.json").write_text(json.dumps(result or {"id": mid}), encoding="utf-8")
    return m


def make_run(base: Path, ts: str = "2026-01-01_00-00-00") -> Path:
    d = base / ".humanize" / "rlcr" / ts
    d.mkdir(parents=True)
    (d / "run.json").write_text(
        json.dumps({"timestamp": ts, "problem": {"name": "toy"}}), encoding="utf-8"
    )
    (d / "events.ndjson").write_text('{"seq":1,"type":"run_start"}\n', encoding="utf-8")
    (d / "loop.log").write_text("dispatch m0\n", encoding="utf-8")
    return d


LINES = [
    ev("text", {"type": "text", "text": "I will read the kernel."}),
    ev("tool_use", tool_part("read", {"filePath": "/wt/submission.py"}, "kernel = 'v1'")),
    ev("reasoning", {"type": "reasoning", "text": "the panel loads dominate"}),
    ev("tool_use", tool_part("bash", {"command": "ncu ./a.out"}, BIG_OUTPUT)),
    "not json at all",
    ev("step_finish", {"type": "step-finish", "cost": 0.25, "tokens": {"total": 10}}),
    ev("step_finish", {"type": "step-finish", "cost": 0.25, "tokens": {"total": 10}}),
]


# --- the fold ----------------------------------------------------------------


def test_parts_normalises_every_kind_and_skips_junk(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    m = make_member(run, 0, lines=LINES)
    items = list(transcript.parts(m / "opencode.ndjson"))

    kinds = [i["kind"] for i in items]
    assert kinds == ["text", "tool", "think", "tool", "step", "step"], "junk line not skipped"
    assert items[0]["text"] == "I will read the kernel."
    assert items[2]["text"] == "the panel loads dominate"
    assert items[1]["input"] == {"filePath": "/wt/submission.py"}
    assert items[1]["output"] == "kernel = 'v1'"
    assert items[1]["line"] == "read /wt/submission.py"
    assert items[1]["status"] == "completed"


def test_parts_reads_the_other_input_shape(tmp_path: Path) -> None:
    """Some events put the args at part.input instead of part.state.input."""
    run = make_run(tmp_path)
    m = make_member(
        run,
        0,
        lines=[ev("tool", {"type": "tool", "tool": "edit", "input": {"filePath": "s.py"}})],
    )
    (item,) = list(transcript.parts(m / "opencode.ndjson"))
    assert item["input"] == {"filePath": "s.py"}
    assert item["line"] == "edit s.py"


def test_parts_tolerates_a_missing_or_half_written_log(tmp_path: Path) -> None:
    """A live run's log is tailed mid-write; a partial last line must not raise."""
    run = make_run(tmp_path)
    m = make_member(run, 0, lines=LINES)
    log = m / "opencode.ndjson"
    log.write_text(log.read_text(encoding="utf-8") + '{"type":"text","part":{"te', encoding="utf-8")
    assert len(list(transcript.parts(log))) == 6
    assert list(transcript.parts(run / "members" / "9" / "opencode.ndjson")) == []


# --- rendering ---------------------------------------------------------------


def test_render_member_is_verbatim_by_default(tmp_path: Path) -> None:
    """The point of this over the web UI's pane: nothing is clipped or dropped."""
    run = make_run(tmp_path)
    m = make_member(run, 3, lines=LINES, result={"id": 3, "metric": 70.2, "correct": True})
    md = transcript.render_member(m)

    assert "make it faster" in md, "the prompt is part of the conversation"
    assert '"metric": 70.2' in md
    assert "I will read the kernel." in md
    assert "the panel loads dominate" in md
    assert "ncu ./a.out" in md
    assert BIG_OUTPUT in md, "tool output was clipped without being asked"
    assert "2 tool call(s), $0.5000" in md


def test_render_member_clips_when_asked(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    m = make_member(run, 0, lines=LINES)
    md = transcript.render_member(m, max_output=100)
    assert BIG_OUTPUT not in md
    assert "chars omitted" in md


def test_render_member_fences_survive_embedded_backticks(tmp_path: Path) -> None:
    """Tool output routinely contains markdown fences (the agent reads .md files);
    a fixed ``` fence would end the block early and mangle everything after it."""
    run = make_run(tmp_path)
    payload = "here is a fence:\n```python\nprint(1)\n```\ndone"
    m = make_member(run, 0, lines=[ev("tool_use", tool_part("read", {"path": "p.md"}, payload))])
    md = transcript.render_member(m)
    assert payload in md
    assert "````\n" in md, "fence was not lengthened past the embedded one"


def test_render_member_includes_stderr_when_the_turn_crashed(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    m = make_member(run, 0, lines=LINES)
    (m / "stderr.log").write_text("Traceback: boom\n", encoding="utf-8")
    assert "Traceback: boom" in transcript.render_member(m)


# --- export ------------------------------------------------------------------


def test_export_writes_every_member_plus_index_and_run_files(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    for i in (0, 1, 2):
        make_member(run, i, lines=LINES, result={"id": i, "op": "explore", "metric": 70 + i})
    (run / "members" / "notamember").mkdir()

    out = tmp_path / "out"
    ids = transcript.export_transcripts(run, out, jsonl=True)

    assert ids == [0, 1, 2], "non-numeric member dir leaked in"
    for i in ids:
        assert (out / f"member-{i}.md").is_file()
        items = [json.loads(x) for x in (out / f"member-{i}.jsonl").read_text().splitlines()]
        assert [x["kind"] for x in items][:2] == ["text", "tool"]
    index = (out / "index.md").read_text(encoding="utf-8")
    assert "toy / 2026-01-01_00-00-00" in index
    assert "[2](member-2.md)" in index
    assert (out / "loop.log").is_file() and (out / "run.json").is_file()


def test_export_of_a_run_with_no_members_is_not_an_error(tmp_path: Path) -> None:
    run = make_run(tmp_path)
    assert transcript.export_transcripts(run, tmp_path / "out") == []
    assert (tmp_path / "out" / "index.md").is_file()


# --- archiving renders transcripts on the way out ----------------------------


def test_archive_renders_transcripts_automatically(tmp_path: Path) -> None:
    """A finished run archives itself; the transcripts must ride along, or the
    only readable copy is one `prepare_problem` away from being rebuilt over."""
    run = make_run(tmp_path / "managed")
    make_member(run, 0, lines=LINES)

    dest = archive.export_run(run, tmp_path / "arch", problem_name="toy")

    assert dest is not None
    md = tmp_path / "arch" / "toy" / archive.TRANSCRIPTS / run.name / "member-0.md"
    assert md.is_file()
    assert BIG_OUTPUT in md.read_text(encoding="utf-8")


def test_archive_survives_a_transcript_failure(tmp_path: Path, monkeypatch) -> None:
    """Every gate here fails open: a malformed log must not cost the run its
    artifacts (the whole reason export_run never raises)."""
    run = make_run(tmp_path / "managed")
    make_member(run, 0, lines=LINES)

    def boom(*a: object, **k: object) -> list[int]:
        raise RuntimeError("nope")

    monkeypatch.setattr(archive.transcript, "export_transcripts", boom)
    dest = archive.export_run(run, tmp_path / "arch", problem_name="toy")

    assert dest is not None and (dest / "members" / "0" / "opencode.ndjson").is_file()
