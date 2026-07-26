"""Command-line entry point: ``kernelthing [<problem-dir> | <objective>]``.

A *problem* is a directory containing a ``problem.json`` manifest (see
problem.py). Pass an existing problem dir to optimize it directly. Omit it (or
pass a natural-language objective instead) and kernelthing first *bootstraps* a
new problem dir with an opencode agent (see bootstrap.py): interactively by
default, or non-interactively with ``--auto-setup`` (which then needs an
objective). Either way it auto-detects the enclosing git repo, runs the loop, and
serves a web UI for watching progress and live-tuning N / the turn cap / stop.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import archive, bootstrap, journal
from .config import Config
from .orchestrator import Orchestrator
from .problem import Problem, load_problem, prepare_problem

BANNER_ART = r"""
 __                                          ___     __      __
/\ \                                        /\_ \   /\ \__  /\ \        __
\ \ \/'\       __    _ __     ___       __  \//\ \  \ \ ,_\ \ \ \___   /\_\     ___       __
 \ \ , <     /'__`\ /\`'__\ /' _ `\   /'__`\  \ \ \  \ \ \/  \ \  _ `\ \/\ \  /' _ `\   /'_ `\
  \ \ \\`\  /\  __/ \ \ \/  /\ \/\ \ /\  __/   \_\ \_ \ \ \_  \ \ \ \ \ \ \ \ /\ \/\ \ /\ \L\ \
   \ \_\ \_\\ \____\ \ \_\  \ \_\ \_\\ \____\  /\____\ \ \__\  \ \_\ \_\ \ \_\\ \_\ \_\\ \____ \
    \/_/\/_/ \/____/  \/_/   \/_/\/_/ \/____/  \/____/  \/__/   \/_/\/_/  \/_/ \/_/\/_/ \/___L\ \
        Evolutionary Autoresearch Optimization of GPU Kernels.                            /\____/
                                                                                          \/___/
"""

# bright green for the figlet, bright blue for the tagline prose.
GREEN, BLUE, RESET = "\033[92m", "\033[94m", "\033[0m"
TAGLINE = "Evolutionary Autoresearch Optimization of GPU Kernels."


def colorize_banner(art: str) -> str:
    """Bright-green the art; recolor the tagline prose bright blue (its line also
    carries art glyphs on the right, so the two halves are colored separately)."""
    lines = []
    for line in art.split("\n"):
        if TAGLINE in line:
            end = line.index(TAGLINE) + len(TAGLINE)
            line = f"{BLUE}{line[:end]}{GREEN}{line[end:]}"
        lines.append(line)
    return f"{GREEN}{chr(10).join(lines)}{RESET}"


# Color only when writing to a terminal; piped/redirected --help stays plain.
BANNER = colorize_banner(BANNER_ART) if sys.stdout.isatty() else BANNER_ART


def duration(text: str) -> int:
    """argparse type for ``-w/--wall-clock``: parse a duration into whole seconds.

    Accepts a number with an optional ``s/m/h/d/w`` suffix (``10m``, ``2h``,
    ``1d``, ``90s``); a bare number is seconds. ``0`` means 'off'. This exists
    because bare-seconds was a footgun -- ``-w 10`` reads as 10 *seconds*, not the
    10 minutes one might expect."""
    from .config import parse_duration

    try:
        return parse_duration(text)
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"invalid duration '{text}'; use e.g. 90s, 10m, 2h, 1d, 1w (a bare number is seconds)"
        ) from err


def read_objective(args: argparse.Namespace) -> str | None:
    """The bootstrap objective: ``--objective-file`` wins, else a non-path positional."""
    if args.objective_file:
        return Path(args.objective_file).read_text(encoding="utf-8")
    return (
        str(args.problem) if args.problem is not None else None
    )  # a positional that wasn't a problem dir is objective text


def resolve_problem(args: argparse.Namespace, cfg: Config) -> Problem:
    """Load an existing problem dir, or bootstrap a new one from an objective."""
    src = args.problem
    if src and not args.objective_file:
        p = Path(src)
        if (p.is_dir() and (p / "problem.json").is_file()) or (
            p.is_file() and p.name == "problem.json"
        ):
            problem = load_problem(p)
            # Keep the problem's bootstrap-prompt.md snapshot current with the live
            # template before the run copies the dir (the copy inherits the refresh).
            bootstrap.refresh_bootstrap_prompt(problem.dir, problem.repo_root)
            return prepare_problem(problem, cfg.problem_root)
        if p.exists():
            raise RuntimeError(
                f"{src} exists but is not a problem dir (no problem.json); "
                "pass a problem dir, or an objective to bootstrap from"
            )
    # Bootstrap mode: build a new problem dir inside a managed repo.
    target = bootstrap.bootstrap_problem(
        read_objective(args), cfg=cfg, auto=args.auto_setup, managed_root=cfg.problem_root
    )
    return load_problem(target)


def run_loop(args: argparse.Namespace) -> int:
    cfg = Config(
        model=args.model,
        opencode_timeout=args.timeout,
        methodology=args.methodology,
        sandbox=not args.no_sandbox,
        parallelism=args.parallelism,
        kernelguard=not args.no_kernelguard,
        ncu=not args.no_ncu,
        wiki=not args.no_wiki,
        veloq=not args.no_veloq,
        ptx=not args.no_ptx,
        mcp_cuda_docs=not args.no_cuda_docs,
        auto_setup=args.auto_setup,
        max_candidates=args.max_candidates,
        wall_clock_s=args.wall_clock,
        elite_k=args.elite_k,
        min_niches=args.min_niches,
        problem_root=args.problem_root,
        archive_root=None if args.no_archive else args.archive_root,
    )

    try:
        problem = resolve_problem(args, cfg)
    except (FileNotFoundError, RuntimeError, KeyError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if not args.no_web:
        from . import webui

        try:
            _httpd, port = webui.start_server(cfg.problem_root, port=args.web_port)
            print(f"[kernelthing] web UI:   http://127.0.0.1:{port}", file=sys.stderr)
        except OSError as e:
            print(f"[kernelthing] web UI disabled ({e}); continuing headless", file=sys.stderr)
    else:
        print("[kernelthing] running headless (--no-web)", file=sys.stderr)

    print(f"[kernelthing] problem:   {problem.name}", file=sys.stderr)
    print(f"[kernelthing] artifacts: {problem.repo_root}", file=sys.stderr)
    from .config import format_duration

    wall_str = format_duration(args.wall_clock) if args.wall_clock else "none"
    budget_str = f"{args.max_candidates} candidates" if args.max_candidates else "unbounded"
    print(
        f"[kernelthing] agents:    {args.parallelism}  wall: {wall_str}  budget: {budget_str}",
        file=sys.stderr,
    )

    orch = Orchestrator(problem, cfg)
    try:
        exit_reason = orch.run()
    except KeyboardInterrupt:
        print("\n[kernelthing] interrupted", file=sys.stderr)
        orch.persist_current_head()
        return 130
    print(f"[kernelthing] loop finished: {exit_reason}", file=sys.stderr)
    # complete / stalled_out / stopped / maxiter all leave a correct HEAD.
    return 0 if exit_reason in ("complete", "stalled_out", "stopped", "maxiter") else 1


def score_command(argv: list[str]) -> int:
    """``kernelthing score [<dir>]``: run the authoritative scorer on a problem dir
    and print its JSON verdict.

    This is the *same* call every loop round scores with -- so a green here means
    the problem (or a kernel edit) scores correct for real.

    Grading is remote: a ``bench.backend == "popcorn"`` problem is submitted to the
    hosted popcorn service and the returned numbers are parsed (see popcorn.py).
    There is no local GPU step, and no other backend is supported.
    """
    p = argparse.ArgumentParser(
        prog="kernelthing score",
        description="Score a problem dir on the hosted popcorn service and print "
        "{correct, metric, unit}. Same code path the loop scores with -- use it to "
        "check a freshly authored problem or a kernel edit.",
    )
    p.add_argument(
        "dir",
        nargs="?",
        default=".",
        help="problem dir containing problem.json (default: current dir)",
    )
    p.add_argument(
        "--test-only",
        action="store_true",
        default=False,
        help="check correctness only, skipping the timing run: the cheap pre-check "
        "(one popcorn submission instead of two). Prints no metric.",
    )
    args = p.parse_args(argv)

    try:
        problem = load_problem(Path(args.dir))
    except (FileNotFoundError, RuntimeError, KeyError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    # Grading is remote. A popcorn-backed problem routes to the hosted service;
    # nothing here ever touches a local GPU.
    if (problem.bench or {}).get("backend") == "popcorn":
        from . import popcorn

        return popcorn.score_command(problem, args)

    print(
        f"error: problem '{problem.name}' has no popcorn backend "
        "(bench.backend != 'popcorn'). Local benchmarking was removed; every "
        "problem must be graded on the hosted popcorn service.",
        file=sys.stderr,
    )
    return 2


def web_command(argv: list[str]) -> int:
    """``kernelthing web``: serve the web UI standalone over a directory of runs.

    Discovers every run (live or finished) under ``--root`` -- each run dir is
    self-describing on disk (run.json / events.ndjson / members/), so no run
    process needs to be alive. Use it to inspect old runs, or as a single UI
    over several concurrent ``kernelthing --no-web`` runs.
    """
    from . import webui

    p = argparse.ArgumentParser(
        prog="kernelthing web",
        description="Serve the kernelthing web UI over a directory of runs "
        "(live and finished). Runs live in <root>/<problem>/.humanize/rlcr/.",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache" / "kernelthing",
        help="directory to discover runs under (default: %(default)s)",
    )
    p.add_argument("--port", type=int, default=8765, help="port (default: %(default)s)")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: %(default)s)")
    args = p.parse_args(argv)

    httpd = webui.make_server(args.root, port=args.port, host=args.host)
    print(
        f"[kernelthing] web UI: http://{args.host}:{httpd.server_address[1]}  "
        f"(runs from {args.root})",
        file=sys.stderr,
    )
    import contextlib

    with contextlib.suppress(KeyboardInterrupt):
        httpd.serve_forever()
    return 0


def archive_command(argv: list[str]) -> int:
    """``kernelthing archive``: export runs out of the managed root by hand.

    A run archives itself when it ends, so this is for the runs that never got
    to end -- a killed process, a machine that restarted mid-search. Their
    artifacts are still sitting in the managed root, intact but one
    ``prepare_problem`` away from being rebuilt over.
    """
    p = argparse.ArgumentParser(
        prog="kernelthing archive",
        description="Copy run artifacts (journal, members, git bundle, best "
        "kernel) out of the disposable managed root into a durable archive. "
        "With no RUN_ID, archives every run found.",
    )
    p.add_argument(
        "run_id",
        nargs="*",
        metavar="RUN_ID",
        help="run ids to archive, as printed by --list (default: all of them)",
    )
    p.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".cache" / "kernelthing",
        help="managed root to read runs from (default: %(default)s)",
    )
    p.add_argument(
        "--archive-root",
        type=Path,
        default=archive.default_archive_root(),
        help="destination archive root (default: %(default)s)",
    )
    p.add_argument("--list", action="store_true", help="list run ids under --root and exit")
    args = p.parse_args(argv)

    runs = journal.discover_runs(args.root)
    if args.list:
        if not runs:
            print(f"no runs under {args.root}", file=sys.stderr)
        for r in runs:
            meta = r.get("run") or {}
            live = " (live)" if r.get("live") else ""
            print(f"{r['id']}{live}  {(meta.get('problem') or {}).get('name', '?')}")
        return 0

    def say(msg: str) -> None:
        print(f"[kernelthing] {msg}", file=sys.stderr)

    if not runs:
        print(f"error: no runs found under {args.root}", file=sys.stderr)
        return 2
    known = {str(r["id"]) for r in runs}
    unknown = [r for r in args.run_id if r not in known]
    if unknown:
        print(f"error: no such run under {args.root}: {', '.join(unknown)}", file=sys.stderr)
        return 2

    done = archive.export_all(
        args.root, args.archive_root, run_ids=args.run_id or None, log=say
    )
    print(
        f"[kernelthing] archived {len(done)} run(s) to {args.archive_root}\n"
        f"[kernelthing] browse:   kernelthing web --root {args.archive_root}",
        file=sys.stderr,
    )
    return 0 if done else 1


def resolve_runs(target: str, roots: list[Path]) -> list[Path]:
    """Run dirs named by ``target``: a path, a run id under a root, or ``all``.

    A path is taken as given (a run dir, or any dir with runs beneath it) so a
    run that was never archived -- or one sitting next to the problem in the
    source repo -- resolves without a --root dance. Otherwise ``target`` is a run
    id as ``kernelthing archive --list`` prints it, looked up in each root.
    """
    p = Path(target).expanduser()
    if p.is_dir():
        if (p / journal.RUN_JSON).is_file():
            return [p.resolve()]
        return [Path(r["dir"]) for r in _runs_under(p)]
    hits = []
    for root in roots:
        for r in _runs_under(root):
            if target in ("all", "*") or str(r["id"]) == target:
                hits.append(Path(r["dir"]))
    return hits


def _runs_under(root: Path) -> list[dict[str, Any]]:
    """``journal.discover_runs`` with the run dir resolved onto each entry."""
    root = Path(root).expanduser()
    return [{**r, "dir": root.resolve() / str(r["id"])} for r in journal.discover_runs(root)]


def transcripts_command(argv: list[str]) -> int:
    """``kernelthing transcripts``: flatten the agents' NDJSON logs to markdown.

    Archiving already does this for every run it exports (``<archive>/<problem>/
    transcripts/<ts>/``), so this is for the cases that miss that path: a run
    still going, a run dir somewhere else on disk, or a re-render with ``--jsonl``
    / a different clip.
    """
    from . import transcript

    p = argparse.ArgumentParser(
        prog="kernelthing transcripts",
        description="Render every agent's full conversation (prompt, assistant "
        "text, reasoning, and every tool call with its complete input and "
        "output) to one markdown file per member.",
    )
    p.add_argument(
        "run",
        nargs="*",
        metavar="RUN",
        help="run ids (as printed by --list) or paths to run dirs; a dir with "
        "runs beneath it works too. Default: every run found under --root.",
    )
    p.add_argument(
        "--root",
        type=Path,
        action="append",
        metavar="DIR",
        help="root to resolve run ids under; repeatable. Default: the archive "
        f"root ({archive.default_archive_root()}) then the managed root.",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("transcripts"),
        help="output dir; each run lands in <out>/<problem>/<timestamp>/ "
        "(default: ./%(default)s)",
    )
    p.add_argument(
        "--jsonl",
        action="store_true",
        help="also write member-<id>.jsonl -- one normalised item per line, for "
        "grep/jq over the stream",
    )
    p.add_argument(
        "--max-output",
        type=int,
        default=0,
        metavar="N",
        help="clip each tool input/output to N characters (default: 0, verbatim)",
    )
    p.add_argument("--list", action="store_true", help="list run ids under the roots and exit")
    args = p.parse_args(argv)

    roots = args.root or [archive.default_archive_root(), Path.home() / ".cache" / "kernelthing"]
    if args.list:
        for root in roots:
            for r in _runs_under(root):
                meta = r.get("run") or {}
                live = " (live)" if r.get("live") else ""
                print(f"{r['id']}{live}  {(meta.get('problem') or {}).get('name', '?')}  {root}")
        return 0

    run_dirs: list[Path] = []
    for target in args.run or ["all"]:
        found = resolve_runs(target, roots)
        if not found:
            print(f"error: no run matching '{target}'", file=sys.stderr)
            return 2
        run_dirs += [d for d in found if d not in run_dirs]

    total = 0
    for run_dir in run_dirs:
        dest = args.out / _problem_name(run_dir) / run_dir.name
        ids = transcript.export_transcripts(
            run_dir, dest, max_output=args.max_output, jsonl=args.jsonl
        )
        total += len(ids)
        print(f"[kernelthing] {len(ids)} member(s) -> {dest}", file=sys.stderr)
    if not total:
        print("[kernelthing] no members found (run had no agents yet?)", file=sys.stderr)
    return 0


def _problem_name(run_dir: Path) -> str:
    """Problem name from run.json, falling back to the layout (<problem>/.humanize/…)."""
    try:
        meta = json.loads((run_dir / journal.RUN_JSON).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = None
    if isinstance(meta, dict):
        name = (meta.get("problem") or {}).get("name")
        if name:
            return str(name)
    return run_dir.parts[-4] if len(run_dir.parts) >= 4 else "run"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "score":
        return score_command(argv[1:])
    if argv and argv[0] == "web":
        return web_command(argv[1:])
    if argv and argv[0] == "archive":
        return archive_command(argv[1:])
    if argv and argv[0] == "transcripts":
        return transcripts_command(argv[1:])

    parser = argparse.ArgumentParser(
        prog="kernelthing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=BANNER,
        epilog="subcommands:\n"
        "  score [<dir>]   score a problem dir with the authoritative benchmark and\n"
        "                  print {correct, metric, unit} -- the same scoring the loop\n"
        "                  uses; run `kernelthing score --help` for details\n"
        "  web             serve the web UI standalone over all runs (live and\n"
        "                  finished) under --root; run `kernelthing web --help`\n"
        "  archive         copy run artifacts out of the disposable managed root\n"
        "                  into a durable archive; for runs that died before they\n"
        "                  could archive themselves (a finished run does it itself)\n"
        "  transcripts     render every agent's full conversation (prompt, text,\n"
        "                  reasoning, tool calls with complete I/O) to markdown;\n"
        "                  archiving already does this, so this is for live runs\n"
        "                  and re-renders",
    )
    parser.add_argument(
        "problem",
        nargs="?",
        help="path to a problem dir (containing problem.json) or the manifest. "
        "Omit it (or pass a natural-language objective instead) to bootstrap "
        "a new problem dir first; without --auto-setup this is interactive.",
    )

    search = parser.add_argument_group(
        "search budget & shape",
        "how long the evolutionary search runs and how wide it goes; "
        "-j/-k/-m/-w are all live-tunable from the web UI while the run is going",
    )
    search.add_argument(
        "-j",
        "--parallelism",
        type=int,
        default=4,
        help="max agents editing and scoring at once (live-tunable up and down "
        "in the web UI); each score is a remote popcorn submission. Default 4.",
    )
    search.add_argument(
        "-k",
        "--elite-k",
        type=int,
        default=4,
        help="size of the top-K frontier (the exploit pool); live-tunable "
        "in the web UI. Default 4.",
    )
    search.add_argument(
        "--min-niches",
        type=int,
        default=4,
        help="below this many strategy niches, bias toward explore "
        "to fill out the diversity grid. Default 4.",
    )
    search.add_argument(
        "-m",
        "--max-candidates",
        type=int,
        default=24,
        help="budget: total candidates to dispatch, then stop. 0 = run until "
        "--wall-clock / web-UI stop. Live-tunable in the web UI. Default 24.",
    )
    search.add_argument(
        "-w",
        "--wall-clock",
        type=duration,
        default=0,
        metavar="DUR",
        help="budget: wall-clock duration, then stop. Accepts an s/m/h/d/w "
        "suffix (e.g. 10m, 2h, 1d); a bare number is seconds. 0 = off. "
        "Live-tunable in the web UI. Default 0.",
    )

    models = parser.add_argument_group("model")
    models.add_argument(
        "--model",
        default="deepseek/deepseek-v4-pro",
        help="opencode model that edits kernels and authors problems (default: %(default)s)",
    )
    models.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="per-opencode-turn timeout in seconds (default: %(default)s)",
    )

    boot = parser.add_argument_group("bootstrap (authoring a new problem)")
    boot.add_argument(
        "--auto-setup",
        action="store_true",
        help="build the new problem dir non-interactively and auto-accept on "
        "validation pass (no interactive review). Requires an objective "
        "(positional arg or --objective-file).",
    )
    boot.add_argument(
        "--objective-file",
        metavar="PATH",
        help="read the objective description from this file instead of the "
        "positional argument (useful for long descriptions)",
    )

    tools = parser.add_argument_group("agent tools & sandboxing")
    tools.add_argument("--no-sandbox", action="store_true", help="disable bwrap (debug only)")
    tools.add_argument(
        "--no-kernelguard",
        action="store_true",
        help="disable kernelguard benchmark-cheat detection (rollback/disqualify)",
    )
    tools.add_argument(
        "--no-ncu",
        action="store_true",
        help="don't offer the agent the Nsight Compute (ncu) skill for reading "
        "the remote popcorn --profile-brev reports",
    )
    tools.add_argument(
        "--no-wiki",
        action="store_true",
        help="don't offer the agent the KernelWiki kernel-optimization knowledge base",
    )
    tools.add_argument(
        "--no-veloq",
        action="store_true",
        help="don't offer the agent veloq for reading the downloaded .ncu-rep as "
        "structured JSON (falls back to the flat ncu-details.txt dump)",
    )
    tools.add_argument(
        "--no-ptx",
        action="store_true",
        help="don't offer the agent the vendored PTX/CUDA ISA reference",
    )
    tools.add_argument(
        "--no-cuda-docs",
        action="store_true",
        help="don't declare NVIDIA's hosted CUDA-docs MCP server for the agent "
        "(it needs a one-time interactive OAuth to be useful)",
    )

    web = parser.add_argument_group("web UI")
    web.add_argument(
        "--web-port", type=int, default=8765, help="web UI port (default: %(default)s)"
    )
    web.add_argument("--no-web", action="store_true", help="run headless (no web UI)")

    runtime = parser.add_argument_group("runtime & output")
    runtime.add_argument(
        "--problem-root",
        type=Path,
        default=Path.home() / ".cache" / "kernelthing",
        help="managed problem repo root (worktrees branch from copies here)",
    )
    runtime.add_argument(
        "--archive-root",
        type=Path,
        default=archive.default_archive_root(),
        help="durable artifact archive: the run's journal, members, git bundle "
        "and best kernel are copied here when the run ends, outside the "
        "disposable --problem-root. Serve it with `kernelthing web --root`. "
        "Default: %(default)s",
    )
    runtime.add_argument(
        "--no-archive",
        action="store_true",
        help="do not copy artifacts out of --problem-root when the run ends",
    )
    runtime.add_argument(
        "--methodology",
        action="store_true",
        help="at loop exit, run a retrospective that writes a sanitized "
        "methodology report (methodology-analysis-report.md) to the loop dir",
    )

    args = parser.parse_args(argv)
    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
