from kernelthing import prompts
from kernelthing.config import PROMPTS_DIR, REPO_ROOT
from kernelthing.orchestrator import Orchestrator, _skill_section


def test_single_pass_no_rescan():
    # A value containing a placeholder must NOT be re-expanded.
    out = prompts.render("{{A}} {{B}}", A="{{B}}", B="value")
    assert out == "{{B}} value"


def test_missing_var_left_intact():
    assert prompts.render("x {{UNKNOWN}} y", FOO="bar") == "x {{UNKNOWN}} y"


def test_basic_substitution():
    assert prompts.render("round {{N}}", N=3) == "round 3"


def test_load_real_prompt_renders_placeholders():
    # A live prompt loads and substitutes. kernel-tools-wiki.md is one the
    # orchestrator actually renders -- the fixture is deliberately not a
    # legacy file, so this test dies with the feature rather than outliving it.
    text = prompts.load("claude/kernel-tools-wiki.md")
    assert text, "kernel-tools-wiki.md should be present"
    rendered = prompts.render(text, PYTHON="/usr/bin/python3", WIKI_DIR="/vendor/KernelWiki")
    assert "/vendor/KernelWiki/scripts/query.py" in rendered
    assert "{{WIKI_DIR}}" not in rendered


def test_working_set_has_no_role_word_claude():
    # The mechanical rename should have removed the literal agent word "Claude".
    bad = []
    for p in PROMPTS_DIR.rglob("*.md"):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if "Claude" in line:
                bad.append(f"{p.name}:{i}")
    assert not bad, f"unexpected 'Claude' occurrences: {bad}"


def test_load_and_render_safe_fallback():
    out = prompts.load_and_render_safe("does/not/exist.md", "fallback {{X}}", X="ok")
    assert out == "fallback ok"


def test_skill_section_stops_at_a_sibling_subsection():
    # ncu-report-skill's `## File index` holds a `### Reference docs` table and a
    # `### Helpers` one. The note asks for the subsection so the seven helpers we spend a
    # line telling the agent to skip are not listed by name one line above it.
    skill = REPO_ROOT / "vendor" / "ncu-report-skill"
    index = _skill_section(skill, "Reference docs (read these when you need details)", 3)
    assert "05-analysis-dimensions.md" in index
    assert "helpers/" not in index
    # ...and the whole section does carry them, which is what makes the level matter
    assert "helpers/" in _skill_section(skill, "File index")
    # a heading that does not exist drops the index rather than raising
    assert _skill_section(skill, "No Such Heading", 3) == ""


def test_the_ncu_report_skill_note_is_prose_only():
    """Its SKILL.md quickstart builds a harness and runs ncu locally, and those binaries
    are installed on this box -- they would run on a consumer GPU at the wrong
    architecture and return numbers that look real. Its helpers/ read only the first
    launch and call rule_speedups() keys Nsight 2026.2 no longer emits."""
    note = Orchestrator._ncu_skill_note(REPO_ROOT / "vendor" / "ncu-report-skill")
    assert note, "vendor/ncu-report-skill should be present"
    assert "Ignore its collection workflow" in note and "helpers/*.py" in note
    assert "05-analysis-dimensions.md" in note and "08-b200-metric-names.md" in note
    # the description's Chinese trigger phrases are skill-router dispatch metadata
    assert "including variants in Chinese" not in note
    # an absent tree drops the note, and with it the whole section (the .md is a heading)
    assert Orchestrator._ncu_skill_note(REPO_ROOT / "vendor" / "does-not-exist") == ""


def test_both_veloq_skills_are_vendored_and_render_the_same_shape():
    """One function serves both, with no per-skill parameter: they are the same document
    shape, and what distinguishes the two sections is the heading, not the prose. A `lead`
    argument here is how a paraphrase creeps back in."""
    for name in ("veloq-ncu-skill", "veloq-nsys-skill"):
        skill = REPO_ROOT / "vendor" / name
        note = Orchestrator._veloq_ref_note(skill)
        assert note.startswith(f"\nVendored at `{skill}/`"), name
        assert f"{skill}/SKILL.md" in note
        assert "VeloQ CLI" in note  # the skill's own description:, read from disk
        assert "references/limitations.md" in note  # its own References index


def test_the_tools_block_is_ordered_turn_loop_then_reference_then_commands():
    """Section order is a decision, so it needs a test -- appending a new `parts.append`
    at the bottom of the method is otherwise invisible until someone reads a prompt.

    Three bands: what the agent does every turn, the reference it consults when stuck,
    then the command surfaces. Commands last is the recency slot, since this block sits
    immediately above the task. The two backward citations must stay backward --
    kernel-tools-veloq.md cites "the profiling section above" for the rejected token, and
    kernel-tools-nsys.md defers to the ncu section on durations-as-ratios.
    """
    from kernelthing.config import Config
    from kernelthing.problem import load_problem

    problem = load_problem(REPO_ROOT / "problems" / "cholesky_b256n128")
    block = Orchestrator(problem, Config())._kernel_tools_block
    headings = [ln[4:] for ln in block.splitlines() if ln.startswith("### ")]
    assert headings == [
        "Timing",
        "Profiling",
        "Rules",
        "Diagnosing an ncu report — `ncu-profile-analysis`",
        "B200 profiling reference — `ncu-report-skill`",
        "Reading an nsys timeline — `nsys-profile-analysis`",
        "CUDA documentation — the `nvidia-cuda-docs` MCP server",
        "PTX / CUDA ISA reference — `ptx-skill`",
        "Reading the Nsight Compute report — `veloq ncu`",
        "Reading the Nsight Systems timeline — `veloq nsys`",
        "KernelWiki — Blackwell/Hopper kernel-optimization knowledge base",
    ]
    assert block.index("### Profiling") < block.index("### Reading the Nsight Compute")
    assert block.index("### Reading the Nsight Compute") < block.index(
        "### Reading the Nsight Systems"
    )


def test_the_dump_renders_the_sections_in_the_order_candidates_get_them():
    """A dump that reordered them would document a prompt no candidate receives."""
    from kernelthing.config import Config
    from kernelthing.dump_prompts import render_kernel_tools
    from kernelthing.problem import load_problem

    cfg = Config()
    live = Orchestrator(load_problem(REPO_ROOT / "problems" / "cholesky_b256n128"), cfg)
    heads = lambda text: [ln for ln in text.splitlines() if ln.startswith("### ")]  # noqa: E731
    assert heads(render_kernel_tools(cfg)) == heads(live._kernel_tools_block)


def test_every_skill_section_renders_through_one_template():
    """Four vendored skills, one heading shape. Drifting formats is what the shared
    template prevents; an absent tree drops the whole section, heading included."""
    notes = {
        "Diagnosing an ncu report — `ncu-profile-analysis`": Orchestrator._veloq_ref_note(
            REPO_ROOT / "vendor" / "veloq-ncu-skill"
        ),
        "Reading an nsys timeline — `nsys-profile-analysis`": Orchestrator._veloq_ref_note(
            REPO_ROOT / "vendor" / "veloq-nsys-skill"
        ),
        "B200 profiling reference — `ncu-report-skill`": Orchestrator._ncu_skill_note(
            REPO_ROOT / "vendor" / "ncu-report-skill"
        ),
    }
    for title, note in notes.items():
        part = Orchestrator._skill_part(title, note)
        assert part.startswith(f"### {title}\n"), title
        # the heading is the only thing that differs -- every note opens the same way
        assert part.splitlines()[2].startswith("Vendored at `"), title
    assert Orchestrator._skill_part("Some Title", "") == ""
    assert Orchestrator._skill_part("Some Title", "   \n ") == ""
