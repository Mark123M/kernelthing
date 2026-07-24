"""Bubblewrap sandbox for edit-capable (and read-only) agent processes.

Basic, not bulletproof: the property that matters is that an edit-capable
opencode run cannot write anywhere except the project worktree, opencode's own
state dir, and /tmp. The whole filesystem is mounted read-only and specific
paths are re-bound writable on top. Network is left shared (the model API and
the popcorn service need it); the filesystem is the confinement boundary. No
GPU device nodes are bound -- kernels are compiled and benchmarked remotely on
the hosted popcorn service, never in the sandbox.
"""

from __future__ import annotations

import shutil
from pathlib import Path

# opencode loads skills from these directories. Mask them entirely so no
# user-level skills (flywheel, prose-restructurer, etc.) load into the agent's
# system prompt. The loop provides its own kernel-domain tooling via prompt
# injection; user skills are pure noise.
SKILL_HOMES = [
    Path.home() / ".claude" / "skills",
    Path.home() / ".agents" / "skills",
]


def available() -> bool:
    return shutil.which("bwrap") is not None


def wrap(
    inner_argv: list[str],
    *,
    project_dir: Path,
    writable: bool,
    writable_extra: list[Path] | tuple[Path, ...] = (),
    enabled: bool = True,
) -> list[str]:
    """Return ``inner_argv`` wrapped in a bwrap invocation (or unchanged if disabled).

    ``writable``: project_dir is bound read-write (implementer) vs read-only
    (reviewer). ``writable_extra`` paths are always bound read-write at their
    real locations -- used for opencode's own session/cache state so the
    implementer's ``-s`` session persists across rounds. No GPU device nodes are
    bound: kernels are compiled and benchmarked remotely on the popcorn service.
    """
    if not enabled:
        return inner_argv
    project_dir = Path(project_dir).resolve()

    args: list[str] = [
        "bwrap",
        "--die-with-parent",
        "--unshare-pid",
        "--ro-bind",
        "/",
        "/",  # everything readable, nothing writable...
        "--proc",
        "/proc",
        "--dev",
        "/dev",  # fresh devtmpfs (null/zero/random/...)
        "--tmpfs",
        "/tmp",
    ]
    # ...then re-bind the few writable paths on top.
    if writable:
        args += ["--bind", str(project_dir), str(project_dir)]
    else:
        args += ["--ro-bind", str(project_dir), str(project_dir)]
    for extra in writable_extra:
        extra = Path(extra)
        extra.mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(extra), str(extra)]

    # Mask skill directories so opencode loads no user-level skills.
    for home in SKILL_HOMES:
        if home.exists():
            args += ["--tmpfs", str(home)]

    # Run inside the project worktree.
    args += ["--chdir", str(project_dir)]
    args += inner_argv
    return args
