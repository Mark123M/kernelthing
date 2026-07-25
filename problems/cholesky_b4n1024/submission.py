"""Seed kernel for cholesky_b4n1024 -- FILL THIS IN.

Contract (enforced by the hosted popcorn evaluator, not by anything local):

- This file is the *only* thing submitted. It must be a single self-contained
  Python module: native code goes through
  ``torch.utils.cpp_extension.load_inline``, never a second ``.cu`` file.
- Entry point is ``custom_kernel(data: input_t) -> output_t``. ``data`` is a
  ``(4, 1024, 1024)`` fp32 CUDA tensor of SPD matrices; return lower-triangular
  ``L`` with a positive diagonal such that ``A = L @ L.T``.
- ``custom_kernel`` is called for *every* one of the 15 leaderboard shapes, not
  just this one. Specialise on the target shape and fall through to the torch
  path for the rest, so the other 14 entries stay constant across attempts and
  only benchmark index 6 moves.
- Compile as **C++20**. C++17 builds locally but breaks on the evaluator's
  PyTorch ABI.
- The evaluator rejects any submission containing one banned substring
  anywhere -- including inside a longer word or a comment. ``plan.md`` names it;
  this file must never carry it, not even in a docstring. ``kernelthing score``
  rejects it locally first so you do not waste a submission.

The torch fallback below is a correct but unoptimised seed: it scores, so the
search has a starting point, and it is what every non-target shape should keep
hitting. Port your kernel from
``/home/mark123/projects/linalg/cholesky/b4n1024/cholesky_b4n1024.py``.
"""

from __future__ import annotations

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # TODO: specialise batch=4, n=1024 (benchmark index 6); everything
    # else must keep taking this fallback.
    return torch.linalg.cholesky_ex(data, check_errors=False).L
