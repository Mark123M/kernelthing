"""Local stand-in for the ``task`` module the popcorn runner injects at evaluation time.

``submission.py`` does ``from task import input_t, output_t``. On the evaluation box that
module is supplied by the harness; here it exists only so the file can be imported for a
syntax/structure check without a network round-trip. It is deliberately inert -- nothing
in this repo evaluates the kernel locally, and nothing should.
"""

from __future__ import annotations

import torch

input_t = torch.Tensor
output_t = torch.Tensor
