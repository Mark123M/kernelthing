# Batched Cholesky, batch=8 n=2048, on B200

## Goal

Minimise the wall-clock time of benchmark entry **index 9** of the gpu-mode
`cholesky` leaderboard: `n: 2048; cond: 2; seed: 512048; batch: 8`, fp32, on a **B200 (sm_100)**.

`custom_kernel(data)` must handle that exact `(8, 2048, 2048)` shape with the
specialised kernel and fall through to
`torch.linalg.cholesky_ex(..., check_errors=False).L` for every other shape, so the
other 14 leaderboard entries stay constant across attempts and only the target entry
moves.

## Problem

Input `A` is a `8 x 2048 x 2048` CUDA tensor in `torch.float32`; every matrix is
symmetric positive definite up to fp32 roundoff. Return a lower-triangular fp32 tensor
`L` with a positive diagonal such that `A = L @ L.T`.

The checker is property-based, not an elementwise comparison against a library result:
it validates shape, dtype, device, finiteness, lower-triangular structure, positive
diagonal, and the reconstruction residual against the original fp32 input. Inputs across
the test set cover dense covariance-like matrices, planted spectra, diagonal, damped
low-rank, scaled rows/columns and tridiagonal SPD matrices — a kernel tuned only for
well-conditioned dense input will fail the harder cases.

<!-- FILL IN: what makes THIS shape different. n=2048 does not fit in shared memory — this is a blocked/tiled factorisation (panel + trailing update), and with batch=8 there is little batch parallelism to hide latency with. -->

## Where you are starting

<!-- FILL IN. `submission.py` currently holds only the torch fallback. Replace this
     section once you seed a real kernel: say what approach it takes, which
     alternatives are already encoded in it, and what has already been ruled out.
     The search reads this to avoid re-deriving what you already know. -->

## Hardware and build constraints

- Native code goes through `torch.utils.cpp_extension.load_inline`. The submission is a
  single self-contained Python file; there is no second source file.
- Compile as **C++20**. C++17 builds locally but breaks on the evaluator's PyTorch ABI.
- The evaluation server **rejects any submission containing the substring `stream`**,
  anywhere — inside a longer word, inside a comment, or assembled from concatenated
  fragments. The scorer rejects it locally first so you do not waste a submission.
- Target `sm_100`.

<!-- FILL IN: the shared-memory / occupancy budget at n=2048. A 2048x2048 fp32 matrix is 16.0 MB — far past shared memory, so the blocking factor and the trailing-update GEMM shape are the design. -->

## Build & test

There is no GPU here worth measuring on — correctness and timing both come from the
hosted evaluator. Score the kernel with:

    kernelthing score .

It prints `{"correct": ..., "metric": ..., "unit": "us", ...}`, where `metric` is the
mean time of benchmark index 9 in microseconds. **Lower is better.**
`"correct": true` requires every public test shape to pass *and* every benchmark shape
to survive its re-check.

While a change is still likely broken, use the cheap pre-check — one submission instead
of two, correctness only:

    kernelthing score . --test-only

Do not build or run the kernel locally, and do not trust any locally produced number.

## Current state

<!-- FILL IN once you have a first scored submission. Give the measured time for the
     target index and for the neighbouring shapes you must not regress, plus the
     submission id and date, e.g.:

| shape | index | time |
|---|---|---|
| **n=2048, batch=8 (target)** | **9** | **? µs** |

Whole-board geometric mean: ? µs. Only index 9 is yours to move — the rest run
the torch fallback and must stay put. -->

## Measurement policy

The scorer reads the evaluator's own numbers from the API, so the metric is the exact
measured mean, not the three-significant-figure value the CLI prints. Resolution is not
your limit; run-to-run noise is.

<!-- FILL IN: the measured standard error for this shape once you have one, and the
     resulting "treat >X% as real, <Y% as noise" threshold. -->
