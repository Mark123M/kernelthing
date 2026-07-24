# Batched Cholesky, batch=256 n=128, on B200

## Goal

Minimise the wall-clock time of benchmark entry **index 2** of the gpu-mode `cholesky`
leaderboard: `batch: 256; n: 128; cond: 2; seed: 41128`, fp32, on a **B200 (sm_100)**.

`custom_kernel(data)` must handle that exact `(256, 128, 128)` shape with the specialised
kernel and fall through to `torch.linalg.cholesky_ex(..., check_errors=False).L` for every
other shape, so the other 14 leaderboard entries stay constant across attempts and only
the target entry moves.

## Problem

Input `A` is a `256 x 128 x 128` CUDA tensor in `torch.float32`; every matrix is symmetric
positive definite up to fp32 roundoff. Return a lower-triangular fp32 tensor `L` with a
positive diagonal such that `A = L @ L.T`.

The checker is property-based, not an elementwise comparison against a library result: it
validates shape, dtype, device, finiteness, lower-triangular structure, positive diagonal,
and the reconstruction residual against the original fp32 input. Inputs across the test set
cover dense covariance-like matrices, planted spectra, diagonal, damped low-rank, scaled
rows/columns and tridiagonal SPD matrices — a kernel tuned only for well-conditioned dense
input will fail the harder cases.

## Where you are starting

`submission.py` already carries a tuned kernel with a table of variants selected by
`_DEFAULT_VARIANT`. Read it before changing anything: the variant table encodes what has
already been tried (phase/panel decompositions, shared-memory layouts, SIMT vs TF32/WMMA
tiles, async copy, warp-tail and row-balanced work splits). Beating it means understanding
which of those is currently winning and why, not sampling the table at random.

## Hardware and build constraints

- Native code goes through `torch.utils.cpp_extension.load_inline`. The submission is a
  single self-contained Python file; there is no second source file.
- Compile as **C++20**. C++17 builds locally but breaks on the evaluator's PyTorch ABI.
- The evaluation server **rejects any submission containing the substring `stream`**,
  anywhere — inside a longer word, inside a comment, or assembled from concatenated
  fragments. The scorer rejects it locally first so you do not waste a submission.
- Target `sm_100`. 128x128 fp32 tiles fit comfortably in shared memory; the interesting
  tradeoffs are occupancy vs per-block work and how the panel factorisation is split.

## Build & test

There is no GPU here worth measuring on — correctness and timing both come from the
hosted evaluator. Score the kernel with:

    kernelthing score .

It prints `{"correct": ..., "metric": ..., "unit": "us", ...}`, where `metric` is the mean
time of benchmark index 2 in microseconds. **Lower is better.** `"correct": true` requires
every public test shape to pass *and* every benchmark shape to survive its re-check.

While a change is still likely broken, use the cheap pre-check — one submission instead of
two, correctness only:

    kernelthing score . --test-only

Do not build or run the kernel locally, and do not trust any locally produced number.

## Current state

Measured on B200 (submission 900990, 2026-07-23), the starting kernel:

| shape | index | time |
|---|---|---|
| **n=128, batch=256 (target)** | **2** | **75.141 µs** |
| n=32, batch=4096 | 0 | 113 µs |
| n=64, batch=1024 | 1 | 110 µs |
| n=256, batch=64 | 3 | 276 µs |

Whole-board geometric mean: 2007 µs. Only index 2 is yours to move — the rest run the
torch fallback and must stay put.

## Measurement policy

The scorer reads the evaluator's own numbers, so the metric is the exact measured mean
(75.14144521620538 µs for the starting kernel), not the three-significant-figure `75.1 µs`
the CLI prints. Resolution is not your limit; run-to-run noise is, and it is small — the
reported standard error is ±0.07 µs over 18 repeats. Treat a >0.3% change as real and
anything under ~0.1% as noise.
