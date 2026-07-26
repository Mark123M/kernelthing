# B200 Batched Cholesky - batch=640 n=512

## Goal

Develop a GPU kernel for an NVIDIA B200 GPU that minimizes latency while preserving numerical correctness. Allowed languages: CUDA C++, cuBLAS, MathDx, CuTe C++/DSL.

## Problem

Implement batched dense Cholesky factorization. Input is `A`, a `batch x N x N` CUDA tensor in `torch.float32`. Every matrix is symmetric positive definite up to FP32 roundoff. Return a lower-triangular FP32 tensor `L` with positive diagonal. Correctness passes when `norm_1(L @ L.T - A) <= 20 * N * float32_epsilon * norm_1(A)`. Among passing submissions, ranking is by the geometric mean of runtime across all benchmark entries.

## Rules

- The submission is a single self-contained Python file `submission.py`. ONLY optimize the kernel for workload `batch x N x N`, everything else should remain `torch.linalg.cholesky_ex`.
- Compile as **C++20**. C++17 builds locally but breaks on the evaluator's PyTorch ABI.
- Do **not** build or run the kernel locally, and do not trust any locally produced performance numbers.
- The evaluation server **rejects any submission containing the substring `stream`**, anywhere inside a longer word, inside a comment, or assembled from concatenated fragments.

## Testing

Score the kernel with:

```bash
kernelthing score .
```

This prints something like: {"correct": ..., "metric": ..., "unit": "us", ...}. `correct: true` requires every public test shape to pass **and** every benchmark shape to survive its re-check. `metric` is the mean runtime in microseconds.

Correctness check:

```bash
kernelthing score . --test-only
```
