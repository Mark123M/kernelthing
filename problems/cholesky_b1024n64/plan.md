# B200 Batched Cholesky - batch=1024 n=64

## Goal

Develop a GPU kernel for an NVIDIA B200 GPU that minimizes latency while preserving numerical correctness. Allowed languages: CUDA C++, cuBLAS, MathDx, CuTe C++/DSL.

## Problem

Implement batched dense Cholesky factorization. Input is `A`, a `1024 x 64 x 64` CUDA tensor in `torch.float32`. Every matrix is symmetric positive definite up to FP32 roundoff. Return a lower-triangular FP32 tensor `L` with positive diagonal. Correctness passes when `norm_1(L @ L.T - A) <= 20 * 64 * float32_epsilon * norm_1(A)`. You are scored on the runtime of this shape alone; the other 14 benchmark entries only have to keep passing.

## Rules

- The submission is a single self-contained Python file `submission.py`. ONLY optimize the kernel for workload `1024 x 64 x 64`, everything else should remain `torch.linalg.cholesky_ex`.
- Compile as **C++20**. C++17 builds locally but breaks on the evaluator's PyTorch ABI.
- Do **not** build or run the kernel locally, and do not trust any locally produced performance numbers.
- The evaluation server **rejects any submission containing the substring `stream`**, anywhere inside a longer word, inside a comment, or assembled from concatenated fragments.
