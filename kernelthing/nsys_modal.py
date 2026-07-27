"""Modal B200 Nsight Systems capture for one Cholesky submission.

This module is intentionally optional: normal kernelthing imports do not touch
``modal``. ``popcorn.profile_nsys_submission`` shells out to ``modal run`` only
after a full score has passed correctness, and treats every failure here as an
unavailable profile rather than a failed score.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

image = (
    modal.Image.from_registry("nvidia/cuda:13.1.1-devel-ubuntu24.04", add_python="3.13")
    .entrypoint([])
    .apt_install("cuda-nsight-systems-13-1")
    .pip_install(
        "torch==2.12.0",
        "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu130",
    )
)

app = modal.App("kernelthing-nsys-b200", image=image)
cache_vol = modal.Volume.from_name("kernelthing-nsys-cache", create_if_missing=True)

RUNNER = r"""
from __future__ import annotations

import argparse
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    sys.path.insert(0, "/workspace")
    import submission

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")

    # A dense SPD batch whose spectrum is exactly linspace(1, 2) -- so condition number
    # exactly 2, which is what every shape on the board is generated at (`cond: 2`).
    #
    # This used to be diag_embed(linspace(...)) alone. That already had the right
    # spectrum, but every off-diagonal was an exact zero, so any kernel that branches on
    # a value -- an early exit on a small pivot, an isfinite guard, a non-convergence
    # fallback -- took a path it would never take on the board, and the timeline was of
    # the wrong code. (Nothing else changed: ptxas never sees the data, zeros are normal
    # values at full FP32 throughput, and the storage is dense either way.)
    #
    # Built with ONE Householder reflector rather than a random orthogonal Q: for
    # Q = I - 2vv^T with v a unit vector, A = QDQ is a rank-2 update of D, so this is
    # O(batch*n^2) with a single extra temporary. Going through torch.linalg.qr would be
    # O(n^3) with a large workspace -- ~minutes of dead time at n=32768, on a capture
    # whose whole budget is PROFILE_TIMEOUT_S. All of it lands before cudaProfilerStart,
    # so it costs the profile nothing either way, but it does cost the job's wall clock.
    d = torch.linspace(1.0, 2.0, args.n, device="cuda", dtype=torch.float32)
    d = d.repeat(args.batch, 1)
    v = torch.randn(args.batch, args.n, 1, device="cuda", dtype=torch.float32)
    v /= v.norm(dim=1, keepdim=True)
    w = d.unsqueeze(-1) * v  # D v
    s = (w * v).sum(dim=1, keepdim=True)  # v^T D v
    data = torch.diag_embed(d)
    data.baddbmm_(w, v.mT, alpha=-2.0)
    data.baddbmm_(v, w.mT, alpha=-2.0)
    data.baddbmm_(4.0 * s * v, v.mT)
    del d, v, w, s
    torch.cuda.synchronize()

    with torch.no_grad():
        out = submission.custom_kernel(data)
    torch.cuda.synchronize()
    del out

    cudart = torch.cuda.cudart()
    cudart.cudaProfilerStart()
    with torch.no_grad():
        out = submission.custom_kernel(data)
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()
    torch.cuda.synchronize()

    if not hasattr(out, "shape") or tuple(out.shape) != (args.batch, args.n, args.n):
        print(f"warning: custom_kernel returned {type(out).__name__}", flush=True)


if __name__ == "__main__":
    main()
"""

TASK = """from __future__ import annotations

import torch

input_t = torch.Tensor
output_t = torch.Tensor
"""


@app.function(image=image, gpu="B200", timeout=3600, volumes={"/cache": cache_vol})
def profile_cholesky(
    source: str,
    batch: int,
    n: int,
    seed: int,
    digest: str,
) -> tuple[str, list[tuple[str, str]]]:
    import os
    import subprocess
    import sys
    import uuid

    work = Path("/workspace")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(source, encoding="utf-8")
    (work / "task.py").write_text(TASK, encoding="utf-8")
    (work / "runner.py").write_text(RUNNER, encoding="utf-8")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_name = f"cholesky_b{batch}_n{n}_{digest[:16]}_{stamp}_{uuid.uuid4().hex[:8]}"
    output_dir = Path("/cache/nsys") / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    report_base = output_dir / "profile"
    report_path = report_base.with_suffix(".nsys-rep")
    sqlite_path = report_base.with_suffix(".sqlite")
    stats_path = output_dir / "stats.txt"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work)
    command = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,cublas",
        "--sample=none",
        "--cpuctxsw=none",
        "--cudabacktrace=none",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--force-overwrite=true",
        "--export=sqlite",
        "--output",
        str(report_base),
        "--",
        sys.executable,
        str(work / "runner.py"),
        "--batch",
        str(batch),
        "--n",
        str(n),
        "--seed",
        str(seed),
    ]
    print("command: " + " ".join(command), flush=True)
    profiled = subprocess.run(command, env=env)
    if profiled.returncode != 0:
        raise RuntimeError(f"nsys profile failed: {profiled.returncode}")
    for path in (report_path, sqlite_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"nsys produced no nonempty artifact: {path}")

    stats = subprocess.run(
        ["nsys", "stats", "--format", "table", "--output", "-", str(sqlite_path)],
        capture_output=True,
        text=True,
    )
    if stats.stdout:
        print(stats.stdout, end="" if stats.stdout.endswith("\n") else "\n", flush=True)
    if stats.stderr:
        print(stats.stderr, end="" if stats.stderr.endswith("\n") else "\n", flush=True)
    if stats.returncode != 0:
        raise RuntimeError(f"nsys stats failed: {stats.returncode}")
    if not stats.stdout.strip():
        raise RuntimeError("nsys stats produced an empty summary")
    stats_path.write_text(stats.stdout, encoding="utf-8")

    required = (report_path, sqlite_path, stats_path)
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing or empty Nsight Systems artifacts: {missing}")

    cache_vol.commit()
    return run_name, [(path.name, str(path.relative_to("/cache"))) for path in required]


@app.local_entrypoint()
def main(
    submission: str,
    output: str,
    batch: int,
    n: int,
    seed: int,
    digest: str = "",
) -> None:
    source = Path(submission).read_text(encoding="utf-8")
    run_name, artifacts = profile_cholesky.remote(source, batch, n, seed, digest)

    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[str] = []
    for relative, remote_path in artifacts:
        local_path = destination / relative
        with local_path.open("wb") as fileobj:
            cache_vol.read_file_into_fileobj(remote_path, fileobj)
        if local_path.stat().st_size == 0:
            raise RuntimeError(f"downloaded empty Nsight Systems artifact: {local_path}")
        downloaded.append(str(local_path))
        print(f"downloaded {local_path}", flush=True)

    print(json.dumps({"run_name": run_name, "artifacts": downloaded}), flush=True)
