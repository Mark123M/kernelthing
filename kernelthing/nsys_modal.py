"""Modal B200 execution and Nsight Systems capture for one Cholesky submission.

This module is intentionally optional: normal kernelthing imports do not touch
``modal``. ``popcorn.deadlock_check_submission`` uses it for one exact-shape
kernel call, while ``popcorn.profile_nsys_submission`` uses it for a warmed
Nsight Systems capture.
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
import json
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--completion-check", action="store_true")
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

    if args.completion_check:
        print("kernelthing: exact-shape custom_kernel launch starting", flush=True)
        with torch.no_grad():
            out = submission.custom_kernel(data)
        torch.cuda.synchronize()
        if not hasattr(out, "shape") or tuple(out.shape) != (args.batch, args.n, args.n):
            raise RuntimeError(
                "custom_kernel returned "
                f"{type(out).__name__}, expected shape {(args.batch, args.n, args.n)}"
            )
        print(
            json.dumps(
                {
                    "kind": "kernelthing-modal-kernel-run",
                    "status": "passed",
                    "shape": [args.batch, args.n, args.n],
                }
            ),
            flush=True,
        )
        return

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


@app.function(image=image, gpu="B200", timeout=300)
def check_cholesky(
    source: str,
    batch: int,
    n: int,
    seed: int,
    timeout_s: int,
) -> dict[str, object]:
    """Run ``custom_kernel`` once at the scored shape with a hard execution cap."""
    import os
    import subprocess
    import sys
    import time

    work = Path("/workspace")
    work.mkdir(parents=True, exist_ok=True)
    (work / "submission.py").write_text(source, encoding="utf-8")
    (work / "task.py").write_text(TASK, encoding="utf-8")
    (work / "runner.py").write_text(RUNNER, encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(work)
    command = [
        sys.executable,
        str(work / "runner.py"),
        "--batch",
        str(batch),
        "--n",
        str(n),
        "--seed",
        str(seed),
        "--completion-check",
    ]
    started = time.monotonic()
    try:
        ran = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        wall_s = time.monotonic() - started
        stderr = (
            exc.stderr.decode(errors="replace")
            if isinstance(exc.stderr, bytes)
            else (exc.stderr or "")
        )
        stdout = (
            exc.stdout.decode(errors="replace")
            if isinstance(exc.stdout, bytes)
            else (exc.stdout or "")
        )
        tail = (stderr + stdout).strip()[-500:]
        launched = "exact-shape custom_kernel launch starting" in (stderr + stdout)
        if not launched:
            return {
                "kind": "kernelthing-deadlock-check",
                "status": "failed",
                "ok": False,
                "timed_out": False,
                "wall_s": wall_s,
                "error": (
                    f"exact-shape Modal setup exceeded {timeout_s}s before custom_kernel "
                    f"launched for batch={batch}, n={n}, seed={seed}; this is not a "
                    "classified GPU kernel deadlock"
                ),
                "tail": tail,
            }
        return {
            "kind": "kernelthing-deadlock-check",
            "status": "timed_out",
            "ok": False,
            "timed_out": True,
            "wall_s": wall_s,
            "error": (
                f"exact-shape Modal run exceeded {timeout_s}s while "
                f"executing custom_kernel for batch={batch}, n={n}, seed={seed}; "
                "likely GPU kernel deadlock (or an exceptionally slow compile/launch). "
                "Do not start NCU or NSYS for this kernel."
            ),
            "tail": tail,
        }

    wall_s = time.monotonic() - started
    tail = ((ran.stderr or "") + (ran.stdout or "")).strip()[-500:]
    if ran.returncode != 0:
        return {
            "kind": "kernelthing-deadlock-check",
            "status": "failed",
            "ok": False,
            "timed_out": False,
            "wall_s": wall_s,
            "error": (
                f"exact-shape Modal run failed for batch={batch}, n={n}, seed={seed} "
                f"(exit {ran.returncode}): {tail}"
            ),
        }
    return {
        "kind": "kernelthing-deadlock-check",
        "status": "passed",
        "ok": True,
        "timed_out": False,
        "wall_s": wall_s,
        "shape": [batch, n, n],
        "error": "",
    }


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
    batch: int,
    n: int,
    seed: int,
    output: str = "",
    digest: str = "",
    completion_check: bool = False,
    deadlock_timeout_s: int = 120,
) -> None:
    source = Path(submission).read_text(encoding="utf-8")
    if completion_check:
        print(
            json.dumps(
                check_cholesky.remote(source, batch, n, seed, deadlock_timeout_s)
            ),
            flush=True,
        )
        return
    if not output:
        raise ValueError("--output is required unless --completion-check is set")

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
