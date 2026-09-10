"""CPU/GPU benchmark for the S(q) kernel.

One configuration per invocation, one CSV row appended per invocation.  Thread
counts are set from the environment before torch is imported, which is the only
reliable way to pin intra-op parallelism, so the sweep is driven by the shell
(see bench_hyak.sbatch) rather than from inside a single process.

Positions are synthetic by default, so the benchmark is reproducible by anyone
who clones the repo -- no private trajectory needed.  Pass --gsd to time a real
trajectory instead.

Examples
--------
    python benchmark_sq.py --device cpu --threads 1  --ngrid 200
    python benchmark_sq.py --device cpu --threads 16 --ngrid 200
    python benchmark_sq.py --device cuda            --ngrid 200
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--threads", type=int, default=0,
                   help="torch intra-op threads for CPU runs; 0 = leave as-is. "
                        "Must match OMP_NUM_THREADS, which the caller sets.")
    p.add_argument("--ngrid", type=int, default=200)
    p.add_argument("--particles", type=int, default=8000)
    p.add_argument("--box", type=float, default=40.0,
                   help="Cubic box edge in reduced units (sigma = 1).")
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gsd", default=None,
                   help="Optional GSD trajectory; its first frame replaces the "
                        "synthetic configuration.")
    p.add_argument("--csv", default="benchmark_results.csv")
    p.add_argument("--label", default="", help="Free-text tag for this row.")
    return p.parse_args()


def gpu_name():
    import torch
    if not torch.cuda.is_available():
        return "none"
    return torch.cuda.get_device_name(0)


def cpu_model():
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        return platform.processor() or "unknown"
    except Exception:
        return "unknown"


def load_positions(args):
    import numpy as np
    if args.gsd:
        from saxsfft.gsdio import extract_positions
        from saxsfft.utils import read_configuration
        txt = Path(args.gsd).with_suffix(".bench.txt")
        extract_positions(args.gsd, str(txt), frames="last:1")
        x, box = read_configuration(str(txt), frames=0)
        return np.ascontiguousarray(x[0]), np.asarray(box[0], dtype=float)
    rng = np.random.default_rng(0)
    box = np.full(3, float(args.box))
    return rng.random((args.particles, 3)) * box, box


def main():
    args = parse_args()

    # Threads must be pinned before the first torch op.  OMP_NUM_THREADS is set
    # by the caller; this asserts the two agree so a mislabelled row is
    # impossible.
    if args.device == "cpu" and args.threads:
        omp = os.environ.get("OMP_NUM_THREADS")
        if omp is not None and int(omp) != args.threads:
            sys.exit(f"OMP_NUM_THREADS={omp} but --threads={args.threads}; "
                     "set them to the same value.")

    import numpy as np
    import torch
    from saxsfft.structurefactor import compute_s_1d

    if args.device == "cpu" and args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("--device cuda requested but torch.cuda.is_available() is False")

    dtype = getattr(torch, args.dtype)
    x, box = load_positions(args)

    def run():
        return compute_s_1d(x, box, args.ngrid, device=args.device, dtype=dtype)

    for _ in range(args.warmup):          # CUDA context + cuFFT plan caching
        run()
    if args.device == "cuda":
        torch.cuda.synchronize()
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(args.repeats):
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        q, num, cnt = run()
        if args.device == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    times = np.asarray(times)
    n_grid_actual = int(round(args.ngrid * max(box) / min(box)))

    row = {
        "label": args.label,
        "device": args.device,
        "dtype": args.dtype,
        "threads": args.threads if args.device == "cpu" else 0,
        "ngrid": args.ngrid,
        "grid_points": n_grid_actual ** 3,
        "particles": len(x),
        "median_s": float(np.median(times)),
        "min_s": float(times.min()),
        "std_s": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "repeats": args.repeats,
        "gpu": gpu_name() if args.device == "cuda" else "",
        "cpu": cpu_model(),
        "torch": torch.__version__,
        "cuda_toolkit": torch.version.cuda or "",
        "peak_gpu_gib": (torch.cuda.max_memory_allocated() / 2 ** 30) if args.device == "cuda" else "",
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
    }

    path = Path(args.csv)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            w.writeheader()
        w.writerow(row)

    print(f"{args.device:4s} dtype={args.dtype} threads={row['threads']:3d} "
          f"ngrid={args.ngrid:4d} -> median {row['median_s']:.4f} s "
          f"(min {row['min_s']:.4f}, sd {row['std_s']:.4f}, n={args.repeats})")


if __name__ == "__main__":
    main()
