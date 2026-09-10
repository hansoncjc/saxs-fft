# Benchmarks

CPU vs GPU timings for the S(q) kernel — grid binning, 3D FFT, and radial
averaging — measured on one Hyak node so that both sides of the comparison
share the same hardware, the same data and the same software stack.

Measured on **saxs-fft v0.3.1**, i.e. after the q-grid dtype fix. Numbers from
earlier versions are not comparable: before that fix the q-grid was built in
single precision even in float64 mode.

| | |
|---|---|
| GPU | NVIDIA A40 (48 GiB) |
| CPU | Intel Xeon Gold 6230R @ 2.10 GHz, 16 cores allocated |
| torch | 2.7.1+cu118, CUDA 11.8 |
| Configuration | 8,000 particles, cubic box, single frame |

## Headline

At the grid size used in production (N_grid = 300, 2.7 × 10⁷ grid points):

| dtype | 16 CPU cores | A40 | speedup |
|---|---|---|---|
| float64 (current default) | 0.489 s | 0.0547 s | **8.9×** |
| float32 | 0.363 s | 0.0245 s | **14.8×** |

The speedup is quoted against a **saturated 16-core CPU**, not against a single
core. Against one core the same measurements give 72× (float64) and 109×
(float32); those are the less meaningful numbers and are included in the tables
below only for completeness.

## Full results

Median of 7 timed runs, after 2 discarded warm-up runs (CUDA context creation
and cuFFT plan caching). `torch.cuda.synchronize()` brackets every GPU
measurement — without it the timings measure kernel launch, not kernel
execution.

### float64

| N_grid | grid points | 1 core | 2 | 4 | 8 | 16 | A40 | GPU vs 16c | GPU vs 1c |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.6 × 10⁵ | 0.0195 | 0.0126 | 0.0088 | 0.0064 | 0.0065 | 0.0020 | 3.3× | 9.8× |
| 100 | 1.0 × 10⁶ | 0.0921 | 0.0555 | 0.0336 | 0.0204 | 0.0188 | 0.0036 | 5.2× | 25.5× |
| 150 | 3.4 × 10⁶ | 0.3360 | 0.2062 | 0.1134 | 0.0685 | 0.0485 | 0.0081 | 6.0× | 41.4× |
| 200 | 8.0 × 10⁶ | 1.1281 | 0.6230 | 0.3449 | 0.2100 | 0.1472 | 0.0170 | 8.7× | 66.4× |
| 300 | 2.7 × 10⁷ | 3.9334 | 2.1312 | 1.1764 | 0.6980 | 0.4886 | 0.0547 | **8.9×** | 71.9× |

### float32

| N_grid | grid points | 1 core | 2 | 4 | 8 | 16 | A40 | GPU vs 16c | GPU vs 1c |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 2.6 × 10⁵ | 0.0128 | 0.0090 | 0.0073 | 0.0049 | 0.0053 | 0.0020 | 2.7× | 6.5× |
| 100 | 1.0 × 10⁶ | 0.0626 | 0.0360 | 0.0222 | 0.0156 | 0.0127 | 0.0024 | 5.2× | 25.8× |
| 150 | 3.4 × 10⁶ | 0.2455 | 0.1333 | 0.0798 | 0.0481 | 0.0405 | 0.0046 | 8.8× | 53.6× |
| 200 | 8.0 × 10⁶ | 0.6520 | 0.3897 | 0.2138 | 0.1155 | 0.0984 | 0.0091 | 10.8× | 71.7× |
| 300 | 2.7 × 10⁷ | 2.6584 | 1.4360 | 0.8339 | 0.5051 | 0.3627 | 0.0245 | **14.8×** | 108.6× |

All times in seconds. The N_grid = 64 rows carry 10–20 % run-to-run scatter and
should not be read closely; everything from N_grid = 100 up is stable to under
1 % on the GPU.

### Peak GPU memory

| N_grid | float64 | float32 | ratio |
|---|---|---|---|
| 64 | 0.018 GiB | 0.009 GiB | 1.98× |
| 100 | 0.070 GiB | 0.035 GiB | 2.01× |
| 150 | 0.229 GiB | 0.114 GiB | 2.00× |
| 200 | 0.538 GiB | 0.272 GiB | 1.98× |
| 300 | 1.811 GiB | 0.906 GiB | 2.00× |

Extrapolating the N_grid = 300 measurement as N_grid³ (**extrapolation, not
measurement**): float64 needs roughly 67 GiB at N_grid = 1000 and does not fit
on a 48 GiB A40, while float32 needs roughly 34 GiB and does. Polydisperse
systems can require grids that large, which is the practical argument for
single precision as a default.

## What is and is not being timed

**Only the compute kernel.** Positions are already resident in memory; the
timed region is `compute_s_1d`, which covers the host-to-device transfer,
periodic wrapping, cell-list binning onto the density grid, the 3D FFT, the
q-grid construction and the radial average.

**Not the trajectory I/O.** Reading a GSD file and parsing it is
device-independent, so including it would dilute what the comparison is trying
to isolate. It is not negligible in absolute terms — `extract_positions` writes
a text file that `read_configuration` then parses back, and for a multi-frame
trajectory that round trip can dominate end-to-end wall time. Removing it is
tracked separately.

**CPU thread counts are pinned**, with `OMP_NUM_THREADS` and
`torch.set_num_threads` set to the same value and asserted to agree, so a
mislabelled row is not possible. Each configuration runs in its own process,
which is why threads are set from the shell rather than switched at runtime.

## Observations

**The GPU wins at every size tested; there is no crossover.** Even at
N_grid = 64, where the grid is small enough that launch overhead and the
host-to-device copy are a visible fraction of the total, the A40 is 3.3×
faster than 16 cores. The margin grows monotonically with grid size as that
fixed overhead is amortised.

**The CPU path saturates near 8 threads at small grids.** At N_grid = 64 and
100, 8 threads is as fast as or faster than 16 (float64: 0.0064 s vs 0.0065 s).
Even at N_grid = 300 the best 16-core scaling is 8.1× — about 50 % parallel
efficiency. This is memory-bandwidth saturation, not a threading bug: the FFT
and the radial average both stream far more bytes than they do arithmetic.

**float64 costs about 2× float32, not the ~64× the A40's FP64 ratio would
suggest.** GA102 runs double precision at 1/64 of its single-precision rate, so
an ALU-bound kernel would be crippled. The measured ratio is 1.0× at
N_grid = 64 rising to 2.24× at N_grid = 300 — tracking the memory ratio (2.00×)
rather than the ALU ratio. The pipeline is bandwidth-bound, so double precision
costs what the extra bytes cost and the FP64 throughput penalty never becomes
the limiter. A more FP64-capable GPU would not help much here.

## Correctness

A speedup is only meaningful if both devices compute the same thing. The test
suite asserts CPU/GPU agreement on S(q) to `rtol=1e-10`, on both a non-cubic
box and a cubic one — the latter being the degenerate geometry where roughly
0.5 % of grid points land exactly on a radial bin edge. See
`tests/test_structurefactor_api.py`.

## Reproducing

```bash
# One configuration per invocation; a CSV row is appended per run.
python benchmarks/benchmark_sq.py --device cpu  --threads 16 --ngrid 300
python benchmarks/benchmark_sq.py --device cuda              --ngrid 300
```

Positions are synthetic by default, so no trajectory file is needed. Pass
`--gsd path/to/traj.gsd` to time a real one instead.

`bench_hyak.example.sbatch` runs the full sweep in a single Slurm allocation.
It is cluster-specific: adjust the account, partition, module and environment
names before submitting. `results_a40.csv` is the raw output behind every number
above.
