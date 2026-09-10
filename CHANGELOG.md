# Changelog

## v0.3.1 (2026-09-10)

### Fixed
- **The q-grid was built in single precision regardless of the requested
  dtype.** `_compute_q3_grid_torch` called `torch.fft.fftfreq` without a
  `dtype`, so it returned the torch default (float32), and multiplying by
  `dq[i]` did not promote the result: PyTorch gives 0-dim operands lower
  priority in type promotion, and `dq[i]` is 0-dim, so `float32 (3-D) *
  float64 (0-D)` stays float32. The q-grid, `|q|`, the radial bin edges and
  the reported q axis therefore carried ~1e-7 relative error even when
  `dtype=torch.float64` was requested. The density grid and the FFT itself
  were always genuine float64, which is why the median S(q) shell still agreed
  to 1e-15 and only a handful of shells were affected.

  Two consequences, both now fixed:

  - **CPU and GPU could disagree.** On a cubic box `|q|/dq` equals
    `sqrt(i^2+j^2+k^2)`, which is exactly an integer whenever `i^2+j^2+k^2` is
    a perfect square — putting 41,086 of 8,000,000 grid points (0.51 %) exactly
    on a radial bin edge at N_grid = 200. Single-precision noise moved them off
    it and the two devices rounded in different directions: 536 grid points
    landed in different shells, changing S(q) by up to 9.4e-4 in the worst
    shell. After the fix, shell occupancies are identical and S(q) agrees to
    1.2e-14.
  - **Silent precision loss.** float64 is documented as the default "for
    numerical fidelity"; the q axis did not honour it.

  Timing impact is negligible — the FFT dominates and was always double
  precision. Peak GPU memory in float64 mode does rise, since the q-grid is now
  genuinely 8 bytes per element.

### Added
- Three regression tests, bringing the suite to 47:
  - `test_q_grid_honours_dtype` — the q-grid must come back in the dtype that
    was asked for, checked for float32 and float64. A dtype assertion rather
    than a numerical one, because no tolerance anywhere else in the suite was
    tight enough to see a 1e-7 error in q.
  - `test_cubic_q_grid_is_exact_multiples_of_dq` — the numerical counterpart:
    on a cubic box every q component must be an exact integer multiple of
    `dq = 2*pi/L`.
  - `test_cuda_matches_cpu_cubic_box` — CPU/GPU agreement on a cubic box. The
    existing `test_cuda_matches_cpu` uses the deliberately non-cubic fixture,
    which is 34x less densely populated with exact bin-edge ties and so could
    not see this class of bug, while production runs are cubic.
- `benchmarks/` — CPU vs GPU measurements taken in a single Hyak allocation
  (NVIDIA A40 against a saturated 16-core Xeon Gold 6230R, same node, same
  data): **8.9x in float64 and 14.8x in float32** at a 300^3 grid
  (2.7e7 points), plus CPU thread-scaling and peak-memory figures, the
  benchmark script and an example Slurm submission. See
  `benchmarks/README.md`.
- GitHub Actions CI running the pytest suite on Python 3.9, 3.10, 3.11 and
  3.12; tests, license and Python badges in the README.

### Changed
- `test_peak_ratio_identifies_lattice` was passing by accident, and the dtype
  fix exposed it. Its FCC case used a 6-cell lattice, where the (111) and (200)
  reflections are only 1.6 radial bins apart; (200) — whose `|G|` is an exact
  integer multiple of `dq`, so it sits exactly on a bin edge — was therefore a
  shoulder of the stronger (111) rather than a local maximum. The test had been
  finding it only because the single-precision q-grid jittered it into the next
  shell up; deterministic binning moved it onto (111)'s shoulder and the peak
  detector fell through to (220), giving a ratio of 1.571 instead of 1.155.
  The FCC case now uses 12 cells on a 128^3 grid, which separates the
  reflections by 3.2 bins while keeping the grid resolution per lattice
  constant (`a/Delta` = 10.667) and the Nyquist limit (16.76) at the values the
  6-cell case had — raising the cell count alone halves both and introduces a
  nearest-grid-point aliasing artifact below (111) that the detector picks up
  instead. BCC is unaffected: its two lowest reflections are 4.7 bins apart.
- `requires-python` raised to `>=3.9`, classifiers updated to span 3.9-3.12.
  Python 3.8 was already unusable in practice: PyTorch dropped it after 2.4.
- `gsd` is constrained to `<3.4` on Python 3.9, where upstream ships no wheel
  and pip would otherwise fall back to building from source.

### Known
- `_compute_q3_grid_torch` materialises three full `N_grid^3` meshgrids, which
  dominates peak memory: 1.81 GiB in float64 at N_grid = 300, extrapolating to
  roughly 67 GiB at N_grid = 1000 — beyond a 48 GiB A40. Computing `|q|` by
  broadcasting instead would cut this substantially.
- The radial bin edges are derived from the minimum and maximum of the computed
  `|q|`, which differ by one ULP between devices, so the edges themselves are
  not bitwise identical across CPU and GPU. This has no observable effect on
  S(q) at any size tested, but the edges are a pure function of `box` and
  `N_grid` and would be better computed analytically from those.

## v0.3.0 (2026-09-08)

### Added
- **Test suite**: 43 tests under `tests/`, run with `pytest`. Three layers —
  interface contracts (`test_structurefactor_api.py`), numerical correctness
  against closed-form references (`test_structurefactor_physics.py`), and the
  GSD I/O pipeline (`test_gsd_pipeline.py`). Physics is verified against
  analytic results — the 1/N normalisation, the ideal-gas limit S(q) → 1,
  translation invariance, and the FCC (111) / BCC (110) Bragg peak positions —
  rather than against stored reference curves. Test-only lattice builders live
  in `tests/conftest.py` and are deliberately **not** part of the public API.
- `[tool.pytest.ini_options]` in `pyproject.toml`, so a bare `pytest` from the
  repository root finds the suite.
- **NumPy frame specifiers**: `frames` now accepts NumPy integers and integer
  arrays, e.g. `frames=np.arange(0, 100, 5)`.
- **Frame specifier validation**: malformed `frames` values raise `ValueError`
  with a message listing the four accepted forms.
- **`step` validation**: `step` must be a positive integer.
- **Out-of-range reporting**: requesting a frame index past the end of a
  trajectory raises `ValueError` naming the file and its actual frame count.

### Changed
- **BREAKING — default `step` changed from `5` to `1`** in `StructureFactor`
  and `Intensity.set_structure_factor`. The previous default silently thinned
  the default `"last:100"` window down to 20 frames. Calls that relied on the
  default now process 5x more frames: better statistics, proportionally slower.
  See the migration guide below.
- **BREAKING — `step` outside `"last:N"` no longer raises.** It is now ignored
  with a `UserWarning`. In particular `StructureFactor(path, N_grid,
  frames="all")` used to raise `ValueError` because of the `step=5` default;
  it now works.
- Expanded the `frames` / `step` docstrings on `read_configuration`,
  `StructureFactor` and `set_structure_factor`, including the non-obvious
  ordering rule: `"last:N"` is applied first, then `[::step]` walks that window
  from its oldest frame, so the newest frame may be excluded.

### Fixed
- `frames=True` was silently treated as frame 1 (`bool` is a subclass of `int`).
- `frames=[0, 2.0]` silently accepted floating-point indices.
- `frames=np.arange(...)` failed with `ValueError: truth value of an array ...
  is ambiguous`; NumPy arrays were unusable as frame specifiers.
- `frames=np.int64(3)` failed with `TypeError: 'numpy.int64' object is not
  iterable`.
- Invalid strings such as `frames="first:2"` raised
  `TypeError: '>' not supported between instances of 'int' and 'str'`.
- Out-of-range and negative frame indices surfaced as NumPy's
  `ValueError: need at least one array to stack`.
- `step=0` and negative `step` values were silently ignored — they fell through
  the internal `if step > 1` guard and behaved as `step=1`.

### Removed
- README references to `tests/single_sim_test.py`,
  `tests/structurefactor_test.py` and `tests/cpu_gpu_bench.py`, which were
  deleted in 99de1e1. `single_sim_test.py` was a bare `unittest` runner
  superseded by `pytest`; `cpu_gpu_bench.py` was a benchmark script with
  hard-coded cluster paths and no assertions — its GPU coverage is replaced by
  the `test_cuda_matches_cpu` correctness test.

### Migration Guide

**1. The `step` default changed from 5 to 1.**

```python
# v0.2.2: the last 100 frames were thinned to 20
sf = StructureFactor(path, N_grid=50)

# v0.3.0: the same call processes all 100 frames (~5x slower, better averaging).
# To keep the old behaviour, pass step explicitly:
sf = StructureFactor(path, N_grid=50, frames="last:100", step=5)
```

**2. `frames="all"` no longer needs `step=1`.**

```python
# v0.2.2
StructureFactor(path, N_grid=50, frames="all")           # ValueError
StructureFactor(path, N_grid=50, frames="all", step=1)   # OK

# v0.3.0
StructureFactor(path, N_grid=50, frames="all")           # OK
StructureFactor(path, N_grid=50, frames="all", step=3)   # OK, warns, step ignored
```

**3. Previously-silent bad input now raises.** If you were passing `frames=True`,
float indices, or a non-positive `step`, those calls will now fail with a
`ValueError` describing the problem.

## v0.2.2 (2026-03-16)
### Added
- **`step` parameter**: Added `step` parameter to `StructureFactor` and `Intensity.set_structure_factor`. This allows sub-sampling frames within a selection window (e.g., `"last:100"` with `step=5` processes 20 frames).
- **Enhanced Logging**: Improved console output during data extraction to show the GSD path, total frames in the file, and exactly how many frames are being extracted.
- **Automatic S(q) Saving**: `compute_s_1d()` now automatically saves the averaged structure factor result to both `average_structure_factor.npy` and `average_structure_factor.txt` in the source GSD file's directory.

### Changed
- **New Defaults**: Default `frames` changed from `"last:150"` to `"last:100"` and default `step` set to `5`. This reduces default computation time by ~7.5x while still providing representative averaging.

### Fixed
- Robustness improvement in `read_configuration` to handle trailing whitespace or malformed lines at the end of text configuration files.

## v0.2.1 (2026-03-06)
### Fixed
- **Unit mismatch in `SphereIntensity.set_form_factor`**: `particle_diameter` is stored in
  **nm**, but `compute_s_1d` returns `q` in **Å⁻¹** (dividing reduced-unit q by
  `diameter × 10`).  The auto-derived radius was previously left in nm, making `qr`
  10× too small and shifting all P(q) features to artificially high q.  The radius is
  now correctly converted to Å (`diameter / 2 × 10`) so that `qr` is dimensionless
  and consistent with the q axis.

### Changed
- Default `frames` value changed from `"last:5"` to `"last:150"` in both
  `StructureFactor` and `SphereIntensity.set_structure_factor` for better
  statistical averaging out of the box.

### Added
- `particle_diameter` parameter on `StructureFactor` and `SphereIntensity.set_structure_factor`
  for automatic physical-unit conversion of q from reduced to Å⁻¹.
- Expanded README with a full usage guide covering S(q), P(q), I(q), unit conventions,
  plotting examples, and an API reference table.

---

## v0.2.0 (2026-02-03)
### Changed
- **BREAKING**: Renamed package from `gsd2sas` to `saxsfft`
- Reorganized project structure as proper pip-installable package
- Updated imports to use relative imports within package
- Changed default frames selection to `'last:5'` in `set_structure_factor()` and `StructureFactor` constructor

### Added
- Added `pyproject.toml` for modern Python packaging
- Added `saxsfft/__init__.py` with public API exports
- Added `.gitignore` for Python projects
- Added comprehensive README with installation and usage instructions
- Package now installable via `pip install -e .`

### Migration Guide
To migrate from v0.1.0 to v0.2.0, update your imports:
```python
# Old (v0.1.0)
from gsd2sas.structurefactor import StructureFactor

# New (v0.2.0)
from saxsfft import StructureFactor
```

## v0.1.0
- Torch backend for structure factor (GPU-capable, float64).
- Added `frames="last:N"` option for tail-frame selection.
- Added `examples/Sphere_pytorch.ipynb` for user to jump-start using the functions.
- Added `tests/cpu_gpu_bench.py` and `tests/single_sim_test.py` for GPU-operation benchmark and unittests.
