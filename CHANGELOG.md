# Changelog

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
