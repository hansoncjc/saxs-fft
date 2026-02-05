# Changelog

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
