# saxs-fft

[![tests](https://github.com/hansoncjc/saxs-fft/actions/workflows/tests.yml/badge.svg)](https://github.com/hansoncjc/saxs-fft/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Structure factor S(q), form factor P(q), and SAS intensity I(q) calculation from HOOMD-Blue GSD trajectory files using GPU-accelerated FFT.

---

## Installation

```bash
# Editable / development install
pip install -e .

# With dev dependencies (tests, linting)
pip install -e ".[dev]"

# Directly from GitHub
pip install git+https://github.com/hansoncjc/saxs-fft.git
```

---

## Quick Start

```python
from saxsfft import StructureFactor, Sphere, SphereIntensity
import matplotlib.pyplot as plt

gsd_path = "path/to/trajectory.gsd"

# --- 1. Structure factor S(q) ---
sf = StructureFactor(
    gsd_path,
    N_grid=50,
    frames="last:100",        # average over last 100 frames
    step=5,                   # optional: keep every 5th frame of the window (default 1)
    particle_diameter=10.0,   # physical diameter in nm (optional)
)
q, Sq = sf.compute_s_1d()    # q in Å⁻¹ when particle_diameter is set

# --- 2. Form factor P(q) (standalone) ---
sphere = Sphere(radius=50.0)  # radius in Å (must match q units)
Pq = sphere.Compute_Pq(q)

# --- 3. Full intensity I(q) = prefactor × V × S(q) × P(q) ---
si = SphereIntensity(
    volume_fraction=0.1,
    sld_sample=10.0,    # scattering length density, 10⁻⁶ Å⁻²
    sld_solvent=0.0,
)
si.set_structure_factor(gsd_path, N_grid=50, frames="last:100", step=5, particle_diameter=10.0)
si.set_form_factor()   # auto-derives radius from particle_diameter
q, Iq = si.compute_Iq()

plt.loglog(q, Sq, label="S(q)")
plt.loglog(q, Pq, label="P(q)")
plt.loglog(q, Iq, label="I(q)")
plt.xlabel(r"$q$ (Å$^{-1}$)")
plt.legend()
plt.show()
```

---

## Usage Guide

### 1. Structure Factor S(q)

`StructureFactor` reads particle positions from a GSD file, bins them onto a 3D grid, computes the 3D FFT, and radially averages to produce S(q).

```python
from saxsfft import StructureFactor

sf = StructureFactor(
    gsd_path      = "trajectory.gsd",
    N_grid        = 50,              # grid points along the smallest box dimension
    frames        = "last:100",      # frame selection (default "last:100")
    step          = 1,               # keep every step-th frame of the window (default 1)
    particle_diameter = 10.0,        # physical diameter in nm (see Unit Conventions)
    trim          = slice(3, -3),    # discard FFT artefacts near q boundaries
    device        = "cuda",          # or "cpu"
    dtype         = None,            # defaults to torch.float64
)

q, Sq = sf.compute_s_1d()
```

> **Note:** Calling `compute_s_1d()` automatically saves the result as `average_structure_factor.npy` and `average_structure_factor.txt` in the same directory as your source GSD file.

#### Frame Selection

| `frames` value | Behaviour |
|----------------|-----------|
| `"all"` | Every frame in the file |
| `"last:N"` | The last *N* frames (default `"last:100"`). *N* is an upper bound — asking for more frames than the file holds returns all of them |
| `int` | A single frame, by 0-based index |
| iterable of `int` | Specific 0-based indices: a list, tuple, `range`, or NumPy integer array such as `np.arange(0, 100, 5)` |

Negative indices are not supported — use `"last:N"` to select from the end of
the trajectory. Anything outside the four forms above raises `ValueError` with a
message listing what is accepted.

##### Stepping

`step` keeps every *step*-th frame and **only applies to `"last:N"`**. Supplying
it alongside any other selector is ignored, with a `UserWarning`.

The order of operations matters: the last *N* frames are selected **first**, and
`[::step]` is then applied to that window **starting from its oldest frame**. The
newest frame is therefore not necessarily included:

```python
# On a 6-frame trajectory:
frames="last:6", step=2   # -> frames 0, 2, 4   (frame 5 is dropped)
frames="last:5", step=2   # -> frames 1, 3, 5
```

#### Unit Conventions

The simulation uses **reduced units** where the particle diameter σ = 1.
When you supply `particle_diameter` (in **nm**), `compute_s_1d` automatically converts q:

```
q [Å⁻¹] = q_reduced / (particle_diameter [nm] × 10)
```

If `particle_diameter` is **not** set, q is returned in reduced units (σ⁻¹).

---

### 2. Form Factor P(q)

`Sphere` computes the analytical form factor for a uniform sphere:

$$F(q) = 3\,\frac{\sin(qr) - qr\cos(qr)}{(qr)^3}, \qquad P(q) = |F(q)|^2$$

```python
from saxsfft import Sphere

# radius must be in the SAME units as q (Å if q is in Å⁻¹, nm⁻¹ if q in nm⁻¹, etc.)
sphere = Sphere(radius=50.0)   # 50 Å = 5 nm radius

Fq = sphere.Compute_Fq(q)     # scattering amplitude F(q)
Pq = sphere.Compute_Pq(q)     # form factor P(q) = |F(q)|²
```

> **Unit note:** `qr` must be dimensionless.  
> If `particle_diameter = 10.0` nm → q is in Å⁻¹ → radius must be in **Å** (= 50 Å for a 10 nm particle).

---

### 3. Full Intensity I(q) with `SphereIntensity`

`SphereIntensity` wraps S(q) and P(q) into the full scattering intensity:

$$I(q) = \phi \,(\Delta\rho)^2 \cdot V \cdot S(q) \cdot P(q)$$

where φ is the volume fraction, Δρ is the contrast (SLD difference), and V is the particle volume.

```python
from saxsfft import SphereIntensity

si = SphereIntensity(
    volume_fraction = 0.1,
    sld_sample      = 10.0,   # 10⁻⁶ Å⁻²
    sld_solvent     = 0.0,
)

# Load trajectory and set S(q) source
si.set_structure_factor(
    "trajectory.gsd",
    N_grid=50,
    frames="last:100",
    step=5,
    particle_diameter=10.0,   # nm
)

# Set P(q) — auto-derives radius in Å from particle_diameter
si.set_form_factor()

# Alternatively, pass radius explicitly in Å:
# si.set_form_factor(radius=50.0)

q, Iq = si.compute_Iq()
```

---

### 4. Plotting

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# S(q)
q, Sq = sf.compute_s_1d()
axes[0].semilogy(q, Sq)
axes[0].set_xlabel(r"$q$ (Å$^{-1}$)")
axes[0].set_ylabel(r"$S(q)$")
axes[0].set_title("Structure Factor")

# P(q)
sphere = Sphere(radius=50.0)   # 5 nm radius → 50 Å
Pq = sphere.Compute_Pq(q)
axes[1].semilogy(q, Pq)
axes[1].set_xlabel(r"$q$ (Å$^{-1}$)")
axes[1].set_ylabel(r"$P(q)$")
axes[1].set_title("Form Factor")

# I(q)
axes[2].loglog(q, Iq)
axes[2].set_xlabel(r"$q$ (Å$^{-1}$)")
axes[2].set_ylabel(r"$I(q)$ (a.u.)")
axes[2].set_title("Intensity")

plt.tight_layout()
plt.show()
```

---

## GPU Support

```python
import torch

sf = StructureFactor(
    "trajectory.gsd",
    N_grid=50,
    device=torch.device("cuda"),   # GPU
    dtype=torch.float64,
)
```

CPU fallback:
```python
sf = StructureFactor("trajectory.gsd", N_grid=50, device="cpu")
```

---

## API Reference

| Symbol | Class / Function | Description |
|--------|-----------------|-------------|
| S(q) | `StructureFactor` | Structure factor from GSD trajectory via FFT |
| P(q) | `Sphere` | Analytical sphere form factor |
| I(q) | `SphereIntensity` | Full SAS intensity = prefactor × V × S(q) × P(q) |
| — | `extract_positions` | Extract particle positions from a GSD file |
| — | `read_configuration` | Read positions + box from a text snapshot |

### `StructureFactor(gsd_path, N_grid, frames, step, particle_diameter, trim, device, dtype)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `gsd_path` | str | — | Path to `.gsd` file |
| `N_grid` | int | — | Grid points along smallest box axis |
| `frames` | str / int / list | `"last:100"` | Frame selection |
| `step` | int | `1` | Keep every step-th frame; only applies to `"last:N"` |
| `particle_diameter` | float | `None` | Physical diameter in **nm**; enables Å⁻¹ output |
| `trim` | slice | `slice(3,-3)` | Trim FFT edge artefacts |
| `device` | str / device | `None` | Torch compute device |
| `dtype` | dtype | `torch.float64` | Floating-point precision |

### `Sphere(radius)`

| Parameter | Description |
|-----------|-------------|
| `radius` | Sphere radius in the **same length units as 1/q** |

### `SphereIntensity(volume_fraction, sld_sample, sld_solvent)`

| Method | Description |
|--------|-------------|
| `.set_structure_factor(...)` | Load GSD and compute S(q) source |
| `.set_form_factor(radius=None)` | Set P(q); auto-converts nm → Å from `particle_diameter` |
| `.compute_Iq()` | Returns `(q, Iq)` |

---

## Example Notebooks

- [`examples/Sphere_pytorch.ipynb`](examples/Sphere_pytorch.ipynb) — Full worked example: S(q), P(q), and I(q) for a sphere system using PyTorch/GPU

---

## Testing

```bash
pip install -e ".[dev]"
pytest
```

The suite is self-contained — no external data files, no GPU, no network — and
runs in about one second.

| File | What it covers |
|------|----------------|
| `tests/test_structurefactor_api.py` | Interface contracts: return shapes, `trim`, `particle_diameter`, `dtype`, and CPU/CUDA agreement |
| `tests/test_structurefactor_physics.py` | Numerical correctness against closed-form references: the 1/N normalisation, the ideal-gas limit S(q) → 1, translation invariance, and FCC/BCC Bragg peak positions |
| `tests/test_gsd_pipeline.py` | The GSD → text → `StructureFactor` path: extraction, frame-selection semantics, input validation, and the HOOMD centered-box convention |

Physics is checked against analytic results rather than stored reference curves:
a perfect FCC crystal must place its (111) reflection at q = 2π√3/a and a BCC
crystal its (110) at q = 2π√2/a, each asserted to within one radial bin width.

The CPU/GPU agreement test is skipped automatically when no CUDA device is
present, so `pytest` on a laptop reports one skip.

To run one file:

```bash
pytest tests/test_structurefactor_physics.py -v
```

---

## Requirements

- Python ≥ 3.8
- `numpy`
- `torch`
- `gsd`
- `matplotlib`
