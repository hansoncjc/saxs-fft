"""Shared helpers for the saxs-fft test suite.

The lattice builders below are copied from the author's simulation scripts
(``FCC.py`` / ``BCC.py``) and deliberately kept **test-only**: they are not
exported from ``saxsfft``, so adding this suite leaves the package's public API
untouched.

Ground truth for the physics tests always comes from closed-form expressions
(Bragg peak positions of the FCC/BCC reciprocal lattices), never from stored
reference data.  These helpers therefore only have to *build* the structures -
a bug in them would move the measured peaks away from the analytic prediction
and still be caught.
"""
from __future__ import annotations

import numpy as np
import pytest

from saxsfft.structurefactor import compute_s_1d

SEED = 0


# ---------------------------------------------------------------------------
# Lattice builders
# ---------------------------------------------------------------------------

def _resolve_phi(phi, rho):
    if (phi is None) == (rho is None):
        raise ValueError("Specify exactly one of phi or rho.")
    return phi if phi is not None else rho * np.pi / 6


def fcc_lattice(N, phi=None, rho=None):
    """FCC lattice with ~N particles, exactly commensurate with a cubic box.

    Returns ``(x, L, a)``: positions in ``[0, L)``, box length, lattice
    constant.  ``a`` is returned (unlike the original helper) because the
    analytic Bragg positions are expressed in terms of it.
    """
    phi = _resolve_phi(phi, rho)
    Ncell = int(np.ceil((N / 4) ** (1 / 3)))
    X, Y, Z = np.meshgrid(range(Ncell), range(Ncell), range(Ncell), indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    unit = np.vstack([
        np.column_stack([X, Y, Z]),
        np.column_stack([X + 0.5, Y + 0.5, Z]),
        np.column_stack([X + 0.5, Y, Z + 0.5]),
        np.column_stack([X, Y + 0.5, Z + 0.5]),
    ])
    # phi = (4/a^3) * pi/6  =>  a = (2*pi/(3*phi))^(1/3)
    a = (2 * np.pi / (3 * phi)) ** (1 / 3)
    return unit * a, Ncell * a, a


def bcc_lattice(N, phi=None, rho=None):
    """BCC lattice with ~N particles, exactly commensurate with a cubic box.

    Returns ``(x, L, a)`` - see :func:`fcc_lattice`.
    """
    phi = _resolve_phi(phi, rho)
    Ncell = int(np.ceil((N / 2) ** (1 / 3)))
    X, Y, Z = np.meshgrid(range(Ncell), range(Ncell), range(Ncell), indexing="ij")
    X, Y, Z = X.ravel(), Y.ravel(), Z.ravel()
    unit = np.vstack([
        np.column_stack([X, Y, Z]),
        np.column_stack([X + 0.5, Y + 0.5, Z + 0.5]),
    ])
    # phi = (2/a^3) * pi/6  =>  a = (pi/(3*phi))^(1/3)
    a = (np.pi / (3 * phi)) ** (1 / 3)
    return unit * a, Ncell * a, a


LATTICE_BUILDERS = {"fcc": fcc_lattice, "bcc": bcc_lattice}

# |G| * a / (2*pi) for the two lowest allowed reflections.
#   FCC reciprocal lattice is BCC -> h,k,l all even or all odd -> (111), (200)
#   BCC reciprocal lattice is FCC -> h+k+l even             -> (110), (200)
REFLECTIONS = {
    "fcc": (np.sqrt(3.0), 2.0),
    "bcc": (np.sqrt(2.0), 2.0),
}


def bragg_q(a, order):
    """|q| of a reflection whose |G|*a/(2*pi) equals ``order``."""
    return 2.0 * np.pi * order / a


def add_gaussian_displacement(x, L, sigma_disp, rng=None):
    """Perturb positions with iid Gaussian noise, wrapped into ``[-L/2, L/2)``.

    This is HOOMD's centered box convention, which is what ``write_gsd`` needs.
    """
    if rng is None:
        rng = np.random.default_rng()
    x_centered = x - L / 2
    x_displaced = x_centered + rng.normal(0.0, sigma_disp, size=x_centered.shape)
    return (x_displaced + L / 2) % L - L / 2


# ---------------------------------------------------------------------------
# GSD writing (only used by the pipeline tests)
# ---------------------------------------------------------------------------

def write_gsd(path, frames, L, sigma=1.0):
    """Write a HOOMD GSD trajectory.

    ``frames`` is either a single ``(N, 3)`` array or a sequence of them; one
    GSD frame is written per array.  Positions must already lie in
    ``[-L/2, L/2)``.
    """
    import gsd.hoomd

    if isinstance(frames, np.ndarray) and frames.ndim == 2:
        frames = [frames]

    with gsd.hoomd.open(str(path), "w") as traj:
        for step, positions in enumerate(frames):
            N = positions.shape[0]
            frame = gsd.hoomd.Frame()
            frame.particles.N = N
            frame.particles.types = ["A"]
            frame.particles.typeid = np.zeros(N, dtype=np.uint32)
            frame.particles.position = positions.astype(np.float32)
            frame.particles.diameter = np.full(N, sigma, dtype=np.float32)
            frame.configuration.box = [L, L, L, 0.0, 0.0, 0.0]
            frame.configuration.step = step
            traj.append(frame)
    return path


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def radial_sq(x, box, N_grid, **kwargs):
    """``compute_s_1d`` plus the ``num / cnt`` division, returning ``(q, S)``.

    The module-level ``compute_s_1d`` returns *unnormalised* ``(q, num, cnt)``;
    only the ``StructureFactor`` class divides.  Tests that care about S(q)
    values go through here so the division convention lives in exactly one
    place.
    """
    box = np.asarray(box, dtype=float)
    if box.ndim == 0:
        box = np.repeat(box, 3)
    q, num, cnt = compute_s_1d(x, box, N_grid, **kwargs)
    S = np.divide(num, cnt, out=np.zeros_like(num), where=cnt > 0)
    return q, S


def lowest_q_peaks(q, S, n_peaks=2, rel_threshold=0.1):
    """q-positions of the ``n_peaks`` lowest-q local maxima above a threshold.

    Peaks are selected by *position order*, not by height: for FCC the (111)
    reflection is roughly 1.8x the (200), but higher-order reflections such as
    (311) are comparable to (200), so ranking by height does not reliably give
    the two lowest reflections.
    """
    threshold = rel_threshold * S.max()
    peaks = []
    for i in range(1, len(S) - 1):
        if S[i] > S[i - 1] and S[i] >= S[i + 1] and S[i] > threshold:
            peaks.append(q[i])
            if len(peaks) == n_peaks:
                break
    return np.asarray(peaks)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture
def random_config(rng):
    """Small, cheap, deliberately non-cubic config for API-contract tests.

    The three box lengths differ so that ``n_grid`` differs per axis, which is
    what catches an x/y/z axis-ordering mistake; a cubic box would not.
    """
    box = np.array([10.0, 12.0, 14.0])
    return rng.random((128, 3)) * box, box


@pytest.fixture
def ideal_gas(rng):
    """Uniform random points: the S(q) = 1 reference."""
    L = 20.0
    return rng.random((20000, 3)) * L, L
