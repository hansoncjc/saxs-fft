"""Numerical correctness of the FFT -> q-grid -> radial-binning chain.

Every reference value here is a closed-form expression, not stored data.

A note on what may and may not be asserted: particles are assigned to the FFT
grid by nearest-grid-point, which is equivalent to displacing each particle to
its voxel centre.  That acts as a Debye-Waller factor - it damps Bragg peak
*heights* but does not move their *positions*.  So peak positions are asserted;
peak heights are not.  The ideal gas is unaffected because its coherent term
vanishes (S = 1 exactly, damping factor included).
"""
from __future__ import annotations

import numpy as np
import pytest

from saxsfft.structurefactor import compute_s_3d

from conftest import (
    LATTICE_BUILDERS,
    REFLECTIONS,
    SEED,
    add_gaussian_displacement,
    bragg_q,
    lowest_q_peaks,
    radial_sq,
)

# Small enough to stay fast, large enough that the Nyquist limit
# (pi * N_grid / L) sits well above the second reflection.
N_GRID = 64
RHO = 0.5
# Perfect lattice sites land on exact multiples of a/2, which can coincide with
# voxel boundaries and make the nearest-grid-point assignment sensitive to
# round-off.  A small displacement breaks that degeneracy and is physically
# realistic besides.
SIGMA_DISP = 0.03


def _lattice_sq(lattice, n_target, n_grid=N_GRID):
    builder = LATTICE_BUILDERS[lattice]
    x, L, a = builder(n_target, rho=RHO)
    x = add_gaussian_displacement(x, L, SIGMA_DISP, rng=np.random.default_rng(SEED))
    q, S = radial_sq(x, L, n_grid)
    return q, S, L, a


def test_q0_equals_N(random_config):
    """S_3d at q = 0 must equal N exactly, pinning the 1/N normalisation.

    All grid weights are 1, so the DC component of the FFT is N and
    |N|^2 / N = N.  Also checks that fftshift really put DC at the centre.
    """
    x, box = random_config
    s3, n_grid = compute_s_3d(x, box, N_grid=8)
    centre = tuple(int(n) // 2 for n in n_grid)

    assert s3[centre] == pytest.approx(x.shape[0], rel=1e-10)


def test_ideal_gas_sq_approaches_one(ideal_gas):
    """Uniform random points must give S(q) = 1 across the whole q range."""
    x, L = ideal_gas
    q, S = radial_sq(x, L, N_GRID)

    # Skip the lowest shells: they contain few q-points each, so their scatter
    # is large (relative noise ~ 1/sqrt(shell occupancy)).
    band = q > 2.0
    assert band.sum() > 10
    mean_S = S[band].mean()

    assert mean_S == pytest.approx(1.0, abs=0.05), (
        f"ideal-gas S(q) averaged {mean_S:.4f} over {band.sum()} shells, expected 1.0"
    )


def test_translation_invariance(ideal_gas):
    """A rigid shift by a whole number of voxels must leave S(q) unchanged.

    Shifting by an integer number of voxels makes the nearest-grid-point
    assignment a pure cyclic permutation of the density grid, so |FFT|^2 is
    unchanged to machine precision.  A non-integer shift would only be
    approximately invariant and would make this a much weaker assertion.
    """
    x, L = ideal_gas
    voxel = L / N_GRID
    shift = voxel * np.array([3.0, 5.0, 7.0])

    q_ref, S_ref = radial_sq(x, L, N_GRID)
    q_shift, S_shift = radial_sq((x + shift) % L, L, N_GRID)

    np.testing.assert_allclose(q_shift, q_ref, rtol=1e-12)
    np.testing.assert_allclose(S_shift, S_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize(
    "lattice,n_target",
    [("fcc", 864), ("bcc", 1024)],
)
def test_first_bragg_peak_position(lattice, n_target):
    """The lowest allowed reflection must land at its analytic |q|.

    FCC -> (111) at 2*pi*sqrt(3)/a;  BCC -> (110) at 2*pi*sqrt(2)/a.
    Tolerance is one radial bin width, dq = 2*pi/L.
    """
    q, S, L, a = _lattice_sq(lattice, n_target)
    dq = 2 * np.pi / L
    expected = bragg_q(a, REFLECTIONS[lattice][0])

    peaks = lowest_q_peaks(q, S, n_peaks=1)
    assert peaks.size == 1, f"no peak found for {lattice}; max S = {S.max():.3e}"

    assert abs(peaks[0] - expected) < dq, (
        f"{lattice.upper()} first peak at q = {peaks[0]:.4f}, "
        f"expected {expected:.4f} (a = {a:.4f}, bin width = {dq:.4f})"
    )


@pytest.mark.parametrize(
    "lattice,n_target",
    [("fcc", 864), ("bcc", 1024)],
)
def test_peak_ratio_identifies_lattice(lattice, n_target):
    """The first two reflections must be spaced the way this lattice requires.

    FCC gives 2/sqrt(3) = 1.1547, BCC gives 2/sqrt(2) = 1.4142.  Unlike the
    absolute position test, this ratio is immune to an overall scale error (a
    wrong 2*pi factor, say), so the two tests fail on disjoint bug classes.
    The assertion is comparative rather than tolerance-based, which keeps it
    insensitive to the half-bin-width error in each peak position.
    """
    q, S, L, a = _lattice_sq(lattice, n_target)
    peaks = lowest_q_peaks(q, S, n_peaks=2)
    assert peaks.size == 2, f"found {peaks.size} peak(s) for {lattice}, need 2"

    measured = peaks[1] / peaks[0]
    ratios = {name: second / first for name, (first, second) in REFLECTIONS.items()}
    other = "bcc" if lattice == "fcc" else "fcc"

    assert abs(measured - ratios[lattice]) < abs(measured - ratios[other]), (
        f"peak ratio {measured:.4f} (peaks at {peaks[0]:.4f}, {peaks[1]:.4f}) is "
        f"closer to the {other.upper()} value {ratios[other]:.4f} than to the "
        f"{lattice.upper()} value {ratios[lattice]:.4f}"
    )
