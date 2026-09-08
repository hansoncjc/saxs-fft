"""Interface contracts of the module-level structure-factor functions.

These tests deliberately assert nothing about physics - only about how the
functions may be called and what shape/kind of thing comes back.  Physical
correctness lives in ``test_structurefactor_physics.py``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from saxsfft.structurefactor import compute_q3_grid, compute_s_1d, compute_s_3d

from conftest import radial_sq


def test_compute_s_3d_shape_and_finite(random_config):
    """S_3d must come back on the grid that ``n_grid`` advertises."""
    x, box = random_config
    s3, n_grid = compute_s_3d(x, box, N_grid=8)

    assert s3.shape == tuple(n_grid)
    # Non-cubic box: the three grid dimensions must differ, otherwise this test
    # could not detect an x/y/z axis swap.
    assert len(set(n_grid.tolist())) == 3
    assert np.isfinite(s3).all()


def test_compute_q3_grid_matches_s_3d_shape(random_config):
    """The q-grid and S_3d grid must be index-compatible."""
    x, box = random_config
    s3, _ = compute_s_3d(x, box, N_grid=8)
    q3x, q3y, q3z = compute_q3_grid(x, box, N_grid=8)

    assert q3x.shape == q3y.shape == q3z.shape == s3.shape


def test_compute_s_1d_returns_q_num_cnt(random_config):
    """``compute_s_1d`` returns unnormalised (q, num, cnt) - not (q, S).

    This is the easiest contract in the package to misread, because the
    ``StructureFactor`` class method of the same name *does* return S(q).
    """
    x, box = random_config
    q, num, cnt = compute_s_1d(x, box, N_grid=32)

    assert q.shape == num.shape == cnt.shape
    assert q.size > 10  # enough bins that the monotonicity check below means something
    assert np.isfinite(q).all()
    assert np.isfinite(num).all()
    assert np.isfinite(cnt).all()
    assert (cnt >= 0).all()
    assert np.any(cnt > 0)
    assert np.all(np.diff(q) > 0)


def test_trim_default_drops_six_points(random_config):
    """The default ``trim=slice(3, -3)`` discards exactly 3 points per end."""
    x, box = random_config
    q_default, _, _ = compute_s_1d(x, box, N_grid=32)
    q_full, _, _ = compute_s_1d(x, box, N_grid=32, trim=slice(None))

    assert q_full.size - q_default.size == 6
    np.testing.assert_allclose(q_default, q_full[3:-3], rtol=0, atol=0)


def test_particle_diameter_scales_q(random_config):
    """``particle_diameter`` (nm) converts q to inverse angstrom via q/(10*d)."""
    x, box = random_config
    diameter_nm = 24.6

    q_reduced, _, _ = compute_s_1d(x, box, N_grid=32)
    q_angstrom, _, _ = compute_s_1d(x, box, N_grid=32, particle_diameter=diameter_nm)

    np.testing.assert_allclose(q_angstrom, q_reduced / (diameter_nm * 10), rtol=1e-15)


def test_float32_matches_float64(random_config):
    """Single precision must not change the answer beyond round-off."""
    x, box = random_config
    q64, s64 = radial_sq(x, box, 32, dtype=torch.float64)
    q32, s32 = radial_sq(x, box, 32, dtype=torch.float32)

    np.testing.assert_allclose(q32, q64, rtol=1e-5)
    np.testing.assert_allclose(s32, s64, rtol=2e-3, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device available")
def test_cuda_matches_cpu(random_config):
    """The GPU path must agree with the CPU path.

    This replaces the deleted ``cpu_gpu_bench.py``: the same code path, but
    asserting agreement instead of printing a timing.
    """
    x, box = random_config
    q_cpu, s_cpu = radial_sq(x, box, 32, device="cpu")
    q_gpu, s_gpu = radial_sq(x, box, 32, device="cuda")

    np.testing.assert_allclose(q_gpu, q_cpu, rtol=1e-12)
    np.testing.assert_allclose(s_gpu, s_cpu, rtol=1e-10, atol=1e-12)
