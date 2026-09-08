"""The GSD -> StructureFactor path: extraction, frame selection, box convention.

Everything here runs inside pytest's ``tmp_path``.  That matters: the code under
test writes files as a side effect - ``_extract_data`` drops a ``.txt`` next to
the GSD, and ``compute_s_1d`` writes ``average_structure_factor.npy/.txt`` - and
none of that may land in the repository.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gsd", reason="gsd is required for the GSD pipeline tests")

from saxsfft.gsdio import extract_positions
from saxsfft.structurefactor import StructureFactor
from saxsfft.utils import read_configuration

from conftest import (
    REFLECTIONS,
    SEED,
    add_gaussian_displacement,
    bragg_q,
    fcc_lattice,
    lowest_q_peaks,
    radial_sq,
    write_gsd,
)

# extract_positions serialises positions with "%.5f", and GSD itself stores
# float32, so nothing survives a round trip better than ~1e-5 absolute.
ROUNDTRIP_ATOL = 1e-5


def _random_frames(n_frames, n_particles, L, seed=SEED):
    rng = np.random.default_rng(seed)
    return [rng.random((n_particles, 3)) * L - L / 2 for _ in range(n_frames)]


def test_extract_positions_roundtrip(tmp_path):
    """Positions written to GSD must survive extraction to the text format."""
    L = 10.0
    frames = _random_frames(3, 64, L)
    gsd_path = write_gsd(tmp_path / "traj.gsd", frames, L)
    txt_path = tmp_path / "traj.txt"

    n_frames = extract_positions(str(gsd_path), str(txt_path))
    assert n_frames == 3

    x, box = read_configuration(str(txt_path), frames="all")
    assert x.shape == (3, 64, 3)
    assert box.shape == (3, 3)
    np.testing.assert_allclose(box[0], [L, L, L])
    np.testing.assert_allclose(x[0], frames[0], atol=ROUNDTRIP_ATOL)
    np.testing.assert_allclose(x[2], frames[2], atol=ROUNDTRIP_ATOL)


@pytest.mark.parametrize(
    "frames,step,expected",
    [
        ("all", 1, 10),
        ("last:4", 1, 4),
        # 'last:N' keeps the last N frames and *then* applies [::step], so
        # 'last:6' with step=2 yields 3 frames, not 6 and not 5.
        ("last:6", 2, 3),
        (0, 1, 1),
        # step outside 'last:N' is ignored (with a warning), not applied.
        pytest.param("all", 3, 10, marks=pytest.mark.filterwarnings("ignore::UserWarning")),
        ([0, 2, 4], 1, 3),
    ],
)
def test_frame_selection_semantics(tmp_path, frames, step, expected):
    """Frame selection must keep the documented - and non-obvious - counts."""
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(10, 32, L), L)

    sf = StructureFactor(str(gsd_path), N_grid=8, frames=frames, step=step)

    assert sf.x.shape[0] == expected
    assert sf.box.shape[0] == expected


def test_step_ignored_outside_last_n(tmp_path):
    """``step`` outside 'last:N' warns and is ignored, rather than being fatal."""
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(10, 32, L), L)

    with pytest.warns(UserWarning, match="only applied to 'last:N'"):
        sf = StructureFactor(str(gsd_path), N_grid=8, frames="all", step=3)

    assert sf.x.shape[0] == 10


def test_default_step_keeps_every_frame(tmp_path):
    """The default ``step=1`` must not silently thin the trajectory.

    Guards the 5 -> 1 default change: with the old default of 5 this returned
    2 frames instead of 10.
    """
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(10, 32, L), L)

    sf = StructureFactor(str(gsd_path), N_grid=8, frames="last:10")

    assert sf.x.shape[0] == 10


@pytest.mark.parametrize(
    "bad_frames",
    ["banana", "first:2", "last:0", "last:x", -1, 3.0, [0, 2.0], True, None, []],
)
def test_frames_rejects_invalid_input(tmp_path, bad_frames):
    """Malformed specifiers must raise ValueError, not TypeError or a numpy error."""
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(3, 32, L), L)

    with pytest.raises(ValueError):
        StructureFactor(str(gsd_path), N_grid=8, frames=bad_frames)


@pytest.mark.parametrize("bad_step", [0, -1, 2.5, "2"])
def test_step_rejects_invalid_input(tmp_path, bad_step):
    """A malformed ``step`` must raise rather than silently doing nothing.

    ``step=0`` and negative values used to fall through the ``if step > 1``
    guard and act as if no stepping had been requested at all.
    """
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(3, 32, L), L)

    with pytest.raises(ValueError):
        StructureFactor(str(gsd_path), N_grid=8, frames="last:3", step=bad_step)


@pytest.mark.parametrize(
    "frames,expected",
    [(np.int64(2), 1), (np.arange(0, 6, 2), 3), (np.array([1, 3]), 2)],
)
def test_numpy_frame_specifiers_accepted(tmp_path, frames, expected):
    """NumPy integers and integer arrays are ordinary frame specifiers.

    ``np.arange(...)`` used to fail outright: the ``frames == 'all'`` comparison
    returned an array and blew up on its ambiguous truth value.
    """
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(6, 32, L), L)

    sf = StructureFactor(str(gsd_path), N_grid=8, frames=frames)

    assert sf.x.shape[0] == expected


def test_out_of_range_frame_reports_frame_count(tmp_path):
    """An out-of-range index must say how many frames the trajectory has.

    Previously this surfaced as numpy's "need at least one array to stack".
    """
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(6, 32, L), L)

    with pytest.raises(ValueError, match="out of range") as excinfo:
        StructureFactor(str(gsd_path), N_grid=8, frames=99)

    assert "6 frame" in str(excinfo.value)


def test_class_matches_module_level(tmp_path):
    """The class must be a faithful wrapper around the module-level function.

    Comparison uses ``sf.x[0]`` rather than the pre-GSD positions so that the
    text round-trip's 1e-5 precision floor does not enter; this isolates the
    wrapper logic itself, which can then be asserted exactly.
    """
    L = 10.0
    gsd_path = write_gsd(tmp_path / "traj.gsd", _random_frames(1, 256, L), L)

    sf = StructureFactor(str(gsd_path), N_grid=16, frames="last:1", step=1)
    q_class, s_class = sf.compute_s_1d()
    q_module, s_module = radial_sq(sf.x[0], sf.box[0], 16)

    np.testing.assert_allclose(q_class, q_module, rtol=1e-12)
    np.testing.assert_allclose(s_class, s_module, rtol=1e-12)


def test_bragg_peak_survives_gsd_pipeline(tmp_path):
    """End-to-end: an FCC crystal keeps its (111) peak through GSD I/O.

    The only coverage this adds over the physics suite is the HOOMD box
    convention: ``write_gsd`` emits positions in [-L/2, L/2), and the cell-list
    has to wrap those negatives back correctly.
    """
    x, L, a = fcc_lattice(864, rho=0.5)
    x = add_gaussian_displacement(x, L, 0.03, rng=np.random.default_rng(SEED))
    assert x.min() < 0, "positions should be in HOOMD's centered box convention"

    gsd_path = write_gsd(tmp_path / "fcc.gsd", x, L)
    sf = StructureFactor(str(gsd_path), N_grid=64, frames="last:1", step=1)
    q, S = sf.compute_s_1d()

    expected = bragg_q(a, REFLECTIONS["fcc"][0])
    dq = 2 * np.pi / L
    peaks = lowest_q_peaks(q, S, n_peaks=1)

    assert peaks.size == 1, f"no peak found; max S = {S.max():.3e}"
    assert abs(peaks[0] - expected) < dq, (
        f"FCC (111) peak at q = {peaks[0]:.4f} after GSD round trip, "
        f"expected {expected:.4f} (bin width {dq:.4f})"
    )
