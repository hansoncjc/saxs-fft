import numpy as np
import warnings
from collections import deque

_FRAMES_HELP = (
    "Valid frames specifiers are:\n"
    "    'all'            - every frame\n"
    "    'last:N'         - the last N frames (N a positive integer)\n"
    "    int              - a single 0-based frame index\n"
    "    iterable of int  - specific 0-based indices, e.g. [0, 2, 4] or np.arange(0, 100, 5)"
)


def _is_integer(value):
    """True for Python and NumPy integers, but not for bool.

    ``bool`` is excluded deliberately: it is a subclass of ``int``, so without
    this guard ``frames=True`` would silently select frame 1.
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _normalize_frames(frames):
    """Validate a frames specifier and reduce it to ``(mode, payload)``.

    Returns one of ``('all', None)``, ``('last', N)`` or
    ``('indices', tuple_of_sorted_unique_indices)``.

    Raises
    ------
    ValueError
        For any specifier that is not one of the four documented forms.
    """
    if isinstance(frames, bytes):
        raise ValueError(f"Invalid frames specifier: {frames!r}.\n{_FRAMES_HELP}")

    if isinstance(frames, str):
        if frames == "all":
            return "all", None
        if frames.startswith("last:"):
            tail = frames.split(":", 1)[1]
            try:
                last_n = int(tail)
            except ValueError:
                raise ValueError(
                    f"Invalid frames specifier: {frames!r}. 'last:N' requires an "
                    f"integer N.\n{_FRAMES_HELP}"
                ) from None
            if last_n <= 0:
                raise ValueError(
                    f"Invalid frames specifier: {frames!r}. 'last:N' requires N > 0."
                )
            return "last", last_n
        raise ValueError(f"Invalid frames specifier: {frames!r}.\n{_FRAMES_HELP}")

    if _is_integer(frames):
        index = int(frames)
        if index < 0:
            raise ValueError(
                f"Negative frame index {index} is not supported; use "
                f"'last:{abs(index)}' to select from the end of the trajectory."
            )
        return "indices", (index,)

    try:
        items = list(frames)
    except TypeError:
        raise ValueError(
            f"Invalid frames specifier: {frames!r} (type {type(frames).__name__})."
            f"\n{_FRAMES_HELP}"
        ) from None

    if not items:
        raise ValueError(
            f"Invalid frames specifier: {frames!r} selects no frames.\n{_FRAMES_HELP}"
        )

    indices = []
    for item in items:
        if not _is_integer(item):
            raise ValueError(
                f"Invalid frame index {item!r} (type {type(item).__name__}) in "
                f"{frames!r}; frame indices must be integers.\n{_FRAMES_HELP}"
            )
        index = int(item)
        if index < 0:
            raise ValueError(
                f"Negative frame index {index} is not supported; use 'last:N' to "
                f"select from the end of the trajectory."
            )
        indices.append(index)

    return "indices", tuple(sorted(set(indices)))


def _normalize_step(step, mode, frames):
    """Validate ``step`` and silence it where it has no meaning.

    ``step`` only ever applies to a ``'last:N'`` selection.  Supplying it
    alongside any other specifier is a no-op that warns rather than raises, so
    that a stray non-default ``step`` cannot break an otherwise valid call.
    """
    if not _is_integer(step) or int(step) < 1:
        raise ValueError(f"step must be a positive integer, got {step!r}.")
    step = int(step)

    if step != 1 and mode != "last":
        warnings.warn(
            f"step is only applied to 'last:N' frame selection; ignoring "
            f"step={step} for frames={frames!r}.",
            UserWarning,
            stacklevel=3,
        )
        step = 1
    return step


def read_configuration(position_file, frames='all', step=1):
    """
    Read particle positions and box dimensions from a text trajectory.

    Parameters
    ----------
    position_file : str
        Path to the text file written by :func:`saxsfft.gsdio.extract_positions`.
    frames : 'all' | 'last:N' | int | Iterable[int]
        Which frames to read.  Indices are 0-based; negative indices are not
        supported (use ``'last:N'`` to select from the end).  ``'last:N'`` is an
        upper bound, so asking for more frames than the file holds returns all
        of them rather than raising.
    step : int, optional
        Keep every ``step``-th frame.  **Only meaningful with ``'last:N'``**;
        with any other specifier it is ignored and a ``UserWarning`` is issued.

        Note the order of operations: the last N frames are selected *first*,
        and ``[::step]`` is then applied to that window starting from its
        oldest frame.  ``'last:6'`` with ``step=2`` on a 6-frame trajectory
        therefore yields frames 0, 2 and 4 - the newest frame is not
        necessarily included.

    Returns
    -------
    x   : (F, N, 3) float64   positions for each requested frame
    box : (F, 3)     float64  box lengths for each frame

    Raises
    ------
    ValueError
        If ``frames`` or ``step`` is malformed, or if a requested frame index
        lies beyond the end of the file.
    """
    mode, payload = _normalize_frames(frames)
    step = _normalize_step(step, mode, frames)

    if mode == 'last':
        x_list = deque(maxlen=payload)
        box_list = deque(maxlen=payload)
    else:
        x_list, box_list = [], []

    target = set(payload) if mode == 'indices' else None
    max_target = max(target) if target else None

    n_read = 0
    with open(position_file) as f:
        while True:
            first = f.readline()
            if not first:
                break
            try:
                N = int(first)
            except ValueError:
                break           # trailing whitespace or an empty line
            box = np.fromstring(f.readline(), sep=' ')
            pos = np.fromfile(f, count=3 * N, sep=' ').reshape(N, 3)

            if mode in ('all', 'last') or n_read in target:
                x_list.append(pos.copy())
                box_list.append(box.copy())
            n_read += 1

            if mode == 'indices' and n_read > max_target:
                break

    if mode == 'indices':
        # Only reachable after hitting EOF, so n_read is the true frame count.
        missing = sorted(i for i in target if i >= n_read)
        if missing:
            raise ValueError(
                f"frame index {missing[0]} out of range: {position_file!r} "
                f"contains {n_read} frame(s)."
            )

    if not x_list:
        raise ValueError(f"No frames were read from {position_file!r}.")

    x_ret = np.stack(list(x_list))
    box_ret = np.stack(list(box_list))

    if step > 1:
        x_ret = x_ret[::step]
        box_ret = box_ret[::step]

    return x_ret, box_ret


def cell_list(x, box, rmax):
    """
    Assign particles to 3D spatial cells.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (N, 3) with particle positions.
    box : array-like
        Box dimensions. Shape: (3,)
    rmax : float or array-like
        Maximum cell size per dimension (scalar or (3,) array).

    Returns
    -------
    cells : np.ndarray
        index of the cell to which each particle belongs
    Ncell : np.ndarray
        (3,) number of cells in each dimension.
    """
    box = np.asarray(box)
    rmax = np.broadcast_to(rmax, box.shape)

    Ncell = np.floor(box / rmax).astype(int)
    Ncell[Ncell < 3] = 3  # Ensure at least 3 cells per dimension

    # Apply periodic boundary conditions
    x_wrapped = np.mod(x, box)

    # Scale position to cell index (0-based indexing)
    cell = np.floor(x_wrapped * Ncell / box).astype(int)
    cells = np.squeeze(cell)

    return cells, Ncell

def load_types(types_file):
    return np.loadtxt(types_file)
