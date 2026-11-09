"""Synthetic blind spots on a reflectivity grid.

Training draws fresh masks from its own random stream. The holdout mask is
built from a separate seed and is never passed into training.
"""

from __future__ import annotations

import numpy as np

HOLDOUT_SEED = 20220322
# The uncertain class starts past this distance. The holdout plants one disk
# deeper than that, so the figure has a core the rule calls uncertain.
ESTIMATED_WITHIN_KM = 12.0
# Radius past the 12 km ring. The holdout disk and the wide training holes
# both use it, so training sees uncertain cores of the same size.
WIDE_HOLE_EXTRA_KM = 16.0
WIDE_HOLE_RADIUS_KM = ESTIMATED_WITHIN_KM + WIDE_HOLE_EXTRA_KM


def disk_mask(
    shape: tuple[int, int],
    center_row: float,
    center_col: float,
    radius_px: float,
) -> np.ndarray:
    """Filled disk, inclusive of the radius."""
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    return (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius_px**2


def sample_hidden_mask(
    valid: np.ndarray,
    rng: np.random.Generator,
    *,
    resolution_km: float,
    n_blobs: tuple[int, int] = (2, 4),
    n_rects: tuple[int, int] = (1, 3),
    blob_radius_km: tuple[float, float] = (4.0, 18.0),
    rect_side_km: tuple[float, float] = (8.0, 22.0),
) -> np.ndarray:
    """Random rectangles and blobs. True means the pixel is hidden.

    Holes are punched only where ``valid`` is True. Centers are chosen from
    valid pixels so the holes land on real radar samples.
    """
    if resolution_km <= 0:
        raise ValueError("resolution_km must be positive")
    hidden = np.zeros(valid.shape, dtype=bool)
    rows, cols = np.nonzero(valid)
    if len(rows) == 0:
        return hidden

    n_blob = int(rng.integers(n_blobs[0], n_blobs[1] + 1))
    n_rect = int(rng.integers(n_rects[0], n_rects[1] + 1))
    for _ in range(n_blob):
        pick = int(rng.integers(0, len(rows)))
        radius_km = float(rng.uniform(blob_radius_km[0], blob_radius_km[1]))
        radius_px = max(1.0, radius_km / resolution_km)
        hidden |= disk_mask(valid.shape, int(rows[pick]), int(cols[pick]), radius_px)
    for _ in range(n_rect):
        pick = int(rng.integers(0, len(rows)))
        height_px = max(1, int(round(float(rng.uniform(*rect_side_km)) / resolution_km)))
        width_px = max(1, int(round(float(rng.uniform(*rect_side_km)) / resolution_km)))
        row0 = int(np.clip(int(rows[pick]) - height_px // 2, 0, valid.shape[0]))
        col0 = int(np.clip(int(cols[pick]) - width_px // 2, 0, valid.shape[1]))
        hidden[row0 : row0 + height_px, col0 : col0 + width_px] = True
    hidden &= valid
    return hidden


def punch_wide_hole(
    valid: np.ndarray,
    rng: np.random.Generator,
    *,
    resolution_km: float,
    radius_km: float = WIDE_HOLE_RADIUS_KM,
    prefer: np.ndarray | None = None,
) -> np.ndarray:
    """One disk as wide as the holdout hole, limited to valid cells.

    The center is inset by the radius when the grid allows it, so the disk
    sits inside the window and its core is far from the window edge. ``prefer``
    marks cells that should host the center when any exist. This is a training
    hole. It does not use the holdout mask seed.
    """
    if resolution_km <= 0 or radius_km <= 0:
        raise ValueError("resolution_km and radius_km must be positive")
    radius_px = radius_km / resolution_km
    inset = int(np.floor(radius_px))
    rows, cols = _inset_centers(valid, prefer, inset)
    if len(rows) == 0:
        rows, cols = _inset_centers(valid, prefer, 0)
    if len(rows) == 0:
        return np.zeros(valid.shape, dtype=bool)
    pick = int(rng.integers(0, len(rows)))
    disk = disk_mask(valid.shape, int(rows[pick]), int(cols[pick]), radius_px)
    return disk & valid


def _inset_centers(
    valid: np.ndarray,
    prefer: np.ndarray | None,
    inset: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = valid.shape
    region = np.ones(valid.shape, dtype=bool)
    if inset > 0:
        if height <= 2 * inset or width <= 2 * inset:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        region[:inset, :] = False
        region[-inset:, :] = False
        region[:, :inset] = False
        region[:, -inset:] = False
    pool = valid & region
    if prefer is not None:
        preferred = pool & np.asarray(prefer, dtype=bool)
        if np.any(preferred):
            pool = preferred
    return np.nonzero(pool)


def _large_disk(
    valid: np.ndarray,
    rng: np.random.Generator,
    *,
    resolution_km: float,
    radius_km: float,
    dbz: np.ndarray | None = None,
) -> np.ndarray:
    """A disk that sits mostly inside valid coverage, so its core is hidden.

    When ``dbz`` is provided, disks that cover precipitation are preferred so
    the holdout hole is not parked in empty sky.
    """
    radius_px = radius_km / resolution_km
    rows, cols = np.nonzero(valid)
    if dbz is not None:
        echo_centers = valid & np.isfinite(dbz) & (dbz >= 20)
        echo_rows, echo_cols = np.nonzero(echo_centers)
        if len(echo_rows) > 0:
            rows, cols = echo_rows, echo_cols
    if len(rows) == 0:
        return np.zeros(valid.shape, dtype=bool)

    n_try = min(40, len(rows))
    choice = rng.choice(len(rows), size=n_try, replace=False)
    best = None
    best_score = -1.0
    fallback = None
    fallback_score = -1.0
    height, width = valid.shape
    for index in np.atleast_1d(choice):
        row = int(rows[int(index)])
        col = int(cols[int(index)])
        disk = disk_mask(valid.shape, row, col, radius_px)
        if not np.any(disk):
            continue
        coverage = float(valid[disk].mean())
        border = min(row, col, height - 1 - row, width - 1 - col)
        echo = 0.0
        if dbz is not None:
            echo = float((np.nan_to_num(dbz[disk], nan=-999.0) >= 20).mean())
        score = coverage + 0.5 * echo + 0.001 * border
        if score > fallback_score:
            fallback_score = score
            fallback = disk
        if coverage >= 0.9 and score > best_score:
            best_score = score
            best = disk
    chosen = best if best is not None else fallback
    if chosen is None:
        return np.zeros(valid.shape, dtype=bool)
    return chosen & valid


def make_holdout_mask(
    valid: np.ndarray,
    *,
    resolution_km: float,
    estimated_within_km: float = ESTIMATED_WITHIN_KM,
    seed: int = HOLDOUT_SEED,
    dbz: np.ndarray | None = None,
) -> np.ndarray:
    """Fixed evaluation mask, including one hole deeper than the estimated range.

    Built from ``seed``, not from the training generator. Training must not
    reuse this array. Pass ``dbz`` to aim the large hole at precipitation.
    """
    rng = np.random.default_rng(seed)
    hidden = sample_hidden_mask(valid, rng, resolution_km=resolution_km)
    large_radius_km = estimated_within_km + WIDE_HOLE_EXTRA_KM
    hidden |= _large_disk(
        valid,
        rng,
        resolution_km=resolution_km,
        radius_km=large_radius_km,
        dbz=dbz,
    )
    hidden &= valid
    return hidden
