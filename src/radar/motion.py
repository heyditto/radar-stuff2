"""Past-only translation alignment for a persistently blocked radar sweep."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as shift_image
from scipy.signal import fftconvolve


def estimate_translation(
    older_dbz: np.ndarray,
    older_valid: np.ndarray,
    recent_dbz: np.ndarray,
    recent_valid: np.ndarray,
    *,
    max_shift_px: int = 12,
) -> tuple[int, int]:
    """Estimate integer row/column travel from older to recent echoes.

    Only cells marked valid enter the correlation. A zero shift is returned
    when there is too little echo texture to identify motion reliably.
    """
    if older_dbz.shape != recent_dbz.shape or older_valid.shape != older_dbz.shape or recent_valid.shape != recent_dbz.shape:
        raise ValueError("past radar fields and flags must share a shape")
    if max_shift_px < 0:
        raise ValueError("max_shift_px must be nonnegative")
    older_ok = np.asarray(older_valid, bool) & np.isfinite(older_dbz)
    recent_ok = np.asarray(recent_valid, bool) & np.isfinite(recent_dbz)
    older = np.where(older_ok, np.clip(older_dbz - 10.0, 0.0, 40.0), 0.0).astype(np.float32)
    recent = np.where(recent_ok, np.clip(recent_dbz - 10.0, 0.0, 40.0), 0.0).astype(np.float32)
    if np.count_nonzero(older > 0) < 32 or np.count_nonzero(recent > 0) < 32:
        return 0, 0
    correlation = fftconvolve(recent, older[::-1, ::-1], mode="full")
    # Normalize each shift by echo energy in the overlapping valid cells.
    older_energy = fftconvolve(recent_ok.astype(np.float32), (older * older)[::-1, ::-1], mode="full")
    recent_energy = fftconvolve(recent * recent, older_ok[::-1, ::-1].astype(np.float32), mode="full")
    center = np.array(older.shape) - 1
    radius = min(max_shift_px, *(dim - 1 for dim in older.shape))
    window = tuple(slice(int(c - radius), int(c + radius + 1)) for c in center)
    denominator = np.sqrt(np.maximum(older_energy[window] * recent_energy[window], 0.0))
    score = np.where(denominator > 1e-6, correlation[window] / np.maximum(denominator, 1e-6), -np.inf)
    if not np.isfinite(score).any() or float(np.max(score)) < 0.15:
        return 0, 0
    best = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(best[0] - radius), int(best[1] - radius)


def align_past_scans(
    temporal_dbz: np.ndarray,
    temporal_valid: np.ndarray,
    hidden: np.ndarray,
    ages_minutes: np.ndarray,
    *,
    max_shift_px: int = 12,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Extrapolate two past scans to target time without reading hidden pixels.

    The blockage is applied in source coordinates before estimating motion or
    shifting images. Shifted values can enter the target gap only when their
    source pixels were visible outside that blockage.
    """
    dbz = np.asarray(temporal_dbz)
    valid = np.asarray(temporal_valid, bool)
    hole = np.asarray(hidden, bool)
    ages = np.asarray(ages_minutes, dtype=np.float32)
    if dbz.shape != valid.shape or dbz.ndim != 3 or dbz.shape[0] < 2 or dbz.shape[1:] != hole.shape:
        raise ValueError("temporal arrays must contain at least two scans on the hidden grid")
    if ages.shape != (2,) or not np.all(np.isfinite(ages)) or not (ages[1] > ages[0] > 0):
        raise ValueError("two positive, increasing past-scan ages are required")
    source_valid = valid[:2] & ~hole[None, :, :]
    movement = estimate_translation(dbz[1], source_valid[1], dbz[0], source_valid[0], max_shift_px=max_shift_px)
    aligned_dbz = np.zeros_like(dbz, dtype=np.float32)
    aligned_valid = np.zeros_like(valid, dtype=bool)
    interval = float(ages[1] - ages[0])
    for slot in range(2):
        factor = float(ages[slot]) / interval
        displacement = tuple(float(axis) * factor for axis in movement)
        source = np.where(source_valid[slot], dbz[slot], 0.0).astype(np.float32)
        warped_valid = shift_image(source_valid[slot].astype(np.uint8), displacement, order=0, mode="constant", cval=0) > 0
        warped = shift_image(source, displacement, order=0, mode="constant", cval=0.0)
        aligned_dbz[slot] = np.where(warped_valid, warped, 0.0)
        aligned_valid[slot] = warped_valid
    return aligned_dbz, aligned_valid, movement
