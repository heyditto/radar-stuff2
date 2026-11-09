"""Load the next reflectivity tilt above the lowest sweep.

The gap-fill experiment uses this on the lowest sweep, which has no tilt below.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radar.grid import ReflectivityGrid, grid_reflectivity
from radar.holdout import Scan
from radar.io import (
    download_volume,
    lowest_reflectivity_sweep_index,
    next_reflectivity_sweep_index,
    read_volume,
    sweep_elevation_deg,
)
# The day-split figure, same storm, lowest sweep only. Cited so this score
# sits next to that one.
PREVIOUS_UNET_MAE = 5.6
PREVIOUS_NEAREST_MAE = 6.3
PREVIOUS_PRECIP_MAE = 6.7
PREVIOUS_ESTIMATED_MAE = 5.0
PREVIOUS_UNCERTAIN_MAE = 7.5
ABOVE_CACHE_VERSION = 1


@dataclass(frozen=True)
class AboveSweep:
    """Next tilt on the same grid as the lowest sweep."""

    dbz: np.ndarray
    valid: np.ndarray
    elevation: float
    sweep_index: int


def _above_cache_path(name: str) -> Path:
    return Path("data") / "grids" / f"{name}.above.npz"


def _read_above(path: Path, shape: tuple[int, int]) -> AboveSweep | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["version"]) != ABOVE_CACHE_VERSION:
            return None
        dbz = np.array(stored["dbz"], dtype=np.float32)
        valid = np.array(stored["valid"], dtype=bool)
        if dbz.shape != shape or valid.shape != shape:
            return None
        return AboveSweep(
            dbz=dbz,
            valid=valid,
            elevation=float(stored["sweep_elevation"]),
            sweep_index=int(stored["sweep_index"]),
        )


def _write_above(path: Path, above: AboveSweep) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.int32(ABOVE_CACHE_VERSION),
        dbz=above.dbz,
        valid=above.valid,
        sweep_elevation=np.float64(above.elevation),
        sweep_index=np.int32(above.sweep_index),
    )


def _assert_above_values(name: str, above: AboveSweep) -> None:
    """Invalid cells stay NaN. They are not written as 0 dBZ."""
    if np.any(above.valid & ~np.isfinite(above.dbz)):
        raise RuntimeError(f"{name} sweep above has a valid cell without reflectivity")
    if np.any(~above.valid & np.isfinite(above.dbz)):
        raise RuntimeError(f"{name} sweep above stored an invalid cell as a reflectivity")


def load_above(scan: Scan, lowest: ReflectivityGrid) -> AboveSweep:
    """Grid the next reflectivity tilt onto the lowest sweep's grid."""
    cached = _read_above(_above_cache_path(scan.name), lowest.shape)
    if cached is not None:
        _assert_above_values(scan.name, cached)
        return cached

    path = download_volume(scan.uri)
    radar = read_volume(path)
    lowest_index = lowest_reflectivity_sweep_index(radar)
    lowest_elevation = sweep_elevation_deg(radar, lowest_index)
    if abs(lowest_elevation - lowest.sweep_elevation) > 0.02:
        raise RuntimeError(
            f"{scan.name} lowest sweep is {lowest_elevation:.2f}°, "
            f"cached grid is {lowest.sweep_elevation:.2f}°"
        )
    above_index = next_reflectivity_sweep_index(radar, lowest_index)
    gridded = grid_reflectivity(radar, above_index, missing_gates="invalid")
    if gridded.shape != lowest.shape or not np.array_equal(gridded.x_km, lowest.x_km):
        raise RuntimeError(f"{scan.name} sweep above is not on the lowest sweep's grid")
    if gridded.sweep_elevation < lowest.sweep_elevation + 0.15:
        raise RuntimeError(
            f"{scan.name} sweep above {gridded.sweep_elevation:.2f}° "
            f"is not above the lowest sweep {lowest.sweep_elevation:.2f}°"
        )
    above = AboveSweep(
        dbz=gridded.dbz,
        valid=gridded.valid,
        elevation=gridded.sweep_elevation,
        sweep_index=above_index,
    )
    _assert_above_values(scan.name, above)
    _write_above(_above_cache_path(scan.name), above)
    return above
