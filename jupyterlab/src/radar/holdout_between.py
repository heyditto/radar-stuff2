"""Load every distinct reflectivity tilt in a volume.

The gap-fill experiment trains on each tilt and reads the tilts beside it.
A missing side is an all-false valid flag. No tilt is invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radar.grid import ReflectivityGrid, grid_reflectivity
from radar.holdout import Scan, _text
from radar.io import (
    SAME_TILT_TOLERANCE_DEG,
    distinct_reflectivity_sweep_indices,
    download_volume,
    lowest_reflectivity_sweep_index,
    next_reflectivity_sweep_index,
    read_volume,
)

# Published hole-fill scores for the tilt-above slice. Cited beside this run.
TILT_ABOVE_UNET_MAE = 2.4
TILT_ABOVE_NEAREST_MAE = 6.3
TILT_ABOVE_COPY_MAE = 3.0
TILT_ABOVE_PRECIP_MAE = 2.7
TILT_ABOVE_ESTIMATED_MAE = 2.5
TILT_ABOVE_UNCERTAIN_MAE = 2.1
TILT_CACHE_VERSION = 1


@dataclass(frozen=True)
class TiltExample:
    """One target tilt and the distinct tilts immediately beside it."""

    target: ReflectivityGrid
    below: ReflectivityGrid | None
    above: ReflectivityGrid | None

    @property
    def kind(self) -> str:
        if self.below is None and self.above is not None:
            return "lowest"
        if self.above is None and self.below is not None:
            return "highest"
        if self.below is not None and self.above is not None:
            return "middle"
        return "none"


def _cache_path(name: str) -> Path:
    return Path("data") / "grids" / f"{name}.tilts.npz"


def _assert_values(name: str, grid: ReflectivityGrid) -> None:
    """Invalid cells stay NaN. They are not written as 0 dBZ."""
    label = f"{name} {grid.sweep_elevation:.2f}°"
    if np.any(grid.valid & ~np.isfinite(grid.dbz)):
        raise RuntimeError(f"{label} has a valid cell without reflectivity")
    if np.any(~grid.valid & np.isfinite(grid.dbz)):
        raise RuntimeError(f"{label} stored an invalid cell as a reflectivity")


def _assert_tilt_order(grids: list[ReflectivityGrid]) -> None:
    if len(grids) < 3:
        raise RuntimeError("a volume needs a lowest tilt, a middle tilt, and a highest tilt")
    first = grids[0]
    for grid in grids:
        if grid.shape != first.shape or grid.resolution_km != first.resolution_km:
            raise RuntimeError("tilts in one volume must share a shape and resolution")
        if not np.array_equal(grid.x_km, first.x_km) or not np.array_equal(grid.y_km, first.y_km):
            raise RuntimeError("tilts in one volume must share a grid")
    for earlier, later in zip(grids, grids[1:]):
        if later.sweep_elevation <= earlier.sweep_elevation + SAME_TILT_TOLERANCE_DEG:
            raise RuntimeError(
                f"tilts {earlier.sweep_elevation:.2f}° and {later.sweep_elevation:.2f}° are not distinct"
            )


def _read_tilts(path: Path) -> list[ReflectivityGrid] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["version"]) != TILT_CACHE_VERSION:
            return None
        dbz = np.array(stored["dbz"], dtype=np.float32)
        valid = np.array(stored["valid"], dtype=bool)
        elevation = np.array(stored["elevation"], dtype=np.float64)
        if dbz.ndim != 3 or valid.shape != dbz.shape or elevation.shape != (dbz.shape[0],):
            return None
        x_km = np.array(stored["x_km"], dtype=np.float32)
        y_km = np.array(stored["y_km"], dtype=np.float32)
        timestamp = _text(stored["timestamp"])
        radar_name = _text(stored["radar_name"])
        resolution_km = float(stored["resolution_km"])
        radar_lat = float(stored["radar_lat"])
        radar_lon = float(stored["radar_lon"])
        grids = []
        for index in range(dbz.shape[0]):
            grids.append(
                ReflectivityGrid(
                    dbz=dbz[index],
                    valid=valid[index],
                    resolution_km=resolution_km,
                    x_km=x_km,
                    y_km=y_km,
                    radar_name=radar_name,
                    radar_lat=radar_lat,
                    radar_lon=radar_lon,
                    sweep_elevation=float(elevation[index]),
                    timestamp=timestamp,
                )
            )
        return grids


def _write_tilts(path: Path, grids: list[ReflectivityGrid]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first = grids[0]
    np.savez_compressed(
        path,
        version=np.int32(TILT_CACHE_VERSION),
        dbz=np.stack([grid.dbz for grid in grids]),
        valid=np.stack([grid.valid for grid in grids]),
        elevation=np.array([grid.sweep_elevation for grid in grids], dtype=np.float64),
        resolution_km=np.float32(first.resolution_km),
        x_km=first.x_km,
        y_km=first.y_km,
        radar_name=np.asarray(first.radar_name),
        radar_lat=np.float64(first.radar_lat),
        radar_lon=np.float64(first.radar_lon),
        timestamp=np.asarray(first.timestamp),
    )


def load_tilts(scan: Scan) -> list[ReflectivityGrid]:
    """Grid every distinct reflectivity tilt in one volume."""
    cached = _read_tilts(_cache_path(scan.name))
    if cached is not None:
        for grid in cached:
            _assert_values(scan.name, grid)
        _assert_tilt_order(cached)
        return cached

    path = download_volume(scan.uri)
    radar = read_volume(path)
    indices = distinct_reflectivity_sweep_indices(radar)
    if len(indices) < 3:
        raise RuntimeError(f"{scan.name} has fewer than three reflectivity tilts")
    lowest_index = lowest_reflectivity_sweep_index(radar)
    next_index = next_reflectivity_sweep_index(radar, lowest_index)
    if indices[1] != next_index:
        raise RuntimeError(
            f"{scan.name} second tilt is sweep {indices[1]}, "
            f"next reflectivity tilt is sweep {next_index}"
        )
    grids = [grid_reflectivity(radar, index, missing_gates="invalid") for index in indices]
    for grid in grids:
        _assert_values(scan.name, grid)
    _assert_tilt_order(grids)
    _write_tilts(_cache_path(scan.name), grids)
    return grids


def examples_from_tilts(grids: list[ReflectivityGrid]) -> list[TiltExample]:
    """Pair each tilt with the next distinct tilt below and above, when present."""
    _assert_tilt_order(grids)
    examples = []
    last = len(grids) - 1
    for index, target in enumerate(grids):
        below = grids[index - 1] if index > 0 else None
        above = grids[index + 1] if index < last else None
        examples.append(TiltExample(target=target, below=below, above=above))
    return examples


def _side(grid: ReflectivityGrid | None, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Reflectivity and valid flag for one neighbor, or a missing side."""
    if grid is None:
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=bool)
    if grid.shape != shape:
        raise RuntimeError("neighbor tilt is not on the target grid")
    return grid.dbz, grid.valid


def _assert_holdout_elevations(grids: list[ReflectivityGrid]) -> None:
    """Ensure the scored KLGX file exposes three usable, distinct low tilts."""
    if len(grids) < 3:
        raise RuntimeError("held-out volume has fewer than three reflectivity tilts")
    lowest, middle, above = (grid.sweep_elevation for grid in grids[:3])
    if lowest < -0.1 or lowest > 0.6:
        raise RuntimeError(f"unexpected KLGX lowest tilt {lowest:.2f}°")
    if not lowest + SAME_TILT_TOLERANCE_DEG < middle < above:
        raise RuntimeError(
            f"held-out low tilts are not distinct: {lowest:.2f}°, "
            f"{middle:.2f}°, {above:.2f}°"
        )
