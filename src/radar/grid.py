"""Grid one reflectivity sweep onto a Cartesian PPI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from radar.io import (
    lowest_reflectivity_sweep_index,
    radar_name,
    sweep_elevation_deg,
    volume_timestamp,
)

# Reflectivity is scaled into this window before it enters the network.
# Clear-air gates (below the radar threshold) are stored as DBZ_MIN.
DBZ_MIN = 0.0
DBZ_MAX = 60.0
CLEAR_AIR_DBZ = DBZ_MIN

DEFAULT_RANGE_KM = 160.0
DEFAULT_RESOLUTION_KM = 1.0
DEFAULT_MAX_DISTANCE_KM = 4.0
# Three pooling stages in the U-Net, so each side must be divisible by 8.
GRID_STRIDE = 8


@dataclass(frozen=True)
class ReflectivityGrid:
    """One PPI reflectivity image in kilometers east and north of the radar.

    ``dbz[row, col]`` is the value at ``(x_km[col], y_km[row])``. Row 0 is the
    south edge. ``valid`` is False outside the radar's sampled area.
    """

    dbz: np.ndarray
    valid: np.ndarray
    resolution_km: float
    x_km: np.ndarray
    y_km: np.ndarray
    radar_name: str
    radar_lat: float
    radar_lon: float
    sweep_elevation: float
    timestamp: str

    @property
    def shape(self) -> tuple[int, int]:
        return self.dbz.shape


def _transition_flags(radar, sweep_slice) -> np.ndarray | None:
    """Ray or gate flags for antenna-transition samples, when the file has them."""
    sources = []
    attribute = getattr(radar, "antenna_transition", None)
    if isinstance(attribute, dict) and "data" in attribute:
        sources.append(attribute["data"])
    fields = getattr(radar, "fields", {})
    if isinstance(fields, dict) and "antenna_transition" in fields:
        sources.append(fields["antenna_transition"]["data"])
    for source in sources:
        flags = np.asarray(source, dtype=bool)[sweep_slice].ravel()
        return flags
    return None


def grid_shape(range_km: float, resolution_km: float) -> tuple[int, float]:
    """Return ``(n, half_width_km)`` with ``n`` divisible by the U-Net stride."""
    if resolution_km <= 0 or range_km <= 0:
        raise ValueError("range_km and resolution_km must be positive")
    n = int(round(2 * range_km / resolution_km))
    n -= n % GRID_STRIDE
    if n < GRID_STRIDE:
        raise ValueError("range is too small for the U-Net grid")
    half_width_km = n * resolution_km / 2
    return n, half_width_km


def _unsampled_gates(reflectivity) -> np.ndarray:
    """True where a gate is range-folded or was not collected.

    Below-threshold gates are real samples. NEXRAD stores them as the lowest
    masked code and stores range-folded or absent gates one half-dBZ step
    higher. Py-ART masks both. A cell with no sample must stay invalid, with
    its own flag, instead of being written as 0 dBZ.
    """
    mask = np.ma.getmaskarray(reflectivity)
    unsampled = np.zeros(np.shape(reflectivity), dtype=bool)
    if not np.any(mask):
        return unsampled
    underlying = np.asarray(np.ma.getdata(reflectivity), dtype=np.float64)
    masked_values = underlying[mask]
    lowest = float(np.min(masked_values))
    higher = masked_values > lowest + 0.25
    if np.any(higher):
        unsampled[mask] = higher
        return unsampled
    # A lone masked code at the upper step has no below-threshold partner.
    # Standard NEXRAD reflectivity puts that step near -32.5 dBZ.
    if lowest >= -32.6:
        unsampled[mask] = True
    return unsampled


def grid_reflectivity(
    radar,
    sweep_index: int | None = None,
    *,
    range_km: float = DEFAULT_RANGE_KM,
    resolution_km: float = DEFAULT_RESOLUTION_KM,
    max_distance_km: float = DEFAULT_MAX_DISTANCE_KM,
    missing_gates: str = "clear_air",
) -> ReflectivityGrid:
    """Nearest-neighbor grid of one PPI sweep.

    Each gate is placed at its ground-range x, y. A grid cell takes the nearest
    gate within ``max_distance_km``. This is a picture of that tilt, not a
    constant-altitude slice: at long range the beam is well above the ground.
    Gates below the detection threshold count as clear air (``DBZ_MIN``),
    because the radar did look there and found no significant echo.

    ``missing_gates="clear_air"`` is the lowest-sweep picture: every masked
    gate inside the sampled sweep is stored as 0 dBZ. ``missing_gates="invalid"``
    keeps range-folded and uncollected gates out of the grid. Those cells are
    NaN with ``valid`` False, not 0 dBZ.
    """
    if missing_gates not in {"clear_air", "invalid"}:
        raise ValueError("missing_gates must be 'clear_air' or 'invalid'")
    if sweep_index is None:
        sweep_index = lowest_reflectivity_sweep_index(radar)
    if "reflectivity" not in radar.fields:
        raise ValueError("volume has no reflectivity field")

    sweep_slice = radar.get_slice(sweep_index)
    reflectivity = radar.fields["reflectivity"]["data"][sweep_slice]
    gate_x, gate_y, _gate_z = radar.get_gate_x_y_z(
        sweep_index, filter_transitions=False
    )
    if gate_x.shape != np.shape(reflectivity):
        raise ValueError(
            f"gate grid {gate_x.shape} does not match reflectivity {np.shape(reflectivity)}"
        )

    gate_x = np.ma.filled(np.ma.asarray(gate_x, dtype=np.float64), np.nan)
    gate_y = np.ma.filled(np.ma.asarray(gate_y, dtype=np.float64), np.nan)
    values = np.ma.filled(np.ma.asarray(reflectivity), CLEAR_AIR_DBZ).astype(np.float32)
    points = np.column_stack((gate_x.ravel(), gate_y.ravel()))
    flat_values = values.ravel()
    usable = np.isfinite(points).all(axis=1) & np.isfinite(flat_values)
    transition = _transition_flags(radar, sweep_slice)
    if transition is not None and transition.shape == usable.shape:
        usable &= ~transition
    if missing_gates == "invalid":
        unsampled = _unsampled_gates(reflectivity).ravel()
        if unsampled.shape != usable.shape:
            raise ValueError("unsampled gate mask does not match the sweep")
        usable &= ~unsampled
    if not np.any(usable):
        raise ValueError("sweep has no usable reflectivity gates")

    n, half_width_km = grid_shape(range_km, resolution_km)
    # Cell centers, so the outer edge of the grid sits at +/- half_width_km.
    x_km = np.linspace(
        -half_width_km + resolution_km / 2,
        half_width_km - resolution_km / 2,
        n,
    )
    y_km = np.linspace(
        -half_width_km + resolution_km / 2,
        half_width_km - resolution_km / 2,
        n,
    )
    east_m, north_m = np.meshgrid(x_km * 1000.0, y_km * 1000.0)
    query = np.column_stack((east_m.ravel(), north_m.ravel()))

    tree = cKDTree(points[usable])
    distance_m, nearest = tree.query(query, k=1)
    dbz = flat_values[usable][nearest].reshape(n, n).astype(np.float32)
    valid = distance_m.reshape(n, n) <= (max_distance_km * 1000.0)
    dbz = np.where(valid, dbz, np.nan).astype(np.float32)

    latitude = float(np.asarray(radar.latitude["data"]).ravel()[0])
    longitude = float(np.asarray(radar.longitude["data"]).ravel()[0])
    return ReflectivityGrid(
        dbz=dbz,
        valid=valid,
        resolution_km=float(resolution_km),
        x_km=x_km.astype(np.float32),
        y_km=y_km.astype(np.float32),
        radar_name=radar_name(radar),
        radar_lat=latitude,
        radar_lon=longitude,
        sweep_elevation=sweep_elevation_deg(radar, sweep_index),
        timestamp=volume_timestamp(radar),
    )
