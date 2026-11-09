"""KLGX training days and held-out storm for the temporal gap-fill experiment.

15 November 2021 and its temporal context are not loaded until training ends.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from radar.grid import ReflectivityGrid, grid_reflectivity
from radar.io import download_volume, lowest_reflectivity_sweep_index, read_volume, volume_name_site, volume_name_time

# The one-volume figure scored the 12:01 UTC scan after training on that same
# scan. Cited beside the gap-fill score.
# Fixed before looking at the score. Not the training seed.
HOLDOUT_MASK_SEED = 20211115
# Pixels at or above 20 dBZ on the 1 km grid. Quiet winter scans stay well
# under the first cut. The storm scans used here clear the second.
QUIET_MAX_ECHO_PIXELS = 2500
STORMY_MIN_ECHO_PIXELS = 4000
CACHE_VERSION = 1


@dataclass(frozen=True)
class Scan:
    """One archived volume and the role it plays in the split."""

    day: str
    kind: str
    name: str
    site: str = "KLGX"

    @property
    def uri(self) -> str:
        year, month, date = self.day.split("-")
        return f"s3://unidata-nexrad-level2/{year}/{month}/{date}/{self.site}/{self.name}"

    def __post_init__(self) -> None:
        if volume_name_time(self.name).date().isoformat() != self.day:
            raise ValueError(f"{self.name} is not from {self.day}")
        if volume_name_site(self.name) != self.site:
            raise ValueError(f"{self.name} is not from {self.site}")

    @property
    def timestamp(self):
        return volume_name_time(self.name)


# UTC dates, which is how the public archive is laid out.
TRAINING_SCANS: tuple[Scan, ...] = (
    Scan("2021-11-11", "stormy", "KLGX20211111_120508_V06"),
    Scan("2021-11-12", "stormy", "KLGX20211112_120745_V06"),
    Scan("2021-11-25", "stormy", "KLGX20211125_175652_V06"),
    Scan("2021-11-27", "stormy", "KLGX20211127_175642_V06"),
    Scan("2021-11-28", "stormy", "KLGX20211128_180032_V06"),
)
HOLDOUT_SCAN = Scan("2021-11-15", "stormy", "KLGX20211115_120233_V06")

DAY_NOTES = {
    "2021-11-11": "first atmospheric-river pulse over coastal Washington",
    "2021-11-12": "continuing first atmospheric-river pulse",
    "2021-11-25": "first late-November atmospheric river",
    "2021-11-27": "second late-November atmospheric river",
    "2021-11-28": "third late-November atmospheric river",
    "2021-11-15": "held out, strongest pulse of the second atmospheric river",
}


def assert_day_split() -> None:
    """Refuse a split that trains on the held-out day or drops 22 March 2022."""
    train_days = {scan.day for scan in TRAINING_SCANS}
    if HOLDOUT_SCAN.day in train_days:
        raise RuntimeError(f"held-out day {HOLDOUT_SCAN.day} is also a training day")
    stamp = HOLDOUT_SCAN.day.replace("-", "")
    for scan in TRAINING_SCANS:
        if stamp in scan.uri or stamp in scan.name:
            raise RuntimeError(f"training volume is from the held-out day: {scan.name}")
    for scan in (*TRAINING_SCANS, HOLDOUT_SCAN):
        if scan.site != "KLGX":
            raise RuntimeError(f"non-KLGX scan in KLGX experiment: {scan.name}")
        if scan.kind not in {"stormy", "quiet"}:
            raise RuntimeError(f"unknown kind {scan.kind} on {scan.name}")


def echo_pixels(grid: ReflectivityGrid) -> int:
    """Count valid grid cells at or above 20 dBZ."""
    echo = grid.valid & np.isfinite(grid.dbz) & (grid.dbz >= 20)
    return int(echo.sum())


def _cache_path(name: str) -> Path:
    return Path("data") / "grids" / f"{name}.npz"


def _text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def _read_cached_grid(path: Path) -> ReflectivityGrid | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["version"]) != CACHE_VERSION:
            return None
        return ReflectivityGrid(
            dbz=np.array(stored["dbz"], dtype=np.float32),
            valid=np.array(stored["valid"], dtype=bool),
            resolution_km=float(stored["resolution_km"]),
            x_km=np.array(stored["x_km"], dtype=np.float32),
            y_km=np.array(stored["y_km"], dtype=np.float32),
            radar_name=_text(stored["radar_name"]),
            radar_lat=float(stored["radar_lat"]),
            radar_lon=float(stored["radar_lon"]),
            sweep_elevation=float(stored["sweep_elevation"]),
            timestamp=_text(stored["timestamp"]),
        )


def _write_cached_grid(path: Path, grid: ReflectivityGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.int32(CACHE_VERSION),
        dbz=grid.dbz,
        valid=grid.valid,
        resolution_km=np.float32(grid.resolution_km),
        x_km=grid.x_km,
        y_km=grid.y_km,
        radar_name=np.asarray(grid.radar_name),
        radar_lat=np.float64(grid.radar_lat),
        radar_lon=np.float64(grid.radar_lon),
        sweep_elevation=np.float64(grid.sweep_elevation),
        timestamp=np.asarray(grid.timestamp),
    )


def load_scan(scan: Scan) -> ReflectivityGrid:
    """Download one volume if needed and grid its lowest reflectivity sweep."""
    cached = _read_cached_grid(_cache_path(scan.name))
    if cached is not None:
        return cached
    path = download_volume(scan.uri)
    radar = read_volume(path)
    sweep_index = lowest_reflectivity_sweep_index(radar)
    grid = grid_reflectivity(radar, sweep_index)
    _write_cached_grid(_cache_path(scan.name), grid)
    return grid


def _check_echo(scan: Scan, grid: ReflectivityGrid) -> int:
    count = echo_pixels(grid)
    if scan.kind == "quiet" and count > QUIET_MAX_ECHO_PIXELS:
        raise RuntimeError(
            f"{scan.name} was labeled quiet but has {count} pixels >= 20 dBZ"
        )
    if scan.kind == "stormy" and count < STORMY_MIN_ECHO_PIXELS:
        raise RuntimeError(
            f"{scan.name} was labeled stormy but has {count} pixels >= 20 dBZ"
        )
    return count


def _log_days(log) -> None:
    seen: list[str] = []
    for scan in TRAINING_SCANS:
        if scan.day not in seen:
            seen.append(scan.day)
    for day in seen:
        count = sum(1 for scan in TRAINING_SCANS if scan.day == day)
        log(f"training  {day}  {count} volumes  {DAY_NOTES[day]}")
    log(f"held out  {HOLDOUT_SCAN.day}  {HOLDOUT_SCAN.name}  {DAY_NOTES[HOLDOUT_SCAN.day]}")
