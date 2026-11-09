"""Causally aligned GOES ABI Band 13 imagery on a radar grid."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer

from radar.environment import parse_volume_time
from radar.grid import ReflectivityGrid

PRODUCT = "ABI-L2-CMIPC"
CHANNEL = "C13"
MAX_AGE_MINUTES = 20.0
CACHE_VERSION = 1
_END = re.compile(r"_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d*_")


@dataclass(frozen=True)
class SatelliteContext:
    brightness_k: np.ndarray
    valid: np.ndarray
    observed_at: datetime | None
    source: str | None


def satellite_bucket(grid: ReflectivityGrid, when: datetime) -> str:
    """Use the operational western satellite for KLGX, eastern for KHGX."""
    if grid.radar_name.startswith("KHGX"):
        return "noaa-goes16"
    if grid.radar_name.startswith("KLGX"):
        transition = datetime(2023, 1, 4, 18, tzinfo=timezone.utc)
        return "noaa-goes17" if when < transition else "noaa-goes18"
    raise ValueError(f"no GOES selection configured for {grid.radar_name}")


def _end_time(key: str) -> datetime | None:
    match = _END.search(key)
    if match is None:
        return None
    year, day, hour, minute, second = map(int, match.groups())
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=day - 1, hours=hour, minutes=minute, seconds=second
    )


def select_image(keys: list[str], when: datetime) -> tuple[str, datetime] | None:
    """Choose the latest completed C13 scene, never one ending after the radar start."""
    candidates = []
    for key in keys:
        if f"{CHANNEL}_" not in key:
            continue
        end = _end_time(key)
        if end is not None and timedelta(0) <= when - end <= timedelta(minutes=MAX_AGE_MINUTES):
            candidates.append((end, key))
    if not candidates:
        return None
    end, key = max(candidates)
    return key, end


def _nearest_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    ascending = axis[0] < axis[-1]
    ordered = axis if ascending else axis[::-1]
    right = np.clip(np.searchsorted(ordered, values), 1, len(axis) - 1)
    left = right - 1
    nearest = np.where(abs(values - ordered[left]) <= abs(values - ordered[right]), left, right)
    return nearest if ascending else len(axis) - 1 - nearest


def project_image(
    brightness: np.ndarray,
    quality: np.ndarray,
    x_radians: np.ndarray,
    y_radians: np.ndarray,
    projection: dict,
    grid: ReflectivityGrid,
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest native ABI pixel at each radar cell; reject outside-domain cells."""
    satellite_crs = CRS.from_cf(projection)
    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={grid.radar_lat} +lon_0={grid.radar_lon} +datum=WGS84 +units=m"
    )
    east, north = np.meshgrid(grid.x_km * 1000, grid.y_km * 1000)
    x_m, y_m = Transformer.from_crs(local_crs, satellite_crs, always_xy=True).transform(east, north)
    height = float(projection["perspective_point_height"])
    x = x_m / height
    y = y_m / height
    inside = np.isfinite(x) & np.isfinite(y)
    inside &= (x >= min(x_radians[0], x_radians[-1])) & (x <= max(x_radians[0], x_radians[-1]))
    inside &= (y >= min(y_radians[0], y_radians[-1])) & (y <= max(y_radians[0], y_radians[-1]))
    col = _nearest_index(x_radians, np.nan_to_num(x, nan=0.0))
    row = _nearest_index(y_radians, np.nan_to_num(y, nan=0.0))
    values = np.asarray(brightness, dtype=np.float32)[row, col]
    flags = np.asarray(quality)[row, col]
    valid = inside & np.isfinite(values) & (flags <= 1)
    return np.where(valid, values, np.nan).astype(np.float32), valid


def blank_satellite_context(grid: ReflectivityGrid) -> SatelliteContext:
    return SatelliteContext(np.full(grid.shape, np.nan, dtype=np.float32), np.zeros(grid.shape, bool), None, None)


def load_satellite_context(grid: ReflectivityGrid, *, cache_dir: Path = Path("data/satellite")) -> SatelliteContext:
    """Fetch and cache one aligned CONUS C13 scene from unsigned NOAA S3."""
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from netCDF4 import Dataset

    when = parse_volume_time(grid.timestamp)
    bucket = satellite_bucket(grid, when)
    cache_dir.mkdir(parents=True, exist_ok=True)
    radar_stamp = when.strftime("%Y%m%dT%H%M%SZ")
    cache = cache_dir / f"{grid.radar_name}_{radar_stamp}_{bucket}_{CHANNEL}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            if int(saved["version"]) == CACHE_VERSION and tuple(saved["brightness_k"].shape) == grid.shape:
                observed = datetime.fromisoformat(str(saved["observed_at"]))
                if timedelta(0) <= when - observed <= timedelta(minutes=MAX_AGE_MINUTES):
                    return SatelliteContext(saved["brightness_k"].copy(), saved["valid"].copy(), observed, str(saved["source"]))

    client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys: list[str] = []
    for hour in (when - timedelta(hours=1), when):
        prefix = f"{PRODUCT}/{hour:%Y}/{hour.timetuple().tm_yday:03d}/{hour:%H}/"
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
    selected = select_image(keys, when)
    if selected is None:
        raise RuntimeError(f"no completed {bucket} {PRODUCT} {CHANNEL} image within {MAX_AGE_MINUTES:g} minutes of {when}")
    key, observed = selected
    raw = cache_dir / Path(key).name
    if not raw.exists():
        partial = raw.with_suffix(raw.suffix + ".partial")
        client.download_file(bucket, key, str(partial))
        partial.replace(raw)
    try:
        with Dataset(raw) as dataset:
            image = np.ma.filled(dataset.variables["CMI"][:], np.nan)
            quality = np.ma.filled(dataset.variables["DQF"][:], 255)
            x = np.asarray(dataset.variables["x"][:])
            y = np.asarray(dataset.variables["y"][:])
            mapping = dataset.variables["goes_imager_projection"]
            projection = {
                "grid_mapping_name": "geostationary",
                "perspective_point_height": mapping.perspective_point_height,
                "longitude_of_projection_origin": mapping.longitude_of_projection_origin,
                "latitude_of_projection_origin": mapping.latitude_of_projection_origin,
                "sweep_angle_axis": mapping.sweep_angle_axis,
                "semi_major_axis": mapping.semi_major_axis,
                "semi_minor_axis": mapping.semi_minor_axis,
            }
        brightness, valid = project_image(image, quality, x, y, projection, grid)
    finally:
        raw.unlink(missing_ok=True)
    np.savez_compressed(cache, version=np.int32(CACHE_VERSION), brightness_k=brightness,
                        valid=valid, observed_at=np.asarray(observed.isoformat()), source=np.asarray(f"s3://{bucket}/{key}"))
    return SatelliteContext(brightness, valid, observed, f"s3://{bucket}/{key}")
