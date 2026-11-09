"""Temperature, humidity, and wind at the height of a radar beam.

NEXRAD Level II does not store these. The nearest routine balloon to KLGX
is Quillayute (KUIL). Iowa State serves that sounding. Each volume uses
the latest 00:00 or 12:00 UTC launch at or before the volume start. The
profile is one column. It changes across the grid because the beam rises
with range.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from radar.grid import ReflectivityGrid
from radar.io import download_volume, read_volume

# Quillayute, Washington. The nearest twice-daily radiosonde to Langley Hill.
SOUNDING_STATION = "UIL"
SOUNDING_SITE = "Quillayute"
SOUNDING_URL = "https://mesonet.agron.iastate.edu/json/raob.py"
# 4/3 Earth radius, the refraction model Py-ART uses for beam height.
EFFECTIVE_EARTH_RADIUS_M = 6_371_000.0 * 4.0 / 3.0
# A beam just below the first reported height keeps that surface measurement.
SURFACE_HOLD_M = 500.0
KNOTS_TO_METERS_PER_SECOND = 0.514444
ENV_CACHE_VERSION = 1
# Metadata sanity bounds for land-based NEXRAD sites.
ALTITUDE_MIN_M = -100.0
ALTITUDE_MAX_M = 5_000.0


@dataclass(frozen=True)
class Sounding:
    """One balloon profile. Missing levels are NaN, not a filled-in value."""

    station: str
    valid: str
    height_m: np.ndarray
    temperature_c: np.ndarray
    dewpoint_c: np.ndarray
    wind_u_ms: np.ndarray
    wind_v_ms: np.ndarray


@dataclass(frozen=True)
class BeamEnvironment:
    """Temperature, relative humidity, and horizontal wind on one tilt.

    ``relative_humidity`` is a fraction from 0 to 1. ``wind_u_ms`` is the
    eastward component and ``wind_v_ms`` is the northward component, both in
    meters per second. ``valid`` is False where the balloon did not cover
    that beam height.
    """

    temperature_c: np.ndarray
    relative_humidity: np.ndarray
    wind_u_ms: np.ndarray
    wind_v_ms: np.ndarray
    valid: np.ndarray
    elevation_deg: float
    station: str
    sounding_valid: str


def parse_volume_time(stamp: str) -> datetime:
    """Parse a volume timestamp that ends in ``Z`` or carries an offset."""
    text = stamp.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    when = datetime.fromisoformat(text)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)


def latest_sounding_time(when: datetime) -> datetime:
    """Latest 00:00 or 12:00 UTC launch at or before ``when``.

    A volume at 10:59 UTC stays on the 00:00 launch. The 12:00 launch is
    still in the future at that minute.
    """
    when = when.astimezone(timezone.utc)
    launch = when.replace(minute=0, second=0, microsecond=0)
    launch = launch.replace(hour=12 if launch.hour >= 12 else 0)
    if launch > when:
        launch -= timedelta(hours=12)
    return launch


def beam_height_m(
    ground_range_m: np.ndarray,
    elevation_deg: float,
    radar_alt_m: float,
) -> np.ndarray:
    """Height above sea level for a beam at fixed elevation.

    ``ground_range_m`` is distance along the surface from the radar. The
    curvature is the same 4/3-Earth model Py-ART uses to place gates.
    """
    ground = np.asarray(ground_range_m, dtype=np.float64)
    if np.any(ground < 0):
        raise ValueError("ground range must be non-negative")
    elevation = float(np.deg2rad(elevation_deg))
    radius = EFFECTIVE_EARTH_RADIUS_M
    psi = ground / radius
    height = np.zeros(ground.shape, dtype=np.float64)
    near = ground < 1.0
    denominator = np.cos(elevation + psi)
    usable = ~near & (denominator > 0.05)
    if np.any(~near & ~usable):
        raise ValueError("beam geometry is outside a NEXRAD sweep")
    slant = radius * np.sin(psi[usable]) / denominator[usable]
    height[usable] = slant * np.cos(elevation) / np.sin(psi[usable]) - radius
    return (height + float(radar_alt_m)).astype(np.float32)


def relative_humidity_fraction(temperature_c: np.ndarray, dewpoint_c: np.ndarray) -> np.ndarray:
    """Relative humidity from the Magnus formula, clipped to 0–1."""
    temperature = np.asarray(temperature_c, dtype=np.float64)
    dewpoint = np.asarray(dewpoint_c, dtype=np.float64)
    humidity = np.full(temperature.shape, np.nan, dtype=np.float64)
    known = np.isfinite(temperature) & np.isfinite(dewpoint)
    vapor = _saturation_vapor(dewpoint[known]) / _saturation_vapor(temperature[known])
    humidity[known] = np.clip(vapor, 0.0, 1.0)
    return humidity.astype(np.float32)


def environment_at_beam(
    sounding: Sounding,
    x_km: np.ndarray,
    y_km: np.ndarray,
    elevation_deg: float,
    radar_alt_m: float,
) -> BeamEnvironment:
    """Sample the balloon at each cell's beam height."""
    east, north = np.meshgrid(np.asarray(x_km, dtype=np.float64), np.asarray(y_km, dtype=np.float64))
    ground_m = np.hypot(east, north) * 1000.0
    height = beam_height_m(ground_m, elevation_deg, radar_alt_m)
    temperature = _interpolate(sounding.height_m, sounding.temperature_c, height)
    dewpoint = _interpolate(sounding.height_m, sounding.dewpoint_c, height)
    wind_u = _interpolate(sounding.height_m, sounding.wind_u_ms, height)
    wind_v = _interpolate(sounding.height_m, sounding.wind_v_ms, height)
    humidity = relative_humidity_fraction(temperature, dewpoint)
    valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(wind_u)
        & np.isfinite(wind_v)
    )
    return BeamEnvironment(
        temperature_c=np.where(valid, temperature, np.nan).astype(np.float32),
        relative_humidity=np.where(valid, humidity, np.nan).astype(np.float32),
        wind_u_ms=np.where(valid, wind_u, np.nan).astype(np.float32),
        wind_v_ms=np.where(valid, wind_v, np.nan).astype(np.float32),
        valid=valid,
        elevation_deg=float(elevation_deg),
        station=sounding.station,
        sounding_valid=sounding.valid,
    )


def load_sounding(when: datetime, station: str = SOUNDING_STATION) -> Sounding:
    """Load the latest available launch at or before ``when``."""
    launch = latest_sounding_time(when)
    for fallback in range(5):
        candidate = launch - timedelta(hours=12 * fallback)
        stamp = candidate.strftime("%Y-%m-%dT%H:%M:%SZ")
        path = (
            Path("data")
            / "soundings"
            / f"{station}_{candidate.strftime('%Y%m%dT%H%M%SZ')}.json"
        )
        url = f"{SOUNDING_URL}?station={station}&ts={stamp}"
        try:
            payload = _cached_json(url, path)
        except RuntimeError as exc:
            if str(exc).startswith("no sounding was returned") and fallback < 4:
                continue
            raise
        return _parse_sounding(payload, stamp)
    raise RuntimeError(f"no {station} sounding available before {when.isoformat()}")


def load_beam_environments(
    name: str,
    uri: str,
    grids: list[ReflectivityGrid],
) -> list[BeamEnvironment]:
    """Sounding sampled on every tilt in one volume.

    The Level II file is opened for the antenna altitude. Reflectivity grids
    that are already cached are not rebuilt.
    """
    if len(grids) == 0:
        raise ValueError("grids must not be empty")
    cached = _read_environment_cache(name, grids)
    if cached is not None:
        _require_lowest_coverage(name, grids, cached)
        return cached

    path = download_volume(uri)
    radar = read_volume(path)
    altitude = _radar_altitude_m(radar)
    when = parse_volume_time(grids[0].timestamp)
    for grid in grids[1:]:
        if grid.timestamp != grids[0].timestamp:
            raise RuntimeError(f"{name} tilts do not share a volume time")
        if grid.shape != grids[0].shape or not np.array_equal(grid.x_km, grids[0].x_km):
            raise RuntimeError(f"{name} tilts do not share a grid")
    sounding = load_sounding(when)
    environments = [
        environment_at_beam(
            sounding,
            grid.x_km,
            grid.y_km,
            grid.sweep_elevation,
            altitude,
        )
        for grid in grids
    ]
    _require_lowest_coverage(name, grids, environments)
    _write_environment_cache(name, environments, altitude)
    return environments


def _saturation_vapor(temperature_c: np.ndarray) -> np.ndarray:
    return np.exp(17.625 * temperature_c / (temperature_c + 243.04))


def _interpolate(height_m: np.ndarray, values: np.ndarray, query_m: np.ndarray) -> np.ndarray:
    """Linear interpolation. Within 500 m under the first level, use that level."""
    known = np.isfinite(height_m) & np.isfinite(values)
    height = np.asarray(height_m[known], dtype=np.float64)
    value = np.asarray(values[known], dtype=np.float64)
    out = np.full(query_m.shape, np.nan, dtype=np.float32)
    if height.size < 2:
        return out
    order = np.argsort(height, kind="mergesort")
    height = height[order]
    value = value[order]
    unique_height, inverse = np.unique(height, return_inverse=True)
    summed = np.zeros(unique_height.shape, dtype=np.float64)
    counts = np.zeros(unique_height.shape, dtype=np.float64)
    np.add.at(summed, inverse, value)
    np.add.at(counts, inverse, 1.0)
    value = summed / counts
    query = np.asarray(query_m, dtype=np.float64)
    inside = (query >= unique_height[0]) & (query <= unique_height[-1])
    out[inside] = np.interp(query[inside], unique_height, value).astype(np.float32)
    below = (query < unique_height[0]) & (query >= unique_height[0] - SURFACE_HOLD_M)
    out[below] = np.float32(value[0])
    return out


def _wind_components(direction_deg: float, speed_knots: float) -> tuple[float, float]:
    """Meteorological direction (where the wind comes from) to east and north."""
    if not np.isfinite(speed_knots):
        return float("nan"), float("nan")
    if speed_knots == 0.0 and not np.isfinite(direction_deg):
        return 0.0, 0.0
    if not np.isfinite(direction_deg):
        return float("nan"), float("nan")
    radians = np.deg2rad(direction_deg)
    speed = speed_knots * KNOTS_TO_METERS_PER_SECOND
    return float(-speed * np.sin(radians)), float(-speed * np.cos(radians))


def _parse_sounding(payload: dict, requested: str) -> Sounding:
    profiles = payload.get("profiles") or []
    if not profiles:
        raise RuntimeError(f"no sounding was returned for {requested}")
    profile = profiles[0]
    returned = str(profile.get("valid", ""))
    if returned != requested:
        raise RuntimeError(f"sounding time {returned or 'missing'} is not {requested}")
    station = str(profile.get("station") or SOUNDING_STATION)
    levels = profile.get("profile") or []
    height = []
    temperature = []
    dewpoint = []
    wind_u = []
    wind_v = []
    for level in levels:
        level_height = _float_or_nan(level.get("hght"))
        if not np.isfinite(level_height):
            continue
        east, north = _wind_components(
            _float_or_nan(level.get("drct")),
            _float_or_nan(level.get("sknt")),
        )
        height.append(level_height)
        temperature.append(_float_or_nan(level.get("tmpc")))
        dewpoint.append(_float_or_nan(level.get("dwpc")))
        wind_u.append(east)
        wind_v.append(north)
    if len(height) < 4:
        raise RuntimeError(f"sounding {station} {requested} has fewer than 4 levels")
    return Sounding(
        station=station,
        valid=requested,
        height_m=np.asarray(height, dtype=np.float64),
        temperature_c=np.asarray(temperature, dtype=np.float64),
        dewpoint_c=np.asarray(dewpoint, dtype=np.float64),
        wind_u_ms=np.asarray(wind_u, dtype=np.float64),
        wind_v_ms=np.asarray(wind_v, dtype=np.float64),
    )


def _text(value) -> str:
    item = np.asarray(value).reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode()
    return str(item)


def _float_or_nan(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _cached_json(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            path.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "typhoonboombonanza"})
    partial = path.with_name(path.name + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url}") from exc
    partial.write_bytes(body)
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"sounding response was not JSON: {url}") from exc
    if not payload.get("profiles"):
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"no sounding was returned for {url}")
    partial.replace(path)
    return payload


def _radar_altitude_m(radar) -> float:
    altitude = getattr(radar, "altitude", None)
    if not isinstance(altitude, dict) or "data" not in altitude:
        raise RuntimeError("volume has no antenna altitude")
    meters = float(np.asarray(altitude["data"]).ravel()[0])
    if not ALTITUDE_MIN_M <= meters <= ALTITUDE_MAX_M:
        raise RuntimeError(f"antenna altitude {meters:.1f} m is outside the expected range")
    return meters


def _environment_cache_path(name: str) -> Path:
    return Path("data") / "environment" / f"{name}.npz"


def _read_environment_cache(name: str, grids: list[ReflectivityGrid]) -> list[BeamEnvironment] | None:
    path = _environment_cache_path(name)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as stored:
        if int(stored["version"]) != ENV_CACHE_VERSION:
            return None
        temperature = np.array(stored["temperature_c"], dtype=np.float32)
        humidity = np.array(stored["relative_humidity"], dtype=np.float32)
        wind_u = np.array(stored["wind_u_ms"], dtype=np.float32)
        wind_v = np.array(stored["wind_v_ms"], dtype=np.float32)
        valid = np.array(stored["valid"], dtype=bool)
        elevation = np.array(stored["elevation"], dtype=np.float64)
        if temperature.shape[0] != len(grids) or elevation.shape != (len(grids),):
            return None
        if any(field.shape != temperature.shape for field in (humidity, wind_u, wind_v, valid)):
            return None
        station = _text(stored["station"])
        sounding_valid = _text(stored["sounding_valid"])
        environments = []
        for index, grid in enumerate(grids):
            if temperature[index].shape != grid.shape:
                return None
            if abs(float(elevation[index]) - grid.sweep_elevation) > 1e-3:
                return None
            environments.append(
                BeamEnvironment(
                    temperature_c=temperature[index],
                    relative_humidity=humidity[index],
                    wind_u_ms=wind_u[index],
                    wind_v_ms=wind_v[index],
                    valid=valid[index],
                    elevation_deg=float(elevation[index]),
                    station=station,
                    sounding_valid=sounding_valid,
                )
            )
        return environments


def _write_environment_cache(name: str, environments: list[BeamEnvironment], altitude_m: float) -> None:
    path = _environment_cache_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    first = environments[0]
    np.savez_compressed(
        path,
        version=np.int32(ENV_CACHE_VERSION),
        temperature_c=np.stack([item.temperature_c for item in environments]),
        relative_humidity=np.stack([item.relative_humidity for item in environments]),
        wind_u_ms=np.stack([item.wind_u_ms for item in environments]),
        wind_v_ms=np.stack([item.wind_v_ms for item in environments]),
        valid=np.stack([item.valid for item in environments]),
        elevation=np.array([item.elevation_deg for item in environments], dtype=np.float64),
        station=np.asarray(first.station),
        sounding_valid=np.asarray(first.sounding_valid),
        radar_alt_m=np.float64(altitude_m),
    )


def _require_lowest_coverage(
    name: str,
    grids: list[ReflectivityGrid],
    environments: list[BeamEnvironment],
) -> None:
    """The lowest tilt has to carry the balloon on most of its sampled cells."""
    lowest = min(range(len(grids)), key=lambda index: grids[index].sweep_elevation)
    grid = grids[lowest]
    environment = environments[lowest]
    sampled = int(grid.valid.sum())
    covered = int((grid.valid & environment.valid).sum())
    if sampled == 0 or covered * 2 < sampled:
        raise RuntimeError(
            f"{name} lowest tilt has temperature, humidity, and wind on "
            f"{covered} of {sampled} sampled pixels"
        )
    if np.any(environment.valid & ~np.isfinite(environment.temperature_c)):
        raise RuntimeError(f"{name} marked a cell valid without temperature")
    humidity = environment.relative_humidity
    if np.any(environment.valid & ((humidity < 0) | (humidity > 1) | ~np.isfinite(humidity))):
        raise RuntimeError(f"{name} relative humidity is outside 0 to 1")
