"""Time-aligned observations from NDBC standard-meteorological archives."""

from __future__ import annotations

import bisect
import gzip
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

NDBC_ARCHIVE_URL = "https://www.ndbc.noaa.gov/data/historical/stdmet"
BUOY_MAX_AGE_MIN = 20.0


@dataclass(frozen=True)
class BuoyStation:
    station: str
    name: str
    latitude: float
    longitude: float


# The requested near-coast stations did not report during November 2021.
# These three Pacific buoys have records on every experiment date.
BUOY_STATIONS: tuple[BuoyStation, ...] = (
    BuoyStation("46005", "West Washington", 46.147, -131.077),
    BuoyStation("46002", "West Oregon", 42.560, -130.523),
    BuoyStation("46050", "Stonewall Bank", 44.679, -124.535),
)

# Wind direction is circular, so it is represented by eastward/northward
# components. The remaining names match NDBC standard-met measurements.
BUOY_FIELDS = (
    "wind_u_ms",
    "wind_v_ms",
    "dewpoint_c",
    "pressure_hpa",
    "sea_temperature_c",
    "air_temperature_c",
)


@dataclass(frozen=True)
class BuoyObservation:
    station: str
    timestamp: datetime
    values: np.ndarray
    valid: np.ndarray


@dataclass(frozen=True)
class BuoyContext:
    """One causal observation per configured station at a radar timestamp."""

    values: np.ndarray
    valid: np.ndarray
    observed_at: tuple[str | None, ...]
    age_minutes: tuple[float, ...]


def load_buoy_context(
    when: datetime,
    *,
    max_age_min: float = BUOY_MAX_AGE_MIN,
) -> BuoyContext:
    """Load the latest observation at or before ``when`` from every station."""
    target = _utc(when)
    observations = [
        select_observation(
            load_station_year(station.station, target.year),
            target,
            max_age_min=max_age_min,
        )
        for station in BUOY_STATIONS
    ]
    return context_from_observations(observations, target)


def blank_buoy_context() -> BuoyContext:
    shape = (len(BUOY_STATIONS), len(BUOY_FIELDS))
    return BuoyContext(
        values=np.zeros(shape, dtype=np.float32),
        valid=np.zeros(shape, dtype=bool),
        observed_at=tuple(None for _ in BUOY_STATIONS),
        age_minutes=tuple(float("nan") for _ in BUOY_STATIONS),
    )


def context_from_observations(
    observations: list[BuoyObservation | None],
    when: datetime,
) -> BuoyContext:
    if len(observations) != len(BUOY_STATIONS):
        raise ValueError("buoy observation count does not match configured stations")
    target = _utc(when)
    values = np.zeros((len(BUOY_STATIONS), len(BUOY_FIELDS)), dtype=np.float32)
    valid = np.zeros_like(values, dtype=bool)
    observed_at: list[str | None] = []
    ages: list[float] = []
    for row, (station, observation) in enumerate(zip(BUOY_STATIONS, observations, strict=True)):
        if observation is None:
            observed_at.append(None)
            ages.append(float("nan"))
            continue
        if observation.station != station.station:
            raise ValueError(
                f"expected station {station.station}, got {observation.station}"
            )
        age = (target - _utc(observation.timestamp)).total_seconds() / 60.0
        if age < 0:
            raise ValueError("a buoy observation is from the future")
        values[row] = np.where(observation.valid, observation.values, 0.0)
        valid[row] = observation.valid
        observed_at.append(_utc(observation.timestamp).isoformat().replace("+00:00", "Z"))
        ages.append(age)
    return BuoyContext(
        values=values,
        valid=valid,
        observed_at=tuple(observed_at),
        age_minutes=tuple(ages),
    )


@lru_cache(maxsize=None)
def load_station_year(station: str, year: int) -> list[BuoyObservation]:
    path = Path("data") / "buoys" / f"{station}h{year}.txt.gz"
    body = _cached_archive(station, year, path)
    try:
        text = gzip.decompress(body).decode("ascii")
    except (gzip.BadGzipFile, UnicodeDecodeError) as exc:
        raise RuntimeError(f"invalid NDBC archive {path}") from exc
    return parse_standard_met(text, station)


def assert_buoy_split(
    training: list[BuoyContext],
    holdout_day: str,
    holdout: BuoyContext | None = None,
) -> None:
    """Reject holdout-day observations in training or exact train/test overlap."""
    training_stamps = {
        stamp for context in training for stamp in context.observed_at if stamp is not None
    }
    leaked = sorted(stamp for stamp in training_stamps if stamp[:10] == holdout_day)
    if leaked:
        raise RuntimeError(f"training buoy observations are from held-out day: {leaked}")
    if holdout is not None:
        overlap = training_stamps & {
            stamp for stamp in holdout.observed_at if stamp is not None
        }
        if overlap:
            raise RuntimeError(f"holdout buoy observations appear in training: {sorted(overlap)}")


def select_observation(
    observations: list[BuoyObservation],
    when: datetime,
    *,
    max_age_min: float = BUOY_MAX_AGE_MIN,
) -> BuoyObservation | None:
    """Select causally: future records are never eligible."""
    if max_age_min <= 0:
        raise ValueError("max_age_min must be positive")
    target = _utc(when)
    stamps = [item.timestamp for item in observations]
    index = bisect.bisect_right(stamps, target) - 1
    if index < 0:
        return None
    chosen = observations[index]
    age = (target - chosen.timestamp).total_seconds() / 60.0
    return chosen if age <= max_age_min else None


def parse_standard_met(text: str, station: str) -> list[BuoyObservation]:
    """Parse an NDBC standard-met text file using its column header."""
    lines = text.splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"NDBC archive for {station} is empty")
    header = next((line for line in lines if line.startswith("#") and "WDIR" in line), None)
    if header is None:
        raise RuntimeError(f"NDBC archive for {station} has no column header")
    columns = header.lstrip("#").split()
    required = {"YY", "MM", "DD", "hh", "mm", "WDIR", "WSPD", "PRES", "ATMP", "WTMP", "DEWP"}
    missing = required - set(columns)
    if missing:
        raise RuntimeError(f"NDBC archive for {station} lacks columns {sorted(missing)}")
    positions = {name: columns.index(name) for name in required}
    observations: list[BuoyObservation] = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < len(columns):
            continue
        try:
            timestamp = datetime(
                int(parts[positions["YY"]]),
                int(parts[positions["MM"]]),
                int(parts[positions["DD"]]),
                int(parts[positions["hh"]]),
                int(parts[positions["mm"]]),
                tzinfo=timezone.utc,
            )
            direction = _measurement(parts[positions["WDIR"]], "WDIR")
            speed = _measurement(parts[positions["WSPD"]], "WSPD")
            dewpoint = _measurement(parts[positions["DEWP"]], "DEWP")
            pressure = _measurement(parts[positions["PRES"]], "PRES")
            sea_temperature = _measurement(parts[positions["WTMP"]], "WTMP")
            air_temperature = _measurement(parts[positions["ATMP"]], "ATMP")
        except (ValueError, OverflowError):
            continue
        wind_u, wind_v = wind_components(direction, speed)
        values = np.asarray(
            [wind_u, wind_v, dewpoint, pressure, sea_temperature, air_temperature],
            dtype=np.float32,
        )
        observations.append(
            BuoyObservation(
                station=station,
                timestamp=timestamp,
                values=values,
                valid=np.isfinite(values),
            )
        )
    observations.sort(key=lambda item: item.timestamp)
    if not observations:
        raise RuntimeError(f"NDBC archive for {station} has no observations")
    return observations


def wind_components(direction_deg: float, speed_ms: float) -> tuple[float, float]:
    """Convert meteorological direction to eastward and northward wind."""
    if not np.isfinite(speed_ms):
        return float("nan"), float("nan")
    if speed_ms == 0 and not np.isfinite(direction_deg):
        return 0.0, 0.0
    if not np.isfinite(direction_deg):
        return float("nan"), float("nan")
    angle = np.deg2rad(direction_deg)
    return float(-speed_ms * np.sin(angle)), float(-speed_ms * np.cos(angle))


def _measurement(text: str, field: str) -> float:
    value = float(text)
    if field == "WDIR":
        return value if 0 <= value <= 360 else float("nan")
    if field == "PRES":
        return value if 800 <= value <= 1100 else float("nan")
    if field in {"ATMP", "WTMP", "DEWP"}:
        return value if -100 < value < 60 else float("nan")
    if field == "WSPD":
        return value if 0 <= value < 99 else float("nan")
    raise KeyError(field)


def _cached_archive(station: str, year: int, path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    url = f"{NDBC_ARCHIVE_URL}/{station}h{year}.txt.gz"
    request = urllib.request.Request(url, headers={"User-Agent": "typhoonboombonanza"})
    partial = path.with_name(path.name + ".partial")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
    except urllib.error.URLError as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"could not download {url}") from exc
    try:
        gzip.decompress(body)
    except gzip.BadGzipFile as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"NDBC response was not gzip: {url}") from exc
    partial.write_bytes(body)
    partial.replace(path)
    return body


def _utc(when: datetime) -> datetime:
    if when.tzinfo is None:
        return when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc)
