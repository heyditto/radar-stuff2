"""Fixed temporal context for the KLGX gap-fill experiment."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from radar.holdout import Scan
from radar.holdout_between import load_tilts
from radar.io import list_archived_volumes, volume_name_time

TEMPORAL_OFFSETS_MIN = tuple(range(-30, 0, 5)) + tuple(range(5, 31, 5))
TEMPORAL_TOLERANCE_MIN = 4.0
TEMPORAL_ELEVATION_TOLERANCE_DEG = 0.18
RECENT_PAST_COUNT = 2


@dataclass(frozen=True)
class TemporalContext:
    """Reflectivity and availability for all fixed time slots at one tilt."""

    dbz: np.ndarray
    valid: np.ndarray
    names: tuple[str | None, ...]


def assign_recent_past_scans(
    center: Scan,
    names: list[str],
    *,
    count: int = RECENT_PAST_COUNT,
    max_age_min: float = 30.0,
) -> tuple[Scan | None, ...]:
    """Put the most recent actual past volumes in the first two slots.

    All remaining slots stay blank so the legacy channel layout can be reused.
    No future volume or volume from a different radar is accepted.
    """
    if not 0 <= count <= RECENT_PAST_COUNT:
        raise ValueError("recent past scan count must be 0, 1, or 2")
    candidates = []
    for name in names:
        if name == center.name or not name.startswith(center.site):
            continue
        timestamp = volume_name_time(name)
        age = (center.timestamp - timestamp).total_seconds() / 60.0
        if 0 < age <= max_age_min:
            candidates.append((age, name, timestamp))
    candidates.sort()
    selected = [
        Scan(day=stamp.date().isoformat(), kind=center.kind, name=name, site=center.site)
        for _, name, stamp in candidates[:count]
    ]
    return tuple(selected + [None] * (len(TEMPORAL_OFFSETS_MIN) - len(selected)))


def assign_temporal_scans(
    center: Scan,
    names: list[str],
    *,
    tolerance_min: float = TEMPORAL_TOLERANCE_MIN,
) -> tuple[Scan | None, ...]:
    """Assign unique archive volumes to nominal offsets by minimum time error."""
    candidates = [
        (name, volume_name_time(name))
        for name in names
        if name != center.name and name.startswith(center.site)
    ]
    pairs: list[tuple[float, int, str]] = []
    for slot, offset in enumerate(TEMPORAL_OFFSETS_MIN):
        desired = center.timestamp + timedelta(minutes=offset)
        for name, timestamp in candidates:
            error = abs((timestamp - desired).total_seconds()) / 60.0
            if error <= tolerance_min:
                pairs.append((error, slot, name))
    pairs.sort(key=lambda item: (item[0], abs(TEMPORAL_OFFSETS_MIN[item[1]]), item[2]))
    assigned: list[str | None] = [None] * len(TEMPORAL_OFFSETS_MIN)
    used: set[str] = set()
    for _, slot, name in pairs:
        if assigned[slot] is None and name not in used:
            assigned[slot] = name
            used.add(name)
    result: list[Scan | None] = []
    for name in assigned:
        if name is None:
            result.append(None)
        else:
            result.append(
                Scan(
                    day=volume_name_time(name).date().isoformat(),
                    kind=center.kind,
                    name=name,
                    site=center.site,
                )
            )
    return tuple(result)


def discover_temporal_scans(center: Scan) -> tuple[Scan | None, ...]:
    """List the archive days touched by the window and assign fixed slots."""
    start = (center.timestamp - timedelta(minutes=30)).date()
    finish = (center.timestamp + timedelta(minutes=30)).date()
    names = list_archived_volumes(center.site, start)
    if finish != start:
        names.extend(list_archived_volumes(center.site, finish))
    return assign_temporal_scans(center, names)


def assert_context_split(
    training: dict[str, tuple[Scan | None, ...]],
    holdout_center: Scan,
    holdout_context: tuple[Scan | None, ...] | None = None,
) -> None:
    """Ensure training context cannot contain holdout-day or holdout files."""
    training_names: set[str] = set()
    for center_name, scans in training.items():
        if center_name == holdout_center.name:
            raise RuntimeError("held-out center appears in training")
        for scan in scans:
            if scan is None:
                continue
            if scan.day == holdout_center.day:
                raise RuntimeError(f"training context is from held-out day: {scan.name}")
            if scan.name in training_names:
                continue
            training_names.add(scan.name)
    if holdout_context is not None:
        overlap = training_names & {scan.name for scan in holdout_context if scan is not None}
        if overlap:
            raise RuntimeError(f"holdout temporal files appear in training: {sorted(overlap)}")


def _same_grid(target, candidate) -> bool:
    return (
        target.shape == candidate.shape
        and target.resolution_km == candidate.resolution_km
        and np.array_equal(target.x_km, candidate.x_km)
        and np.array_equal(target.y_km, candidate.y_km)
        and target.radar_name == candidate.radar_name
    )


def load_temporal_contexts(
    center: Scan,
    target_tilts: list,
    scans: tuple[Scan | None, ...] | None = None,
) -> list[TemporalContext]:
    """Load each temporal volume once and match its nearest target elevation."""
    if scans is None:
        scans = discover_temporal_scans(center)
    if len(scans) != len(TEMPORAL_OFFSETS_MIN):
        raise ValueError("temporal scan assignment has the wrong number of slots")
    loaded = [None if scan is None else load_tilts(scan) for scan in scans]
    contexts: list[TemporalContext] = []
    for target in target_tilts:
        dbz_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        names: list[str | None] = []
        for scan, grids in zip(scans, loaded, strict=True):
            if scan is None or grids is None:
                dbz_rows.append(np.zeros(target.shape, dtype=np.float32))
                valid_rows.append(np.zeros(target.shape, dtype=bool))
                names.append(None)
                continue
            match = min(grids, key=lambda grid: abs(grid.sweep_elevation - target.sweep_elevation))
            difference = abs(match.sweep_elevation - target.sweep_elevation)
            if difference > TEMPORAL_ELEVATION_TOLERANCE_DEG or not _same_grid(target, match):
                dbz_rows.append(np.zeros(target.shape, dtype=np.float32))
                valid_rows.append(np.zeros(target.shape, dtype=bool))
                names.append(None)
                continue
            dbz_rows.append(match.dbz)
            valid_rows.append(match.valid)
            names.append(scan.name)
        contexts.append(
            TemporalContext(
                dbz=np.stack(dbz_rows).astype(np.float32),
                valid=np.stack(valid_rows).astype(bool),
                names=tuple(names),
            )
        )
    return contexts


def blank_temporal_contexts(grids: list) -> list[TemporalContext]:
    """Temporal channels with every slot unavailable."""
    slots = len(TEMPORAL_OFFSETS_MIN)
    return [
        TemporalContext(
            dbz=np.zeros((slots, *grid.shape), dtype=np.float32),
            valid=np.zeros((slots, *grid.shape), dtype=bool),
            names=tuple(None for _ in range(slots)),
        )
        for grid in grids
    ]


def temporal_mean_dbz(context: TemporalContext) -> np.ndarray:
    """Mean available temporal reflectivity at each pixel."""
    ok = context.valid & np.isfinite(context.dbz)
    total = np.where(ok, context.dbz, 0.0).sum(axis=0)
    count = ok.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
