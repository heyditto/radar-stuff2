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


@dataclass(frozen=True)
class TemporalContext:
    dbz: np.ndarray
    valid: np.ndarray
    names: tuple[str | None, ...]


def assign_temporal_scans(center: Scan, names: list[str], *, tolerance_min: float = TEMPORAL_TOLERANCE_MIN) -> tuple[Scan | None, ...]:
    candidates = [(name, volume_name_time(name)) for name in names if name != center.name and name.startswith(center.site)]
    pairs = []
    for slot, offset in enumerate(TEMPORAL_OFFSETS_MIN):
        desired = center.timestamp + timedelta(minutes=offset)
        for name, timestamp in candidates:
            error = abs((timestamp - desired).total_seconds()) / 60.0
            if error <= tolerance_min:
                pairs.append((error, slot, name))
    pairs.sort(key=lambda item: (item[0], abs(TEMPORAL_OFFSETS_MIN[item[1]]), item[2]))
    assigned = [None] * len(TEMPORAL_OFFSETS_MIN)
    used = set()
    for _, slot, name in pairs:
        if assigned[slot] is None and name not in used:
            assigned[slot] = name
            used.add(name)
    return tuple(
        None if name is None else Scan(volume_name_time(name).date().isoformat(), center.kind, name, center.site)
        for name in assigned
    )


def discover_temporal_scans(center: Scan) -> tuple[Scan | None, ...]:
    start = (center.timestamp - timedelta(minutes=30)).date()
    finish = (center.timestamp + timedelta(minutes=30)).date()
    names = list_archived_volumes(center.site, start)
    if finish != start:
        names.extend(list_archived_volumes(center.site, finish))
    return assign_temporal_scans(center, names)


def assert_context_split(training, holdout_center: Scan, holdout_context=None) -> None:
    training_names = set()
    for center_name, scans in training.items():
        if center_name == holdout_center.name:
            raise RuntimeError("held-out center appears in training")
        for scan in scans:
            if scan is None:
                continue
            if scan.day == holdout_center.day:
                raise RuntimeError(f"training context is from held-out day: {scan.name}")
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


def load_temporal_contexts(center: Scan, target_tilts: list, scans=None) -> list[TemporalContext]:
    scans = discover_temporal_scans(center) if scans is None else scans
    if len(scans) != len(TEMPORAL_OFFSETS_MIN):
        raise ValueError("temporal scan assignment has the wrong number of slots")
    loaded = [None if scan is None else load_tilts(scan) for scan in scans]
    contexts = []
    for target in target_tilts:
        dbz_rows, valid_rows, names = [], [], []
        for scan, grids in zip(scans, loaded, strict=True):
            if scan is None or grids is None:
                dbz_rows.append(np.zeros(target.shape, dtype=np.float32))
                valid_rows.append(np.zeros(target.shape, dtype=bool))
                names.append(None)
                continue
            match = min(grids, key=lambda grid: abs(grid.sweep_elevation - target.sweep_elevation))
            if abs(match.sweep_elevation - target.sweep_elevation) > TEMPORAL_ELEVATION_TOLERANCE_DEG or not _same_grid(target, match):
                dbz_rows.append(np.zeros(target.shape, dtype=np.float32))
                valid_rows.append(np.zeros(target.shape, dtype=bool))
                names.append(None)
            else:
                dbz_rows.append(match.dbz)
                valid_rows.append(match.valid)
                names.append(scan.name)
        contexts.append(TemporalContext(np.stack(dbz_rows).astype(np.float32), np.stack(valid_rows).astype(bool), tuple(names)))
    return contexts


def blank_temporal_contexts(grids: list) -> list[TemporalContext]:
    slots = len(TEMPORAL_OFFSETS_MIN)
    return [
        TemporalContext(
            np.zeros((slots, *grid.shape), dtype=np.float32),
            np.zeros((slots, *grid.shape), dtype=bool),
            tuple(None for _ in range(slots)),
        )
        for grid in grids
    ]


def temporal_mean_dbz(context: TemporalContext) -> np.ndarray:
    ok = context.valid & np.isfinite(context.dbz)
    total = np.where(ok, context.dbz, 0.0).sum(axis=0)
    count = ok.sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan).astype(np.float32)
