"""Test cross-day storm analog retrieval on the existing blockage masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from radar.analog import Analog, retrieve_analogs
from radar.holdout import HOLDOUT_SCAN, TRAINING_SCANS
from radar.holdout_between import load_tilts
from radar.io import volume_name_time
from radar.masks import make_holdout_mask
from radar.train import masked_mae_dbz

ALL_SCANS = (*TRAINING_SCANS, HOLDOUT_SCAN)
HOLDOUT_DAYS = ("2021-11-15", "2021-11-25", "2021-11-28")
MAX_PRIOR_AGE_MIN = 12.0
OUTPUT = Path("outputs/analog_retrieval")


def cached_pairs(day: str) -> list[tuple[str, str, float]]:
    """Return center, nearest preceding scan, and age from cached grids."""
    names = sorted(
        (p.name.removesuffix(".tilts.npz") for p in Path("data/grids").glob(f"KLGX{day.replace('-', '')}_*.tilts.npz")),
        key=volume_name_time,
    )
    pairs = []
    for previous, center in zip(names, names[1:]):
        age = (volume_name_time(center) - volume_name_time(previous)).total_seconds() / 60.0
        if 0 < age <= MAX_PRIOR_AGE_MIN:
            pairs.append((center, previous, age))
    return pairs


def _tilt(name: str, index: int, cache: dict):
    if name not in cache:
        timestamp = volume_name_time(name)
        from radar.holdout import Scan
        scan = Scan(timestamp.date().isoformat(), "stormy", name)
        cache[name] = load_tilts(scan)
    return cache[name][index]


def _library(holdout_day: str, index: int, cache: dict) -> list[Analog]:
    library = []
    for day in sorted(scan.day for scan in ALL_SCANS if scan.day < holdout_day):
        for center_name, previous_name, age in cached_pairs(day):
            current = _tilt(center_name, index, cache)
            previous = _tilt(previous_name, index, cache)
            if abs(current.sweep_elevation - previous.sweep_elevation) > 0.18:
                continue
            library.append(Analog(
                center_name, day, current.dbz, current.valid,
                previous.dbz, previous.valid, age,
            ))
    return library


def run() -> list[dict]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    baseline = {
        (day, kind): json.loads(Path(f"outputs/temporal_recency/{day}_none.json").read_text())[kind]["holdout_mae_dbz"]
        for day in HOLDOUT_DAYS for kind in ("lowest", "middle")
    }
    rows = []
    details = []
    for day in HOLDOUT_DAYS:
        center = next(scan for scan in ALL_SCANS if scan.day == day)
        pairs = {name: (previous, age) for name, previous, age in cached_pairs(day)}
        if center.name not in pairs:
            raise RuntimeError(f"{center.name} has no cached recent scan")
        previous_name, age = pairs[center.name]
        for kind, index in (("lowest", 0), ("middle", 1)):
            target = _tilt(center.name, index, cache)
            previous = _tilt(previous_name, index, cache)
            hidden = make_holdout_mask(
                target.valid, resolution_km=target.resolution_km,
                dbz=target.dbz, seed=int(day.replace("-", "")),
            )
            library = _library(day, index, cache)
            if any(item.day >= day for item in library):
                raise RuntimeError("current or future day leaked into analog library")
            for mode, use_temporal in (("current_only", False), ("current_and_past", True)):
                prediction, covered, neighbors = retrieve_analogs(
                    target.dbz, target.valid,
                    previous.dbz, previous.valid,
                    age, hidden, library, temporal=use_temporal,
                )
                scored = hidden & target.valid & covered
                coverage = float(scored.sum() / (hidden & target.valid).sum())
                if coverage < 0.99:
                    raise RuntimeError(f"analog coverage is only {coverage:.1%} on {day} {kind}")
                row = {
                    "holdout_day": day, "tilt": kind, "mode": mode,
                    "analog_mae_dbz": masked_mae_dbz(prediction, target.dbz, scored),
                    "no_past_unet_mae_dbz": baseline[day, kind],
                    "gap_coverage": coverage, "library_size": len(library),
                    "past_age_minutes": age,
                }
                rows.append(row)
                details.append({**row, "retrieved": neighbors})
                print(row, flush=True)
    with (OUTPUT / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "retrieved.json").write_text(json.dumps(details, indent=2) + "\n")
    return rows


if __name__ == "__main__":
    run()
