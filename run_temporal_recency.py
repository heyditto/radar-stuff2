"""Compare past-only radar inputs under a persistent artificial blockage.

Run one fold and one input mode per process to keep memory use bounded. All
modes use the same 73-channel U-Net, training schedule, storm-day split,
and hole mask. The two age channels are zero when a scan is withheld.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from radar.buoy import load_buoy_context
from radar.classify import UNCERTAIN, classify_pixels
from radar.environment import load_beam_environments
from radar.holdout import HOLDOUT_SCAN, TRAINING_SCANS, Scan
from radar.holdout_between import examples_from_tilts, load_tilts, _side
from radar.holdout_gaps import _training_row, evaluate
from radar.io import volume_name_time
from radar.masks import ESTIMATED_WITHIN_KM, WIDE_HOLE_RADIUS_KM, make_holdout_mask
from radar.satellite import load_satellite_context
from radar.temporal import (
    RECENT_PAST_COUNT,
    assign_recent_past_scans,
    assert_context_split,
    blank_temporal_contexts,
    load_temporal_contexts,
)
from radar.train import predict_dbz_gaps, train_gapfill_gaps

ALL_SCANS = (*TRAINING_SCANS, HOLDOUT_SCAN)
MODES = {"none": 0, "one": 1, "two": 2, "aligned": 2}
EPOCHS = 20
STEPS_PER_EPOCH = 40
CROP = 128
WIDE_PROBABILITY = 0.5
PRECIP_WEIGHT = 4.0


def recent_selection(center: Scan, count: int):
    day = center.day.replace("-", "")
    names = [
        path.name for path in Path("data").glob(f"{center.site}{day}_*_V*")
        if path.is_file()
    ]
    selected = assign_recent_past_scans(center, names, count=count)
    if count and any(scan is None for scan in selected[:count]):
        raise RuntimeError(f"{center.name} has fewer than {count} cached past scans")
    ages = np.zeros(RECENT_PAST_COUNT, dtype=np.float32)
    for index, scan in enumerate(selected[:RECENT_PAST_COUNT]):
        if scan is not None:
            ages[index] = (center.timestamp - volume_name_time(scan.name)).total_seconds() / 60.0
    return selected, ages


def data_for_scan(center: Scan, count: int):
    tilts = load_tilts(center)
    selected, ages = recent_selection(center, count)
    contexts = (
        load_temporal_contexts(center, tilts, selected)
        if count else blank_temporal_contexts(tilts)
    )
    environment = load_beam_environments(center.name, center.uri, tilts)
    satellite = load_satellite_context(tilts[0])
    buoy = load_buoy_context(center.timestamp)
    return tilts, selected, ages, contexts, environment, satellite, buoy


def run(holdout_day: str, mode: str, output_dir: Path) -> dict:
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}")
    holdout = next((scan for scan in ALL_SCANS if scan.day == holdout_day), None)
    if holdout is None:
        raise ValueError(f"unknown holdout day {holdout_day}")
    training = [scan for scan in ALL_SCANS if scan.day != holdout_day]
    if len(training) != 5 or len({scan.day for scan in training}) != 5:
        raise RuntimeError("each fold must train on five other storm days")
    count = MODES[mode]
    motion_aligned = mode == "aligned"
    print(f"fold {holdout_day}  mode {mode}  train {[scan.day for scan in training]}", flush=True)

    rows = []
    training_selections = {}
    resolution_km = None
    for scan in training:
        tilts, selected, ages, contexts, environments, satellite, buoy = data_for_scan(scan, count)
        if resolution_km is None:
            resolution_km = tilts[0].resolution_km
        elif resolution_km != tilts[0].resolution_km:
            raise RuntimeError("training volumes use different grid resolutions")
        training_selections[scan.name] = selected
        for example, temporal, environment in zip(
            examples_from_tilts(tilts), contexts, environments, strict=True
        ):
            row = _training_row(example, environment, temporal, buoy, satellite)
            rows.append(replace(row, temporal_age_minutes=ages.copy()))
        print(f"loaded {scan.day}: {len(tilts)} tilts, past ages {ages.tolist()}", flush=True)
    assert_context_split(training_selections, holdout)
    trained = train_gapfill_gaps(
        rows,
        resolution_km=resolution_km,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        crop=CROP,
        wide_radius_km=WIDE_HOLE_RADIUS_KM,
        wide_probability=WIDE_PROBABILITY,
        precip_weight=PRECIP_WEIGHT,
        mask_temporal_inside_hole=True,
        align_temporal_motion=motion_aligned,
    )
    del rows

    held = data_for_scan(holdout, count)
    holdout_tilts, selected, ages, contexts, environments, satellite, buoy = held
    assert_context_split(training_selections, holdout, selected)
    examples = examples_from_tilts(holdout_tilts)
    results = {}
    mask_seed = int(holdout_day.replace("-", ""))
    for kind, index in (("lowest", 0), ("middle", 1)):
        example = examples[index]
        target = example.target
        hidden = make_holdout_mask(
            target.valid,
            resolution_km=target.resolution_km,
            dbz=target.dbz,
            seed=mask_seed,
        )
        labels = classify_pixels(
            hidden,
            target.valid,
            resolution_km=target.resolution_km,
            estimated_within_km=ESTIMATED_WITHIN_KM,
        )
        if int((labels == UNCERTAIN).sum()) == 0:
            raise RuntimeError(f"{holdout.day} {kind}: mask has no uncertain core")
        below_dbz, below_valid = _side(example.below, target.shape)
        above_dbz, above_valid = _side(example.above, target.shape)
        env = environments[index]
        temporal = contexts[index]
        predicted = predict_dbz_gaps(
            trained.model,
            target.dbz,
            hidden,
            below_dbz,
            below_valid,
            above_dbz,
            above_valid,
            temporal.dbz,
            temporal.valid,
            env.temperature_c,
            env.relative_humidity,
            env.wind_u_ms,
            env.wind_v_ms,
            env.valid,
            buoy.values,
            buoy.valid,
            satellite.brightness_k,
            satellite.valid,
            device=trained.device,
            temporal_age_minutes=ages,
            mask_temporal_inside_hole=True,
            align_temporal_motion=motion_aligned,
        )
        metrics = evaluate(target, hidden, predicted, labels)
        metrics["hidden_pixels"] = int((hidden & target.valid).sum())
        metrics["rain_hidden_pixels"] = int((hidden & target.valid & (target.dbz >= 20)).sum())
        results[kind] = metrics
        print(f"{kind} MAE {metrics['holdout_mae_dbz']:.3f}  rain {metrics['precip_mae_dbz']:.3f}", flush=True)

    report = {
        "holdout_day": holdout_day,
        "training_days": [scan.day for scan in training],
        "mode": mode,
        "past_scans": [scan.name if scan else None for scan in selected[:RECENT_PAST_COUNT]],
        "past_ages_minutes": ages.tolist(),
        "persistent_temporal_mask": True,
        "motion_aligned": motion_aligned,
        "motion_method": "past-pair masked echo correlation and translation extrapolation" if motion_aligned else None,
        "future_scans_used": False,
        "satellite": True,
        "epochs": EPOCHS,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "crop": CROP,
        "wide_hole_radius_km": WIDE_HOLE_RADIUS_KM,
        "rain_loss_weight": PRECIP_WEIGHT,
        "middle": results["middle"],
        "lowest": results["lowest"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{holdout_day}_{mode}.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {destination}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", required=True, choices=[scan.day for scan in ALL_SCANS])
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/temporal_recency"))
    args = parser.parse_args()
    run(args.holdout, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
