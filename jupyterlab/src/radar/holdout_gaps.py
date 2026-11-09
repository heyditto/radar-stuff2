"""Fill blockage gaps using vertical neighbors and ±30-minute KLGX scans.

The target sweep stays partly visible. Holes are the only cells to predict.
A middle tilt can read the sweep below and the sweep above. Every tilt can
read uniquely assigned temporal scans at fixed five-minute offsets. The lowest tilt
has no sweep below, so that flag stays off and the sweep above is the only
vertical input. The 12 km labels use the target sweep only. The held-out
file is loaded after training.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from radar.buoy import (
    BUOY_FIELDS,
    BUOY_STATIONS,
    BuoyContext,
    assert_buoy_split,
    blank_buoy_context,
    load_buoy_context,
)
from radar.classify import ESTIMATED, OBSERVED, UNCERTAIN, classify_pixels, nearest_observed_fill
from radar.environment import SOUNDING_SITE, BeamEnvironment, load_beam_environments
from radar.holdout import (
    HOLDOUT_MASK_SEED,
    HOLDOUT_SCAN,
    TRAINING_SCANS,
    _check_echo,
    _log_days,
    assert_day_split,
    load_scan,
)
from radar.holdout_above import load_above
from radar.holdout_between import (
    TiltExample,
    _assert_holdout_elevations,
    _side,
    examples_from_tilts,
    load_tilts,
)
from radar.masks import ESTIMATED_WITHIN_KM, WIDE_HOLE_RADIUS_KM, make_holdout_mask
from radar.train import (
    TRAIN_SEED,
    GapExample,
    masked_mae_dbz,
    neighbor_average_dbz,
    predict_dbz_gaps,
    train_gapfill_gaps,
)
from radar.temporal import (
    TEMPORAL_OFFSETS_MIN,
    TemporalContext,
    assert_context_split,
    blank_temporal_contexts,
    discover_temporal_scans,
    load_temporal_contexts,
    temporal_mean_dbz,
)

OUTPUT_PATH = Path("outputs/gapfill_holdout_gaps.png")
EPOCHS = 20
STEPS_PER_EPOCH = 40
CROP = 128
WIDE_PROBABILITY = 0.5
PRECIP_WEIGHT = 4.0


def _training_row(
    example: TiltExample,
    environment: BeamEnvironment,
    temporal: TemporalContext,
    buoy: BuoyContext,
    *,
    use_neighbors: bool = True,
    use_sounding: bool = True,
) -> GapExample:
    shape = example.target.shape
    below_dbz, below_valid = _side(example.below, shape)
    above_dbz, above_valid = _side(example.above, shape)
    if example.kind == "lowest" and np.any(below_valid):
        raise RuntimeError("lowest tilt invented a sweep below")
    if example.kind == "highest" and np.any(above_valid):
        raise RuntimeError("highest tilt invented a sweep above")
    if not use_neighbors:
        below_dbz = np.zeros(shape, dtype=np.float32)
        below_valid = np.zeros(shape, dtype=bool)
        above_dbz = np.zeros(shape, dtype=np.float32)
        above_valid = np.zeros(shape, dtype=bool)
    if environment.temperature_c.shape != shape:
        raise RuntimeError("sounding grid is not on the target tilt")
    if abs(environment.elevation_deg - example.target.sweep_elevation) > 1e-3:
        raise RuntimeError("sounding was sampled on a different tilt")
    temperature = environment.temperature_c
    humidity = environment.relative_humidity
    wind_u = environment.wind_u_ms
    wind_v = environment.wind_v_ms
    environment_valid = environment.valid
    if not use_sounding:
        temperature = np.zeros(shape, dtype=np.float32)
        humidity = np.zeros(shape, dtype=np.float32)
        wind_u = np.zeros(shape, dtype=np.float32)
        wind_v = np.zeros(shape, dtype=np.float32)
        environment_valid = np.zeros(shape, dtype=bool)
    return GapExample(
        target_dbz=example.target.dbz,
        target_valid=example.target.valid,
        below_dbz=below_dbz,
        below_valid=below_valid,
        above_dbz=above_dbz,
        above_valid=above_valid,
        temporal_dbz=temporal.dbz,
        temporal_valid=temporal.valid,
        temperature_c=temperature,
        relative_humidity=humidity,
        wind_u_ms=wind_u,
        wind_v_ms=wind_v,
        environment_valid=environment_valid,
        buoy_values=buoy.values,
        buoy_valid=buoy.valid,
    )


def _model_environment(environment: BeamEnvironment, use_sounding: bool):
    """Sounding arrays for the network. A withheld sounding is flagged off."""
    if use_sounding:
        return (
            environment.temperature_c,
            environment.relative_humidity,
            environment.wind_u_ms,
            environment.wind_v_ms,
            environment.valid,
        )
    shape = environment.temperature_c.shape
    zeros = np.zeros(shape, dtype=np.float32)
    return zeros, zeros, zeros, zeros, np.zeros(shape, dtype=bool)


def _sounding_note(environment: BeamEnvironment, grid) -> str:
    sampled = int(grid.valid.sum())
    covered = int((environment.valid & grid.valid).sum())
    return (
        f"sounding {environment.station} {environment.sounding_valid}  "
        f"lowest-tilt pixels with temperature, humidity, and wind {covered} of {sampled}"
    )


def _buoy_note(context: BuoyContext) -> str:
    notes = []
    for row, station in enumerate(BUOY_STATIONS):
        available = [
            name for name, valid in zip(BUOY_FIELDS, context.valid[row], strict=True) if valid
        ]
        age = context.age_minutes[row]
        age_text = "missing" if not np.isfinite(age) else f"{age:.1f} min old"
        notes.append(f"{station.station} {age_text} {len(available)}/{len(BUOY_FIELDS)} fields")
    return "buoys " + "; ".join(notes)


def _neighbor_mae(target_dbz, holes, below_dbz, below_valid, above_dbz, above_valid) -> float:
    """Mean absolute error of the available-neighbor average inside the holes."""
    average = neighbor_average_dbz(below_dbz, below_valid, above_dbz, above_valid)
    covered = holes & np.isfinite(average)
    if not np.any(covered):
        raise RuntimeError("neighbor average covers no blocked cell")
    return masked_mae_dbz(average, target_dbz, covered)


def evaluate(grid, hidden, prediction, labels) -> dict[str, float]:
    """MAE in dBZ for the U-Net and the nearest-observed baseline."""
    holes = hidden & grid.valid
    precip = holes & np.isfinite(grid.dbz) & (grid.dbz >= 20)
    baseline = nearest_observed_fill(grid.dbz, hidden, grid.valid)
    return {
        "holdout_mae_dbz": masked_mae_dbz(prediction, grid.dbz, holes),
        "estimated_mae_dbz": masked_mae_dbz(prediction, grid.dbz, labels == ESTIMATED),
        "uncertain_mae_dbz": masked_mae_dbz(prediction, grid.dbz, labels == UNCERTAIN),
        "baseline_mae_dbz": masked_mae_dbz(baseline, grid.dbz, holes),
        "precip_mae_dbz": masked_mae_dbz(prediction, grid.dbz, precip),
    }


def run(
    output_path: Path = OUTPUT_PATH,
    log=print,
    *,
    use_temporal: bool = True,
    use_buoy: bool = True,
    use_neighbors: bool = True,
    use_sounding: bool = True,
) -> dict:
    """Train on holes in every training tilt, then score the held-out storm."""
    import matplotlib

    matplotlib.use("Agg")

    output_path = Path(output_path)
    assert_day_split()
    if log:
        _log_days(log)
        log(
            f"gap training  epochs {EPOCHS}  steps {STEPS_PER_EPOCH}  crop {CROP}  "
            f"wide holes {WIDE_HOLE_RADIUS_KM:.0f} km  probability {WIDE_PROBABILITY:.2f}  "
            f">=20 dBZ loss weight {PRECIP_WEIGHT:.0f}  "
            f"training seed {TRAIN_SEED}  holdout mask seed {HOLDOUT_MASK_SEED}  "
            f"sounding {SOUNDING_SITE}  sounding inputs {use_sounding}  "
            f"temporal inputs {use_temporal}  buoy inputs {use_buoy}  "
            f"vertical neighbors {use_neighbors}"
        )
        log("the held-out file is loaded only after training")

    rows = []
    training_context_scans = {
        scan.name: discover_temporal_scans(scan) if use_temporal else tuple(None for _ in TEMPORAL_OFFSETS_MIN)
        for scan in TRAINING_SCANS
    }
    training_buoys = {
        scan.name: load_buoy_context(scan.timestamp) if use_buoy else blank_buoy_context()
        for scan in TRAINING_SCANS
    }
    assert_buoy_split(list(training_buoys.values()), HOLDOUT_SCAN.day)
    if use_temporal:
        assert_context_split(training_context_scans, HOLDOUT_SCAN)
    counts = {"lowest": 0, "middle": 0, "highest": 0}
    resolution_km = None
    for scan in TRAINING_SCANS:
        lowest_grid = load_scan(scan)
        echo = _check_echo(scan, lowest_grid)
        tilts = load_tilts(scan)
        if resolution_km is None:
            resolution_km = tilts[0].resolution_km
        elif tilts[0].resolution_km != resolution_km or tilts[0].shape != rows[0].target_dbz.shape:
            raise RuntimeError("training grids must share one shape and resolution")
        environments = load_beam_environments(scan.name, scan.uri, tilts)
        buoy = training_buoys[scan.name]
        temporal_contexts = (
            load_temporal_contexts(scan, tilts, training_context_scans[scan.name])
            if use_temporal
            else blank_temporal_contexts(tilts)
        )
        if len(environments) != len(tilts):
            raise RuntimeError(f"{scan.name} sounding count does not match its tilts")
        for example, environment, temporal in zip(
            examples_from_tilts(tilts), environments, temporal_contexts, strict=True
        ):
            counts[example.kind] += 1
            rows.append(
                _training_row(
                    example,
                    environment,
                    temporal,
                    buoy,
                    use_neighbors=use_neighbors,
                    use_sounding=use_sounding,
                )
            )
        if log:
            available = sum(
                context is not None for context in training_context_scans[scan.name]
            )
            log(
                f"loaded {scan.name}  {tilts[0].timestamp}  "
                f"{len(tilts)} tilts  lowest-sweep pixels >= 20 dBZ {echo}  "
                f"temporal slots {available}/{len(TEMPORAL_OFFSETS_MIN)}  "
                f"{_sounding_note(environments[0], tilts[0])}  {_buoy_note(buoy)}"
            )
    if counts["lowest"] == 0 or counts["middle"] == 0 or counts["highest"] == 0:
        raise RuntimeError(f"training cases are incomplete: {counts}")
    if log:
        log(
            f"training tilts  lowest {counts['lowest']}  "
            f"middle {counts['middle']}  highest {counts['highest']}"
        )

    trained = train_gapfill_gaps(
        rows,
        resolution_km=resolution_km,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        crop=CROP,
        wide_radius_km=WIDE_HOLE_RADIUS_KM,
        wide_probability=WIDE_PROBABILITY,
        precip_weight=PRECIP_WEIGHT,
        require_neighbors=use_neighbors,
        require_environment=use_sounding,
        log=log,
    )

    holdout_buoy = (
        load_buoy_context(HOLDOUT_SCAN.timestamp) if use_buoy else blank_buoy_context()
    )
    assert_buoy_split(
        list(training_buoys.values()), HOLDOUT_SCAN.day, holdout_buoy
    )
    holdout_context_scans = (
        discover_temporal_scans(HOLDOUT_SCAN) if use_temporal else tuple(None for _ in TEMPORAL_OFFSETS_MIN)
    )
    if use_temporal:
        assert_context_split(
            training_context_scans, HOLDOUT_SCAN, holdout_context_scans
        )
    holdout = load_scan(HOLDOUT_SCAN)
    above = load_above(HOLDOUT_SCAN, holdout)
    holdout_echo = _check_echo(HOLDOUT_SCAN, holdout)
    lowest_hidden = make_holdout_mask(
        holdout.valid,
        resolution_km=holdout.resolution_km,
        dbz=holdout.dbz,
        seed=HOLDOUT_MASK_SEED,
    )
    lowest_labels = classify_pixels(
        lowest_hidden,
        holdout.valid,
        resolution_km=holdout.resolution_km,
        estimated_within_km=ESTIMATED_WITHIN_KM,
    )
    if int((lowest_labels == UNCERTAIN).sum()) == 0:
        raise RuntimeError("lowest-sweep holdout mask has no uncertain core")
    missing = np.zeros(holdout.shape, dtype=np.float32)
    missing_valid = np.zeros(holdout.shape, dtype=bool)
    holdout_tilts = load_tilts(HOLDOUT_SCAN)
    holdout_temporal = (
        load_temporal_contexts(HOLDOUT_SCAN, holdout_tilts, holdout_context_scans)
        if use_temporal
        else blank_temporal_contexts(holdout_tilts)
    )
    holdout_environments = load_beam_environments(HOLDOUT_SCAN.name, HOLDOUT_SCAN.uri, holdout_tilts)
    lowest_environment = holdout_environments[0]
    if abs(lowest_environment.elevation_deg - holdout.sweep_elevation) > 1e-3:
        raise RuntimeError("lowest-sweep sounding is on a different tilt")
    model_above_dbz = above.dbz if use_neighbors else missing
    model_above_valid = above.valid if use_neighbors else missing_valid
    lowest_prediction = predict_dbz_gaps(
        trained.model,
        holdout.dbz,
        lowest_hidden,
        missing,
        missing_valid,
        model_above_dbz,
        model_above_valid,
        holdout_temporal[0].dbz,
        holdout_temporal[0].valid,
        *_model_environment(lowest_environment, use_sounding),
        holdout_buoy.values,
        holdout_buoy.valid,
        device=trained.device,
    )
    lowest_metrics = evaluate(holdout, lowest_hidden, lowest_prediction, lowest_labels)
    lowest_holes = lowest_hidden & holdout.valid
    lowest_neighbor = _neighbor_mae(
        holdout.dbz,
        lowest_holes,
        missing,
        missing_valid,
        above.dbz,
        above.valid,
    )
    lowest_metrics["neighbor_mae_dbz"] = lowest_neighbor
    lowest_temporal_mean = temporal_mean_dbz(holdout_temporal[0])
    lowest_temporal_mask = lowest_holes & np.isfinite(lowest_temporal_mean)
    lowest_metrics["temporal_mean_mae_dbz"] = masked_mae_dbz(
        lowest_temporal_mean, holdout.dbz, lowest_temporal_mask
    )

    tilts = holdout_tilts
    _assert_holdout_elevations(tilts)
    examples = examples_from_tilts(tilts)
    middle = examples[1]
    if middle.kind != "middle" or middle.below is None or middle.above is None:
        raise RuntimeError("held-out middle tilt is missing a neighbor")
    target = middle.target
    hidden = make_holdout_mask(
        target.valid,
        resolution_km=target.resolution_km,
        dbz=target.dbz,
        seed=HOLDOUT_MASK_SEED,
    )
    labels = classify_pixels(
        hidden,
        target.valid,
        resolution_km=target.resolution_km,
        estimated_within_km=ESTIMATED_WITHIN_KM,
    )
    if int((labels == UNCERTAIN).sum()) == 0:
        raise RuntimeError("middle-tilt holdout mask has no uncertain core")
    below_dbz, below_valid = _side(middle.below, target.shape)
    above_dbz, above_valid = _side(middle.above, target.shape)
    middle_environment = holdout_environments[1]
    if abs(middle_environment.elevation_deg - target.sweep_elevation) > 1e-3:
        raise RuntimeError("middle-tilt sounding is on a different tilt")
    model_below_dbz = below_dbz if use_neighbors else np.zeros(target.shape, dtype=np.float32)
    model_below_valid = below_valid if use_neighbors else np.zeros(target.shape, dtype=bool)
    model_above_dbz = above_dbz if use_neighbors else np.zeros(target.shape, dtype=np.float32)
    model_above_valid = above_valid if use_neighbors else np.zeros(target.shape, dtype=bool)
    prediction = predict_dbz_gaps(
        trained.model,
        target.dbz,
        hidden,
        model_below_dbz,
        model_below_valid,
        model_above_dbz,
        model_above_valid,
        holdout_temporal[1].dbz,
        holdout_temporal[1].valid,
        *_model_environment(middle_environment, use_sounding),
        holdout_buoy.values,
        holdout_buoy.valid,
        device=trained.device,
    )
    metrics = evaluate(target, hidden, prediction, labels)
    holes = hidden & target.valid
    neighbor_mae = _neighbor_mae(target.dbz, holes, below_dbz, below_valid, above_dbz, above_valid)
    metrics["neighbor_mae_dbz"] = neighbor_mae
    temporal_mean = temporal_mean_dbz(holdout_temporal[1])
    temporal_mask = holes & np.isfinite(temporal_mean)
    metrics["temporal_mean_mae_dbz"] = masked_mae_dbz(
        temporal_mean, target.dbz, temporal_mask
    )
    if not np.isfinite(prediction[holes]).all():
        raise RuntimeError("prediction has a non-finite blocked cell")

    if log:
        log(
            f"loaded {HOLDOUT_SCAN.name}  {holdout.timestamp}  "
            f"lowest-sweep pixels >= 20 dBZ {holdout_echo}  {_buoy_note(holdout_buoy)}"
        )
        log(
            "lowest sweep, same holes as the tilt-above slice  "
            f"U-Net {lowest_metrics['holdout_mae_dbz']:.2f}  "
            f"nearest {lowest_metrics['baseline_mae_dbz']:.2f}  "
            f"{'neighbor copy' if use_neighbors else 'withheld neighbor copy'} {lowest_neighbor:.2f}  "
            f"temporal mean {lowest_metrics['temporal_mean_mae_dbz']:.2f}  "
            f">=20 {lowest_metrics['precip_mae_dbz']:.2f}  "
            f"estimated {lowest_metrics['estimated_mae_dbz']:.2f}  "
            f"uncertain {lowest_metrics['uncertain_mae_dbz']:.2f}  "
            f"above {above.elevation:.2f}°  no tilt below"
        )
        log(
            f"middle tilt {target.sweep_elevation:.2f}° gaps  "
            f"below {middle.below.sweep_elevation:.2f}°  "
            f"above {middle.above.sweep_elevation:.2f}°  "
            f"U-Net {metrics['holdout_mae_dbz']:.2f}  "
            f"nearest {metrics['baseline_mae_dbz']:.2f}  "
            f"{'neighbor average' if use_neighbors else 'withheld neighbor average'} {neighbor_mae:.2f}  "
            f"temporal mean {metrics['temporal_mean_mae_dbz']:.2f}  "
            f">=20 {metrics['precip_mae_dbz']:.2f}  "
            f"estimated {metrics['estimated_mae_dbz']:.2f}  "
            f"uncertain {metrics['uncertain_mae_dbz']:.2f}"
        )
        log(
            "middle pixels  "
            f"observed {int((labels == OBSERVED).sum())}  "
            f"estimated {int((labels == ESTIMATED).sum())}  "
            f"uncertain {int((labels == UNCERTAIN).sum())}"
        )
        comparison = "neighbor average" if use_neighbors else "withheld neighbor average"
        if metrics["holdout_mae_dbz"] < neighbor_mae:
            log(f"On the middle tilt, U-Net beats the {comparison}")
        else:
            log(f"On the middle tilt, U-Net does not beat the {comparison}")

    if metrics["holdout_mae_dbz"] < neighbor_mae and metrics["holdout_mae_dbz"] < metrics["baseline_mae_dbz"]:
        verdict = "U-Net beats the neighbor average and the nearest observed pixel."
    elif metrics["holdout_mae_dbz"] < neighbor_mae:
        verdict = "U-Net beats the neighbor average and does not beat the nearest observed pixel."
    elif metrics["holdout_mae_dbz"] < metrics["baseline_mae_dbz"]:
        verdict = "U-Net beats the nearest observed pixel and does not beat the neighbor average."
    else:
        verdict = "U-Net does not beat the neighbor average or the nearest observed pixel."
    if use_neighbors and use_temporal:
        context_note = "Both neighbor tilts and scans from -30 to +30 minutes are inputs. "
    elif use_neighbors:
        context_note = "Both neighbor tilts are inputs. Temporal scans are withheld. "
    elif use_temporal:
        context_note = "Vertical neighbor tilts are withheld. Scans from -30 to +30 minutes are inputs. "
    else:
        context_note = "Vertical neighbor tilts and temporal scans are withheld. "
    caption = (
        f"Held-out KLGX 15 Nov 2021 12:02 UTC. Gaps in the {target.sweep_elevation:.2f}° tilt. "
        f"{context_note}"
        f"{'Temperature, humidity, and wind are the ' + middle_environment.station + ' sounding at ' + middle_environment.sounding_valid + '. ' if use_sounding else 'The radiosonde is withheld. '}"
        f"{'Three NDBC buoys are inputs. ' if use_buoy else 'Buoy inputs are withheld. '}"
        f"{verdict}\n"
        f"{'Temporal mean ' + format(metrics['temporal_mean_mae_dbz'], '.1f') + '. ' if use_temporal else ''}"
        f"Lowest sweep, same holes, no tilt below: U-Net {lowest_metrics['holdout_mae_dbz']:.1f}, "
        f"nearest {lowest_metrics['baseline_mae_dbz']:.1f}, "
        f"{'above copy' if use_neighbors else 'withheld above copy'} {lowest_neighbor:.1f}."
    )

    from matplotlib import pyplot as plt

    from radar.plot import plot_gap_neighbors

    figure = plot_gap_neighbors(
        target,
        hidden,
        prediction,
        below_dbz,
        below_valid,
        middle.below.sweep_elevation,
        above_dbz,
        above_valid,
        middle.above.sweep_elevation,
        output_path=output_path,
        unet_mae=metrics["holdout_mae_dbz"],
        nearest_mae=metrics["baseline_mae_dbz"],
        neighbor_mae=neighbor_mae,
        temporal_mae=metrics["temporal_mean_mae_dbz"],
        precip_mae=metrics["precip_mae_dbz"],
        estimated_mae=metrics["estimated_mae_dbz"],
        uncertain_mae=metrics["uncertain_mae_dbz"],
        caption=caption,
    )
    plt.close(figure)
    if log:
        log(f"wrote {output_path}")
    return {
        "middle_metrics": metrics,
        "lowest_metrics": lowest_metrics,
        "output_path": output_path,
    }


def main() -> None:
    run()


if __name__ == "__main__":
    main()
