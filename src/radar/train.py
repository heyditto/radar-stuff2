"""Train the gap-filling U-Net.

``train_gapfill_gaps`` fills holes in a target sweep. It can also read the
tilt below and the tilt above when those tilts exist, plus temperature,
relative humidity, and horizontal wind at the beam. The evaluation mask
is not an input. The caller keeps the held-out day out of the list.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from radar.grid import DBZ_MAX, DBZ_MIN
from radar.masks import WIDE_HOLE_RADIUS_KM, punch_wide_hole, sample_hidden_mask
from radar.motion import align_past_scans
from radar.model import (
    ABOVE_CHANNEL,
    ABOVE_FLAG_CHANNEL,
    BELOW_CHANNEL,
    BELOW_FLAG_CHANNEL,
    BUOY_FIELD_COUNT,
    BUOY_STATION_COUNT,
    ENVIRONMENT_FLAG_CHANNEL,
    HUMIDITY_CHANNEL,
    INPUT_CHANNELS,
    RECENCY_AGE_START_CHANNEL,
    RECENCY_INPUT_CHANNELS,
    SATELLITE_CHANNEL,
    SATELLITE_FLAG_CHANNEL,
    TEMPERATURE_CHANNEL,
    TEMPORAL_SLOT_COUNT,
    UNet,
    WIND_U_CHANNEL,
    WIND_V_CHANNEL,
    buoy_flag_channel,
    buoy_value_channel,
    temporal_dbz_channel,
    temporal_flag_channel,
)

TRAIN_SEED = 7
DEFAULT_EPOCHS = 15
DEFAULT_STEPS_PER_EPOCH = 30
DEFAULT_BATCH_SIZE = 8
DEFAULT_CROP = 64
DEFAULT_LR = 1e-3
# Fixed scales so a cold day and a warm day stay comparable.
TEMPERATURE_MIN_C = -80.0
TEMPERATURE_MAX_C = 40.0
WIND_MIN_MS = -60.0
WIND_MAX_MS = 60.0
BUOY_LOW = np.array([-60.0, -60.0, -50.0, 850.0, -2.0, -50.0], dtype=np.float32)
BUOY_HIGH = np.array([60.0, 60.0, 35.0, 1080.0, 35.0, 45.0], dtype=np.float32)


@dataclass(frozen=True)
class GapExample:
    """One tilt, the tilts beside it, and the sounding at this beam."""

    target_dbz: np.ndarray
    target_valid: np.ndarray
    below_dbz: np.ndarray
    below_valid: np.ndarray
    above_dbz: np.ndarray
    above_valid: np.ndarray
    temporal_dbz: np.ndarray
    temporal_valid: np.ndarray
    temperature_c: np.ndarray
    relative_humidity: np.ndarray
    wind_u_ms: np.ndarray
    wind_v_ms: np.ndarray
    environment_valid: np.ndarray
    buoy_values: np.ndarray
    buoy_valid: np.ndarray
    satellite_brightness_k: np.ndarray | None = None
    satellite_valid: np.ndarray | None = None
    temporal_age_minutes: np.ndarray | None = None


@dataclass
class TrainResult:
    model: UNet
    history_mae_dbz: list[float]
    device: str


def normalize_dbz(dbz: np.ndarray) -> np.ndarray:
    """Scale dBZ into ``[0, 1]``. Invalid (NaN) cells become ``DBZ_MIN``."""
    span = DBZ_MAX - DBZ_MIN
    filled = np.nan_to_num(dbz, nan=DBZ_MIN)
    scaled = (filled - DBZ_MIN) / span
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)


def denormalize_dbz(values: np.ndarray) -> np.ndarray:
    """Invert :func:`normalize_dbz`."""
    return (values * (DBZ_MAX - DBZ_MIN) + DBZ_MIN).astype(np.float32)


def masked_mae_dbz(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute error in dBZ over the True cells of ``mask``."""
    selected = np.asarray(mask, dtype=bool)
    if not np.any(selected):
        return float("nan")
    error = np.abs(prediction[selected] - target[selected])
    return float(np.mean(error))


def _masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.sum().clamp(min=1.0)
    return ((prediction - target).abs() * mask).sum() / weight


def _echo_threshold() -> float:
    return (20.0 - DBZ_MIN) / (DBZ_MAX - DBZ_MIN)


def _weighted_hidden_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    precip_weight: float,
) -> torch.Tensor:
    """L1 on hidden pixels. Echoes at or above 20 dBZ take ``precip_weight``.

    Clear-air hidden pixels keep weight 1. Observed pixels stay out of the loss.
    """
    if precip_weight < 1.0:
        raise ValueError("precip_weight must be at least 1 so clear air stays in the loss")
    precip = (target >= _echo_threshold()).to(dtype=prediction.dtype)
    weights = mask * (1.0 + (precip_weight - 1.0) * precip)
    return (prediction - target).abs().mul(weights).sum() / weights.sum().clamp(min=1.0)


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def neighbor_average_dbz(
    below_dbz: np.ndarray,
    below_valid: np.ndarray,
    above_dbz: np.ndarray,
    above_valid: np.ndarray,
) -> np.ndarray:
    """Average the neighbor cells that were sampled.

    Both neighbors valid: their mean. One neighbor valid: that neighbor.
    Neither: NaN. A missing tilt is an all-false valid flag, not a made-up sweep.
    """
    if below_dbz.shape != above_dbz.shape or below_valid.shape != below_dbz.shape:
        raise ValueError("neighbor tilts must share a shape")
    if above_valid.shape != above_dbz.shape:
        raise ValueError("neighbor tilts must share a shape")
    below_ok = np.asarray(below_valid, dtype=bool) & np.isfinite(below_dbz)
    above_ok = np.asarray(above_valid, dtype=bool) & np.isfinite(above_dbz)
    total = np.where(below_ok, below_dbz, 0.0) + np.where(above_ok, above_dbz, 0.0)
    count = below_ok.astype(np.float32) + above_ok.astype(np.float32)
    average = total / np.maximum(count, 1.0)
    return np.where(count > 0, average, np.nan).astype(np.float32)


def neighbor_score_mask(
    target_valid: np.ndarray,
    below_valid: np.ndarray,
    above_valid: np.ndarray,
) -> np.ndarray:
    """True where the target was sampled and at least one neighbor was sampled."""
    if target_valid.shape != below_valid.shape or target_valid.shape != above_valid.shape:
        raise ValueError("target and neighbor flags must share a shape")
    target = np.asarray(target_valid, dtype=bool)
    return target & (np.asarray(below_valid, dtype=bool) | np.asarray(above_valid, dtype=bool))


def _scale_unit(values: np.ndarray, low: float, high: float, valid: np.ndarray) -> np.ndarray:
    scaled = (np.asarray(values, dtype=np.float32) - low) / (high - low)
    return np.where(valid, np.clip(scaled, 0.0, 1.0), 0.0).astype(np.float32)


def build_gap_input(
    target_norm: np.ndarray,
    hidden: np.ndarray,
    below_norm: np.ndarray,
    below_valid: np.ndarray,
    above_norm: np.ndarray,
    above_valid: np.ndarray,
    temporal_norm: np.ndarray,
    temporal_valid: np.ndarray,
    temperature_c: np.ndarray,
    relative_humidity: np.ndarray,
    wind_u_ms: np.ndarray,
    wind_v_ms: np.ndarray,
    environment_valid: np.ndarray,
    buoy_values: np.ndarray,
    buoy_valid: np.ndarray,
    satellite_brightness_k: np.ndarray | None = None,
    satellite_valid: np.ndarray | None = None,
    *,
    temporal_age_minutes: np.ndarray | None = None,
    mask_temporal_inside_hole: bool = False,
) -> np.ndarray:
    """Stack visible target, vertical/temporal neighbors, and sounding.

    Hidden target cells are 0. A missing neighbor, or a cell that neighbor
    did not sample, is 0 with its flag unset. The neighbor tilts stay visible
    inside the hole. A below-threshold gate on a neighbor is 0 dBZ with the
    flag set. Temperature, humidity, and wind stay visible inside the hole.
    A height the balloon did not cover is 0 with the environment flag unset.
    """
    below_flag = np.asarray(below_valid, dtype=bool)
    above_flag = np.asarray(above_valid, dtype=bool)
    temporal_flag = np.asarray(temporal_valid, dtype=bool)
    hole = np.asarray(hidden, dtype=bool)
    if mask_temporal_inside_hole:
        temporal_flag = temporal_flag & ~hole[None, :, :]
    environment_flag = np.asarray(environment_valid, dtype=bool) & np.isfinite(temperature_c)
    environment_flag &= np.isfinite(relative_humidity) & np.isfinite(wind_u_ms) & np.isfinite(wind_v_ms)
    arrays = (
        target_norm,
        hole,
        below_norm,
        below_flag,
        above_norm,
        above_flag,
        temperature_c,
        relative_humidity,
        wind_u_ms,
        wind_v_ms,
        environment_flag,
    )
    if any(array.shape != target_norm.shape for array in arrays):
        raise ValueError("target, hole, neighbor tilts, and sounding must share a shape")
    expected_temporal = (TEMPORAL_SLOT_COUNT, *target_norm.shape)
    if temporal_norm.shape != expected_temporal or temporal_flag.shape != expected_temporal:
        raise ValueError(f"temporal arrays must have shape {expected_temporal}")
    expected_buoy = (BUOY_STATION_COUNT, BUOY_FIELD_COUNT)
    if buoy_values.shape != expected_buoy or buoy_valid.shape != expected_buoy:
        raise ValueError(f"buoy arrays must have shape {expected_buoy}")
    if temporal_age_minutes is not None:
        if np.asarray(temporal_age_minutes).shape != (2,):
            raise ValueError("temporal ages must have one value for each of the two recent scans")
        if np.any(temporal_flag[2:]):
            raise ValueError("recency layout cannot overwrite other temporal scans")
    stacked = np.zeros((INPUT_CHANNELS, *target_norm.shape), dtype=np.float32)
    stacked[0] = np.where(hole, 0.0, target_norm)
    stacked[1] = hole.astype(np.float32)
    stacked[BELOW_CHANNEL] = np.where(below_flag, below_norm, 0.0)
    stacked[BELOW_FLAG_CHANNEL] = below_flag.astype(np.float32)
    stacked[ABOVE_CHANNEL] = np.where(above_flag, above_norm, 0.0)
    stacked[ABOVE_FLAG_CHANNEL] = above_flag.astype(np.float32)
    stacked[TEMPERATURE_CHANNEL] = _scale_unit(
        temperature_c, TEMPERATURE_MIN_C, TEMPERATURE_MAX_C, environment_flag
    )
    stacked[HUMIDITY_CHANNEL] = np.where(
        environment_flag, np.clip(relative_humidity, 0.0, 1.0), 0.0
    )
    stacked[WIND_U_CHANNEL] = _scale_unit(wind_u_ms, WIND_MIN_MS, WIND_MAX_MS, environment_flag)
    stacked[WIND_V_CHANNEL] = _scale_unit(wind_v_ms, WIND_MIN_MS, WIND_MAX_MS, environment_flag)
    stacked[ENVIRONMENT_FLAG_CHANNEL] = environment_flag.astype(np.float32)
    for slot in range(TEMPORAL_SLOT_COUNT):
        stacked[temporal_dbz_channel(slot)] = np.where(
            temporal_flag[slot], temporal_norm[slot], 0.0
        )
        stacked[temporal_flag_channel(slot)] = temporal_flag[slot].astype(np.float32)
    if temporal_age_minutes is not None:
        for slot, age in enumerate(temporal_age_minutes):
            if not np.isfinite(age) or age < 0:
                raise ValueError("temporal ages must be finite nonnegative minutes")
            stacked[RECENCY_AGE_START_CHANNEL + slot].fill(min(float(age) / 30.0, 1.0))
    scaled_buoy = (np.asarray(buoy_values, dtype=np.float32) - BUOY_LOW) / (
        BUOY_HIGH - BUOY_LOW
    )
    buoy_flag = np.asarray(buoy_valid, dtype=bool) & np.isfinite(buoy_values)
    for station in range(BUOY_STATION_COUNT):
        for field in range(BUOY_FIELD_COUNT):
            value = (
                float(np.clip(scaled_buoy[station, field], 0.0, 1.0))
                if buoy_flag[station, field]
                else 0.0
            )
            stacked[buoy_value_channel(station, field)].fill(value)
            stacked[buoy_flag_channel(station, field)].fill(
                float(buoy_flag[station, field])
            )
    if (satellite_brightness_k is None) != (satellite_valid is None):
        raise ValueError("satellite brightness and validity must be supplied together")
    if satellite_brightness_k is not None:
        if satellite_brightness_k.shape != target_norm.shape or satellite_valid.shape != target_norm.shape:
            raise ValueError("satellite arrays must share the target shape")
        flag = np.asarray(satellite_valid, bool) & np.isfinite(satellite_brightness_k)
        # Colder cloud tops are higher values. IR works during day and night.
        stacked[SATELLITE_CHANNEL] = np.where(
            flag, np.clip((330.0 - satellite_brightness_k) / 150.0, 0.0, 1.0), 0.0
        )
        stacked[SATELLITE_FLAG_CHANNEL] = flag.astype(np.float32)
    return stacked


def _sample_gap_window(
    target_norm: np.ndarray,
    target_valid: np.ndarray,
    below_norm: np.ndarray,
    below_valid: np.ndarray,
    above_norm: np.ndarray,
    above_valid: np.ndarray,
    temporal_norm: np.ndarray,
    temporal_valid: np.ndarray,
    temperature_c: np.ndarray,
    relative_humidity: np.ndarray,
    wind_u_ms: np.ndarray,
    wind_v_ms: np.ndarray,
    environment_valid: np.ndarray,
    buoy_values: np.ndarray,
    buoy_valid: np.ndarray,
    satellite_brightness_k: np.ndarray,
    satellite_valid: np.ndarray,
    temporal_age_minutes: np.ndarray | None,
    rng: np.random.Generator,
    *,
    crop: int,
    resolution_km: float,
    wide_radius_km: float,
    wide_probability: float,
    mask_temporal_inside_hole: bool = False,
    align_temporal_motion: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    height, width = target_valid.shape
    if height < crop or width < crop:
        raise ValueError(f"grid {height}x{width} is smaller than crop {crop}")
    echo_threshold = _echo_threshold()
    fallback = None
    for _ in range(40):
        row0 = int(rng.integers(0, height - crop + 1))
        col0 = int(rng.integers(0, width - crop + 1))
        window = (slice(row0, row0 + crop), slice(col0, col0 + crop))
        crop_valid = target_valid[window]
        if int(crop_valid.sum()) < 64:
            continue
        hidden = sample_hidden_mask(crop_valid, rng, resolution_km=resolution_km)
        if rng.random() < wide_probability:
            hidden |= punch_wide_hole(
                crop_valid,
                rng,
                resolution_km=resolution_km,
                radius_km=wide_radius_km,
                prefer=target_norm[window] >= echo_threshold,
            )
            hidden &= crop_valid
        if int(hidden.sum()) < 16:
            continue
        crop_norm = target_norm[window]
        crop_temporal = temporal_norm[:, window[0], window[1]]
        crop_temporal_valid = temporal_valid[:, window[0], window[1]]
        if align_temporal_motion:
            # Convert normalized values back to dBZ for the echo correlation.
            aligned, crop_temporal_valid, _ = align_past_scans(
                crop_temporal * (DBZ_MAX - DBZ_MIN) + DBZ_MIN,
                crop_temporal_valid,
                hidden,
                temporal_age_minutes,
            )
            crop_temporal = normalize_dbz(aligned)
        sample = (
            build_gap_input(
                crop_norm,
                hidden,
                below_norm[window],
                below_valid[window],
                above_norm[window],
                above_valid[window],
                crop_temporal,
                crop_temporal_valid,
                temperature_c[window],
                relative_humidity[window],
                wind_u_ms[window],
                wind_v_ms[window],
                environment_valid[window],
                buoy_values,
                buoy_valid,
                satellite_brightness_k[window],
                satellite_valid[window],
                temporal_age_minutes=temporal_age_minutes,
                mask_temporal_inside_hole=mask_temporal_inside_hole and not align_temporal_motion,
            ),
            crop_norm[None, ...],
            hidden[None, ...].astype(np.float32),
        )
        if int((hidden & (crop_norm >= echo_threshold)).sum()) >= 8:
            return sample
        if fallback is None:
            fallback = sample
    return fallback


def _assert_gap_residual_starts_at_average(device: str) -> None:
    """Before training, a blocked cell copies the available neighbor average.

    Temperature, humidity, and wind are present and must not move that
    starting value. The correction head is still zero.
    """
    model = UNet().to(torch.device(device))
    target = np.linspace(0.0, 30.0, 64, dtype=np.float32).reshape(8, 8)
    below = np.linspace(0.0, 40.0, 64, dtype=np.float32).reshape(8, 8)
    above = np.linspace(10.0, 50.0, 64, dtype=np.float32).reshape(8, 8)
    hidden = np.zeros((8, 8), dtype=bool)
    hidden[:, :4] = True
    below_valid = np.zeros((8, 8), dtype=bool)
    above_valid = np.zeros((8, 8), dtype=bool)
    below_valid[:6, :] = True
    above_valid[2:, :] = True
    temperature = np.linspace(-5.0, 25.0, 64, dtype=np.float32).reshape(8, 8)
    humidity = np.full((8, 8), 0.72, dtype=np.float32)
    wind_u = np.full((8, 8), 12.0, dtype=np.float32)
    wind_v = np.full((8, 8), -4.0, dtype=np.float32)
    environment_valid = np.ones((8, 8), dtype=bool)
    temporal = np.zeros((TEMPORAL_SLOT_COUNT, 8, 8), dtype=np.float32)
    temporal_valid = np.zeros((TEMPORAL_SLOT_COUNT, 8, 8), dtype=bool)
    buoy_values = np.zeros((BUOY_STATION_COUNT, BUOY_FIELD_COUNT), dtype=np.float32)
    buoy_valid = np.zeros_like(buoy_values, dtype=bool)
    satellite_brightness = np.zeros((8, 8), dtype=np.float32)
    satellite_valid = np.zeros((8, 8), dtype=bool)
    stacked = build_gap_input(
        normalize_dbz(target),
        hidden,
        normalize_dbz(below),
        below_valid,
        normalize_dbz(above),
        above_valid,
        temporal,
        temporal_valid,
        temperature,
        humidity,
        wind_u,
        wind_v,
        environment_valid,
        buoy_values,
        buoy_valid,
        satellite_brightness,
        satellite_valid,
    )
    expected_temperature = (temperature - TEMPERATURE_MIN_C) / (TEMPERATURE_MAX_C - TEMPERATURE_MIN_C)
    if stacked.shape[0] != INPUT_CHANNELS or not np.allclose(stacked[TEMPERATURE_CHANNEL], expected_temperature):
        raise RuntimeError("temperature is not the sounding channel")
    if not np.allclose(stacked[HUMIDITY_CHANNEL], humidity):
        raise RuntimeError("humidity is not the sounding channel")
    predicted = predict_dbz_gaps(
        model,
        target,
        hidden,
        below,
        below_valid,
        above,
        above_valid,
        temporal,
        temporal_valid,
        temperature,
        humidity,
        wind_u,
        wind_v,
        environment_valid,
        buoy_values,
        buoy_valid,
        satellite_brightness,
        satellite_valid,
        device=device,
    )
    average = neighbor_average_dbz(below, below_valid, above, above_valid)
    scored = neighbor_score_mask(np.ones((8, 8), dtype=bool), below_valid, above_valid)
    if not np.allclose(predicted[scored], average[scored], rtol=1e-4, atol=1e-3):
        raise RuntimeError("gap fill does not start at the neighbor average")


def train_gapfill_gaps(
    examples: Sequence[GapExample],
    *,
    resolution_km: float,
    epochs: int = DEFAULT_EPOCHS,
    steps_per_epoch: int = DEFAULT_STEPS_PER_EPOCH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    crop: int = 128,
    lr: float = DEFAULT_LR,
    seed: int = TRAIN_SEED,
    device: str | None = None,
    wide_radius_km: float = WIDE_HOLE_RADIUS_KM,
    wide_probability: float = 0.5,
    precip_weight: float = 4.0,
    require_neighbors: bool = True,
    require_environment: bool = True,
    mask_temporal_inside_hole: bool = False,
    align_temporal_motion: bool = False,
    log=print,
) -> TrainResult:
    """Fill holes in a sweep, using the tilts beside it and the sounding.

    The holes are new every step. A missing side is zeros with an all-false
    valid flag. Hidden pixels at or above 20 dBZ count ``precip_weight``
    times. Clear-air hidden pixels stay in the loss. The logged MAE is
    unweighted, in dBZ, on the hidden pixels. Temperature, humidity, and
    wind stay available inside the hole. ``require_neighbors`` is false only
    when an experiment intentionally withholds both vertical tilts.
    ``require_environment`` is false when that experiment also withholds the sounding.
    """
    if crop % 8 != 0:
        raise ValueError("crop must be divisible by 8")
    if len(examples) == 0:
        raise ValueError("examples must not be empty")
    if not 0.0 <= wide_probability <= 1.0:
        raise ValueError("wide_probability must be between 0 and 1")
    if wide_radius_km <= 0 or resolution_km <= 0:
        raise ValueError("wide_radius_km and resolution_km must be positive")
    if align_temporal_motion and not mask_temporal_inside_hole:
        raise ValueError("motion alignment requires a persistent temporal blockage")

    prepared = []
    age_enabled = examples[0].temporal_age_minutes is not None
    if align_temporal_motion and not age_enabled:
        raise ValueError("motion alignment requires two past-scan ages")
    saw_both = False
    saw_below_only = False
    saw_above_only = False
    saw_environment = False
    for example in examples:
        if not isinstance(example, GapExample):
            raise TypeError("examples must be GapExample values")
        if (example.temporal_age_minutes is not None) != age_enabled:
            raise ValueError("all examples must use the same temporal-age feature layout")
        arrays = (
            example.target_dbz,
            example.target_valid,
            example.below_dbz,
            example.below_valid,
            example.above_dbz,
            example.above_valid,
            example.temperature_c,
            example.relative_humidity,
            example.wind_u_ms,
            example.wind_v_ms,
            example.environment_valid,
        )
        shape = example.target_dbz.shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("target, neighbor tilts, and sounding must share a shape")
        temporal_shape = (TEMPORAL_SLOT_COUNT, *shape)
        if (
            example.temporal_dbz.shape != temporal_shape
            or example.temporal_valid.shape != temporal_shape
        ):
            raise ValueError(f"temporal inputs must have shape {temporal_shape}")
        buoy_shape = (BUOY_STATION_COUNT, BUOY_FIELD_COUNT)
        if example.buoy_values.shape != buoy_shape or example.buoy_valid.shape != buoy_shape:
            raise ValueError(f"buoy inputs must have shape {buoy_shape}")
        if (example.satellite_brightness_k is None) != (example.satellite_valid is None):
            raise ValueError("satellite brightness and validity must be supplied together")
        if example.satellite_brightness_k is not None and (
            example.satellite_brightness_k.shape != shape or example.satellite_valid.shape != shape
        ):
            raise ValueError("satellite arrays must share the target shape")
        if shape[0] % 8 != 0 or shape[1] % 8 != 0:
            raise ValueError("grid sides must be divisible by 8")
        if np.shares_memory(example.target_dbz, example.below_dbz) or np.shares_memory(
            example.target_dbz, example.above_dbz
        ):
            raise ValueError("target tilt leaked into a neighbor input")
        target_flag = np.asarray(example.target_valid, dtype=bool)
        below_flag = np.asarray(example.below_valid, dtype=bool)
        above_flag = np.asarray(example.above_valid, dtype=bool)
        temporal_flag = np.asarray(example.temporal_valid, dtype=bool)
        environment_flag = np.asarray(example.environment_valid, dtype=bool)
        if np.any(environment_flag):
            saw_environment = True
            humidity = example.relative_humidity
            if np.any(environment_flag & ((humidity < 0) | (humidity > 1) | ~np.isfinite(humidity))):
                raise ValueError("relative humidity must be a fraction from 0 to 1")
        if np.any(below_flag) and np.any(above_flag):
            saw_both = True
        elif np.any(below_flag):
            saw_below_only = True
        elif np.any(above_flag):
            saw_above_only = True
        elif require_neighbors:
            raise ValueError("a training tilt has no neighbor on either side")
        below_norm = normalize_dbz(np.where(below_flag, example.below_dbz, DBZ_MIN))
        above_norm = normalize_dbz(np.where(above_flag, example.above_dbz, DBZ_MIN))
        temporal_norm = normalize_dbz(
            np.where(temporal_flag, example.temporal_dbz, DBZ_MIN)
        )
        prepared.append(
            (
                normalize_dbz(example.target_dbz),
                target_flag,
                np.where(below_flag, below_norm, 0.0).astype(np.float32),
                below_flag,
                np.where(above_flag, above_norm, 0.0).astype(np.float32),
                above_flag,
                np.where(temporal_flag, temporal_norm, 0.0).astype(np.float32),
                temporal_flag,
                np.asarray(example.temperature_c, dtype=np.float32),
                np.asarray(example.relative_humidity, dtype=np.float32),
                np.asarray(example.wind_u_ms, dtype=np.float32),
                np.asarray(example.wind_v_ms, dtype=np.float32),
                environment_flag,
                np.asarray(example.buoy_values, dtype=np.float32),
                np.asarray(example.buoy_valid, dtype=bool),
                (np.asarray(example.satellite_brightness_k, dtype=np.float32)
                 if example.satellite_brightness_k is not None else np.zeros(shape, np.float32)),
                (np.asarray(example.satellite_valid, dtype=bool)
                 if example.satellite_valid is not None else np.zeros(shape, bool)),
                (np.asarray(example.temporal_age_minutes, dtype=np.float32)
                 if age_enabled else None),
            )
        )
    if require_neighbors and not (saw_both and saw_below_only and saw_above_only):
        raise ValueError("training must include a middle tilt, the lowest tilt, and the highest tilt")
    if require_environment and not saw_environment:
        raise ValueError("training has no temperature, humidity, or wind")

    chosen = device or _default_device()
    _assert_gap_residual_starts_at_average(chosen)
    torch_device = torch.device(chosen)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    model = UNet().to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[float] = []
    choose_grid = len(prepared) > 1
    echo_threshold = _echo_threshold()
    span = DBZ_MAX - DBZ_MIN

    model.train()
    for epoch in range(epochs):
        step_mae: list[float] = []
        step_precip: list[float] = []
        for _ in range(steps_per_epoch):
            inputs: list[np.ndarray] = []
            targets: list[np.ndarray] = []
            masks: list[np.ndarray] = []
            for _ in range(batch_size):
                fields = prepared[int(rng.integers(0, len(prepared)))] if choose_grid else prepared[0]
                sample = _sample_gap_window(
                    *fields,
                    rng,
                    crop=crop,
                    resolution_km=resolution_km,
                    wide_radius_km=wide_radius_km,
                    wide_probability=wide_probability,
                    mask_temporal_inside_hole=mask_temporal_inside_hole,
                    align_temporal_motion=align_temporal_motion,
                )
                if sample is None:
                    continue
                inputs.append(sample[0])
                targets.append(sample[1])
                masks.append(sample[2])
            if len(inputs) < 2:
                continue

            batch_in = torch.from_numpy(np.ascontiguousarray(np.stack(inputs))).to(torch_device)
            batch_target = torch.from_numpy(np.ascontiguousarray(np.stack(targets))).to(torch_device)
            batch_mask = torch.from_numpy(np.ascontiguousarray(np.stack(masks))).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_in)
            loss = _weighted_hidden_l1(prediction, batch_target, batch_mask, precip_weight)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                step_mae.append(float(_masked_l1(prediction, batch_target, batch_mask)) * span)
                precip = batch_mask * (batch_target >= echo_threshold).to(dtype=batch_mask.dtype)
                if float(precip.sum()) > 0:
                    step_precip.append(float(_masked_l1(prediction, batch_target, precip)) * span)

        mean_mae = float(np.mean(step_mae)) if step_mae else float("nan")
        mean_precip = float(np.mean(step_precip)) if step_precip else float("nan")
        history.append(mean_mae)
        if log:
            log(
                f"epoch {epoch + 1}/{epochs}  masked MAE {mean_mae:.2f} dBZ  "
                f">=20 dBZ {mean_precip:.2f}"
            )

    return TrainResult(model=model, history_mae_dbz=history, device=chosen)


@torch.no_grad()
def predict_dbz_gaps(
    model: UNet,
    target_dbz: np.ndarray,
    hidden: np.ndarray,
    below_dbz: np.ndarray,
    below_valid: np.ndarray,
    above_dbz: np.ndarray,
    above_valid: np.ndarray,
    temporal_dbz: np.ndarray,
    temporal_valid: np.ndarray,
    temperature_c: np.ndarray,
    relative_humidity: np.ndarray,
    wind_u_ms: np.ndarray,
    wind_v_ms: np.ndarray,
    environment_valid: np.ndarray,
    buoy_values: np.ndarray,
    buoy_valid: np.ndarray,
    satellite_brightness_k: np.ndarray | None = None,
    satellite_valid: np.ndarray | None = None,
    *,
    device: str,
    temporal_age_minutes: np.ndarray | None = None,
    mask_temporal_inside_hole: bool = False,
    align_temporal_motion: bool = False,
) -> np.ndarray:
    """Fill holes using vertical/temporal neighbors and the sounding."""
    arrays = (
        target_dbz,
        hidden,
        below_dbz,
        below_valid,
        above_dbz,
        above_valid,
        temperature_c,
        relative_humidity,
        wind_u_ms,
        wind_v_ms,
        environment_valid,
    )
    if any(array.shape != target_dbz.shape for array in arrays):
        raise ValueError("target, hole, neighbor tilts, and sounding must share a shape")
    temporal_shape = (TEMPORAL_SLOT_COUNT, *target_dbz.shape)
    if temporal_dbz.shape != temporal_shape or temporal_valid.shape != temporal_shape:
        raise ValueError(f"temporal inputs must have shape {temporal_shape}")
    buoy_shape = (BUOY_STATION_COUNT, BUOY_FIELD_COUNT)
    if buoy_values.shape != buoy_shape or buoy_valid.shape != buoy_shape:
        raise ValueError(f"buoy inputs must have shape {buoy_shape}")
    model.eval()
    below_flag = np.asarray(below_valid, dtype=bool)
    above_flag = np.asarray(above_valid, dtype=bool)
    below_norm = normalize_dbz(np.where(below_flag, below_dbz, DBZ_MIN))
    above_norm = normalize_dbz(np.where(above_flag, above_dbz, DBZ_MIN))
    temporal_flag = np.asarray(temporal_valid, dtype=bool)
    if align_temporal_motion:
        if not mask_temporal_inside_hole:
            raise ValueError("motion alignment requires a persistent temporal blockage")
        temporal_dbz, temporal_flag, _ = align_past_scans(
            temporal_dbz, temporal_flag, hidden, temporal_age_minutes
        )
    temporal_norm = normalize_dbz(np.where(temporal_flag, temporal_dbz, DBZ_MIN))
    inputs = build_gap_input(
        normalize_dbz(target_dbz),
        np.asarray(hidden, dtype=bool),
        below_norm,
        below_flag,
        above_norm,
        above_flag,
        temporal_norm,
        temporal_flag,
        temperature_c,
        relative_humidity,
        wind_u_ms,
        wind_v_ms,
        environment_valid,
        buoy_values,
        buoy_valid,
        satellite_brightness_k,
        satellite_valid,
        temporal_age_minutes=temporal_age_minutes,
        mask_temporal_inside_hole=mask_temporal_inside_hole and not align_temporal_motion,
    )[None, ...]
    tensor = torch.from_numpy(np.ascontiguousarray(inputs)).to(torch.device(device))
    predicted = model(tensor)[0, 0].detach().cpu().numpy()
    return denormalize_dbz(predicted)
