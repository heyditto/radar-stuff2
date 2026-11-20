"""Retrieve past storm analogs using only currently visible radar pixels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter, shift as shift_image

from radar.motion import estimate_translation


@dataclass(frozen=True)
class Analog:
    name: str
    day: str
    current_dbz: np.ndarray
    current_valid: np.ndarray
    previous_dbz: np.ndarray
    previous_valid: np.ndarray
    age_minutes: float


def _warp(values: np.ndarray, valid: np.ndarray, movement: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    flag = shift_image(valid.astype(np.uint8), movement, order=0, mode="constant", cval=0) > 0
    data = shift_image(np.where(valid, values, 0.0), movement, order=0, mode="constant", cval=0.0)
    return data.astype(np.float32), flag


def _echo_distance(
    query: np.ndarray,
    query_valid: np.ndarray,
    candidate: np.ndarray,
    candidate_valid: np.ndarray,
    *,
    hidden: np.ndarray,
) -> float:
    visible = query_valid & candidate_valid & ~hidden
    if visible.sum() < 1000:
        return float("inf")
    # Smooth range-gate noise while retaining the storm's broad structure.
    q = gaussian_filter(np.where(query_valid & ~hidden, query, 0.0), 2.0)
    c = gaussian_filter(np.where(candidate_valid & ~hidden, candidate, 0.0), 2.0)
    echo = visible & ((q >= 10.0) | (c >= 10.0))
    if echo.sum() < 100:
        return float("inf")
    return float(np.mean(np.abs(q[echo] - c[echo])))


def retrieve_analogs(
    query_current: np.ndarray,
    query_current_valid: np.ndarray,
    query_previous: np.ndarray,
    query_previous_valid: np.ndarray,
    query_age_minutes: float,
    hidden: np.ndarray,
    library: list[Analog],
    *,
    temporal: bool,
    k: int = 3,
    max_shift_px: int = 32,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Select K historical storms and transfer their current blocked pixels.

    Candidate alignment and ranking see only visible query and candidate
    pixels. Their full current fields are read only after ranking to supply
    historical outcomes at the blocked coordinates.
    """
    shape = query_current.shape
    if any(a.shape != shape for a in (query_current_valid, query_previous, query_previous_valid, hidden)):
        raise ValueError("query fields and mask must share a grid")
    if k < 1 or not library:
        raise ValueError("a nonempty analog library and positive k are required")
    if query_age_minutes <= 0:
        raise ValueError("query past-scan age must be positive")
    query_visible = np.asarray(query_current_valid, bool) & ~hidden
    past_visible = np.asarray(query_previous_valid, bool) & ~hidden
    matches = []
    for analog in library:
        if temporal and abs(analog.age_minutes - query_age_minutes) > 3.0:
            continue
        if any(a.shape != shape for a in (analog.current_dbz, analog.current_valid, analog.previous_dbz, analog.previous_valid)):
            raise ValueError(f"analog {analog.name} has a different grid")
        candidate_visible = np.asarray(analog.current_valid, bool) & ~hidden
        movement = estimate_translation(
            analog.current_dbz, candidate_visible,
            query_current, query_visible, max_shift_px=max_shift_px,
        )
        # Apply the mask before shifting for matching. Historical gap values
        # cannot influence either the translation or similarity score.
        current_masked, current_flag = _warp(analog.current_dbz, candidate_visible, movement)
        static_score = _echo_distance(
            query_current, query_visible, current_masked, current_flag, hidden=hidden
        )
        if not np.isfinite(static_score):
            continue
        past_score = 0.0
        if temporal:
            previous_masked, previous_flag = _warp(
                analog.previous_dbz, np.asarray(analog.previous_valid, bool) & ~hidden, movement
            )
            # Comparing both frames gives the analog search an explicit view
            # of recent evolution. Time gaps differ slightly, so the score
            # does not claim that pixelwise changes have identical cadence.
            past_score = _echo_distance(
                query_previous, past_visible, previous_masked, previous_flag, hidden=hidden
            )
            if not np.isfinite(past_score):
                continue
        matches.append((static_score + past_score, analog, movement, static_score, past_score))
    matches.sort(key=lambda row: (row[0], row[1].name))
    selected = matches[:k]
    if not selected:
        raise RuntimeError("no analog has enough visible echo overlap")
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    details = []
    for score, analog, movement, static_score, past_score in selected:
        full, flag = _warp(analog.current_dbz, analog.current_valid, movement)
        weight = 1.0 / max(score, 0.1)
        numerator += np.where(flag, full * weight, 0.0)
        denominator += flag * weight
        details.append({
            "scan": analog.name,
            "day": analog.day,
            "score_dbz": score,
            "current_score_dbz": static_score,
            "previous_score_dbz": past_score,
            "shift_row_px": movement[0],
            "shift_col_px": movement[1],
        })
    prediction = np.where(denominator > 0, numerator / np.maximum(denominator, 1e-8), np.nan)
    return prediction.astype(np.float32), denominator > 0, details
