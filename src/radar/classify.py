"""Label hidden pixels as estimated or uncertain from distance to real data."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from radar.masks import ESTIMATED_WITHIN_KM

OBSERVED = 0
ESTIMATED = 1
UNCERTAIN = 2
NODATA = -1

CLASS_NAMES = {
    OBSERVED: "observed",
    ESTIMATED: "estimated",
    UNCERTAIN: "uncertain",
}


def classify_pixels(
    hidden: np.ndarray,
    valid: np.ndarray,
    *,
    resolution_km: float,
    estimated_within_km: float = ESTIMATED_WITHIN_KM,
) -> np.ndarray:
    """Label each pixel as observed, estimated, uncertain, or nodata.

    Valid pixels that were not hidden are observed. Hidden pixels within
    ``estimated_within_km`` of an observed pixel are estimated. Hidden pixels
    farther away are uncertain. Cells the radar never sampled stay nodata
    (``-1``), which is separate from a synthetic hole.
    """
    if hidden.shape != valid.shape:
        raise ValueError("hidden and valid must have the same shape")
    if resolution_km <= 0:
        raise ValueError("resolution_km must be positive")

    labels = np.full(hidden.shape, NODATA, dtype=np.int8)
    observed = valid & ~hidden
    labels[observed] = OBSERVED
    hidden_valid = hidden & valid
    if not np.any(hidden_valid):
        return labels
    if not np.any(observed):
        labels[hidden_valid] = UNCERTAIN
        return labels

    # Distance is 0 on observed pixels and grows toward the nearest one.
    distance_km = distance_transform_edt(
        np.where(observed, 0, 1),
        sampling=resolution_km,
    )
    close = hidden_valid & (distance_km <= estimated_within_km)
    far = hidden_valid & (distance_km > estimated_within_km)
    labels[close] = ESTIMATED
    labels[far] = UNCERTAIN
    return labels


def nearest_observed_fill(
    dbz: np.ndarray,
    hidden: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Copy the nearest observed reflectivity into each hidden valid pixel.

    This is the baseline the network has to beat. It uses the true neighbors,
    not a learned prior.
    """
    filled = np.array(dbz, dtype=np.float32, copy=True)
    observed = valid & ~hidden
    holes = hidden & valid
    if not np.any(holes) or not np.any(observed):
        return filled
    _distance, indices = distance_transform_edt(
        np.where(observed, 0, 1),
        return_indices=True,
    )
    nearest = dbz[indices[0], indices[1]]
    filled[holes] = nearest[holes]
    return filled
